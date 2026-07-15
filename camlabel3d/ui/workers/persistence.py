"""Single-writer, per-path coalescing CSV persistence queue."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Condition
from time import monotonic

from PySide6.QtCore import QThread, Signal

from camlabel3d.core.models import DetectionRecord
from camlabel3d.io.csv_store import CSVStore


class SaveStatus(str, Enum):
    """Lifecycle state for one exact ``(path, revision)`` save request."""

    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        """Whether no further state transition will occur for this result."""

        return self in {
            SaveStatus.SUCCEEDED,
            SaveStatus.FAILED,
            SaveStatus.SUPERSEDED,
            SaveStatus.CANCELLED,
            SaveStatus.REJECTED,
            SaveStatus.UNKNOWN,
        }


@dataclass(frozen=True, slots=True)
class SaveSubmission:
    """Immediate acknowledgement returned by :meth:`CSVSaveWorker.submit`."""

    path: Path
    revision: int
    accepted: bool
    status: SaveStatus
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SaveResult:
    """Observed state or terminal outcome for an exact save revision."""

    path: Path
    revision: int
    status: SaveStatus
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status is SaveStatus.SUCCEEDED


@dataclass(slots=True)
class _SaveRequest:
    path: Path
    records: list[DetectionRecord]
    revision: int
    backup_enabled: bool

    @property
    def key(self) -> tuple[str, int]:
        return (str(self.path), self.revision)


class CSVSaveWorker(QThread):
    """Serialize saves and retain only the newest pending revision per path.

    The queue and completed-result history are bounded. Callers that need a
    durability decision should retain the :class:`SaveSubmission` returned by
    :meth:`submit` and pass its path/revision to :meth:`wait_for_revision`.

    ``flush`` remains a queue-wide compatibility API. In addition to waiting
    for the queue to drain, it returns ``False`` for I/O failures or cancelled
    accepted requests not reported by an earlier completed ``flush`` call.
    """

    saveCompleted = Signal(str, int)
    saveFailed = Signal(str, int, str)

    def __init__(
        self,
        parent=None,
        *,
        history_limit: int = 256,
        max_pending_paths: int = 64,
    ) -> None:
        super().__init__(parent)
        if history_limit < 1:
            raise ValueError("history_limit must be at least 1")
        if max_pending_paths < 1:
            raise ValueError("max_pending_paths must be at least 1")
        self._condition = Condition()
        self._pending: OrderedDict[str, _SaveRequest] = OrderedDict()
        self._active: _SaveRequest | None = None
        self._stopping = False
        self._history_limit = int(history_limit)
        self._max_pending_paths = int(max_pending_paths)
        self._results: OrderedDict[tuple[str, int], SaveResult] = OrderedDict()
        self._waiters: dict[tuple[str, int], int] = {}
        self._failure_sequence = 0
        self._acknowledged_failure_sequence = 0

    def submit(
        self,
        path: str | Path,
        records: list[DetectionRecord],
        revision: int,
        *,
        backup_enabled: bool = True,
    ) -> SaveSubmission:
        """Queue a snapshot and return an explicit acceptance acknowledgement."""

        resolved = Path(path).resolve()
        normalized_revision = int(revision)
        request = _SaveRequest(
            resolved,
            records,
            normalized_revision,
            bool(backup_enabled),
        )
        path_key = str(resolved)
        with self._condition:
            if self._stopping:
                return self._reject_locked(request, "save worker is stopping")

            current = self._state_locked(request.key)
            if current.status in {SaveStatus.PENDING, SaveStatus.ACTIVE}:
                return SaveSubmission(
                    path=resolved,
                    revision=normalized_revision,
                    accepted=False,
                    status=current.status,
                    detail="this path/revision is already queued",
                )

            previous = self._pending.get(path_key)
            if previous is None and len(self._pending) >= self._max_pending_paths:
                return self._reject_locked(request, "pending save queue is full")

            # Reusing a revision after a terminal result starts a new lifecycle.
            self._results.pop(request.key, None)
            if previous is not None:
                self._remember_result_locked(
                    SaveResult(
                        path=previous.path,
                        revision=previous.revision,
                        status=SaveStatus.SUPERSEDED,
                        detail=f"replaced by revision {normalized_revision}",
                    )
                )

            self._pending[path_key] = request
            self._pending.move_to_end(path_key)
            self._condition.notify_all()
            return SaveSubmission(
                path=resolved,
                revision=normalized_revision,
                accepted=True,
                status=SaveStatus.PENDING,
            )

    def status_for_revision(self, path: str | Path, revision: int) -> SaveResult:
        """Return the current state without waiting."""

        resolved = Path(path).resolve()
        with self._condition:
            return self._state_locked((str(resolved), int(revision)))

    def wait_for_revision(
        self,
        path: str | Path,
        revision: int,
        timeout_ms: int = 5000,
    ) -> SaveResult:
        """Wait for an exact revision, distinguishing all terminal outcomes."""

        resolved = Path(path).resolve()
        normalized_revision = int(revision)
        key = (str(resolved), normalized_revision)
        deadline = monotonic() + max(0, int(timeout_ms)) / 1000.0
        with self._condition:
            state = self._state_locked(key)
            if state.status.terminal:
                return state

            self._waiters[key] = self._waiters.get(key, 0) + 1
            try:
                while True:
                    state = self._state_locked(key)
                    if state.status.terminal:
                        return state
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return SaveResult(
                            path=resolved,
                            revision=normalized_revision,
                            status=SaveStatus.TIMEOUT,
                            detail=f"save did not finish within {max(0, int(timeout_ms))} ms",
                        )
                    self._condition.wait(remaining)
            finally:
                remaining_waiters = self._waiters[key] - 1
                if remaining_waiters:
                    self._waiters[key] = remaining_waiters
                else:
                    self._waiters.pop(key, None)
                self._prune_results_locked()

    def flush(self, timeout_ms: int = 5000) -> bool:
        """Drain the queue and report unacknowledged failures accurately.

        A completed call acknowledges failures that made it return ``False``;
        this prevents one old failure from making every future flush fail. A
        timeout never acknowledges failures.
        """

        deadline = monotonic() + max(0, int(timeout_ms)) / 1000.0
        with self._condition:
            observed_acknowledgement = self._acknowledged_failure_sequence
            while self._active is not None or self._pending:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)

            failure_sequence = self._failure_sequence
            failed = failure_sequence > observed_acknowledgement
            if failed:
                self._acknowledged_failure_sequence = max(
                    self._acknowledged_failure_sequence,
                    failure_sequence,
                )
            return not failed

    def stop(self, *, discard_pending: bool = False) -> None:
        """Reject future submissions and optionally cancel queued requests."""

        with self._condition:
            self._stopping = True
            if discard_pending:
                cancelled = list(self._pending.values())
                self._pending.clear()
                for request in cancelled:
                    self._remember_result_locked(
                        SaveResult(
                            path=request.path,
                            revision=request.revision,
                            status=SaveStatus.CANCELLED,
                            detail="pending save discarded while stopping",
                        )
                    )
                    self._failure_sequence += 1
            self._condition.notify_all()

    def run(self) -> None:  # noqa: D401, N802 - Qt naming
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._pending:
                    return
                _, request = self._pending.popitem(last=False)
                self._active = request
                self._condition.notify_all()

            error = ""
            try:
                CSVStore(request.path, backup_enabled=request.backup_enabled).save_records(
                    request.records
                )
                status = SaveStatus.SUCCEEDED
            except Exception as exc:
                status = SaveStatus.FAILED
                error = str(exc)

            with self._condition:
                self._active = None
                self._remember_result_locked(
                    SaveResult(
                        path=request.path,
                        revision=request.revision,
                        status=status,
                        detail=error,
                    )
                )
                if status is SaveStatus.FAILED:
                    self._failure_sequence += 1
                self._condition.notify_all()

            if status is SaveStatus.SUCCEEDED:
                self.saveCompleted.emit(str(request.path), request.revision)
            else:
                self.saveFailed.emit(str(request.path), request.revision, error)

    def _reject_locked(self, request: _SaveRequest, detail: str) -> SaveSubmission:
        result = SaveResult(
            path=request.path,
            revision=request.revision,
            status=SaveStatus.REJECTED,
            detail=detail,
        )
        self._remember_result_locked(result)
        self._condition.notify_all()
        return SaveSubmission(
            path=request.path,
            revision=request.revision,
            accepted=False,
            status=SaveStatus.REJECTED,
            detail=detail,
        )

    def _state_locked(self, key: tuple[str, int]) -> SaveResult:
        path_key, revision = key
        if self._active is not None and self._active.key == key:
            return SaveResult(self._active.path, revision, SaveStatus.ACTIVE)
        pending = self._pending.get(path_key)
        if pending is not None and pending.key == key:
            return SaveResult(pending.path, revision, SaveStatus.PENDING)
        result = self._results.get(key)
        if result is not None:
            self._results.move_to_end(key)
            return result
        return SaveResult(Path(path_key), revision, SaveStatus.UNKNOWN)

    def _remember_result_locked(self, result: SaveResult) -> None:
        key = (str(result.path), result.revision)
        self._results[key] = result
        self._results.move_to_end(key)
        self._prune_results_locked()

    def _prune_results_locked(self) -> None:
        while len(self._results) > self._history_limit:
            removable = next(
                (key for key in self._results if not self._waiters.get(key)),
                None,
            )
            if removable is None:
                return
            self._results.pop(removable, None)
