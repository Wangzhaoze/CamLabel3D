from __future__ import annotations

from pathlib import Path
from threading import Event, Lock

import pytest

pytest.importorskip("PySide6.QtCore")

from camlabel3d.core.models import DetectionRecord
from camlabel3d.ui.workers import persistence
from camlabel3d.ui.workers.persistence import CSVSaveWorker, SaveStatus


def _record(category: str = "car") -> DetectionRecord:
    return DetectionRecord(
        frame_index=0,
        category=category,
        score=0.9,
        score_2d=0.9,
        score_3d=0.8,
        box2d_x1=1.0,
        box2d_y1=2.0,
        box2d_x2=10.0,
        box2d_y2=20.0,
        center_x=0.0,
        center_y=0.0,
        center_z=10.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=2.0,
        size_l=4.0,
        size_h=1.5,
    )


def _stop(worker: CSVSaveWorker) -> None:
    worker.stop(discard_pending=True)
    assert worker.wait(2_000)


def test_submit_and_wait_for_success(tmp_path: Path) -> None:
    worker = CSVSaveWorker()
    worker.start()
    try:
        target = tmp_path / "annotations.csv"
        submission = worker.submit(target, [_record()], revision=17)

        assert submission.accepted
        assert submission.revision == 17
        assert submission.status is SaveStatus.PENDING

        result = worker.wait_for_revision(
            submission.path,
            submission.revision,
            timeout_ms=2_000,
        )
        assert result.status is SaveStatus.SUCCEEDED
        assert result.succeeded
        assert target.is_file()
        assert worker.flush(timeout_ms=100)
    finally:
        _stop(worker)


def test_failed_revision_is_reported_and_flush_does_not_claim_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BrokenCSVStore:
        def __init__(self, path: Path, *, backup_enabled: bool) -> None:
            del path, backup_enabled

        def save_records(self, records: list[DetectionRecord]) -> None:
            del records
            raise OSError("disk is full")

    monkeypatch.setattr(persistence, "CSVStore", BrokenCSVStore)
    worker = CSVSaveWorker()
    worker.start()
    try:
        submission = worker.submit(tmp_path / "broken.csv", [_record()], revision=3)
        result = worker.wait_for_revision(
            submission.path,
            submission.revision,
            timeout_ms=2_000,
        )

        assert result.status is SaveStatus.FAILED
        assert "disk is full" in result.detail
        # A failure remains visible even if the queue drained before flush began.
        assert not worker.flush(timeout_ms=100)
        # A completed flush acknowledges it; unrelated future flushes can succeed.
        assert worker.flush(timeout_ms=100)
    finally:
        _stop(worker)


def test_pending_revisions_are_coalesced_with_explicit_superseded_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    saved_categories: list[str] = []
    saved_lock = Lock()

    class BlockingCSVStore:
        def __init__(self, path: Path, *, backup_enabled: bool) -> None:
            del path, backup_enabled

        def save_records(self, records: list[DetectionRecord]) -> None:
            entered.set()
            assert release.wait(2.0)
            with saved_lock:
                saved_categories.append(records[0].category)

    monkeypatch.setattr(persistence, "CSVStore", BlockingCSVStore)
    worker = CSVSaveWorker()
    worker.start()
    target = tmp_path / "coalesced.csv"
    try:
        first = worker.submit(target, [_record("first")], revision=1)
        assert entered.wait(2.0)

        second = worker.submit(target, [_record("second")], revision=2)
        third = worker.submit(target, [_record("third")], revision=3)
        assert first.accepted and second.accepted and third.accepted

        superseded = worker.wait_for_revision(target, revision=2, timeout_ms=100)
        assert superseded.status is SaveStatus.SUPERSEDED
        assert "revision 3" in superseded.detail

        release.set()
        assert worker.wait_for_revision(target, 1, 2_000).succeeded
        assert worker.wait_for_revision(target, 3, 2_000).succeeded
        assert worker.flush(timeout_ms=2_000)
        assert saved_categories == ["first", "third"]
    finally:
        release.set()
        _stop(worker)


def test_timeout_cancel_and_submit_after_stop_are_distinct(tmp_path: Path) -> None:
    worker = CSVSaveWorker()
    target = tmp_path / "stopped.csv"

    queued = worker.submit(target, [_record()], revision=1)
    assert queued.accepted
    assert worker.wait_for_revision(target, 1, timeout_ms=1).status is SaveStatus.TIMEOUT

    worker.stop(discard_pending=True)
    cancelled = worker.wait_for_revision(target, 1, timeout_ms=100)
    assert cancelled.status is SaveStatus.CANCELLED
    assert not worker.flush(timeout_ms=100)

    rejected = worker.submit(target, [_record()], revision=2)
    assert not rejected.accepted
    assert rejected.status is SaveStatus.REJECTED
    assert "stopping" in rejected.detail
    assert worker.wait_for_revision(target, 2, timeout_ms=100).status is SaveStatus.REJECTED


def test_completed_result_history_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class NoopCSVStore:
        def __init__(self, path: Path, *, backup_enabled: bool) -> None:
            del path, backup_enabled

        def save_records(self, records: list[DetectionRecord]) -> None:
            del records

    monkeypatch.setattr(persistence, "CSVStore", NoopCSVStore)
    worker = CSVSaveWorker(history_limit=2)
    worker.start()
    try:
        targets = [tmp_path / f"{index}.csv" for index in range(3)]
        for revision, target in enumerate(targets):
            worker.submit(target, [_record()], revision)
            assert worker.wait_for_revision(target, revision, 2_000).succeeded

        assert worker.status_for_revision(targets[0], 0).status is SaveStatus.UNKNOWN
        assert worker.status_for_revision(targets[1], 1).status is SaveStatus.SUCCEEDED
        assert worker.status_for_revision(targets[2], 2).status is SaveStatus.SUCCEEDED
    finally:
        _stop(worker)
