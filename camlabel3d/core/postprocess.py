"""Postprocessing session helpers for raw/latest CSV workflows."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from camlabel3d.io.csv_store import CSVStore

from .models import DetectionRecord, natural_sort_key


class WorkflowStage(str, Enum):
    """Editing stage for the current source."""

    DETECTION = "Detection stage"
    POSTPROCESSING = "Postprocessing stage"


@dataclass(frozen=True)
class FilterConfig:
    """Global filter thresholds applied during postprocessing."""

    min_score: float = 0.0
    min_score_3d: float = 0.0
    max_center_z: float = 0.0
    max_range_xz: float = 0.0

    def has_active_threshold(self) -> bool:
        return any(
            value > 0.0
            for value in (
                float(self.min_score),
                float(self.min_score_3d),
                float(self.max_center_z),
                float(self.max_range_xz),
            )
        )

    def matches(self, record: DetectionRecord) -> bool:
        if float(self.min_score) > 0.0 and record.score < float(self.min_score):
            return False
        if float(self.min_score_3d) > 0.0 and record.score_3d < float(self.min_score_3d):
            return False
        if float(self.max_center_z) > 0.0 and record.center_z > float(self.max_center_z):
            return False
        if float(self.max_range_xz) > 0.0:
            range_xz = (record.center_x ** 2 + record.center_z ** 2) ** 0.5
            if range_xz > float(self.max_range_xz):
                return False
        return True


@dataclass(frozen=True)
class TrackSummary:
    """Track-level summary used by the Track Manager table."""

    track_id: str
    category: str
    enabled_count: int
    first_frame: int
    last_frame: int
    status: str


def clone_records(records: list[DetectionRecord]) -> list[DetectionRecord]:
    """Deep-copy the current record list for undo snapshots."""

    return [replace(record) for record in records]


class PostprocessSession:
    """Tracks raw/latest paths, stage, and undo state for one source."""

    def __init__(self, undo_limit: int = 20) -> None:
        self.raw_path: Path | None = None
        self.latest_path: Path | None = None
        self.active_path: Path | None = None
        self.stage = WorkflowStage.DETECTION
        self._undo_stack: list[list[DetectionRecord]] = []
        self.undo_limit = max(1, int(undo_limit))
        self._track_unlock_restore: dict[str, str] = {}

    @staticmethod
    def derive_latest_path(raw_path: str | Path) -> Path:
        raw = Path(raw_path).resolve()
        suffix = ".camlabel3d.csv"
        if raw.name.endswith(suffix):
            stem = raw.name[: -len(suffix)]
            return raw.with_name(f"{stem}.latest{suffix}")
        return raw.with_name(f"{raw.stem}.latest{raw.suffix}")

    @staticmethod
    def derive_raw_path(csv_path: str | Path) -> Path:
        csv = Path(csv_path).resolve()
        latest_suffix = ".latest.camlabel3d.csv"
        if csv.name.endswith(latest_suffix):
            stem = csv.name[: -len(latest_suffix)]
            return csv.with_name(f"{stem}.camlabel3d.csv")
        return csv

    @classmethod
    def is_latest_csv_path(cls, csv_path: str | Path) -> bool:
        csv = Path(csv_path).resolve()
        return csv == cls.derive_latest_path(cls.derive_raw_path(csv))

    def clear(self) -> None:
        self.raw_path = None
        self.latest_path = None
        self.active_path = None
        self.stage = WorkflowStage.DETECTION
        self.clear_undo()

    def configure_source(self, raw_path: str | Path) -> tuple[WorkflowStage, Path]:
        base_raw_path = Path(raw_path).resolve()
        self.clear_undo()
        self.raw_path = base_raw_path
        self.latest_path = self.derive_latest_path(self.raw_path)
        self.active_path = self.raw_path
        self.stage = WorkflowStage.DETECTION
        return self.stage, self.active_path

    def activate(
        self,
        raw_path: str | Path,
        selected_csv_path: str | Path | None = None,
    ) -> tuple[WorkflowStage, Path, list[DetectionRecord]]:
        stage, active_path = self.configure_activation(raw_path, selected_csv_path)
        records = self._load_from(active_path) if active_path.exists() else []
        return stage, active_path, records

    def configure_activation(
        self,
        raw_path: str | Path,
        selected_csv_path: str | Path | None = None,
    ) -> tuple[WorkflowStage, Path]:
        """Configure workflow paths without parsing the potentially large CSV."""

        base_raw_path = Path(raw_path).resolve()
        self.clear_undo()

        if selected_csv_path is not None:
            selected_path = Path(selected_csv_path).resolve()
            if not selected_path.exists():
                raise FileNotFoundError(f"Annotation CSV not found: {selected_path}")
            if self.is_latest_csv_path(selected_path):
                self.raw_path = self.derive_raw_path(selected_path)
                self.latest_path = selected_path
                self.stage = WorkflowStage.POSTPROCESSING
            else:
                self.raw_path = selected_path
                self.latest_path = self.derive_latest_path(selected_path)
                self.stage = WorkflowStage.DETECTION
            self.active_path = selected_path
            return self.stage, self.active_path

        self.raw_path = base_raw_path
        self.latest_path = self.derive_latest_path(self.raw_path)
        if self.latest_path.exists():
            self.stage = WorkflowStage.POSTPROCESSING
            self.active_path = self.latest_path
        else:
            self.stage = WorkflowStage.DETECTION
            self.active_path = self.raw_path
        return self.stage, self.active_path

    def latest_exists(self) -> bool:
        return bool(self.latest_path and self.latest_path.exists())

    def can_start_postprocessing(self, records: list[DetectionRecord]) -> bool:
        if self.stage == WorkflowStage.POSTPROCESSING:
            return False
        if self.raw_path is None:
            return False
        return bool(records)

    def start_postprocessing(self, records: list[DetectionRecord]) -> tuple[Path, list[DetectionRecord]]:
        if self.raw_path is None or self.latest_path is None:
            raise ValueError("No active source is loaded.")
        if not self.raw_path.exists():
            if not records:
                raise ValueError("Run detection first so the raw CSV exists.")
            CSVStore(self.raw_path, backup_enabled=False).save_records(records)
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.raw_path, self.latest_path)
        active_path = self.commit_postprocessing(clear_undo=True)
        return active_path, self._load_from(active_path)

    def reset_to_raw(self) -> tuple[Path, list[DetectionRecord]]:
        if self.raw_path is None or self.latest_path is None:
            raise ValueError("No active source is loaded.")
        if not self.raw_path.exists():
            raise ValueError("The raw detection CSV does not exist yet.")
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.raw_path, self.latest_path)
        active_path = self.commit_postprocessing(clear_undo=False)
        return active_path, self._load_from(active_path)

    def postprocessing_paths(self) -> tuple[Path, Path]:
        """Return a stable raw/latest path pair without changing session state."""

        if self.raw_path is None or self.latest_path is None:
            raise ValueError("No active source is loaded.")
        return self.raw_path.resolve(), self.latest_path.resolve()

    def commit_postprocessing(self, *, clear_undo: bool) -> Path:
        """Commit a completed external CSV transition on the owning thread."""

        if self.latest_path is None:
            raise ValueError("No active source is loaded.")
        self.stage = WorkflowStage.POSTPROCESSING
        self.active_path = self.latest_path
        if clear_undo:
            self.clear_undo()
        return self.active_path

    def save_records(self, records: list[DetectionRecord]) -> Path:
        path, backup_enabled = self.save_target()
        CSVStore(path, backup_enabled=backup_enabled).save_records(records)
        return path

    def save_target(self) -> tuple[Path, bool]:
        """Return the active path and backup policy without performing I/O."""

        if self.active_path is None:
            raise ValueError("No active CSV is available for saving.")
        return self.active_path, self.stage == WorkflowStage.POSTPROCESSING

    def push_undo_snapshot(
        self,
        snapshot: list[DetectionRecord],
        *,
        snapshot_owned: bool = False,
    ) -> None:
        """Store a bounded snapshot, optionally taking ownership of a clone."""

        self._undo_stack.append(list(snapshot) if snapshot_owned else clone_records(snapshot))
        overflow = len(self._undo_stack) - self.undo_limit
        if overflow > 0:
            del self._undo_stack[:overflow]

    def has_undo(self) -> bool:
        return bool(self._undo_stack)

    def undo(self) -> list[DetectionRecord]:
        if not self._undo_stack:
            raise ValueError("Nothing to undo.")
        return self._undo_stack.pop()

    def clear_undo(self) -> None:
        self._undo_stack.clear()
        self._track_unlock_restore.clear()

    def apply_filter(self, records: list[DetectionRecord], config: FilterConfig) -> int:
        disabled = 0
        for record in records:
            if not record.is_enabled:
                continue
            if not config.matches(record):
                record.is_enabled = False
                record.is_visible = False
                disabled += 1
        return disabled

    def delete_track(self, records: list[DetectionRecord], track_id: str) -> int:
        track_id = str(track_id).strip()
        if not track_id:
            raise ValueError("Select a non-empty track first.")
        disabled = 0
        for record in records:
            if record.track_id == track_id and record.is_enabled:
                record.is_enabled = False
                record.is_visible = False
                disabled += 1
        if disabled == 0:
            raise ValueError(f"Track '{track_id}' has no enabled detections.")
        return disabled

    def merge_tracks(self, records: list[DetectionRecord], source_track_id: str, target_track_id: str) -> int:
        source_track_id = str(source_track_id).strip()
        target_track_id = str(target_track_id).strip()
        if not source_track_id or not target_track_id:
            raise ValueError("Both source and target track IDs are required.")
        if source_track_id == target_track_id:
            raise ValueError("Choose two different track IDs to merge.")

        source_records = [
            record
            for record in records
            if record.track_id == source_track_id and record.is_enabled
        ]
        target_records = [
            record
            for record in records
            if record.track_id == target_track_id and record.is_enabled
        ]
        if not source_records:
            raise ValueError(f"Track '{source_track_id}' has no enabled detections.")
        if not target_records:
            raise ValueError(f"Track '{target_track_id}' has no enabled detections.")

        source_categories = {record.category for record in source_records}
        target_categories = {record.category for record in target_records}
        if len(source_categories) != 1 or len(target_categories) != 1:
            raise ValueError("Tracks with mixed categories cannot be merged.")
        if next(iter(source_categories)) != next(iter(target_categories)):
            raise ValueError("Only tracks from the same category can be merged.")

        source_frames = {record.frame_index for record in source_records}
        target_frames = {record.frame_index for record in target_records}
        overlap = sorted(source_frames & target_frames)
        if overlap:
            raise ValueError(
                f"Tracks overlap on frames {overlap[:5]}; disable conflicting rows before merging."
            )

        updated = 0
        for record in records:
            if record.track_id == source_track_id and record.is_enabled:
                record.track_id = target_track_id
                record.track_status = "manual" if target_track_id else ""
                updated += 1
        return updated

    def lock_track(self, records: list[DetectionRecord], track_id: str) -> int:
        track_id = str(track_id).strip()
        if not track_id:
            raise ValueError("Select a non-empty track first.")
        affected = [
            record
            for record in records
            if record.track_id == track_id and record.is_enabled
        ]
        if not affected:
            raise ValueError(f"Track '{track_id}' has no enabled detections.")

        prior_status = "auto"
        for record in affected:
            if record.track_status in {"manual", "auto"}:
                prior_status = record.track_status
                break
        self._track_unlock_restore[track_id] = prior_status
        for record in affected:
            record.track_status = "locked"
        return len(affected)

    def unlock_track(self, records: list[DetectionRecord], track_id: str) -> int:
        track_id = str(track_id).strip()
        if not track_id:
            raise ValueError("Select a non-empty track first.")
        affected = [
            record
            for record in records
            if record.track_id == track_id and record.is_enabled and record.track_status == "locked"
        ]
        if not affected:
            raise ValueError(f"Track '{track_id}' is not locked.")
        restore_status = self._track_unlock_restore.get(track_id, "auto")
        for record in affected:
            record.track_status = restore_status if record.track_id else ""
        return len(affected)

    def build_track_summaries(self, records: list[DetectionRecord]) -> list[TrackSummary]:
        grouped: dict[str, list[DetectionRecord]] = {}
        for record in records:
            if not record.is_enabled or not record.track_id.strip():
                continue
            grouped.setdefault(record.track_id.strip(), []).append(record)

        summaries: list[TrackSummary] = []
        for track_id, group in grouped.items():
            frames = sorted(record.frame_index for record in group)
            categories = sorted({record.category for record in group})
            statuses = {record.track_status.strip() for record in group if record.track_status.strip()}
            if "locked" in statuses:
                status = "locked"
            elif statuses == {"manual"}:
                status = "manual"
            elif statuses == {"auto"}:
                status = "auto"
            elif not statuses:
                status = ""
            else:
                status = "mixed"
            summaries.append(
                TrackSummary(
                    track_id=track_id,
                    category=categories[0] if len(categories) == 1 else "<mixed>",
                    enabled_count=len(group),
                    first_frame=frames[0],
                    last_frame=frames[-1],
                    status=status,
                )
            )

        return sorted(summaries, key=lambda item: natural_sort_key(item.track_id))

    @staticmethod
    def _load_from(path: Path) -> list[DetectionRecord]:
        return CSVStore(path).load_records()
