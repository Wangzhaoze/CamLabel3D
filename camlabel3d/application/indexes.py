"""Indexed, reusable read models for annotation and outlier presentation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from camlabel3d.core.models import DetectionRecord
from camlabel3d.core.processing import OutlierHit


class RecordIndex:
    """O(1) lookup by detection id and O(k) lookup by frame/track."""

    def __init__(self, records: Sequence[DetectionRecord] = ()) -> None:
        self._revision = 0
        self._by_id: dict[str, DetectionRecord] = {}
        self._by_frame: dict[int, tuple[DetectionRecord, ...]] = {}
        self._by_track: dict[str, tuple[DetectionRecord, ...]] = {}
        self.rebuild(records)

    @property
    def revision(self) -> int:
        return self._revision

    def rebuild(self, records: Sequence[DetectionRecord]) -> None:
        by_frame: dict[int, list[DetectionRecord]] = defaultdict(list)
        by_track: dict[str, list[DetectionRecord]] = defaultdict(list)
        by_id: dict[str, DetectionRecord] = {}
        for record in records:
            by_id[record.det_id] = record
            by_frame[int(record.frame_index)].append(record)
            track_id = str(record.track_id).strip()
            if track_id:
                by_track[track_id].append(record)

        self._by_id = by_id
        self._by_frame = {
            frame_index: tuple(sorted(items, key=lambda item: (-item.score, item.det_id)))
            for frame_index, items in by_frame.items()
        }
        self._by_track = {
            track_id: tuple(sorted(items, key=lambda item: (item.frame_index, item.det_id)))
            for track_id, items in by_track.items()
        }
        self._revision += 1

    def by_id(self, det_id: str) -> DetectionRecord | None:
        return self._by_id.get(str(det_id))

    def for_frame(self, frame_index: int) -> list[DetectionRecord]:
        return list(self._by_frame.get(int(frame_index), ()))

    def for_track(self, track_id: str) -> list[DetectionRecord]:
        return list(self._by_track.get(str(track_id).strip(), ()))


@dataclass(slots=True)
class OutlierIndex:
    """Pre-grouped outlier results shared by UI and future report adapters."""

    hits: list[OutlierHit] = field(default_factory=list)
    by_det_id: dict[str, list[OutlierHit]] = field(default_factory=dict)
    by_frame: dict[int, list[OutlierHit]] = field(default_factory=dict)
    by_track_id: dict[str, list[OutlierHit]] = field(default_factory=dict)
    by_rule_id: dict[str, list[OutlierHit]] = field(default_factory=dict)
    frames: set[int] = field(default_factory=set)

    @classmethod
    def build(cls, hits: Iterable[OutlierHit]) -> "OutlierIndex":
        result = cls(hits=list(hits))
        for hit in result.hits:
            result.by_det_id.setdefault(hit.det_id, []).append(hit)
            result.by_frame.setdefault(int(hit.frame_index), []).append(hit)
            track_id = str(hit.track_id).strip()
            if track_id:
                result.by_track_id.setdefault(track_id, []).append(hit)
            result.by_rule_id.setdefault(hit.rule_id, []).append(hit)
            result.frames.add(int(hit.frame_index))
        return result
