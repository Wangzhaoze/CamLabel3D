"""Offline 3D tracking over edited CamLabel3D detections."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .geometry import box9d_to_box10_quaternion, boxes10_quaternion_to_corners
from .models import DetectionRecord

try:  # pragma: no cover - exercised in runtime
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for minimal environments
    linear_sum_assignment = None

EPS = 1e-6


@dataclass(frozen=True)
class TrackingConfig:
    """Offline association parameters for rerunnable tracking."""

    max_gap_frames: int = 8
    max_center_distance_m: float = 6.0
    min_2d_iou: float = 0.01
    min_3d_iou: float = 0.01
    previous_id_bonus: float = 0.12


@dataclass
class _Observation:
    record_index: int
    frame_index: int
    category: str
    original_track_id: str
    box2d: np.ndarray
    box3d: np.ndarray
    center: np.ndarray
    dims: np.ndarray
    yaw_deg: float
    corners_xz: np.ndarray
    y_min: float
    y_max: float


@dataclass
class _TrackState:
    assigned_id: str
    category: str
    observations: list[_Observation] = field(default_factory=list)
    has_locked: bool = False

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_index

    def add(self, observation: _Observation, locked: bool = False) -> None:
        self.observations.append(observation)
        self.observations.sort(key=lambda item: item.frame_index)
        self.has_locked = self.has_locked or bool(locked)

    def latest_observation(self) -> _Observation:
        return self.observations[-1]

    def first_observation(self) -> _Observation:
        return self.observations[0]

    def predicted_center(self, next_frame_index: int) -> np.ndarray:
        latest = self.latest_observation()
        if len(self.observations) < 2:
            return latest.center.copy()
        previous = self.observations[-2]
        frame_delta = max(1, latest.frame_index - previous.frame_index)
        velocity = (latest.center - previous.center) / float(frame_delta)
        future_delta = max(1, int(next_frame_index) - int(latest.frame_index))
        return latest.center + velocity * float(future_delta)


class TrackingEngine:
    """Runs category-wise offline association on enabled detections."""

    COST_DISTANCE = 0.40
    COST_IOU3D = 0.25
    COST_IOU2D = 0.15
    COST_SIZE = 0.10
    COST_YAW = 0.10

    def run(
        self,
        records: list[DetectionRecord],
        config: TrackingConfig | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[DetectionRecord]:
        params = config or TrackingConfig()
        updated_records = [replace(record) for record in records]

        enabled_indices = [idx for idx, record in enumerate(updated_records) if record.is_enabled]
        if not enabled_indices:
            return updated_records

        features = self._build_observations(updated_records, enabled_indices)
        locked_indices = {
            idx
            for idx, record in enumerate(updated_records)
            if record.is_enabled and record.track_status == "locked" and record.track_id.strip()
        }
        existing_ids = {
            record.track_id.strip()
            for record in updated_records
            if record.track_id.strip()
        }
        used_ids = {
            updated_records[idx].track_id.strip()
            for idx in locked_indices
            if updated_records[idx].track_id.strip()
        }
        next_numeric_id = self._next_numeric_id(existing_ids)

        categories = sorted({obs.category for obs in features.values()})
        total_frames = sum(
            len({obs.frame_index for obs in features.values() if obs.category == category})
            for category in categories
        )
        progress_step = 0

        assignment_by_record: dict[int, str] = {}
        tracks_by_category: dict[str, list[_TrackState]] = {}

        for category in categories:
            if should_cancel and should_cancel():
                return updated_records
            category_observations = [obs for obs in features.values() if obs.category == category]
            locked_by_frame: dict[int, list[_Observation]] = {}
            unlocked_by_frame: dict[int, list[_Observation]] = {}
            for obs in category_observations:
                if obs.record_index in locked_indices:
                    locked_by_frame.setdefault(obs.frame_index, []).append(obs)
                else:
                    unlocked_by_frame.setdefault(obs.frame_index, []).append(obs)
            frames = sorted({obs.frame_index for obs in category_observations})
            tracks: list[_TrackState] = []
            track_by_id: dict[str, _TrackState] = {}

            for frame_index in frames:
                if progress_callback:
                    progress_callback(progress_step, max(1, total_frames), f"Tracking {category} frame {frame_index}")
                progress_step += 1

                for locked_obs in sorted(
                    locked_by_frame.get(frame_index, []),
                    key=lambda item: (item.frame_index, item.record_index),
                ):
                    track_id = locked_obs.original_track_id.strip()
                    if not track_id:
                        continue
                    track = track_by_id.get(track_id)
                    if track is None:
                        track = _TrackState(assigned_id=track_id, category=category)
                        tracks.append(track)
                        track_by_id[track_id] = track
                    track.add(locked_obs, locked=True)
                    assignment_by_record[locked_obs.record_index] = track_id
                    used_ids.add(track_id)

                detections = sorted(
                    unlocked_by_frame.get(frame_index, []),
                    key=lambda item: (item.frame_index, item.record_index),
                )
                if not detections:
                    continue

                candidate_tracks = [
                    track
                    for track in tracks
                    if track.last_frame < frame_index
                    and (frame_index - track.last_frame) <= int(params.max_gap_frames)
                ]
                matches, unmatched_track_indices, unmatched_detection_indices = self._associate_frame(
                    candidate_tracks,
                    detections,
                    frame_index,
                    params,
                )

                for track_idx, det_idx in matches:
                    track = candidate_tracks[track_idx]
                    detection = detections[det_idx]
                    track.add(detection, locked=False)
                    assignment_by_record[detection.record_index] = track.assigned_id

                del unmatched_track_indices

                for det_idx in unmatched_detection_indices:
                    detection = detections[det_idx]
                    preferred_id = detection.original_track_id.strip()
                    track_id, next_numeric_id = self._allocate_track_id(
                        preferred_id=preferred_id,
                        used_ids=used_ids,
                        next_numeric_id=next_numeric_id,
                    )
                    new_track = _TrackState(assigned_id=track_id, category=category)
                    new_track.add(detection, locked=False)
                    tracks.append(new_track)
                    track_by_id[track_id] = new_track
                    assignment_by_record[detection.record_index] = track_id

            self._gap_close_tracks(tracks, assignment_by_record, params)
            tracks_by_category[category] = tracks

        for idx, record in enumerate(updated_records):
            if not record.is_enabled:
                continue
            if idx in locked_indices:
                record.track_status = "locked"
                continue
            assigned_id = assignment_by_record.get(idx, "").strip()
            record.track_id = assigned_id
            record.track_status = "auto" if assigned_id else ""

        if progress_callback:
            progress_callback(max(1, total_frames), max(1, total_frames), "Tracking finished")
        return updated_records

    def _build_observations(
        self,
        records: list[DetectionRecord],
        enabled_indices: list[int],
    ) -> dict[int, _Observation]:
        boxes3d = np.array(
            [
                box9d_to_box10_quaternion(
                    center_x=records[idx].center_x,
                    center_y=records[idx].center_y,
                    center_z=records[idx].center_z,
                    size_w=records[idx].size_w,
                    size_l=records[idx].size_l,
                    size_h=records[idx].size_h,
                    yaw_deg=records[idx].yaw_deg,
                    pitch_deg=records[idx].pitch_deg,
                    roll_deg=records[idx].roll_deg,
                )
                for idx in enabled_indices
            ],
            dtype=np.float32,
        )
        corners_all = boxes10_quaternion_to_corners(boxes3d)
        features: dict[int, _Observation] = {}
        for offset, record_index in enumerate(enabled_indices):
            record = records[record_index]
            corners = np.asarray(corners_all[offset], dtype=np.float64)
            corners_xz = self._convex_hull(corners[:, [0, 2]])
            features[record_index] = _Observation(
                record_index=record_index,
                frame_index=int(record.frame_index),
                category=str(record.category),
                original_track_id=str(record.track_id).strip(),
                box2d=np.array(record.box2d_xyxy(), dtype=np.float64),
                box3d=np.asarray(boxes3d[offset], dtype=np.float64),
                center=np.array([record.center_x, record.center_y, record.center_z], dtype=np.float64),
                dims=np.array([record.size_w, record.size_l, record.size_h], dtype=np.float64),
                yaw_deg=float(record.yaw_deg),
                corners_xz=corners_xz,
                y_min=float(np.min(corners[:, 1])),
                y_max=float(np.max(corners[:, 1])),
            )
        return features

    def _associate_frame(
        self,
        tracks: list[_TrackState],
        detections: list[_Observation],
        frame_index: int,
        config: TrackingConfig,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.full((len(tracks), len(detections)), fill_value=np.inf, dtype=np.float64)
        for track_idx, track in enumerate(tracks):
            for det_idx, detection in enumerate(detections):
                cost = self._association_cost(track, detection, frame_index, config)
                cost_matrix[track_idx, det_idx] = cost

        assignment = self._solve_assignment(cost_matrix)
        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()
        accepted_matches: list[tuple[int, int]] = []
        for track_idx, det_idx in assignment:
            if not np.isfinite(cost_matrix[track_idx, det_idx]):
                continue
            accepted_matches.append((track_idx, det_idx))
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)

        unmatched_tracks = [idx for idx in range(len(tracks)) if idx not in matched_track_indices]
        unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_det_indices]
        return accepted_matches, unmatched_tracks, unmatched_detections

    def _association_cost(
        self,
        track: _TrackState,
        detection: _Observation,
        frame_index: int,
        config: TrackingConfig,
    ) -> float:
        last_obs = track.latest_observation()
        gap = int(frame_index) - int(last_obs.frame_index)
        if gap <= 0 or gap > int(config.max_gap_frames):
            return math.inf

        predicted_center = track.predicted_center(frame_index)
        center_distance = float(np.linalg.norm(predicted_center - detection.center))
        if center_distance > float(config.max_center_distance_m):
            return math.inf

        predicted_box2d = last_obs.box2d
        iou2d = self._box2d_iou(predicted_box2d, detection.box2d)

        delta_center = predicted_center - last_obs.center
        predicted_poly = last_obs.corners_xz + np.array([delta_center[0], delta_center[2]], dtype=np.float64)
        predicted_y_min = last_obs.y_min + float(delta_center[1])
        predicted_y_max = last_obs.y_max + float(delta_center[1])
        iou3d = self._box3d_iou_from_features(
            predicted_poly,
            predicted_y_min,
            predicted_y_max,
            detection.corners_xz,
            detection.y_min,
            detection.y_max,
        )
        if iou2d < float(config.min_2d_iou) and iou3d < float(config.min_3d_iou):
            return math.inf

        normalized_center_distance = min(
            center_distance / max(float(config.max_center_distance_m), EPS),
            1.0,
        )
        size_delta = float(
            np.mean(
                np.abs(detection.dims - last_obs.dims)
                / np.maximum(np.abs(last_obs.dims), EPS)
            )
        )
        yaw_delta = self._wrapped_abs_delta_deg(last_obs.yaw_deg, detection.yaw_deg) / 180.0

        cost = (
            self.COST_DISTANCE * normalized_center_distance
            + self.COST_IOU3D * (1.0 - iou3d)
            + self.COST_IOU2D * (1.0 - iou2d)
            + self.COST_SIZE * min(size_delta, 1.0)
            + self.COST_YAW * min(yaw_delta, 1.0)
        )
        if detection.original_track_id and detection.original_track_id == track.assigned_id:
            cost -= float(config.previous_id_bonus)
        return cost

    def _gap_close_tracks(
        self,
        tracks: list[_TrackState],
        assignment_by_record: dict[int, str],
        config: TrackingConfig,
    ) -> None:
        if len(tracks) <= 1:
            return

        ordered_tracks = sorted(tracks, key=lambda item: (item.first_frame, item.last_frame, item.assigned_id))
        candidates: list[tuple[int, int]] = []
        costs: list[float] = []
        for source_idx, source in enumerate(ordered_tracks):
            for target_idx, target in enumerate(ordered_tracks):
                if source_idx == target_idx:
                    continue
                if source.category != target.category:
                    continue
                if source.last_frame >= target.first_frame:
                    continue
                if target.first_frame - source.last_frame > int(config.max_gap_frames):
                    continue
                if source.has_locked and target.has_locked and source.assigned_id != target.assigned_id:
                    continue
                cost = self._tracklet_link_cost(source, target, config)
                if np.isfinite(cost):
                    candidates.append((source_idx, target_idx))
                    costs.append(cost)

        if not candidates:
            return

        cost_matrix = np.full((len(ordered_tracks), len(ordered_tracks)), np.inf, dtype=np.float64)
        for (source_idx, target_idx), cost in zip(candidates, costs):
            cost_matrix[source_idx, target_idx] = cost

        accepted = self._solve_assignment(cost_matrix)
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        for source_idx, target_idx in accepted:
            if not np.isfinite(cost_matrix[source_idx, target_idx]):
                continue
            if source_idx in used_sources or target_idx in used_targets:
                continue
            source = ordered_tracks[source_idx]
            target = ordered_tracks[target_idx]
            keep_id = self._preferred_gap_close_id(source, target)
            if keep_id is None:
                continue
            source.assigned_id = keep_id
            target.assigned_id = keep_id
            source.has_locked = source.has_locked or target.has_locked
            source.observations.extend(target.observations)
            source.observations.sort(key=lambda item: item.frame_index)
            for observation in target.observations:
                assignment_by_record[observation.record_index] = keep_id
            for observation in source.observations:
                assignment_by_record[observation.record_index] = keep_id
            used_sources.add(source_idx)
            used_targets.add(target_idx)

    def _tracklet_link_cost(
        self,
        source: _TrackState,
        target: _TrackState,
        config: TrackingConfig,
    ) -> float:
        target_obs = target.first_observation()
        return self._association_cost(
            track=source,
            detection=target_obs,
            frame_index=target_obs.frame_index,
            config=config,
        )

    @staticmethod
    def _preferred_gap_close_id(source: _TrackState, target: _TrackState) -> str | None:
        if source.has_locked and target.has_locked and source.assigned_id != target.assigned_id:
            return None
        if target.has_locked:
            return target.assigned_id
        return source.assigned_id

    @staticmethod
    def _next_numeric_id(used_ids: set[str]) -> int:
        numeric_values = [int(value) for value in used_ids if str(value).isdigit()]
        return (max(numeric_values) + 1) if numeric_values else 1

    @staticmethod
    def _allocate_track_id(
        preferred_id: str,
        used_ids: set[str],
        next_numeric_id: int,
    ) -> tuple[str, int]:
        preferred = str(preferred_id).strip()
        if preferred and preferred not in used_ids:
            used_ids.add(preferred)
            return preferred, next_numeric_id
        candidate = int(next_numeric_id)
        while str(candidate) in used_ids:
            candidate += 1
        used_ids.add(str(candidate))
        return str(candidate), candidate + 1

    @staticmethod
    def _solve_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        finite_mask = np.isfinite(cost_matrix)
        if not finite_mask.any():
            return []
        if linear_sum_assignment is not None:
            safe_costs = np.where(finite_mask, cost_matrix, 1e6)
            rows, cols = linear_sum_assignment(safe_costs)
            return list(zip(rows.tolist(), cols.tolist()))

        pairs: list[tuple[int, int, float]] = []
        for row_idx in range(cost_matrix.shape[0]):
            for col_idx in range(cost_matrix.shape[1]):
                value = cost_matrix[row_idx, col_idx]
                if np.isfinite(value):
                    pairs.append((row_idx, col_idx, float(value)))
        pairs.sort(key=lambda item: item[2])
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        greedy: list[tuple[int, int]] = []
        for row_idx, col_idx, _ in pairs:
            if row_idx in used_rows or col_idx in used_cols:
                continue
            used_rows.add(row_idx)
            used_cols.add(col_idx)
            greedy.append((row_idx, col_idx))
        return greedy

    @staticmethod
    def _box2d_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
        area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
        union = area_a + area_b - inter
        if union <= EPS:
            return 0.0
        return float(np.clip(inter / union, 0.0, 1.0))

    def _box3d_iou_from_features(
        self,
        a_poly: np.ndarray,
        a_y_min: float,
        a_y_max: float,
        b_poly: np.ndarray,
        b_y_min: float,
        b_y_max: float,
    ) -> float:
        inter_area = self._convex_polygon_intersection_area(a_poly, b_poly)
        if inter_area <= EPS:
            return 0.0
        overlap_h = max(0.0, min(float(a_y_max), float(b_y_max)) - max(float(a_y_min), float(b_y_min)))
        if overlap_h <= EPS:
            return 0.0
        inter_volume = inter_area * overlap_h
        volume_a = self._convex_polygon_area(a_poly) * max(float(a_y_max) - float(a_y_min), 0.0)
        volume_b = self._convex_polygon_area(b_poly) * max(float(b_y_max) - float(b_y_min), 0.0)
        union = volume_a + volume_b - inter_volume
        if union <= EPS:
            return 0.0
        return float(np.clip(inter_volume / union, 0.0, 1.0))

    @staticmethod
    def _wrapped_abs_delta_deg(a_deg: float, b_deg: float) -> float:
        delta = abs(float(a_deg) - float(b_deg)) % 360.0
        return min(delta, 360.0 - delta)

    @staticmethod
    def _convex_hull(points: np.ndarray) -> np.ndarray:
        pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
        if len(pts) <= 2:
            return pts

        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

        def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

        lower: list[np.ndarray] = []
        for point in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)

        upper: list[np.ndarray] = []
        for point in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)

        return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    @staticmethod
    def _convex_polygon_area(polygon: np.ndarray) -> float:
        if polygon is None or len(polygon) < 3:
            return 0.0
        x = polygon[:, 0]
        y = polygon[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

    @classmethod
    def _convex_polygon_intersection_area(cls, subject: np.ndarray, clip: np.ndarray) -> float:
        if len(subject) < 3 or len(clip) < 3:
            return 0.0
        output = np.asarray(subject, dtype=np.float64)
        clip_poly = cls._ensure_ccw(np.asarray(clip, dtype=np.float64))
        for idx in range(len(clip_poly)):
            edge_start = clip_poly[idx]
            edge_end = clip_poly[(idx + 1) % len(clip_poly)]
            input_poly = output
            if len(input_poly) == 0:
                break
            output_list: list[np.ndarray] = []
            s = input_poly[-1]
            for e in input_poly:
                if cls._inside(e, edge_start, edge_end):
                    if not cls._inside(s, edge_start, edge_end):
                        output_list.append(cls._intersection(s, e, edge_start, edge_end))
                    output_list.append(e)
                elif cls._inside(s, edge_start, edge_end):
                    output_list.append(cls._intersection(s, e, edge_start, edge_end))
                s = e
            output = np.asarray(output_list, dtype=np.float64) if output_list else np.empty((0, 2), dtype=np.float64)
        return cls._convex_polygon_area(output)

    @classmethod
    def _ensure_ccw(cls, polygon: np.ndarray) -> np.ndarray:
        if cls._signed_area(polygon) >= 0.0:
            return polygon
        return polygon[::-1]

    @staticmethod
    def _signed_area(polygon: np.ndarray) -> float:
        if len(polygon) < 3:
            return 0.0
        x = polygon[:, 0]
        y = polygon[:, 1]
        return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

    @staticmethod
    def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
        return (
            (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
        ) >= -EPS

    @staticmethod
    def _intersection(
        start: np.ndarray,
        end: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
    ) -> np.ndarray:
        x1, y1 = start
        x2, y2 = end
        x3, y3 = edge_start
        x4, y4 = edge_end
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) <= EPS:
            return np.asarray(end, dtype=np.float64)
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denominator
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denominator
        return np.array([px, py], dtype=np.float64)
