"""Extensible postprocessing rules and bulk operations for CamLabel3D."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Sequence

import numpy as np

from .models import DetectionRecord, natural_sort_key
from .postprocess import TrackSummary

EPS = 1e-6


class ProcessingScope(str, Enum):
    """Shared scope selector for rules and bulk operations."""

    CURRENT_FRAME = "Current Frame"
    SELECTED_TRACK = "Selected Track"
    GLOBAL = "Global"


OutlierScope = ProcessingScope
OperationScope = ProcessingScope


class TrackBatchOperationKind(str, Enum):
    """Track-scoped spreadsheet-style numeric edit modes."""

    SMOOTH = "smooth"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


TRACK_BATCH_NUMERIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("Score", "score"),
    ("2D Score", "score_2d"),
    ("3D Score", "score_3d"),
    ("box2d_x1", "box2d_x1"),
    ("box2d_y1", "box2d_y1"),
    ("box2d_x2", "box2d_x2"),
    ("box2d_y2", "box2d_y2"),
    ("center_x", "center_x"),
    ("center_y", "center_y"),
    ("center_z", "center_z"),
    ("yaw_deg", "yaw_deg"),
    ("pitch_deg", "pitch_deg"),
    ("roll_deg", "roll_deg"),
    ("size_w", "size_w"),
    ("size_l", "size_l"),
    ("size_h", "size_h"),
)
TRACK_BATCH_NUMERIC_FIELD_SET = {field_name for _, field_name in TRACK_BATCH_NUMERIC_FIELDS}
TRACK_BATCH_GEOMETRY_FIELDS = {
    "center_x",
    "center_y",
    "center_z",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "size_w",
    "size_l",
    "size_h",
}
TRACK_BATCH_ANGLE_FIELDS = {"yaw_deg", "pitch_deg", "roll_deg"}
TRACK_BATCH_POSITIVE_FIELDS = {"size_w", "size_l", "size_h"}


@dataclass(frozen=True)
class TrackBatchEditRequest:
    """One batch edit request for a selected track and frame range."""

    track_id: str
    field_name: str
    operation: TrackBatchOperationKind
    frame_start: int
    frame_end: int
    operand: float = 0.0
    smooth_window: int = 5


@dataclass(frozen=True)
class ParameterSpec:
    """UI-friendly numeric parameter definition."""

    key: str
    label: str
    kind: str
    default: float
    minimum: float
    maximum: float
    step: float
    decimals: int = 3

    def normalize(self, value: float | int | str | None) -> float | int:
        numeric = self.default if value is None else float(value)
        numeric = min(max(numeric, self.minimum), self.maximum)
        if self.kind == "int":
            return int(round(numeric))
        return float(numeric)


@dataclass
class ProcessingContext:
    """Runtime-only helpers shared by rules, bulk ops, GUI, and CLI."""

    records: list[DetectionRecord]
    current_frame_index: int | None = None
    selected_track_id: str = ""
    track_summaries: list[TrackSummary] = field(default_factory=list)
    reproject_record: Callable[[DetectionRecord], None] | None = None
    _enabled_tracked_cache: list[DetectionRecord] | None = field(default=None, init=False, repr=False)
    _track_groups_cache: dict[str, list[DetectionRecord]] | None = field(default=None, init=False, repr=False)
    _records_by_id_cache: dict[str, DetectionRecord] | None = field(default=None, init=False, repr=False)

    def enabled_tracked_records(self) -> list[DetectionRecord]:
        if self._enabled_tracked_cache is None:
            self._enabled_tracked_cache = [
                record
                for record in self.records
                if record.is_enabled and str(record.track_id).strip()
            ]
        return self._enabled_tracked_cache

    def track_groups(self) -> dict[str, list[DetectionRecord]]:
        if self._track_groups_cache is not None:
            return self._track_groups_cache
        grouped: dict[str, list[DetectionRecord]] = {}
        for record in self.enabled_tracked_records():
            grouped.setdefault(record.track_id.strip(), []).append(record)
        for track_id, group in grouped.items():
            grouped[track_id] = sorted(group, key=lambda item: (item.frame_index, item.det_id))
        self._track_groups_cache = grouped
        return grouped

    def record_by_id(self, det_id: str) -> DetectionRecord | None:
        if self._records_by_id_cache is None:
            self._records_by_id_cache = {record.det_id: record for record in self.records}
        return self._records_by_id_cache.get(str(det_id))

    def target_records(self, scope: ProcessingScope) -> list[DetectionRecord]:
        tracked = self.enabled_tracked_records()
        if scope == ProcessingScope.CURRENT_FRAME:
            if self.current_frame_index is None:
                return []
            return [
                record
                for record in tracked
                if int(record.frame_index) == int(self.current_frame_index)
            ]
        if scope == ProcessingScope.SELECTED_TRACK:
            track_id = str(self.selected_track_id or "").strip()
            if not track_id:
                return []
            return [
                record
                for record in tracked
                if record.track_id.strip() == track_id
            ]
        return tracked

    def target_det_ids(self, scope: ProcessingScope) -> set[str]:
        return {record.det_id for record in self.target_records(scope)}

    def try_reproject(self, record: DetectionRecord) -> None:
        if self.reproject_record is not None:
            self.reproject_record(record)


@dataclass(frozen=True)
class OutlierHit:
    """One derived outlier finding for a single detection row."""

    rule_id: str
    frame_index: int
    det_id: str
    track_id: str
    category: str
    severity: float
    message: str
    fixable: bool
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    def sort_key(self) -> tuple[int, str, float, str]:
        return (self.frame_index, natural_sort_key(self.track_id), -float(self.severity), self.rule_id)

    def to_report_row(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "frame_index": int(self.frame_index),
            "det_id": self.det_id,
            "track_id": self.track_id,
            "category": self.category,
            "severity": float(self.severity),
            "message": self.message,
            "fixable": bool(self.fixable),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OperationResult:
    """Summary returned after applying a rule fix or bulk operation."""

    updated_count: int
    message: str
    affected_det_ids: tuple[str, ...] = ()


class OutlierRule:
    """Base class for extensible outlier rules."""

    rule_id = ""
    display_name = ""
    default_enabled = False
    supported_scopes = (
        ProcessingScope.CURRENT_FRAME,
        ProcessingScope.SELECTED_TRACK,
        ProcessingScope.GLOBAL,
    )
    param_specs: tuple[ParameterSpec, ...] = ()

    def default_params(self) -> dict[str, float | int]:
        return {spec.key: spec.normalize(spec.default) for spec in self.param_specs}

    def normalize_params(self, params: dict[str, float | int] | None = None) -> dict[str, float | int]:
        merged = self.default_params()
        for spec in self.param_specs:
            merged[spec.key] = spec.normalize((params or {}).get(spec.key, merged[spec.key]))
        return merged

    def analyze(
        self,
        records: list[DetectionRecord],
        scope: OutlierScope,
        context: ProcessingContext,
        params: dict[str, float | int] | None = None,
    ) -> list[OutlierHit]:
        raise NotImplementedError

    def fix(
        self,
        records: list[DetectionRecord],
        hits: Sequence[OutlierHit],
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        raise NotImplementedError


class BulkOperation:
    """Base class for extensible bulk geometry helpers."""

    operation_id = ""
    display_name = ""
    supported_scopes = (
        ProcessingScope.CURRENT_FRAME,
        ProcessingScope.SELECTED_TRACK,
        ProcessingScope.GLOBAL,
    )
    param_specs: tuple[ParameterSpec, ...] = ()

    def default_params(self) -> dict[str, float | int]:
        return {spec.key: spec.normalize(spec.default) for spec in self.param_specs}

    def normalize_params(self, params: dict[str, float | int] | None = None) -> dict[str, float | int]:
        merged = self.default_params()
        for spec in self.param_specs:
            merged[spec.key] = spec.normalize((params or {}).get(spec.key, merged[spec.key]))
        return merged

    def apply(
        self,
        records: list[DetectionRecord],
        scope: OperationScope,
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        raise NotImplementedError


class OutlierRuleRegistry:
    """Code-registered outlier rules."""

    def __init__(self) -> None:
        self._rules: dict[str, OutlierRule] = {}

    def register(self, rule: OutlierRule) -> None:
        if not rule.rule_id:
            raise ValueError("Outlier rules must define rule_id.")
        self._rules[rule.rule_id] = rule

    def get(self, rule_id: str) -> OutlierRule:
        return self._rules[rule_id]

    def all(self) -> list[OutlierRule]:
        return list(self._rules.values())


class BulkOperationRegistry:
    """Code-registered bulk operations."""

    def __init__(self) -> None:
        self._operations: dict[str, BulkOperation] = {}

    def register(self, operation: BulkOperation) -> None:
        if not operation.operation_id:
            raise ValueError("Bulk operations must define operation_id.")
        self._operations[operation.operation_id] = operation

    def get(self, operation_id: str) -> BulkOperation:
        return self._operations[operation_id]

    def all(self) -> list[BulkOperation]:
        return list(self._operations.values())


class ProcessingEngine:
    """Coordinates rules and bulk ops against the current latest records."""

    def __init__(
        self,
        outlier_registry: OutlierRuleRegistry | None = None,
        bulk_operation_registry: BulkOperationRegistry | None = None,
        max_workers: int = 1,
    ) -> None:
        self.outlier_registry = outlier_registry or build_default_outlier_registry()
        self.bulk_operation_registry = bulk_operation_registry or build_default_bulk_operation_registry()
        self.max_workers = max(1, int(max_workers))

    def analyze_outliers(
        self,
        records: list[DetectionRecord],
        scope: OutlierScope,
        enabled_rule_ids: Iterable[str],
        params_by_rule: dict[str, dict[str, float | int]] | None,
        context: ProcessingContext,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[OutlierHit]:
        rule_ids = list(enabled_rule_ids)
        # Materialize shared read indexes once before optional parallel rule
        # evaluation, avoiding duplicate grouping and benign cache races.
        context.track_groups()
        context.record_by_id("")

        def analyze_rule(rule_id: str) -> list[OutlierHit]:
            if should_cancel and should_cancel():
                return []
            rule = self.outlier_registry.get(rule_id)
            params = (params_by_rule or {}).get(rule_id)
            return rule.analyze(records, scope, context, params)

        if self.max_workers > 1 and len(rule_ids) > 1:
            worker_count = min(self.max_workers, len(rule_ids))
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="outlier-rule") as executor:
                grouped_hits = list(executor.map(analyze_rule, rule_ids))
        else:
            grouped_hits = [analyze_rule(rule_id) for rule_id in rule_ids]

        hits = [hit for rule_hits in grouped_hits for hit in rule_hits]
        hits.sort(key=lambda hit: hit.sort_key())
        return hits

    def filter_hits_for_scope(
        self,
        hits: Sequence[OutlierHit],
        scope: OutlierScope,
        context: ProcessingContext,
        rule_id: str | None = None,
    ) -> list[OutlierHit]:
        target_ids = context.target_det_ids(scope)
        filtered = [
            hit
            for hit in hits
            if hit.det_id in target_ids and (rule_id is None or hit.rule_id == rule_id)
        ]
        filtered.sort(key=lambda hit: hit.sort_key())
        return filtered

    def fix_hits(
        self,
        records: list[DetectionRecord],
        hits: Sequence[OutlierHit],
        params_by_rule: dict[str, dict[str, float | int]] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        grouped: dict[str, list[OutlierHit]] = {}
        for hit in hits:
            if hit.fixable:
                grouped.setdefault(hit.rule_id, []).append(hit)
        total_updated = 0
        affected: list[str] = []
        messages: list[str] = []
        for rule_id, rule_hits in grouped.items():
            rule = self.outlier_registry.get(rule_id)
            result = rule.fix(records, rule_hits, (params_by_rule or {}).get(rule_id), context)
            total_updated += int(result.updated_count)
            affected.extend(result.affected_det_ids)
            if result.message:
                messages.append(result.message)
        summary = "; ".join(messages) if messages else "No fixable outliers were updated."
        return OperationResult(
            updated_count=total_updated,
            message=summary,
            affected_det_ids=tuple(sorted(set(affected))),
        )

    def apply_operation(
        self,
        operation_id: str,
        records: list[DetectionRecord],
        scope: OperationScope,
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        operation = self.bulk_operation_registry.get(operation_id)
        return operation.apply(records, scope, params, context)


def _normalize_angle_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _wrapped_abs_delta_deg(a_deg: float, b_deg: float) -> float:
    delta = abs(float(a_deg) - float(b_deg)) % 360.0
    return min(delta, 360.0 - delta)


def _unwrap_angles_deg(values_deg: Sequence[float]) -> np.ndarray:
    radians = np.radians(np.asarray(values_deg, dtype=np.float64))
    return np.degrees(np.unwrap(radians))


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("weighted_median requires at least one value.")
    if w.size != arr.size:
        raise ValueError("weights must match values.")
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)
    order = np.argsort(arr)
    sorted_values = arr[order]
    sorted_weights = w[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(max(index, 0), len(sorted_values) - 1)
    return float(sorted_values[index])


def _track_reference_size(track_records: Sequence[DetectionRecord]) -> tuple[float, float, float]:
    weights = [
        float(record.score_3d) if float(record.score_3d) > 0.0 else max(float(record.score), EPS)
        for record in track_records
    ]
    return (
        _weighted_median([record.size_w for record in track_records], weights),
        _weighted_median([record.size_l for record in track_records], weights),
        _weighted_median([record.size_h for record in track_records], weights),
    )


def _track_size_map(track_groups: dict[str, list[DetectionRecord]]) -> dict[str, tuple[float, float, float]]:
    return {
        track_id: _track_reference_size(track_records)
        for track_id, track_records in track_groups.items()
        if track_records
    }


def _linear_interpolate_scalar(
    frame_prev: int,
    value_prev: float,
    frame_next: int,
    value_next: float,
    frame_current: int,
) -> float:
    if frame_next == frame_prev:
        return float(value_prev)
    ratio = (float(frame_current) - float(frame_prev)) / (float(frame_next) - float(frame_prev))
    return float(value_prev) + ratio * (float(value_next) - float(value_prev))


def _linear_interpolate_vector(
    frame_prev: int,
    value_prev: np.ndarray,
    frame_next: int,
    value_next: np.ndarray,
    frame_current: int,
) -> np.ndarray:
    if frame_next == frame_prev:
        return np.asarray(value_prev, dtype=np.float64).copy()
    ratio = (float(frame_current) - float(frame_prev)) / (float(frame_next) - float(frame_prev))
    return np.asarray(value_prev, dtype=np.float64) + ratio * (
        np.asarray(value_next, dtype=np.float64) - np.asarray(value_prev, dtype=np.float64)
    )


def _restore_record(dst: DetectionRecord, src: DetectionRecord) -> None:
    for field_name in dst.__dataclass_fields__:
        setattr(dst, field_name, getattr(src, field_name))


def _centered_moving_average(values: Sequence[float], window_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return arr.copy()
    window = max(1, int(window_size))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(arr, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _try_update_geometry(
    record: DetectionRecord,
    context: ProcessingContext,
    update_fn: Callable[[DetectionRecord], None],
) -> bool:
    backup = replace(record)
    try:
        update_fn(record)
        context.try_reproject(record)
        return True
    except Exception:
        _restore_record(record, backup)
        return False


def _assign_batch_numeric_value(record: DetectionRecord, field_name: str, value: float) -> None:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{field_name} must stay finite.")
    if field_name in TRACK_BATCH_POSITIVE_FIELDS and numeric <= 0.0:
        raise ValueError(f"{field_name} must be greater than 0.")
    if field_name in TRACK_BATCH_ANGLE_FIELDS:
        numeric = _normalize_angle_deg(numeric)
    setattr(record, field_name, numeric)


def _try_update_numeric_field(
    record: DetectionRecord,
    field_name: str,
    value: float,
    context: ProcessingContext,
) -> bool:
    def mutate(item: DetectionRecord) -> None:
        _assign_batch_numeric_value(item, field_name, value)

    if field_name in TRACK_BATCH_GEOMETRY_FIELDS:
        return _try_update_geometry(record, context, mutate)

    backup = replace(record)
    try:
        mutate(record)
        return True
    except Exception:
        _restore_record(record, backup)
        return False


def track_batch_records(
    records: Sequence[DetectionRecord],
    *,
    track_id: str,
    frame_start: int | None = None,
    frame_end: int | None = None,
    enabled_only: bool = True,
) -> list[DetectionRecord]:
    selected_track_id = str(track_id or "").strip()
    if not selected_track_id:
        return []
    start = None if frame_start is None else int(frame_start)
    end = None if frame_end is None else int(frame_end)
    return sorted(
        [
            record
            for record in records
            if str(record.track_id).strip() == selected_track_id
            and (not enabled_only or record.is_enabled)
            and (start is None or int(record.frame_index) >= start)
            and (end is None or int(record.frame_index) <= end)
        ],
        key=lambda item: (int(item.frame_index), item.det_id),
    )


def apply_track_batch_edit(
    records: list[DetectionRecord],
    request: TrackBatchEditRequest,
    context: ProcessingContext,
) -> OperationResult:
    track_id = str(request.track_id or "").strip()
    field_name = str(request.field_name or "").strip()
    if not track_id:
        raise ValueError("Select a track ID first.")
    if field_name not in TRACK_BATCH_NUMERIC_FIELD_SET:
        raise ValueError(f"Unsupported batch-edit field: {field_name}")
    frame_start = int(request.frame_start)
    frame_end = int(request.frame_end)
    if frame_end < frame_start:
        raise ValueError("End frame must be greater than or equal to start frame.")

    target_records = track_batch_records(
        records,
        track_id=track_id,
        frame_start=frame_start,
        frame_end=frame_end,
        enabled_only=False,
    )
    if not target_records:
        raise ValueError("No enabled detections matched the selected track and frame range.")

    operation = request.operation
    if not isinstance(operation, TrackBatchOperationKind):
        operation = TrackBatchOperationKind(str(operation))

    values = np.asarray([float(getattr(record, field_name)) for record in target_records], dtype=np.float64)
    if operation == TrackBatchOperationKind.SMOOTH:
        window = max(1, int(request.smooth_window))
        if field_name in TRACK_BATCH_ANGLE_FIELDS:
            working = _unwrap_angles_deg(values)
            updated_values = _centered_moving_average(working, window)
        else:
            updated_values = _centered_moving_average(values, window)
    else:
        operand = float(request.operand)
        if not np.isfinite(operand):
            raise ValueError("Operand must be finite.")
        if operation == TrackBatchOperationKind.ADD:
            updated_values = values + operand
        elif operation == TrackBatchOperationKind.SUBTRACT:
            updated_values = values - operand
        elif operation == TrackBatchOperationKind.MULTIPLY:
            updated_values = values * operand
        elif operation == TrackBatchOperationKind.DIVIDE:
            if abs(operand) <= EPS:
                raise ValueError("Division by zero is not allowed.")
            updated_values = values / operand
        else:
            raise ValueError(f"Unsupported batch operation: {operation}")

    updated = 0
    affected: list[str] = []
    for record, value in zip(target_records, updated_values, strict=False):
        if _try_update_numeric_field(record, field_name, float(value), context):
            updated += 1
            affected.append(record.det_id)

    label_by_operation = {
        TrackBatchOperationKind.SMOOTH: "Smoothed",
        TrackBatchOperationKind.ADD: "Added",
        TrackBatchOperationKind.SUBTRACT: "Subtracted",
        TrackBatchOperationKind.MULTIPLY: "Multiplied",
        TrackBatchOperationKind.DIVIDE: "Divided",
    }
    return OperationResult(
        updated_count=updated,
        message=(
            f"{label_by_operation[operation]} {field_name} for {updated} detections "
            f"in track {track_id} across frames {frame_start}-{frame_end}."
        ),
        affected_det_ids=tuple(sorted(set(affected))),
    )


class EulerAngleSpikeRule(OutlierRule):
    """Track-level Euler-angle jump detector for one axis."""

    _ANGLE_LABELS = {
        "yaw": "Yaw",
        "pitch": "Pitch",
        "roll": "Roll",
    }

    def __init__(self, axis_name: str) -> None:
        if axis_name not in self._ANGLE_LABELS:
            raise ValueError(f"Unsupported Euler axis: {axis_name}")
        self.axis_name = axis_name
        self.angle_field = f"{axis_name}_deg"
        angle_label = self._ANGLE_LABELS[axis_name]
        self.rule_id = f"{axis_name}_spike"
        self.display_name = f"{angle_label} Spike"
        self._jump_key = f"{axis_name}_jump_deg"
        self._residual_key = f"{axis_name}_residual_deg"
        self.param_specs = (
            ParameterSpec(self._jump_key, f"{angle_label} Jump (deg)", "float", 60.0, 5.0, 360.0, 1.0, 1),
            ParameterSpec(
                self._residual_key,
                f"{angle_label} Residual (deg)",
                "float",
                45.0,
                1.0,
                360.0,
                1.0,
                1,
            ),
            ParameterSpec("min_track_length", "Min Track Length", "int", 3, 3, 10000, 1, 0),
        )

    def analyze(
        self,
        records: list[DetectionRecord],
        scope: OutlierScope,
        context: ProcessingContext,
        params: dict[str, float | int] | None = None,
    ) -> list[OutlierHit]:
        settings = self.normalize_params(params)
        jump_threshold = float(settings[self._jump_key])
        residual_threshold = float(settings[self._residual_key])
        min_track_length = int(settings["min_track_length"])
        target_ids = context.target_det_ids(scope)
        hits: list[OutlierHit] = []
        angle_label = self._ANGLE_LABELS[self.axis_name]
        for track_id, track_records in context.track_groups().items():
            if len(track_records) < min_track_length:
                continue
            angle_values = [float(getattr(record, self.angle_field)) for record in track_records]
            unwrapped = _unwrap_angles_deg(angle_values)
            for index in range(1, len(track_records) - 1):
                record = track_records[index]
                if record.det_id not in target_ids:
                    continue
                prev_record = track_records[index - 1]
                next_record = track_records[index + 1]
                expected_angle = _linear_interpolate_scalar(
                    prev_record.frame_index,
                    float(unwrapped[index - 1]),
                    next_record.frame_index,
                    float(unwrapped[index + 1]),
                    record.frame_index,
                )
                residual = abs(float(unwrapped[index]) - float(expected_angle))
                jump_prev = _wrapped_abs_delta_deg(
                    float(getattr(record, self.angle_field)),
                    float(getattr(prev_record, self.angle_field)),
                )
                jump_next = _wrapped_abs_delta_deg(
                    float(getattr(record, self.angle_field)),
                    float(getattr(next_record, self.angle_field)),
                )
                if residual < residual_threshold or max(jump_prev, jump_next) < jump_threshold:
                    continue
                severity = max(
                    residual / max(residual_threshold, EPS),
                    max(jump_prev, jump_next) / max(jump_threshold, EPS),
                )
                hits.append(
                    OutlierHit(
                        rule_id=self.rule_id,
                        frame_index=record.frame_index,
                        det_id=record.det_id,
                        track_id=track_id,
                        category=record.category,
                        severity=float(severity),
                        message=(
                            f"{angle_label} spike: expected {_normalize_angle_deg(expected_angle):.1f} deg, "
                            f"residual {residual:.1f} deg, "
                            f"neighbor jumps {jump_prev:.1f}/{jump_next:.1f} deg"
                        ),
                        fixable=True,
                        metadata={
                            "expected_angle_temporal_deg": _normalize_angle_deg(expected_angle),
                            "angle_field": self.angle_field,
                            "residual_deg": residual,
                            "jump_prev_deg": jump_prev,
                            "jump_next_deg": jump_next,
                        },
                    )
                )
                if self.axis_name == "yaw":
                    hits[-1].metadata["expected_yaw_temporal_deg"] = _normalize_angle_deg(expected_angle)
        return hits

    def fix(
        self,
        records: list[DetectionRecord],
        hits: Sequence[OutlierHit],
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        updated = 0
        affected: list[str] = []
        angle_label = self._ANGLE_LABELS[self.axis_name]
        for hit in hits:
            record = context.record_by_id(hit.det_id)
            if record is None:
                continue
            target_angle = float(hit.metadata.get("expected_angle_temporal_deg", getattr(record, self.angle_field)))

            def mutate(item: DetectionRecord) -> None:
                setattr(item, self.angle_field, _normalize_angle_deg(target_angle))

            if _try_update_geometry(record, context, mutate):
                updated += 1
                affected.append(record.det_id)
        return OperationResult(
            updated_count=updated,
            message=f"{angle_label} spike fix updated {updated} detections.",
            affected_det_ids=tuple(sorted(set(affected))),
        )


class SizeSpikeRule(OutlierRule):
    rule_id = "size_spike"
    display_name = "Size Spike"
    param_specs = (
        ParameterSpec("size_rel_delta", "Relative Delta", "float", 0.25, 0.01, 10.0, 0.01, 3),
        ParameterSpec("min_track_length", "Min Track Length", "int", 3, 2, 10000, 1, 0),
    )

    def analyze(
        self,
        records: list[DetectionRecord],
        scope: OutlierScope,
        context: ProcessingContext,
        params: dict[str, float | int] | None = None,
    ) -> list[OutlierHit]:
        settings = self.normalize_params(params)
        rel_threshold = float(settings["size_rel_delta"])
        min_track_length = int(settings["min_track_length"])
        target_ids = context.target_det_ids(scope)
        hits: list[OutlierHit] = []
        size_refs = _track_size_map(context.track_groups())
        for track_id, track_records in context.track_groups().items():
            if len(track_records) < min_track_length or track_id not in size_refs:
                continue
            ref_w, ref_l, ref_h = size_refs[track_id]
            ref_dims = np.array([ref_w, ref_l, ref_h], dtype=np.float64)
            for record in track_records:
                if record.det_id not in target_ids:
                    continue
                dims = np.array([record.size_w, record.size_l, record.size_h], dtype=np.float64)
                rel = np.abs(dims - ref_dims) / np.maximum(np.abs(ref_dims), EPS)
                max_rel = float(np.max(rel))
                if max_rel < rel_threshold:
                    continue
                hits.append(
                    OutlierHit(
                        rule_id=self.rule_id,
                        frame_index=record.frame_index,
                        det_id=record.det_id,
                        track_id=track_id,
                        category=record.category,
                        severity=max_rel / max(rel_threshold, EPS),
                        message=f"Size spike: max relative delta {max_rel:.3f}",
                        fixable=True,
                        metadata={
                            "ref_size_w": ref_w,
                            "ref_size_l": ref_l,
                            "ref_size_h": ref_h,
                            "max_rel_delta": max_rel,
                        },
                    )
                )
        return hits

    def fix(
        self,
        records: list[DetectionRecord],
        hits: Sequence[OutlierHit],
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        updated = 0
        affected: list[str] = []
        for hit in hits:
            record = context.record_by_id(hit.det_id)
            if record is None:
                continue
            ref_w = float(hit.metadata.get("ref_size_w", record.size_w))
            ref_l = float(hit.metadata.get("ref_size_l", record.size_l))
            ref_h = float(hit.metadata.get("ref_size_h", record.size_h))

            def mutate(item: DetectionRecord) -> None:
                item.size_w = ref_w
                item.size_l = ref_l
                item.size_h = ref_h

            if _try_update_geometry(record, context, mutate):
                updated += 1
                affected.append(record.det_id)
        return OperationResult(
            updated_count=updated,
            message=f"Size spike fix updated {updated} detections.",
            affected_det_ids=tuple(sorted(set(affected))),
        )


class CenterSpikeRule(OutlierRule):
    rule_id = "center_spike"
    display_name = "Center Spike"
    param_specs = (
        ParameterSpec("center_residual_m", "Center Residual (m)", "float", 1.5, 0.05, 1000.0, 0.1, 3),
        ParameterSpec("min_track_length", "Min Track Length", "int", 3, 3, 10000, 1, 0),
    )

    def analyze(
        self,
        records: list[DetectionRecord],
        scope: OutlierScope,
        context: ProcessingContext,
        params: dict[str, float | int] | None = None,
    ) -> list[OutlierHit]:
        settings = self.normalize_params(params)
        residual_threshold = float(settings["center_residual_m"])
        min_track_length = int(settings["min_track_length"])
        target_ids = context.target_det_ids(scope)
        hits: list[OutlierHit] = []
        for track_id, track_records in context.track_groups().items():
            if len(track_records) < min_track_length:
                continue
            for index in range(1, len(track_records) - 1):
                record = track_records[index]
                if record.det_id not in target_ids:
                    continue
                prev_record = track_records[index - 1]
                next_record = track_records[index + 1]
                pred_center = _linear_interpolate_vector(
                    prev_record.frame_index,
                    np.array([prev_record.center_x, prev_record.center_y, prev_record.center_z], dtype=np.float64),
                    next_record.frame_index,
                    np.array([next_record.center_x, next_record.center_y, next_record.center_z], dtype=np.float64),
                    record.frame_index,
                )
                center = np.array([record.center_x, record.center_y, record.center_z], dtype=np.float64)
                residual = float(np.linalg.norm(center - pred_center))
                if residual < residual_threshold:
                    continue
                hits.append(
                    OutlierHit(
                        rule_id=self.rule_id,
                        frame_index=record.frame_index,
                        det_id=record.det_id,
                        track_id=track_id,
                        category=record.category,
                        severity=residual / max(residual_threshold, EPS),
                        message=f"Center spike: residual {residual:.3f} m",
                        fixable=True,
                        metadata={
                            "pred_center_x": float(pred_center[0]),
                            "pred_center_y": float(pred_center[1]),
                            "pred_center_z": float(pred_center[2]),
                            "residual_m": residual,
                        },
                    )
                )
        return hits

    def fix(
        self,
        records: list[DetectionRecord],
        hits: Sequence[OutlierHit],
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        updated = 0
        affected: list[str] = []
        for hit in hits:
            record = context.record_by_id(hit.det_id)
            if record is None:
                continue
            pred_x = float(hit.metadata.get("pred_center_x", record.center_x))
            pred_y = float(hit.metadata.get("pred_center_y", record.center_y))
            pred_z = float(hit.metadata.get("pred_center_z", record.center_z))

            def mutate(item: DetectionRecord) -> None:
                item.center_x = pred_x
                item.center_y = pred_y
                item.center_z = pred_z

            if _try_update_geometry(record, context, mutate):
                updated += 1
                affected.append(record.det_id)
        return OperationResult(
            updated_count=updated,
            message=f"Center spike fix updated {updated} detections.",
            affected_det_ids=tuple(sorted(set(affected))),
        )


class SmoothAnglesOperation(BulkOperation):
    operation_id = "smooth_angles"
    display_name = "Smooth Angles"
    param_specs = (
        ParameterSpec("yaw_alpha", "Yaw Alpha", "float", 0.55, 0.0, 0.99, 0.01, 2),
        ParameterSpec("pitch_roll_alpha", "Pitch/Roll Alpha", "float", 0.30, 0.0, 0.99, 0.01, 2),
    )

    @staticmethod
    def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
        smoothed = np.asarray(values, dtype=np.float64).copy()
        for index in range(1, len(smoothed)):
            smoothed[index] = alpha * smoothed[index - 1] + (1.0 - alpha) * smoothed[index]
        return smoothed

    def analyze_track(self, track_records: Sequence[DetectionRecord], yaw_alpha: float, pr_alpha: float) -> dict[str, np.ndarray]:
        yaw_values = _unwrap_angles_deg([record.yaw_deg for record in track_records])
        yaw_forward = self._ema(yaw_values, yaw_alpha)
        yaw_backward = self._ema(yaw_values[::-1], yaw_alpha)[::-1]
        yaw_smoothed = (yaw_forward + yaw_backward) * 0.5

        pitch_values = np.asarray([record.pitch_deg for record in track_records], dtype=np.float64)
        roll_values = np.asarray([record.roll_deg for record in track_records], dtype=np.float64)
        pitch_forward = self._ema(pitch_values, pr_alpha)
        pitch_backward = self._ema(pitch_values[::-1], pr_alpha)[::-1]
        roll_forward = self._ema(roll_values, pr_alpha)
        roll_backward = self._ema(roll_values[::-1], pr_alpha)[::-1]

        return {
            "yaw": yaw_smoothed,
            "pitch": (pitch_forward + pitch_backward) * 0.5,
            "roll": (roll_forward + roll_backward) * 0.5,
        }

    def apply(
        self,
        records: list[DetectionRecord],
        scope: OperationScope,
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        settings = self.normalize_params(params)
        yaw_alpha = float(settings["yaw_alpha"])
        pr_alpha = float(settings["pitch_roll_alpha"])
        target_ids = context.target_det_ids(scope)
        updated = 0
        affected: list[str] = []
        for track_records in context.track_groups().values():
            if len(track_records) < 2:
                continue
            smoothed = self.analyze_track(track_records, yaw_alpha, pr_alpha)
            for index, record in enumerate(track_records):
                if record.det_id not in target_ids:
                    continue

                def mutate(item: DetectionRecord, i: int = index) -> None:
                    item.yaw_deg = _normalize_angle_deg(float(smoothed["yaw"][i]))
                    item.pitch_deg = float(smoothed["pitch"][i])
                    item.roll_deg = float(smoothed["roll"][i])

                if _try_update_geometry(record, context, mutate):
                    updated += 1
                    affected.append(record.det_id)
        return OperationResult(
            updated_count=updated,
            message=f"Angle smoothing updated {updated} detections.",
            affected_det_ids=tuple(sorted(set(affected))),
        )


class FixTrackSizeOperation(BulkOperation):
    operation_id = "fix_track_size"
    display_name = "Fix Track Size"

    def apply(
        self,
        records: list[DetectionRecord],
        scope: OperationScope,
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        del params
        target_ids = context.target_det_ids(scope)
        refs = _track_size_map(context.track_groups())
        updated = 0
        affected: list[str] = []
        for track_id, track_records in context.track_groups().items():
            if track_id not in refs:
                continue
            ref_w, ref_l, ref_h = refs[track_id]
            for record in track_records:
                if record.det_id not in target_ids:
                    continue

                def mutate(item: DetectionRecord) -> None:
                    item.size_w = ref_w
                    item.size_l = ref_l
                    item.size_h = ref_h

                if _try_update_geometry(record, context, mutate):
                    updated += 1
                    affected.append(record.det_id)
        return OperationResult(
            updated_count=updated,
            message=f"Track size fix updated {updated} detections.",
            affected_det_ids=tuple(sorted(set(affected))),
        )


class _CenterKalmanFilter:
    def __init__(self, process_noise_pos: float, process_noise_vel: float, measurement_noise_pos: float) -> None:
        self.dim_x = 6
        self.dim_z = 3
        self.x = np.zeros(self.dim_x, dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64)
        self.initialized = False
        self.process_noise_pos = float(process_noise_pos)
        self.process_noise_vel = float(process_noise_vel)
        self.measurement_noise_pos = float(measurement_noise_pos)

    def _transition(self, frame_delta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = max(1.0, float(frame_delta))
        F = np.eye(self.dim_x, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        H[:, :3] = np.eye(3, dtype=np.float64)

        Q = np.diag(
            [
                self.process_noise_pos ** 2,
                self.process_noise_pos ** 2,
                self.process_noise_pos ** 2,
                self.process_noise_vel ** 2,
                self.process_noise_vel ** 2,
                self.process_noise_vel ** 2,
            ]
        )
        return F, H, Q

    def init_state(self, center: np.ndarray) -> None:
        self.x = np.zeros(self.dim_x, dtype=np.float64)
        self.x[:3] = np.asarray(center, dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64)
        self.P[:3, :3] *= 1.0
        self.P[3:, 3:] *= 10.0
        self.initialized = True

    def predict(self, frame_delta: int) -> None:
        if not self.initialized:
            return
        F, _, Q = self._transition(frame_delta)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, center: np.ndarray, frame_delta: int) -> None:
        observation = np.asarray(center, dtype=np.float64)
        if not self.initialized:
            self.init_state(observation)
            return
        F, H, Q = self._transition(frame_delta)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        R = np.diag([self.measurement_noise_pos ** 2] * 3)
        innovation = observation - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.dim_x, dtype=np.float64) - K @ H) @ self.P

    def state(self) -> np.ndarray:
        return self.x[:3].copy()


class SmoothTrackCentersOperation(BulkOperation):
    operation_id = "smooth_track_centers"
    display_name = "Smooth Track Centers"
    param_specs = (
        ParameterSpec("process_noise_pos", "Process Noise Pos", "float", 0.5, 0.01, 100.0, 0.05, 3),
        ParameterSpec("process_noise_vel", "Process Noise Vel", "float", 1.0, 0.01, 100.0, 0.05, 3),
        ParameterSpec("measurement_noise_pos", "Measurement Noise", "float", 1.0, 0.01, 100.0, 0.05, 3),
    )

    def apply(
        self,
        records: list[DetectionRecord],
        scope: OperationScope,
        params: dict[str, float | int] | None,
        context: ProcessingContext,
    ) -> OperationResult:
        settings = self.normalize_params(params)
        target_ids = context.target_det_ids(scope)
        updated = 0
        affected: list[str] = []
        for track_records in context.track_groups().values():
            if len(track_records) < 2:
                continue
            kf = _CenterKalmanFilter(
                process_noise_pos=float(settings["process_noise_pos"]),
                process_noise_vel=float(settings["process_noise_vel"]),
                measurement_noise_pos=float(settings["measurement_noise_pos"]),
            )
            smoothed_centers: list[np.ndarray] = []
            prev_frame = None
            for record in track_records:
                center = np.array([record.center_x, record.center_y, record.center_z], dtype=np.float64)
                frame_delta = 1 if prev_frame is None else max(1, int(record.frame_index) - int(prev_frame))
                kf.update(center, frame_delta)
                smoothed_centers.append(kf.state())
                prev_frame = record.frame_index

            for index, record in enumerate(track_records):
                if record.det_id not in target_ids:
                    continue
                smoothed_center = smoothed_centers[index]

                def mutate(item: DetectionRecord, value: np.ndarray = smoothed_center) -> None:
                    item.center_x = float(value[0])
                    item.center_y = float(value[1])
                    item.center_z = float(value[2])

                if _try_update_geometry(record, context, mutate):
                    updated += 1
                    affected.append(record.det_id)
        return OperationResult(
            updated_count=updated,
            message=f"Track center smoothing updated {updated} detections.",
            affected_det_ids=tuple(sorted(set(affected))),
        )


def build_default_outlier_registry() -> OutlierRuleRegistry:
    registry = OutlierRuleRegistry()
    registry.register(EulerAngleSpikeRule("yaw"))
    registry.register(EulerAngleSpikeRule("pitch"))
    registry.register(EulerAngleSpikeRule("roll"))
    registry.register(SizeSpikeRule())
    registry.register(CenterSpikeRule())
    return registry


def build_default_bulk_operation_registry() -> BulkOperationRegistry:
    registry = BulkOperationRegistry()
    registry.register(SmoothAnglesOperation())
    registry.register(FixTrackSizeOperation())
    registry.register(SmoothTrackCentersOperation())
    return registry


def hits_to_report_json(
    hits: Sequence[OutlierHit],
    *,
    csv_path: str,
    scope: str,
    enabled_rule_ids: Sequence[str],
) -> str:
    payload = {
        "csv_path": csv_path,
        "scope": scope,
        "enabled_rules": list(enabled_rule_ids),
        "hit_count": len(hits),
        "hits": [hit.to_report_row() for hit in hits],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
