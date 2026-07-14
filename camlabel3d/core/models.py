"""Data models used across UI, inference, and CSV persistence."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .geometry import box10_quaternion_to_box9d


class PromptMode(str, Enum):
    """Supported WildDet3D prompt modes."""

    TEXT = "Text"
    BOX_MULTI = "Box-to-Multi-Object"
    BOX_SINGLE = "Box-to-Single-Object"
    POINT = "Point"


class SourceMode(str, Enum):
    """Supported input-source modes in the desktop UI."""

    DATASET = "Dataset"
    VIDEO = "Video"
    IMAGE_FOLDER = "Image Folder"


@dataclass
class PointPrompt:
    """Single image-space point prompt."""

    x: float
    y: float
    label: int

    def to_tuple(self) -> tuple[float, float, int]:
        return (float(self.x), float(self.y), int(self.label))

    @classmethod
    def from_tuple(cls, value: tuple[float, float, int]) -> "PointPrompt":
        return cls(x=float(value[0]), y=float(value[1]), label=int(value[2]))


@dataclass
class PromptSpec:
    """Prompt state captured from the current UI session."""

    mode: PromptMode = PromptMode.TEXT
    text_prompt: str = ""
    prompt_label: str = ""
    box: tuple[float, float, float, float] | None = None
    points: list[PointPrompt] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mode = PromptMode(self.mode)
        if self.box is not None:
            self.box = tuple(float(value) for value in self.box)
        normalized_points: list[PointPrompt] = []
        for point in self.points:
            if isinstance(point, PointPrompt):
                normalized_points.append(point)
            else:
                normalized_points.append(PointPrompt.from_tuple(point))
        self.points = normalized_points

    def is_valid(self) -> bool:
        if self.mode == PromptMode.TEXT:
            return bool(self.parsed_texts())
        if self.mode in (PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE):
            return self.box is not None
        if self.mode == PromptMode.POINT:
            return bool(self.points)
        return False

    def parsed_texts(self) -> list[str]:
        raw = self.text_prompt.replace("\n", ".").replace(",", ".")
        parts = [item.strip() for item in raw.split(".")]
        parsed = [item for item in parts if item]
        return parsed or ["object"]

    def prompt_text_for_model(self) -> str:
        label = self.prompt_label.strip()
        if self.mode == PromptMode.BOX_MULTI:
            prefix = "visual"
        else:
            prefix = "geometric"
        return f"{prefix}: {label}" if label else prefix

    def class_name_fallback(self) -> str:
        label = self.prompt_label.strip()
        if label:
            return label
        if self.mode == PromptMode.TEXT:
            texts = self.parsed_texts()
            return texts[0] if texts else "object"
        return "object"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode.value,
            "prompt_label": self.prompt_label.strip(),
        }
        if self.mode == PromptMode.TEXT:
            payload["texts"] = self.parsed_texts()
        elif self.mode in (PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE):
            payload["box_xyxy"] = list(self.box) if self.box is not None else None
        elif self.mode == PromptMode.POINT:
            payload["points"] = [asdict(point) for point in self.points]
        return payload

    def payload_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def clone(self) -> "PromptSpec":
        return PromptSpec(
            mode=self.mode,
            text_prompt=self.text_prompt,
            prompt_label=self.prompt_label,
            box=tuple(self.box) if self.box is not None else None,
            points=[PointPrompt(point.x, point.y, point.label) for point in self.points],
        )

    @classmethod
    def from_payload_json(cls, payload_json: str) -> "PromptSpec":
        payload = json.loads(payload_json) if payload_json else {}
        mode = PromptMode(payload.get("mode", PromptMode.TEXT.value))
        text_prompt = ".".join(payload.get("texts", [])) if mode == PromptMode.TEXT else ""
        box = payload.get("box_xyxy")
        points = [PointPrompt(**item) for item in payload.get("points", [])]
        return cls(
            mode=mode,
            text_prompt=text_prompt,
            prompt_label=str(payload.get("prompt_label", "")),
            box=tuple(float(v) for v in box) if box is not None else None,
            points=points,
        )


@dataclass
class DetectionConfig:
    """Runtime inference controls exposed in the UI."""

    start_frame: int = 0
    end_frame: int = 0
    frame_step: int = 1
    score_threshold: float = 0.3
    score_3d_threshold: float = 0.1
    cross_category_nms_iou: float = 0.8
    use_actual_intrinsics: bool = False
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    def to_intrinsics_matrix(self) -> np.ndarray | None:
        if not self.use_actual_intrinsics:
            return None
        if None in (self.fx, self.fy, self.cx, self.cy):
            raise ValueError("Actual intrinsics are enabled but fx/fy/cx/cy are missing.")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Actual intrinsics require fx and fy greater than 0.")
        return np.array(
            [
                [float(self.fx), 0.0, float(self.cx)],
                [0.0, float(self.fy), float(self.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def clamped_range(self, frame_count: int) -> list[int]:
        if frame_count <= 0:
            return []
        start = max(0, min(int(self.start_frame), frame_count - 1))
        end = max(0, min(int(self.end_frame), frame_count - 1))
        step = max(1, int(self.frame_step))
        if end < start:
            start, end = end, start
        return list(range(start, end + 1, step))


@dataclass
class SourceContext:
    """Metadata describing the currently active source."""

    source_mode: SourceMode
    source_type: str
    dataset_id: str = ""
    recording_id: str = ""

    def __post_init__(self) -> None:
        self.source_mode = SourceMode(self.source_mode)


CSV_FIELD_ORDER = [
    "frame_index",
    "category",
    "score",
    "score_2d",
    "score_3d",
    "box2d_x1",
    "box2d_y1",
    "box2d_x2",
    "box2d_y2",
    "center_x",
    "center_y",
    "center_z",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "size_w",
    "size_l",
    "size_h",
    "is_enabled",
    "track_id",
    "track_status",
]


@dataclass
class DetectionRecord:
    """Single detection row persisted to the unified CSV."""

    frame_index: int
    category: str
    score: float
    score_2d: float
    score_3d: float
    box2d_x1: float
    box2d_y1: float
    box2d_x2: float
    box2d_y2: float
    center_x: float
    center_y: float
    center_z: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    size_w: float
    size_l: float
    size_h: float
    is_enabled: bool = True
    track_id: str = ""
    track_status: str = ""
    det_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    source_type: str = ""
    source_path: str = ""
    dataset_id: str = ""
    recording_id: str = ""
    timestamp_ms: float | None = None
    image_path: str = ""
    prompt_mode: str = PromptMode.TEXT.value
    prompt_label: str = ""
    prompt_payload_json: str = ""
    prompt_group_id: str = ""
    input_fx: float | None = None
    input_fy: float | None = None
    input_cx: float | None = None
    input_cy: float | None = None
    pred_fx: float | None = None
    pred_fy: float | None = None
    pred_cx: float | None = None
    pred_cy: float | None = None
    use_actual_intrinsics: bool = False

    @staticmethod
    def new_det_id() -> str:
        return str(uuid.uuid4())

    def box2d_xyxy(self) -> tuple[float, float, float, float]:
        return (self.box2d_x1, self.box2d_y1, self.box2d_x2, self.box2d_y2)

    def box9d_array(self) -> list[float]:
        return [
            self.center_x,
            self.center_y,
            self.center_z,
            self.yaw_deg,
            self.pitch_deg,
            self.roll_deg,
            self.size_w,
            self.size_l,
            self.size_h,
        ]

    def intrinsics_for_preview(self, image_shape: tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        if self.use_actual_intrinsics and None not in (
            self.input_fx,
            self.input_fy,
            self.input_cx,
            self.input_cy,
        ):
            return np.array(
                [
                    [self.input_fx, 0.0, self.input_cx],
                    [0.0, self.input_fy, self.input_cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        if None not in (self.pred_fx, self.pred_fy, self.pred_cx, self.pred_cy):
            return np.array(
                [
                    [self.pred_fx, 0.0, self.pred_cx],
                    [0.0, self.pred_fy, self.pred_cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        focal = max(float(h), float(w))
        return np.array(
            [
                [focal, 0.0, w / 2.0],
                [0.0, focal, h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def to_row(self) -> dict[str, str]:
        row: dict[str, str] = {}
        for field_name in CSV_FIELD_ORDER:
            value = getattr(self, field_name)
            if isinstance(value, bool):
                row[field_name] = "1" if value else "0"
            elif value is None:
                row[field_name] = ""
            elif isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    row[field_name] = ""
                else:
                    row[field_name] = f"{value:.10g}"
            else:
                row[field_name] = str(value)
        return row

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "DetectionRecord":
        def parse_float(value: str) -> float | None:
            value = (value or "").strip()
            return float(value) if value else None

        def parse_int(value: str) -> int:
            return int(float(value or "0"))

        def parse_bool(value: str) -> bool:
            return str(value).strip().lower() in {"1", "true", "yes", "y"}

        kwargs = {
            "frame_index": parse_int(row.get("frame_index", "0")),
            "category": row.get("category", "object"),
            "score": float(row.get("score", "0") or 0.0),
            "score_2d": float(row.get("score_2d", "0") or 0.0),
            "score_3d": float(row.get("score_3d", "0") or 0.0),
            "box2d_x1": float(row.get("box2d_x1", "0") or 0.0),
            "box2d_y1": float(row.get("box2d_y1", "0") or 0.0),
            "box2d_x2": float(row.get("box2d_x2", "0") or 0.0),
            "box2d_y2": float(row.get("box2d_y2", "0") or 0.0),
            "center_x": float(row.get("center_x", "0") or 0.0),
            "center_y": float(row.get("center_y", "0") or 0.0),
            "center_z": float(row.get("center_z", "0") or 0.0),
            "yaw_deg": parse_float(row.get("yaw_deg", "")),
            "pitch_deg": parse_float(row.get("pitch_deg", "")),
            "roll_deg": parse_float(row.get("roll_deg", "")),
            "size_w": float(row.get("size_w", "0") or 0.0),
            "size_l": float(row.get("size_l", "0") or 0.0),
            "size_h": float(row.get("size_h", "0") or 0.0),
            "is_enabled": parse_bool(row.get("is_enabled", "1")),
            "track_id": row.get("track_id", ""),
            "track_status": row.get("track_status", ""),
            "det_id": cls.new_det_id(),
            "source_id": row.get("source_id", ""),
            "source_type": row.get("source_type", ""),
            "source_path": row.get("source_path", ""),
            "dataset_id": row.get("dataset_id", ""),
            "recording_id": row.get("recording_id", ""),
            "timestamp_ms": parse_float(row.get("timestamp_ms", "")),
            "image_path": row.get("image_path", ""),
            "prompt_mode": row.get("prompt_mode", PromptMode.TEXT.value),
            "prompt_label": row.get("prompt_label", ""),
            "prompt_payload_json": row.get("prompt_payload_json", ""),
            "prompt_group_id": row.get("prompt_group_id", ""),
            "input_fx": parse_float(row.get("input_fx", "")),
            "input_fy": parse_float(row.get("input_fy", "")),
            "input_cx": parse_float(row.get("input_cx", "")),
            "input_cy": parse_float(row.get("input_cy", "")),
            "pred_fx": parse_float(row.get("pred_fx", "")),
            "pred_fy": parse_float(row.get("pred_fy", "")),
            "pred_cx": parse_float(row.get("pred_cx", "")),
            "pred_cy": parse_float(row.get("pred_cy", "")),
            "use_actual_intrinsics": parse_bool(row.get("use_actual_intrinsics", "0")),
        }
        if None in (kwargs["yaw_deg"], kwargs["pitch_deg"], kwargs["roll_deg"]):
            legacy_quat = [
                float(row.get("quat_w", "1") or 1.0),
                float(row.get("quat_x", "0") or 0.0),
                float(row.get("quat_y", "0") or 0.0),
                float(row.get("quat_z", "0") or 0.0),
            ]
            legacy_box = [
                kwargs["center_x"],
                kwargs["center_y"],
                kwargs["center_z"],
                kwargs["size_w"],
                kwargs["size_l"],
                kwargs["size_h"],
                legacy_quat[0],
                legacy_quat[1],
                legacy_quat[2],
                legacy_quat[3],
            ]
            legacy_9d = box10_quaternion_to_box9d(legacy_box)
            kwargs["yaw_deg"] = legacy_9d["yaw_deg"]
            kwargs["pitch_deg"] = legacy_9d["pitch_deg"]
            kwargs["roll_deg"] = legacy_9d["roll_deg"]
        kwargs["yaw_deg"] = float(kwargs["yaw_deg"] or 0.0)
        kwargs["pitch_deg"] = float(kwargs["pitch_deg"] or 0.0)
        kwargs["roll_deg"] = float(kwargs["roll_deg"] or 0.0)
        return cls(**kwargs)

    @classmethod
    def sort_key(cls, record: "DetectionRecord") -> tuple[int, str, str]:
        return (record.frame_index, record.category.lower(), record.det_id)


def natural_sort_key(value: str) -> list[Any]:
    """Return a human-friendly sort key for filenames."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]
