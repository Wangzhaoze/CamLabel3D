#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_WILDDET3D_SOURCE_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = DEFAULT_WILDDET3D_SOURCE_ROOT.parent.parent
WILDDET3D_SOURCE_ROOT = Path(
    os.getenv("WILDDET3D_SOURCE_ROOT", str(DEFAULT_WILDDET3D_SOURCE_ROOT))
).resolve()
DEFAULT_WILDDET3D_CKPT_ROOT = DEFAULT_PROJECT_ROOT / "ckpts"
WILDDET3D_CKPT_ROOT = Path(os.getenv("WILDDET3D_CKPT_ROOT", str(DEFAULT_WILDDET3D_CKPT_ROOT))).resolve()
WILDDET3D_CKPT_PATH = WILDDET3D_CKPT_ROOT / "wilddet3d_alldata_all_prompt_v1.0.pt"
DEFAULT_WILDDET3D_MODEL_ID = "wilddet3d_camera_only"
DEFAULT_TOOLKIT_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
DEFAULT_PROMPT_TEXTS = ["car", "person", "bicycle"]

local_lingbot_checkpoint = WILDDET3D_CKPT_ROOT / "lingbot_depth_model.pt"
if local_lingbot_checkpoint.exists():
    os.environ.setdefault("WILDDET3D_LINGBOT_DEPTH_MODEL", str(local_lingbot_checkpoint))

if str(WILDDET3D_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDDET3D_SOURCE_ROOT))


class WildDet3DInferenceItem(BaseModel):
    frame_index: int
    image_path: str
    intrinsics: list[list[float]]
    lidar_to_camera: list[list[float]]

    @field_validator("intrinsics")
    @classmethod
    def validate_intrinsics(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("WildDet3D requires a 3x3 camera intrinsics matrix")
        return value

    @field_validator("lidar_to_camera")
    @classmethod
    def validate_lidar_to_camera(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("WildDet3D requires a 4x4 lidar_to_camera transform")
        return value


class WildDet3DInferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str | None = None
    model_name: str | None = None
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    class_names: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLKIT_CLASS_NAMES))
    prompt_texts: list[str] | None = None
    items: list[WildDet3DInferenceItem]


class WildDet3DProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    name: str
    architecture: str
    framework: str
    task: str
    weight_name: str
    description: str
    ckpt_path: str
    class_names: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLKIT_CLASS_NAMES))
    prompt_texts: list[str] = Field(default_factory=lambda: list(DEFAULT_PROMPT_TEXTS))
    default_score_threshold: float = 0.3
    requires: list[str] = Field(
        default_factory=lambda: ["camera", "camera_intrinsics", "lidar_to_camera_transform"]
    )
    is_available: bool


class WorkerBox3D(BaseModel):
    id: str
    class_name: str
    score: float
    center: dict[str, float]
    size: dict[str, float]
    yaw: float
    source_model: str
    coord_frame: Literal["lidar"] = "lidar"


class WildDet3DInferenceResult(BaseModel):
    frame_index: int
    annotations: list[WorkerBox3D]


class WildDet3DInferenceResponse(BaseModel):
    provider: str = "wilddet3d"
    model_id: str
    model_name: str
    task: str = "3d_camera_bbox"
    results: list[WildDet3DInferenceResult]


class WildDet3DCatalogResponse(BaseModel):
    provider: str = "wilddet3d"
    default_model_id: str
    profiles: list[WildDet3DProfile]


WILDDET3D_PROFILES = {
    DEFAULT_WILDDET3D_MODEL_ID: {
        "id": DEFAULT_WILDDET3D_MODEL_ID,
        "name": "WildDet3D Camera Only",
        "architecture": "WildDet3D",
        "framework": "WildDet3D",
        "task": "3d_camera_bbox",
        "weight_name": WILDDET3D_CKPT_PATH.name,
        "description": "Camera-only 3D bounding box detector. Requires image, intrinsics, and lidar-camera calibration.",
        "ckpt_path": str(WILDDET3D_CKPT_PATH),
        "class_names": DEFAULT_TOOLKIT_CLASS_NAMES,
        "prompt_texts": DEFAULT_PROMPT_TEXTS,
        "default_score_threshold": 0.3,
    }
}


def _tensor_to_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quaternion_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_float_list(matrix: list[list[float]], shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _class_prompt_pairs(class_names: list[str], prompt_texts: list[str] | None) -> tuple[list[str], list[str]]:
    if prompt_texts:
        prompts = [str(item).strip() for item in prompt_texts if str(item).strip()]
    else:
        aliases = {
            "car": "car",
            "vehicle": "car",
            "pedestrian": "person",
            "person": "person",
            "cyclist": "bicycle",
            "bicycle": "bicycle",
        }
        prompts = [aliases.get(str(name).strip().lower(), str(name).strip().lower()) for name in class_names]

    classes = [str(name).strip() for name in class_names if str(name).strip()]
    if not classes:
        classes = list(DEFAULT_TOOLKIT_CLASS_NAMES)
    if len(prompts) != len(classes):
        classes = list(DEFAULT_TOOLKIT_CLASS_NAMES)
        prompts = list(DEFAULT_PROMPT_TEXTS)
    return classes, prompts


def _decode_class_name(class_ids: np.ndarray, index: int, class_names: list[str]) -> str:
    if index >= len(class_ids):
        return class_names[0]
    class_id = int(class_ids[index])
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    if 1 <= class_id <= len(class_names):
        return class_names[class_id - 1]
    return class_names[0]


def _camera_box_to_lidar(box3d: np.ndarray, lidar_to_camera: np.ndarray) -> tuple[dict[str, float], dict[str, float], float]:
    if box3d.shape[0] < 10:
        raise ValueError(f"WildDet3D box must have at least 10 values, got {box3d.shape[0]}")

    camera_to_lidar = np.linalg.inv(lidar_to_camera)
    center_camera = np.array([box3d[0], box3d[1], box3d[2], 1.0], dtype=np.float64)
    center_lidar = camera_to_lidar @ center_camera

    dims_wlh = np.maximum(np.asarray(box3d[3:6], dtype=np.float64), 1e-3)
    width, length, height = [float(value) for value in dims_wlh]

    rotation_camera_box = _quaternion_wxyz_to_matrix(np.asarray(box3d[6:10], dtype=np.float64))
    rotation_lidar_box = camera_to_lidar[:3, :3] @ rotation_camera_box
    heading = rotation_lidar_box[:, 1]
    yaw = math.atan2(float(heading[1]), float(heading[0]))

    center = {
        "x": float(center_lidar[0]),
        "y": float(center_lidar[1]),
        "z": float(center_lidar[2]),
    }
    size = {
        "x": length,
        "y": width,
        "z": height,
    }
    return center, size, float(yaw)


class WildDet3DEngine:
    def catalog(self) -> WildDet3DCatalogResponse:
        profiles = [
            WildDet3DProfile(
                **profile,
                is_available=Path(profile["ckpt_path"]).exists(),
            )
            for profile in WILDDET3D_PROFILES.values()
        ]
        return WildDet3DCatalogResponse(default_model_id=DEFAULT_WILDDET3D_MODEL_ID, profiles=profiles)

    def _resolve_profile(self, model_id: str | None) -> dict:
        resolved_model_id = model_id or DEFAULT_WILDDET3D_MODEL_ID
        profile = WILDDET3D_PROFILES.get(resolved_model_id)
        if profile is None:
            raise ValueError(f"Unknown WildDet3D model profile: {resolved_model_id}")
        return profile

    @lru_cache(maxsize=2)
    def _load_model(self, ckpt_path: str, score_threshold: float):
        import torch
        from wilddet3d import build_model

        checkpoint = Path(ckpt_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"WildDet3D checkpoint not found: {checkpoint}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(
            checkpoint=str(checkpoint),
            score_threshold=float(score_threshold),
            skip_pretrained=True,
            use_predicted_intrinsics=False,
        )
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model, device

    def _infer_item(
        self,
        *,
        item: WildDet3DInferenceItem,
        profile: dict,
        score_threshold: float,
        class_names: list[str],
        prompt_texts: list[str],
    ) -> WildDet3DInferenceResult:
        import torch
        from wilddet3d import preprocess

        image_path = Path(item.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Camera image file not found: {item.image_path}")

        intrinsics = _matrix_to_float_list(item.intrinsics, (3, 3), "intrinsics").astype(np.float32)
        lidar_to_camera = _matrix_to_float_list(item.lidar_to_camera, (4, 4), "lidar_to_camera")

        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.float32)

        model, device = self._load_model(profile["ckpt_path"], float(score_threshold))
        data = preprocess(image, intrinsics)

        with torch.no_grad():
            outputs = model(
                images=data["images"].to(device),
                intrinsics=data["intrinsics"].to(device)[None],
                input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]],
                padding=[data["padding"]],
                input_texts=prompt_texts,
            )

        if len(outputs) >= 7:
            _, boxes3d, scores, scores_2d, scores_3d, raw_class_ids, _ = outputs[:7]
        elif len(outputs) >= 5:
            _, boxes3d, scores, raw_class_ids, _ = outputs[:5]
            scores_2d = scores
            scores_3d = scores
        else:
            raise ValueError("WildDet3D returned an unexpected output tuple")

        boxes3d_np = _tensor_to_numpy(boxes3d[0]).astype(np.float64, copy=False)
        scores_np = _tensor_to_numpy(scores[0]).astype(np.float64, copy=False)
        scores_2d_np = _tensor_to_numpy(scores_2d[0]).astype(np.float64, copy=False)
        scores_3d_np = _tensor_to_numpy(scores_3d[0]).astype(np.float64, copy=False)
        class_ids_np = _tensor_to_numpy(raw_class_ids[0]).astype(np.int64, copy=False)

        annotations: list[WorkerBox3D] = []
        for annotation_index, box3d in enumerate(boxes3d_np):
            if box3d.shape[0] < 10:
                continue
            combined_score = float(scores_np[annotation_index]) if annotation_index < len(scores_np) else 0.0
            score_3d = float(scores_3d_np[annotation_index]) if annotation_index < len(scores_3d_np) else combined_score
            score_2d = float(scores_2d_np[annotation_index]) if annotation_index < len(scores_2d_np) else combined_score
            score = max(combined_score, score_2d, score_3d)
            if score < score_threshold:
                continue
            score = min(1.0, max(0.0, score))
            center, size, yaw = _camera_box_to_lidar(box3d, lidar_to_camera)
            annotations.append(
                WorkerBox3D(
                    id=f"wilddet3d-{item.frame_index}-{annotation_index}",
                    class_name=_decode_class_name(class_ids_np, annotation_index, class_names),
                    score=score,
                    center=center,
                    size=size,
                    yaw=yaw,
                    source_model=profile["name"],
                )
            )

        return WildDet3DInferenceResult(frame_index=item.frame_index, annotations=annotations)

    def infer(self, request: WildDet3DInferenceRequest) -> WildDet3DInferenceResponse:
        profile = self._resolve_profile(request.model_id)
        score_threshold = float(request.score_threshold if request.score_threshold is not None else profile["default_score_threshold"])
        class_names, prompt_texts = _class_prompt_pairs(request.class_names, request.prompt_texts or profile.get("prompt_texts"))
        results = [
            self._infer_item(
                item=item,
                profile=profile,
                score_threshold=score_threshold,
                class_names=class_names,
                prompt_texts=prompt_texts,
            )
            for item in request.items
        ]
        return WildDet3DInferenceResponse(
            model_id=profile["id"],
            model_name=request.model_name or profile["name"],
            results=results,
        )


engine = WildDet3DEngine()
