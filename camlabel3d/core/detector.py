"""WildDet3D inference adapter and preview rendering helpers."""

from __future__ import annotations

import gc
import traceback
import uuid
from contextlib import nullcontext
from pathlib import Path
from threading import Condition, Lock
from typing import Any, Callable

import numpy as np
from PIL import Image

from camlabel3d.diagnostics.gpu import format_result, run_gpu_diagnostic_isolated
from camlabel3d.runtime import default_checkpoint_path, ensure_wilddet3d_on_path
from camlabel3d.runtime_config import RuntimeConfig

from .frame_provider import FrameProvider
from .geometry import (
    Box3DOverlay,
    box10_quaternion_to_box9d,
    box9d_to_box10_quaternion,
    project_box3d_quaternion_to_2d_bounds,
    render_box3d_scene_rgb,
)
from .models import (
    DetectionConfig,
    DetectionRecord,
    PromptMode,
    PromptSpec,
    SourceContext,
    SourceMode,
)


_PREVIEW_CATEGORY_COLORS: tuple[tuple[int, int, int], ...] = (
    (66, 165, 245),
    (255, 159, 67),
    (46, 204, 113),
    (238, 90, 111),
    (165, 94, 234),
    (38, 222, 190),
    (255, 206, 84),
    (84, 160, 255),
    (255, 107, 129),
    (72, 219, 251),
    (29, 209, 161),
    (190, 190, 190),
)


class DetectorAdapter:
    """Thin inference layer over WildDet3D with CSV-friendly outputs."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self.runtime_config = runtime_config or RuntimeConfig.from_env()
        self.checkpoint_path = Path(checkpoint_path or default_checkpoint_path()).resolve()
        self.device = device or self.runtime_config.device
        self._models: dict[bool, Any] = {}
        self._build_error: str | None = None
        self._model_condition = Condition()
        self._building_variant: bool | None = None
        self._resolved_device: str | None = None
        self._torch_configured = False
        self._cuda_preflight_lock = Lock()
        self._cuda_preflight_device: str | None = None
        self._cuda_preflight_error: str | None = None

    def run_range(
        self,
        provider: FrameProvider,
        prompt_spec: PromptSpec,
        config: DetectionConfig,
        source_context: SourceContext | None = None,
        frame_indices: list[int] | None = None,
        prompt_group_id: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[DetectionRecord]:
        indices = frame_indices or config.clamped_range(provider.frame_count)
        if not indices:
            return []
        group_id = prompt_group_id or str(uuid.uuid4())
        results: list[DetectionRecord] = []
        total = len(indices)
        for offset, frame_index in enumerate(indices, start=1):
            if should_cancel and should_cancel():
                break
            if progress_callback:
                progress_callback(offset - 1, total, f"Running frame {frame_index}")
            results.extend(
                self.run_frame(
                    provider=provider,
                    frame_index=frame_index,
                    prompt_spec=prompt_spec,
                    config=config,
                    source_context=source_context,
                    prompt_group_id=group_id,
                )
            )
        if progress_callback:
            progress_callback(total, total, "Detection finished")
        return results

    def run_frame(
        self,
        provider: FrameProvider,
        frame_index: int,
        prompt_spec: PromptSpec,
        config: DetectionConfig,
        source_context: SourceContext | None = None,
        prompt_group_id: str | None = None,
    ) -> list[DetectionRecord]:
        if not prompt_spec.is_valid():
            raise ValueError("Current prompt is incomplete.")

        ensure_wilddet3d_on_path()
        from wilddet3d.preprocessing import preprocess

        frame_rgb = provider.get_frame(frame_index).astype(np.float32)
        intrinsics = config.to_intrinsics_matrix()
        data = preprocess(frame_rgb, intrinsics)

        use_predicted_intrinsics = not config.use_actual_intrinsics
        model = self._get_model(use_predicted_intrinsics=use_predicted_intrinsics)
        results = self._run_model(model, data, prompt_spec)
        filtered = self._postprocess_results(results, prompt_spec, config)

        predicted_k_scaled = self._scale_intrinsics_to_original(
            filtered["predicted_intrinsics"],
            data["input_hw"],
            data["original_hw"],
        )

        image_path = provider.get_image_path(frame_index)
        timestamp_ms = provider.get_timestamp_ms(frame_index)
        group_id = prompt_group_id or str(uuid.uuid4())
        input_k = intrinsics if config.use_actual_intrinsics else None
        context = source_context or SourceContext(
            source_mode=SourceMode.IMAGE_FOLDER,
            source_type=provider.source_type,
        )

        class_names = filtered["class_names"]
        records: list[DetectionRecord] = []
        for det_index in range(len(filtered["boxes"])):
            box2d = filtered["boxes"][det_index]
            box3d = filtered["boxes3d"][det_index]
            box9d = box10_quaternion_to_box9d(box3d)
            category = self._class_name_for_id(class_names, int(filtered["class_ids"][det_index]))
            records.append(
                DetectionRecord(
                    frame_index=int(frame_index),
                    category=category,
                    score=float(filtered["scores"][det_index]),
                    score_2d=float(filtered["scores_2d"][det_index]),
                    score_3d=float(filtered["scores_3d"][det_index]),
                    box2d_x1=float(box2d[0]),
                    box2d_y1=float(box2d[1]),
                    box2d_x2=float(box2d[2]),
                    box2d_y2=float(box2d[3]),
                    center_x=box9d["center_x"],
                    center_y=box9d["center_y"],
                    center_z=box9d["center_z"],
                    yaw_deg=box9d["yaw_deg"],
                    pitch_deg=box9d["pitch_deg"],
                    roll_deg=box9d["roll_deg"],
                    size_w=box9d["size_w"],
                    size_l=box9d["size_l"],
                    size_h=box9d["size_h"],
                    det_id=DetectionRecord.new_det_id(),
                    source_id=provider.source_id,
                    source_type=context.source_type,
                    source_path=str(provider.path),
                    dataset_id=context.dataset_id,
                    recording_id=context.recording_id,
                    timestamp_ms=timestamp_ms,
                    image_path=image_path,
                    prompt_mode=prompt_spec.mode.value,
                    prompt_label=prompt_spec.prompt_label.strip(),
                    prompt_payload_json=prompt_spec.payload_json(),
                    prompt_group_id=group_id,
                    input_fx=float(input_k[0, 0]) if input_k is not None else None,
                    input_fy=float(input_k[1, 1]) if input_k is not None else None,
                    input_cx=float(input_k[0, 2]) if input_k is not None else None,
                    input_cy=float(input_k[1, 2]) if input_k is not None else None,
                    pred_fx=float(predicted_k_scaled[0, 0]) if predicted_k_scaled is not None else None,
                    pred_fy=float(predicted_k_scaled[1, 1]) if predicted_k_scaled is not None else None,
                    pred_cx=float(predicted_k_scaled[0, 2]) if predicted_k_scaled is not None else None,
                    pred_cy=float(predicted_k_scaled[1, 2]) if predicted_k_scaled is not None else None,
                    use_actual_intrinsics=config.use_actual_intrinsics,
                    is_enabled=True,
                    track_id="",
                    track_status="",
                )
            )
        return records

    def render_frame_preview(
        self,
        frame_rgb: np.ndarray,
        records: list[DetectionRecord],
        prompt_spec: PromptSpec | None = None,
        highlight_det_id: str | None = None,
        intrinsics_override: np.ndarray | None = None,
    ) -> Image.Image:
        """Compatibility wrapper around the direct RGB preview renderer."""

        return Image.fromarray(
            self.render_frame_preview_rgb(
                frame_rgb=frame_rgb,
                records=records,
                prompt_spec=prompt_spec,
                highlight_det_id=highlight_det_id,
                intrinsics_override=intrinsics_override,
            )
        )

    def render_frame_preview_rgb(
        self,
        frame_rgb: np.ndarray,
        records: list[DetectionRecord],
        prompt_spec: PromptSpec | None = None,
        highlight_det_id: str | None = None,
        intrinsics_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw all visible 3D boxes in one CPU-efficient full-frame pass."""

        active_records = [record for record in records if record.is_visible]
        intrinsics = intrinsics_override
        if intrinsics is None and active_records:
            intrinsics = active_records[0].intrinsics_for_preview(frame_rgb.shape[:2])
        if intrinsics is None:
            height, width = frame_rgb.shape[:2]
            focal = float(max(height, width))
            intrinsics = np.array(
                [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

        categories = sorted({record.category for record in active_records})
        category_colors = {
            category: _PREVIEW_CATEGORY_COLORS[offset % len(_PREVIEW_CATEGORY_COLORS)]
            for offset, category in enumerate(categories)
        }
        overlays: list[Box3DOverlay] = []
        for record in active_records:
            highlighted = bool(highlight_det_id and record.det_id == highlight_det_id)
            color = (32, 255, 32) if highlighted else category_colors[record.category]
            overlays.append(
                Box3DOverlay(
                    box3d=self.record_to_box3d_quaternion(record),
                    color_rgb=color,
                    line_width=3 if highlighted else 2,
                    draw_edges=True,
                    draw_heading=True,
                    label=(
                        f"{record.category} "
                        f"2D:{float(record.score_2d):.2f} "
                        f"3D:{float(record.score_3d):.2f}"
                    ),
                    heading_color_rgb=color if highlighted else (88, 190, 255),
                )
            )

        prompt_box = prompt_spec.box if prompt_spec and prompt_spec.box else None
        prompt_points = (
            (point.x, point.y, point.label)
            for point in (prompt_spec.points if prompt_spec else ())
        )
        return render_box3d_scene_rgb(
            image_rgb=frame_rgb,
            overlays=overlays,
            intrinsics=intrinsics,
            prompt_box=prompt_box,
            prompt_points=prompt_points,
        )

    @staticmethod
    def record_to_box3d_quaternion(record: DetectionRecord) -> np.ndarray:
        return box9d_to_box10_quaternion(
            center_x=record.center_x,
            center_y=record.center_y,
            center_z=record.center_z,
            size_w=record.size_w,
            size_l=record.size_l,
            size_h=record.size_h,
            yaw_deg=record.yaw_deg,
            pitch_deg=record.pitch_deg,
            roll_deg=record.roll_deg,
        )

    @classmethod
    def project_record_to_box2d(
        cls,
        record: DetectionRecord,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        return project_box3d_quaternion_to_2d_bounds(
            box3d=cls.record_to_box3d_quaternion(record),
            intrinsics=intrinsics,
            image_shape=image_shape,
        )

    def warmup(self, use_predicted_intrinsics: bool) -> None:
        """Build and cache the requested model variant."""
        self._get_model(use_predicted_intrinsics=use_predicted_intrinsics)

    def has_model_variant(self, use_predicted_intrinsics: bool) -> bool:
        with self._model_condition:
            return use_predicted_intrinsics in self._models

    def release_models(self) -> None:
        """Release any cached model variants and free GPU memory if possible."""
        with self._model_condition:
            while self._building_variant is not None:
                self._model_condition.wait()
            models = list(self._models.values())
            self._models = {}
        if not models:
            return
        self._dispose_models(models)

    @staticmethod
    def _dispose_models(models: list[Any]) -> None:
        """Drop all strong references before asking CUDA to release its cache."""

        models.clear()
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _get_model(self, use_predicted_intrinsics: bool) -> Any:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"WildDet3D checkpoint not found: {self.checkpoint_path}")

        stale_models: list[Any] = []
        with self._model_condition:
            if use_predicted_intrinsics in self._models:
                return self._models[use_predicted_intrinsics]
            while self._building_variant is not None:
                if self._building_variant == use_predicted_intrinsics:
                    self._model_condition.wait()
                    if use_predicted_intrinsics in self._models:
                        return self._models[use_predicted_intrinsics]
                else:
                    self._model_condition.wait()
                    if use_predicted_intrinsics in self._models:
                        return self._models[use_predicted_intrinsics]
            # Only one 4+ GB variant is retained. Releasing it before building
            # another avoids a transient two-model VRAM peak on 12 GB cards.
            stale_models = list(self._models.values())
            self._models = {}
            self._building_variant = use_predicted_intrinsics

        try:
            if stale_models:
                self._dispose_models(stale_models)

            # Everything after publishing ``_building_variant`` must stay in
            # this guarded section.  Import/toolchain failures are just as
            # capable of occurring as model construction failures; leaving the
            # flag set would make every waiter (including application shutdown)
            # block forever.
            ensure_wilddet3d_on_path()
            resolved_device = self._resolve_device()
            self._ensure_cuda_runtime_ready(resolved_device)
            from wilddet3d.inference import build_model

            self._configure_torch_runtime()
            model = build_model(
                checkpoint=str(self.checkpoint_path),
                score_threshold=0.0,
                score_3d_threshold=0.0,
                canonical_rotation=True,
                skip_pretrained=True,
                use_predicted_intrinsics=use_predicted_intrinsics,
                device=resolved_device,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self._build_error = "".join(traceback.format_exception(exc))
            with self._model_condition:
                self._building_variant = None
                self._model_condition.notify_all()
            raise RuntimeError(f"Failed to build WildDet3D model:\n{self._build_error}") from exc

        with self._model_condition:
            self._models = {use_predicted_intrinsics: model}
            self._build_error = None
            self._building_variant = None
            self._model_condition.notify_all()
            return model

    def _resolve_device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        requested = str(self.device or "auto").strip().lower()
        try:
            import torch
        except ImportError:
            if requested not in {"", "auto", "cpu"}:
                raise RuntimeError(f"Requested device '{requested}' requires PyTorch.")
            self._resolved_device = "cpu"
            return self._resolved_device

        if requested in {"", "auto"}:
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{requested}', but CUDA is unavailable. "
                "Set CAMLABEL3D_DEVICE=cpu or install a compatible CUDA build."
            )
        else:
            self._resolved_device = requested
        return self._resolved_device

    @staticmethod
    def _cuda_device_index(device: str) -> int | None:
        if device == "cuda":
            return None
        prefix, separator, raw_index = device.partition(":")
        if prefix != "cuda" or not separator or not raw_index:
            raise RuntimeError(
                f"Invalid CUDA device '{device}'. Use 'cuda' or an indexed device such as 'cuda:0'."
            )
        try:
            index = int(raw_index)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid CUDA device '{device}'. The CUDA device index must be an integer."
            ) from error
        if index < 0:
            raise RuntimeError(f"Invalid CUDA device '{device}'. The CUDA device index cannot be negative.")
        return index

    def _ensure_cuda_runtime_ready(self, device: str) -> None:
        """Validate the extension once before this adapter uses a CUDA device."""
        if not device.startswith("cuda"):
            return
        with self._cuda_preflight_lock:
            if self._cuda_preflight_device == device:
                if self._cuda_preflight_error is not None:
                    raise RuntimeError(self._cuda_preflight_error)
                return

            # Keep extension discovery ahead of model imports. The isolated
            # child inherits the amended path and imports the extension itself.
            ensure_wilddet3d_on_path()
            device_index = self._cuda_device_index(device)
            result = run_gpu_diagnostic_isolated(device_index)
            self._cuda_preflight_device = device
            if result.ok:
                self._cuda_preflight_error = None
                return

            diagnostic_command = "python -m camlabel3d.diagnostics.gpu"
            if device_index is not None:
                diagnostic_command += f" --device {device_index}"
            self._cuda_preflight_error = (
                f"CUDA compatibility check failed for '{device}' before the WildDet3D model was loaded.\n"
                f"{format_result(result)}\n"
                f"Diagnostic command: {diagnostic_command}\n"
                "After repairing or rebuilding vis4d_cuda_ops, restart CamLabel3D before retrying."
            )
            raise RuntimeError(self._cuda_preflight_error)

    def _configure_torch_runtime(self) -> None:
        if self._torch_configured:
            return
        try:
            import torch
        except ImportError:
            self._torch_configured = True
            return
        torch.set_num_threads(max(1, int(self.runtime_config.torch_threads)))
        try:
            torch.set_num_interop_threads(max(1, int(self.runtime_config.torch_interop_threads)))
        except RuntimeError:
            # PyTorch only permits setting this before inter-op work starts.
            pass
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self._torch_configured = True

    def _run_model(self, model: Any, data: dict[str, Any], prompt_spec: PromptSpec) -> dict[str, Any]:
        device = self._resolve_device()
        self._ensure_cuda_runtime_ready(device)
        images = data["images"].to(device)
        intrinsics = data["intrinsics"].to(device)[None]
        common_kwargs = {
            "images": images,
            "intrinsics": intrinsics,
            "input_hw": [data["input_hw"]],
            "original_hw": [data["original_hw"]],
            "padding": [data["padding"]],
            "return_predicted_intrinsics": True,
        }
        inference_context = nullcontext()
        autocast_context = nullcontext()
        try:
            import torch
        except ImportError:
            pass
        else:
            inference_context = torch.inference_mode()
            if self.runtime_config.enable_amp and device.startswith("cuda"):
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)

        with inference_context, autocast_context:
            if prompt_spec.mode == PromptMode.TEXT:
                outputs = model(
                    **common_kwargs,
                    input_texts=prompt_spec.parsed_texts(),
                )
                class_names = prompt_spec.parsed_texts()
            elif prompt_spec.mode in (PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE):
                outputs = model(
                    **common_kwargs,
                    input_boxes=[list(prompt_spec.box)],
                    prompt_text=prompt_spec.prompt_text_for_model(),
                )
                class_names = [prompt_spec.class_name_fallback()]
            elif prompt_spec.mode == PromptMode.POINT:
                outputs = model(
                    **common_kwargs,
                    input_points=[[point.to_tuple() for point in prompt_spec.points]],
                    prompt_text=prompt_spec.prompt_text_for_model(),
                )
                class_names = [prompt_spec.class_name_fallback()]
            else:
                raise ValueError(f"Unsupported prompt mode: {prompt_spec.mode}")

        (
            boxes,
            boxes3d,
            scores,
            scores_2d,
            scores_3d,
            class_ids,
            _depth_maps,
            predicted_intrinsics,
            _confidence_maps,
        ) = outputs
        return {
            "boxes": boxes[0],
            "boxes3d": boxes3d[0],
            "scores": scores[0],
            "scores_2d": scores_2d[0] if scores_2d is not None else None,
            "scores_3d": scores_3d[0] if scores_3d is not None else None,
            "class_ids": class_ids[0],
            "predicted_intrinsics": predicted_intrinsics,
            "class_names": class_names,
        }

    def _postprocess_results(
        self,
        raw: dict[str, Any],
        prompt_spec: PromptSpec,
        config: DetectionConfig,
    ) -> dict[str, np.ndarray]:
        boxes = self._tensor_to_numpy(raw["boxes"])
        boxes3d = self._tensor_to_numpy(raw["boxes3d"])
        scores = self._tensor_to_numpy(raw["scores"])
        scores_2d = self._tensor_to_numpy(raw["scores_2d"], fallback_shape=(len(scores),))
        scores_3d = self._tensor_to_numpy(raw["scores_3d"], fallback_shape=(len(scores),))
        class_ids = self._tensor_to_numpy(raw["class_ids"]).astype(np.int64, copy=False)

        if boxes.size == 0:
            return {
                "boxes": boxes.reshape(0, 4),
                "boxes3d": boxes3d.reshape(0, 10),
                "scores": scores.reshape(0),
                "scores_2d": scores_2d.reshape(0),
                "scores_3d": scores_3d.reshape(0),
                "class_ids": class_ids.reshape(0),
                "predicted_intrinsics": raw["predicted_intrinsics"],
                "class_names": raw["class_names"],
            }

        keep = np.ones(len(scores), dtype=bool)
        if config.score_threshold > 0:
            keep &= scores >= float(config.score_threshold)
        if config.score_3d_threshold > 0 and scores_3d.size:
            keep &= scores_3d >= float(config.score_3d_threshold)

        boxes = boxes[keep]
        boxes3d = boxes3d[keep]
        scores = scores[keep]
        scores_2d = scores_2d[keep] if scores_2d.size else np.zeros((len(scores),), dtype=np.float32)
        scores_3d = scores_3d[keep] if scores_3d.size else np.zeros((len(scores),), dtype=np.float32)
        class_ids = class_ids[keep]

        if len(scores) > 1 and config.cross_category_nms_iou > 0:
            boxes, boxes3d, scores, scores_2d, scores_3d, class_ids = self._cross_category_nms(
                boxes,
                boxes3d,
                scores,
                scores_2d,
                scores_3d,
                class_ids,
                float(config.cross_category_nms_iou),
            )

        if prompt_spec.mode in (PromptMode.BOX_SINGLE, PromptMode.POINT) and len(scores) > 1:
            best = int(np.argmax(scores))
            boxes = boxes[best : best + 1]
            boxes3d = boxes3d[best : best + 1]
            scores = scores[best : best + 1]
            scores_2d = scores_2d[best : best + 1]
            scores_3d = scores_3d[best : best + 1]
            class_ids = class_ids[best : best + 1]

        class_names = list(raw["class_names"])
        if prompt_spec.mode in (PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE, PromptMode.POINT):
            class_names = [prompt_spec.class_name_fallback()]

        return {
            "boxes": boxes,
            "boxes3d": boxes3d,
            "scores": scores,
            "scores_2d": scores_2d,
            "scores_3d": scores_3d,
            "class_ids": class_ids,
            "predicted_intrinsics": raw["predicted_intrinsics"],
            "class_names": class_names,
        }

    @staticmethod
    def _tensor_to_numpy(value: Any, fallback_shape: tuple[int, ...] | None = None) -> np.ndarray:
        if value is None:
            return np.zeros(fallback_shape or (0,), dtype=np.float32)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        return array

    @staticmethod
    def _cross_category_nms(
        boxes: np.ndarray,
        boxes3d: np.ndarray,
        scores: np.ndarray,
        scores_2d: np.ndarray,
        scores_3d: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(boxes) <= 1:
            return boxes, boxes3d, scores, scores_2d, scores_3d, class_ids
        order = np.argsort(-scores)
        boxes = boxes[order]
        boxes3d = boxes3d[order]
        scores = scores[order]
        scores_2d = scores_2d[order]
        scores_3d = scores_3d[order]
        class_ids = class_ids[order]

        x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
        y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
        y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
        union = area[:, None] + area[None, :] - inter
        iou = inter / np.maximum(union, 1e-6)

        keep: list[int] = []
        suppressed: set[int] = set()
        for i in range(len(boxes)):
            if i in suppressed:
                continue
            keep.append(i)
            for j in range(i + 1, len(boxes)):
                if j not in suppressed and iou[i, j] >= iou_threshold:
                    suppressed.add(j)
        keep_idx = np.array(keep, dtype=np.int64)
        return (
            boxes[keep_idx],
            boxes3d[keep_idx],
            scores[keep_idx],
            scores_2d[keep_idx],
            scores_3d[keep_idx],
            class_ids[keep_idx],
        )

    @staticmethod
    def _scale_intrinsics_to_original(
        intrinsics: Any,
        input_hw: tuple[int, int],
        original_hw: tuple[int, int],
    ) -> np.ndarray | None:
        if intrinsics is None:
            return None
        if hasattr(intrinsics, "detach"):
            intrinsics = intrinsics.detach().cpu().numpy()
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]
        scaled = intrinsics.copy()
        input_h, input_w = input_hw
        orig_h, orig_w = original_hw
        scaled[0, 0] *= orig_w / float(input_w)
        scaled[1, 1] *= orig_h / float(input_h)
        scaled[0, 2] *= orig_w / float(input_w)
        scaled[1, 2] *= orig_h / float(input_h)
        return scaled

    @staticmethod
    def _class_name_for_id(class_names: list[str], class_id: int) -> str:
        if not class_names:
            return "object"
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        if 1 <= class_id <= len(class_names):
            return class_names[class_id - 1]
        return class_names[0]
