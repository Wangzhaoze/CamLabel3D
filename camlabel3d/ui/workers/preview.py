"""Latest-wins preview renderer running on one long-lived thread."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition
from time import monotonic

import numpy as np
from PySide6.QtCore import QThread, Signal

from camlabel3d.core.detector import DetectorAdapter
from camlabel3d.core.frame_provider import FrameProvider
from camlabel3d.core.models import DetectionRecord, PromptSpec
from camlabel3d.ui.image_utils import pil_to_qimage, rgb_array_to_qimage


@dataclass(slots=True)
class PreviewRequest:
    generation: int
    source_id: str
    provider: FrameProvider
    frame_index: int
    scrubbing: bool
    records: list[DetectionRecord]
    prompt_spec: PromptSpec | None
    highlight_det_id: str | None
    intrinsics_override: np.ndarray | None = None


class PreviewWorker(QThread):
    """Coalesce rapid scrubbing into a single pending render request."""

    previewReady = Signal(int, str, int, bool, object)
    previewFailed = Signal(int, str, int, bool, str)

    def __init__(self, renderer: DetectorAdapter, parent=None) -> None:
        super().__init__(parent)
        self.renderer = renderer
        self._condition = Condition()
        self._pending: PreviewRequest | None = None
        self._active = False
        self._stopping = False

    def submit(self, request: PreviewRequest) -> None:
        with self._condition:
            if self._stopping:
                return
            self._pending = request
            self._condition.notify_all()

    def discard_pending(self) -> None:
        with self._condition:
            self._pending = None
            self._condition.notify_all()

    def wait_until_idle(self, timeout_ms: int = 2000) -> bool:
        deadline = monotonic() + max(0, int(timeout_ms)) / 1000.0
        with self._condition:
            while self._active or self._pending is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()

    def run(self) -> None:  # noqa: D401, N802 - Qt naming
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                request = self._pending
                self._pending = None
                self._active = True

            assert request is not None
            try:
                if request.scrubbing:
                    frame_rgb = request.provider.get_preview_frame(request.frame_index)
                else:
                    frame_rgb = request.provider.get_frame(request.frame_index)
                # Scrubbing is a faster *decode scheduling* mode, never a
                # reduced-content preview. Every published frame includes the
                # source image and its complete 3D bounding-box rendering.
                render_rgb = getattr(self.renderer, "render_frame_preview_rgb", None)
                render_kwargs = {
                    "frame_rgb": frame_rgb,
                    "records": request.records,
                    "prompt_spec": request.prompt_spec,
                    "highlight_det_id": request.highlight_det_id,
                    "intrinsics_override": request.intrinsics_override,
                }
                if callable(render_rgb):
                    # The production renderer returns an owning full-resolution
                    # RGB result, avoiding PIL and two color-space conversions.
                    qimage = rgb_array_to_qimage(render_rgb(**render_kwargs))
                else:
                    # Keep renderer adapters extensible for tests and plugins.
                    preview = self.renderer.render_frame_preview(**render_kwargs)
                    qimage = pil_to_qimage(preview)
                request.provider.prefetch((request.frame_index + 1, request.frame_index - 1))
                self.previewReady.emit(
                    request.generation,
                    request.source_id,
                    request.frame_index,
                    request.scrubbing,
                    qimage,
                )
            except Exception as exc:
                self.previewFailed.emit(
                    request.generation,
                    request.source_id,
                    request.frame_index,
                    request.scrubbing,
                    str(exc),
                )
            finally:
                with self._condition:
                    self._active = False
                    self._condition.notify_all()
