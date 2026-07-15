"""Inference and tracking workers with cooperative cancellation."""

from __future__ import annotations

from threading import Event

from PySide6.QtCore import QThread, Signal

from camlabel3d.core.detector import DetectorAdapter
from camlabel3d.core.frame_provider import FrameProvider
from camlabel3d.core.models import DetectionConfig, PromptSpec, SourceContext
from camlabel3d.core.tracking import TrackingConfig, TrackingEngine


class DetectionWorker(QThread):
    """Run one frame/range request on the single model host."""

    progressChanged = Signal(int, int, str)
    runCompleted = Signal(object)
    runFailed = Signal(str)

    def __init__(
        self,
        detector: DetectorAdapter,
        provider: FrameProvider,
        frame_indices: list[int],
        prompt_spec: PromptSpec,
        config: DetectionConfig,
        source_context: SourceContext,
        replace_target_id: str | None = None,
        release_models_after_run: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.detector = detector
        self.provider = provider
        self.frame_indices = list(frame_indices)
        self.prompt_spec = prompt_spec.clone()
        self.config = config
        self.source_context = source_context
        self.replace_target_id = replace_target_id
        self.release_models_after_run = bool(release_models_after_run)
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:  # noqa: D401, N802 - Qt naming
        try:
            records = self.detector.run_range(
                provider=self.provider,
                prompt_spec=self.prompt_spec,
                config=self.config,
                source_context=self.source_context,
                frame_indices=self.frame_indices,
                progress_callback=self._emit_progress,
                should_cancel=self._cancel_event.is_set,
            )
            self.runCompleted.emit(
                {
                    "records": records,
                    "frame_indices": list(self.frame_indices),
                    "replace_target_id": self.replace_target_id,
                    "canceled": self._cancel_event.is_set(),
                    "prompt_spec": self.prompt_spec,
                }
            )
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self.runFailed.emit(str(exc))
        finally:
            if self.release_models_after_run:
                try:
                    self.detector.release_models()
                except Exception:
                    pass

    def _emit_progress(self, current: int, total: int, message: str) -> None:
        self.progressChanged.emit(int(current), int(total), str(message))


class WarmupWorker(QThread):
    """Warm up one model variant without blocking the event loop."""

    warmupCompleted = Signal(bool)
    warmupFailed = Signal(bool, str)

    def __init__(self, detector: DetectorAdapter, use_predicted_intrinsics: bool, parent=None) -> None:
        super().__init__(parent)
        self.detector = detector
        self.use_predicted_intrinsics = bool(use_predicted_intrinsics)

    def run(self) -> None:  # noqa: D401, N802 - Qt naming
        try:
            self.detector.warmup(self.use_predicted_intrinsics)
            self.warmupCompleted.emit(self.use_predicted_intrinsics)
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self.warmupFailed.emit(self.use_predicted_intrinsics, str(exc))


class TrackingWorker(QThread):
    """Run offline tracking on a detached record snapshot."""

    progressChanged = Signal(int, int, str)
    runCompleted = Signal(object)
    runFailed = Signal(str)

    def __init__(self, records, config: TrackingConfig | None = None, parent=None) -> None:
        super().__init__(parent)
        self.records = list(records)
        self.config = config or TrackingConfig()
        self.engine = TrackingEngine()
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:  # noqa: D401, N802 - Qt naming
        try:
            records = self.engine.run(
                records=self.records,
                config=self.config,
                progress_callback=self._emit_progress,
                should_cancel=self._cancel_event.is_set,
            )
            self.runCompleted.emit({"records": records, "canceled": self._cancel_event.is_set()})
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self.runFailed.emit(str(exc))

    def _emit_progress(self, current: int, total: int, message: str) -> None:
        self.progressChanged.emit(int(current), int(total), str(message))
