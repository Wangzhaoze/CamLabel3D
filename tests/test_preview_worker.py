from __future__ import annotations

from threading import Event

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, Qt

from camlabel3d.core.models import DetectionRecord
from camlabel3d.ui.workers import preview as preview_module
from camlabel3d.ui.workers.preview import PreviewRequest, PreviewWorker


class _BlockingPreviewProvider:
    def __init__(self) -> None:
        self.first_started = Event()
        self.release_first = Event()
        self.processed_indices: list[int] = []

    def get_preview_frame(self, index: int) -> np.ndarray:
        self.processed_indices.append(int(index))
        if index == 0:
            self.first_started.set()
            if not self.release_first.wait(timeout=2.0):
                raise TimeoutError("test did not release the first preview frame")
        return np.full((2, 3, 3), index, dtype=np.uint8)

    def prefetch(self, indices: object) -> None:
        del indices


class _RecordingOverlayRenderer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def render_frame_preview(
        self,
        *,
        frame_rgb: np.ndarray,
        records: list[DetectionRecord],
        prompt_spec: object,
        highlight_det_id: str | None,
        intrinsics_override: np.ndarray | None,
    ) -> np.ndarray:
        self.calls.append(
            {
                "frame_rgb": frame_rgb,
                "records": records,
                "prompt_spec": prompt_spec,
                "highlight_det_id": highlight_det_id,
                "intrinsics_override": intrinsics_override,
            }
        )
        # Deliberately differ from the source pixels. The emitted result then
        # proves that scrubbing used the overlay renderer instead of returning
        # the raw frame through a fast-path shortcut.
        return np.full_like(frame_rgb, 222)


def _record(frame_index: int, *, det_id: str | None = None) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        category="car",
        score=0.95,
        score_2d=0.94,
        score_3d=0.93,
        box2d_x1=1.0,
        box2d_y1=2.0,
        box2d_x2=20.0,
        box2d_y2=16.0,
        center_x=0.0,
        center_y=0.0,
        center_z=10.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=2.0,
        size_l=4.0,
        size_h=1.5,
        det_id=det_id or f"det-{frame_index}",
    )


def _scrub_request(provider: _BlockingPreviewProvider, frame_index: int) -> PreviewRequest:
    return PreviewRequest(
        generation=frame_index + 1,
        source_id="source-instance",
        provider=provider,  # type: ignore[arg-type]
        frame_index=frame_index,
        scrubbing=True,
        records=[_record(frame_index)],
        prompt_spec=None,
        highlight_det_id=None,
    )


def test_preview_worker_coalesces_burst_to_active_and_latest_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """A burst cannot queue 49 obsolete video decodes behind the active one."""

    app = QCoreApplication.instance() or QCoreApplication([])
    del app
    monkeypatch.setattr(preview_module, "pil_to_qimage", lambda frame: frame)

    provider = _BlockingPreviewProvider()
    renderer = _RecordingOverlayRenderer()
    worker = PreviewWorker(renderer=renderer)  # type: ignore[arg-type]
    completed: list[tuple[int, int, bool]] = []
    worker.previewReady.connect(
        lambda generation, _source, frame_index, scrubbing, _image: completed.append(
            (int(generation), int(frame_index), bool(scrubbing))
        ),
        Qt.ConnectionType.DirectConnection,
    )
    worker.start()
    try:
        worker.submit(_scrub_request(provider, 0))
        assert provider.first_started.wait(timeout=2.0)

        for frame_index in range(1, 51):
            worker.submit(_scrub_request(provider, frame_index))

        provider.release_first.set()
        assert worker.wait_until_idle(timeout_ms=2000)

        assert provider.processed_indices == [0, 50]
        assert completed == [(1, 0, True), (51, 50, True)]
        assert [call["records"][0].frame_index for call in renderer.calls] == [0, 50]  # type: ignore[index]
    finally:
        provider.release_first.set()
        worker.stop()
        assert worker.wait(2000)


def test_scrubbing_renders_original_frame_and_current_3d_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live dragging must never trade the 3D-box overlay for a raw frame."""

    app = QCoreApplication.instance() or QCoreApplication([])
    del app
    monkeypatch.setattr(preview_module, "pil_to_qimage", lambda frame: frame)

    provider = _BlockingPreviewProvider()
    renderer = _RecordingOverlayRenderer()
    worker = PreviewWorker(renderer=renderer)  # type: ignore[arg-type]
    emitted_images: list[np.ndarray] = []
    worker.previewReady.connect(
        lambda _generation, _source, _frame, _scrubbing, image: emitted_images.append(image),
        Qt.ConnectionType.DirectConnection,
    )
    request = _scrub_request(provider, 7)
    request.highlight_det_id = request.records[0].det_id
    request.intrinsics_override = np.eye(3, dtype=np.float32)

    worker.start()
    try:
        worker.submit(request)
        assert worker.wait_until_idle(timeout_ms=2000)

        assert provider.processed_indices == [7]
        assert len(renderer.calls) == 1
        call = renderer.calls[0]
        assert np.array_equal(call["frame_rgb"], np.full((2, 3, 3), 7, dtype=np.uint8))
        assert call["records"] == request.records
        assert call["highlight_det_id"] == request.records[0].det_id
        assert np.array_equal(call["intrinsics_override"], np.eye(3, dtype=np.float32))
        assert len(emitted_images) == 1
        assert np.array_equal(emitted_images[0], np.full((2, 3, 3), 222, dtype=np.uint8))
    finally:
        worker.stop()
        assert worker.wait(2000)
