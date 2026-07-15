from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from camlabel3d.core.frame_provider import VideoFrameProvider


class _FakeCapture:
    def __init__(self, cv2_module: ModuleType) -> None:
        self.cv2 = cv2_module
        self.position = 0
        self.grabbed_positions: list[int] = []
        self.read_positions: list[int] = []
        self.seek_positions: list[int] = []
        self.release_calls = 0

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return True

    def get(self, prop: int) -> float:
        values = {
            self.cv2.CAP_PROP_FRAME_COUNT: 100.0,
            self.cv2.CAP_PROP_FPS: 25.0,
            self.cv2.CAP_PROP_FRAME_WIDTH: 3.0,
            self.cv2.CAP_PROP_FRAME_HEIGHT: 2.0,
            self.cv2.CAP_PROP_POS_FRAMES: float(self.position),
        }
        return values.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        assert prop == self.cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        self.seek_positions.append(self.position)
        return True

    def grab(self) -> bool:
        self.grabbed_positions.append(self.position)
        self.position += 1
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        index = self.position
        self.read_positions.append(index)
        self.position += 1
        pixel = np.array([index, index + 1, index + 2], dtype=np.uint8)
        return True, np.tile(pixel, (2, 3, 1))

    def release(self) -> None:
        self.release_calls += 1


def _fake_cv2() -> tuple[ModuleType, _FakeCapture]:
    module = ModuleType("cv2")
    module.CAP_PROP_FRAME_COUNT = 1
    module.CAP_PROP_FPS = 2
    module.CAP_PROP_FRAME_WIDTH = 3
    module.CAP_PROP_FRAME_HEIGHT = 4
    module.CAP_PROP_POS_FRAMES = 5
    module.COLOR_BGR2RGB = 6
    capture = _FakeCapture(module)
    module.VideoCapture = lambda _path: capture
    module.cvtColor = lambda frame, _code: frame[..., ::-1].copy()
    return module, capture


def test_video_provider_uses_forward_grab_seek_and_decoded_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cv2_module, capture = _fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", cv2_module)
    video_path = tmp_path / "fixture.mp4"
    video_path.write_bytes(b"fake video payload")

    provider = VideoFrameProvider(video_path, forward_scan_limit=3, preload_all_frames=False)
    try:
        provider.get_frame(0)
        provider.get_frame(3)

        assert capture.grabbed_positions == [1, 2]
        assert capture.read_positions == [0, 3]
        assert capture.seek_positions == []

        provider.get_frame(10)
        assert capture.seek_positions == [10]
        assert capture.read_positions == [0, 3, 10]

        provider.get_frame(8)
        assert capture.seek_positions == [10, 8]
        assert capture.read_positions == [0, 3, 10, 8]

        operations_before_cache_hit = (
            list(capture.grabbed_positions),
            list(capture.read_positions),
            list(capture.seek_positions),
        )
        cached_frame = provider.get_frame(3)

        assert (
            capture.grabbed_positions,
            capture.read_positions,
            capture.seek_positions,
        ) == operations_before_cache_hit
        assert cached_frame[0, 0].tolist() == [5, 4, 3]
    finally:
        provider.close()

    assert capture.release_calls == 1


def test_video_provider_preloads_every_full_resolution_frame_when_cache_fits(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("cv2")
    module.CAP_PROP_FRAME_COUNT = 1
    module.CAP_PROP_FPS = 2
    module.CAP_PROP_FRAME_WIDTH = 3
    module.CAP_PROP_FRAME_HEIGHT = 4
    module.CAP_PROP_POS_FRAMES = 5
    module.COLOR_BGR2RGB = 6
    captures: list[_FakeCapture] = []

    def make_capture(_path: str) -> _FakeCapture:
        capture = _FakeCapture(module)
        captures.append(capture)
        return capture

    module.VideoCapture = make_capture
    module.cvtColor = lambda frame, _code: frame[..., ::-1].copy()
    monkeypatch.setitem(sys.modules, "cv2", module)
    video_path = tmp_path / "preload.mp4"
    video_path.write_bytes(b"fake video payload")

    # 100 frames * 2 rows * 3 columns * 3 RGB bytes = 1,800 bytes.
    provider = VideoFrameProvider(video_path, cache_bytes=1800, preload_all_frames=True)
    try:
        assert provider.wait_until_preloaded(timeout_seconds=2.0)
        assert len(captures) == 2
        assert captures[1].read_positions == list(range(100))
        assert provider._frame_cache.stats().entries == 100

        frame = provider.get_frame(50)
        assert captures[0].read_positions == []
        assert frame[0, 0].tolist() == [52, 51, 50]
    finally:
        provider.close()

    assert [capture.release_calls for capture in captures] == [1, 1]
