from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from camlabel3d.ui.main_window import MainWindow
from camlabel3d.core.models import DetectionRecord, PromptSpec


class _FakeTimer:
    def __init__(self) -> None:
        self.active = False
        self.started_with: list[int] = []
        self.stop_calls = 0

    def isActive(self) -> bool:
        return self.active

    def start(self, delay: int) -> None:
        self.active = True
        self.started_with.append(int(delay))

    def stop(self) -> None:
        self.active = False
        self.stop_calls += 1


class _FakeSlider:
    def __init__(self, down: bool) -> None:
        self.down = down

    def isSliderDown(self) -> bool:
        return self.down


class _PreviewHarness:
    def __init__(self, *, slider_down: bool, delay: int = 35) -> None:
        self.current_provider = object()
        self.preview_timer = _FakeTimer()
        self.frame_slider = _FakeSlider(slider_down)
        self.runtime_config = SimpleNamespace(preview_debounce_ms=delay)
        self.submissions = 0
        self.full_submissions = 0
        self.table_refreshes = 0
        self.info_refreshes = 0
        self.action_refreshes = 0

    def _submit_preview_request(self, scrubbing: bool | None = None) -> None:
        self.submissions += 1
        if scrubbing is False:
            self.full_submissions += 1

    def _refresh_table(self) -> None:
        self.table_refreshes += 1

    def _refresh_info_panel(self) -> None:
        self.info_refreshes += 1

    def _update_action_states(self) -> None:
        self.action_refreshes += 1


def test_live_scrubbing_submits_every_latest_position_without_timer_queue() -> None:
    harness = _PreviewHarness(slider_down=True)
    harness.preview_timer.active = True

    MainWindow._refresh_preview(harness)
    MainWindow._refresh_preview(harness)

    assert harness.submissions == 2
    assert harness.preview_timer.started_with == []
    assert harness.preview_timer.stop_calls == 2


def test_non_scrub_refresh_keeps_trailing_debounce_behavior() -> None:
    harness = _PreviewHarness(slider_down=False)

    MainWindow._refresh_preview(harness)
    MainWindow._refresh_preview(harness)

    assert harness.preview_timer.started_with == [35, 35]


def test_slider_release_submits_exact_final_frame_immediately() -> None:
    harness = _PreviewHarness(slider_down=True)
    harness.preview_timer.active = True

    MainWindow._on_frame_slider_released(harness)

    assert harness.preview_timer.stop_calls == 1
    assert harness.submissions == 1
    assert harness.full_submissions == 1
    assert harness.table_refreshes == 1
    assert harness.info_refreshes == 1
    assert harness.action_refreshes == 1


def test_zero_delay_submits_immediately() -> None:
    harness = _PreviewHarness(slider_down=True, delay=0)

    MainWindow._refresh_preview(harness)

    assert harness.submissions == 1
    assert harness.preview_timer.started_with == []


class _FakeProvider:
    source_id = "video-source"


class _PreviewPolicyHarness:
    def __init__(self, *, slider_down: bool) -> None:
        self.current_provider = _FakeProvider()
        self.current_frame_index = 40
        self._latest_preview_request_seq = 12
        self._last_displayed_preview_seq = 4
        self.frame_slider = _FakeSlider(slider_down)

    @staticmethod
    def _preview_provider_token(provider: object) -> str:
        return MainWindow._preview_provider_token(provider)


def test_scrubbing_accepts_newest_completed_intermediate_frame() -> None:
    harness = _PreviewPolicyHarness(slider_down=True)
    token = MainWindow._preview_provider_token(harness.current_provider)

    for request_seq, frame_index in ((5, 12), (8, 25), (11, 37)):
        assert MainWindow._should_display_preview(
            harness,
            request_seq,
            token,
            frame_index,
            True,
        )
        # Mirror _on_preview_ready after each accepted result. The slider stays
        # down throughout, so several completed intermediate frames should be
        # displayable before the final mouse release.
        harness._last_displayed_preview_seq = request_seq

    assert not MainWindow._should_display_preview(harness, 10, token, 25, True)
    assert not MainWindow._should_display_preview(harness, 13, "other-source", 25, True)


def test_non_scrub_preview_requires_exact_generation_and_frame() -> None:
    harness = _PreviewPolicyHarness(slider_down=False)
    token = MainWindow._preview_provider_token(harness.current_provider)

    assert MainWindow._should_display_preview(harness, 12, token, 40, False)
    assert not MainWindow._should_display_preview(harness, 11, token, 40, False)
    assert not MainWindow._should_display_preview(harness, 12, token, 39, False)


def test_released_slider_rejects_old_scrub_result_even_for_exact_frame() -> None:
    harness = _PreviewPolicyHarness(slider_down=False)
    token = MainWindow._preview_provider_token(harness.current_provider)

    assert not MainWindow._should_display_preview(harness, 12, token, 40, True)


def _record(frame_index: int, det_id: str) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        category="car",
        score=0.9,
        score_2d=0.8,
        score_3d=0.7,
        box2d_x1=1.0,
        box2d_y1=2.0,
        box2d_x2=12.0,
        box2d_y2=14.0,
        center_x=0.0,
        center_y=0.0,
        center_z=10.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=2.0,
        size_l=4.0,
        size_h=1.5,
        det_id=det_id,
    )


class _RequestCollector:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def submit(self, request: object) -> None:
        self.requests.append(request)


class _CheckedBox:
    @staticmethod
    def isChecked() -> bool:
        return True


class _ScrubSubmissionHarness:
    def __init__(self) -> None:
        self.current_provider = _FakeProvider()
        self.current_frame_index = 40
        self.frame_slider = _FakeSlider(True)
        self._preview_request_seq = 0
        self._latest_preview_request_seq = 0
        self.preview_worker = _RequestCollector()
        self.use_actual_k_checkbox = _CheckedBox()
        self.frame_records = [_record(40, "box-on-current-frame")]

    def _current_frame_records(self) -> list[DetectionRecord]:
        return self.frame_records

    @staticmethod
    def _is_detection_stage() -> bool:
        return True

    @staticmethod
    def _current_prompt_spec() -> PromptSpec:
        return PromptSpec(text_prompt="car")

    @staticmethod
    def _current_detection_config() -> SimpleNamespace:
        return SimpleNamespace(to_intrinsics_matrix=lambda: np.eye(3, dtype=np.float32))

    @staticmethod
    def _selected_det_id() -> str:
        return "box-on-current-frame"

    @staticmethod
    def _preview_provider_token(provider: object) -> str:
        return MainWindow._preview_provider_token(provider)


def test_live_scrub_request_carries_current_frame_records_for_3d_overlay() -> None:
    harness = _ScrubSubmissionHarness()

    MainWindow._submit_preview_request(harness, scrubbing=True)

    assert len(harness.preview_worker.requests) == 1
    request = harness.preview_worker.requests[0]
    assert request.scrubbing is True
    assert request.frame_index == 40
    assert [record.det_id for record in request.records] == ["box-on-current-frame"]
    # The background renderer owns a stable snapshot; UI edits must not mutate
    # an in-flight 3D overlay request.
    assert request.records is not harness.frame_records
    assert request.records[0] is not harness.frame_records[0]
