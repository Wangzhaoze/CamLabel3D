from __future__ import annotations

import os

import pytest

from camlabel3d.runtime_config import RuntimeConfig


CONFIG_ENV_NAMES = (
    "CAMLABEL3D_CPU_WORKERS",
    "CAMLABEL3D_TORCH_THREADS",
    "CAMLABEL3D_TORCH_INTEROP_THREADS",
    "CAMLABEL3D_FRAME_CACHE_MB",
    "CAMLABEL3D_PRELOAD_VIDEO_FRAMES",
    "CAMLABEL3D_DEVICE",
    "CAMLABEL3D_PREVIEW_DEBOUNCE_MS",
    "CAMLABEL3D_AUTOSAVE_DEBOUNCE_MS",
    "CAMLABEL3D_KEEP_MODEL_LOADED",
    "CAMLABEL3D_ENABLE_AMP",
)


@pytest.fixture(autouse=True)
def clean_runtime_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CONFIG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_leave_cpu_capacity_for_ui_and_bound_worker_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 12)

    config = RuntimeConfig.from_env()

    assert config.cpu_workers == 8
    assert config.torch_threads == 8
    assert config.torch_interop_threads == 2
    assert config.frame_cache_bytes == 2048 * 1024 * 1024
    assert config.preload_video_frames is True
    assert config.preview_debounce_ms == 35
    assert config.autosave_debounce_ms == 350
    assert config.keep_model_loaded is True
    assert config.enable_amp is False


def test_numeric_environment_values_are_trimmed_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    values = {
        "CAMLABEL3D_CPU_WORKERS": " 99 ",
        "CAMLABEL3D_TORCH_THREADS": "-3",
        "CAMLABEL3D_TORCH_INTEROP_THREADS": "20",
        "CAMLABEL3D_FRAME_CACHE_MB": "9000",
        "CAMLABEL3D_PREVIEW_DEBOUNCE_MS": "-10",
        "CAMLABEL3D_AUTOSAVE_DEBOUNCE_MS": "99999",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = RuntimeConfig.from_env()

    assert config.cpu_workers == 8
    assert config.torch_threads == 1
    assert config.torch_interop_threads == 4
    assert config.frame_cache_bytes == 8192 * 1024 * 1024
    assert config.preview_debounce_ms == 0
    assert config.autosave_debounce_ms == 5000


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("YES", True), (" on ", True), ("0", False), ("False", False), ("off", False)],
)
def test_boolean_environment_values(raw: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAMLABEL3D_KEEP_MODEL_LOADED", raw)
    monkeypatch.setenv("CAMLABEL3D_ENABLE_AMP", raw)
    monkeypatch.setenv("CAMLABEL3D_PRELOAD_VIDEO_FRAMES", raw)

    config = RuntimeConfig.from_env()

    assert config.keep_model_loaded is expected
    assert config.enable_amp is expected
    assert config.preload_video_frames is expected


def test_invalid_values_fall_back_safely_when_cpu_count_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    monkeypatch.setenv("CAMLABEL3D_CPU_WORKERS", "many")
    monkeypatch.setenv("CAMLABEL3D_FRAME_CACHE_MB", "2.5")
    monkeypatch.setenv("CAMLABEL3D_KEEP_MODEL_LOADED", "sometimes")
    monkeypatch.setenv("CAMLABEL3D_ENABLE_AMP", "maybe")
    monkeypatch.setenv("CAMLABEL3D_DEVICE", " CUDA:1 ")

    config = RuntimeConfig.from_env()

    assert config.cpu_workers == 1
    assert config.torch_threads == 1
    assert config.torch_interop_threads == 1
    assert config.frame_cache_bytes == 2048 * 1024 * 1024
    assert config.keep_model_loaded is True
    assert config.enable_amp is False
    assert config.device == "cuda:1"
