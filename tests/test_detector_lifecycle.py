from __future__ import annotations

import pytest

from camlabel3d.core import detector as detector_module
from camlabel3d.core.detector import DetectorAdapter
from camlabel3d.diagnostics.gpu import GpuDiagnosticResult


def test_model_build_import_failure_releases_waiters(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    adapter = DetectorAdapter(checkpoint_path=checkpoint, device="cpu")

    def fail_runtime_setup() -> None:
        raise ImportError("synthetic WildDet3D import failure")

    monkeypatch.setattr(detector_module, "ensure_wilddet3d_on_path", fail_runtime_setup)

    with pytest.raises(RuntimeError, match="Failed to build WildDet3D model"):
        adapter._get_model(use_predicted_intrinsics=False)

    assert adapter._building_variant is None
    assert adapter._models == {}


def test_stale_model_disposal_failure_releases_waiters(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    adapter = DetectorAdapter(checkpoint_path=checkpoint, device="cpu")
    adapter._models = {False: object()}

    def fail_dispose(_models) -> None:
        raise RuntimeError("synthetic disposal failure")

    monkeypatch.setattr(adapter, "_dispose_models", fail_dispose)

    with pytest.raises(RuntimeError, match="Failed to build WildDet3D model"):
        adapter._get_model(use_predicted_intrinsics=True)

    assert adapter._building_variant is None
    assert adapter._models == {}


def test_cpu_path_never_runs_gpu_preflight(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    adapter = DetectorAdapter(checkpoint_path=checkpoint, device="cpu")
    calls: list[int | None] = []

    monkeypatch.setattr(
        detector_module,
        "run_gpu_diagnostic_isolated",
        lambda device_index: calls.append(device_index),
    )

    adapter._ensure_cuda_runtime_ready("cpu")

    assert calls == []


def test_cuda_preflight_runs_only_once_per_adapter(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    adapter = DetectorAdapter(checkpoint_path=checkpoint, device="cuda:1")
    calls: list[int | None] = []

    def pass_preflight(device_index: int | None) -> GpuDiagnosticResult:
        calls.append(device_index)
        return GpuDiagnosticResult(ok=True, stage="complete", summary="ok")

    monkeypatch.setattr(detector_module, "run_gpu_diagnostic_isolated", pass_preflight)

    adapter._ensure_cuda_runtime_ready("cuda:1")
    adapter._ensure_cuda_runtime_ready("cuda:1")

    assert calls == [1]


def test_cuda_preflight_failure_is_clear_and_cached(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    adapter = DetectorAdapter(checkpoint_path=checkpoint, device="cuda:0")
    calls: list[int | None] = []

    def fail_preflight(device_index: int | None) -> GpuDiagnosticResult:
        calls.append(device_index)
        return GpuDiagnosticResult(
            ok=False,
            stage="kernel_execution",
            summary="The CUDA extension does not contain a usable kernel for sm_86.",
            recommendations=('Set TORCH_CUDA_ARCH_LIST="8.6".',),
        )

    monkeypatch.setattr(detector_module, "run_gpu_diagnostic_isolated", fail_preflight)

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="sm_86") as error:
            adapter._ensure_cuda_runtime_ready("cuda:0")
        assert "before the WildDet3D model was loaded" in str(error.value)
        assert 'TORCH_CUDA_ARCH_LIST="8.6"' in str(error.value)
        assert "restart CamLabel3D" in str(error.value)

    assert calls == [0]


def test_model_build_preflights_cuda_before_importing_model(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    adapter = DetectorAdapter(checkpoint_path=checkpoint, device="cuda")
    events: list[str] = []

    monkeypatch.setattr(detector_module, "ensure_wilddet3d_on_path", lambda: events.append("path"))
    monkeypatch.setattr(adapter, "_resolve_device", lambda: "cuda")

    def reject_cuda(_device: str) -> None:
        events.append("preflight")
        raise RuntimeError("synthetic CUDA compatibility failure")

    monkeypatch.setattr(adapter, "_ensure_cuda_runtime_ready", reject_cuda)

    with pytest.raises(RuntimeError, match="synthetic CUDA compatibility failure"):
        adapter._get_model(use_predicted_intrinsics=False)

    assert events == ["path", "preflight"]
    assert adapter._building_variant is None
