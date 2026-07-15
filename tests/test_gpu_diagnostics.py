from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from camlabel3d.diagnostics.gpu import (
    GpuDiagnosticResult,
    capability_labels,
    format_result,
    is_architecture_failure,
    run_gpu_diagnostic,
    run_gpu_diagnostic_isolated,
)


class _CudaUnavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class _CudaEnabledTorchWithoutGpu:
    __version__ = "test"
    cuda = _CudaUnavailable()

    class version:
        cuda = "12.8"


class _CudaAvailable:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(_index: int) -> str:
        return "Test RTX 3060"

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (8, 6)


class _CudaEnabledTorchWithGpu(_CudaEnabledTorchWithoutGpu):
    cuda = _CudaAvailable()


class _FakeOps:
    __file__ = "vis4d_cuda_ops.test.pyd"

    @staticmethod
    def iou_box3d() -> None:
        return None


class GpuDiagnosticTests(unittest.TestCase):
    def test_capability_labels_for_rtx_3060(self) -> None:
        self.assertEqual(capability_labels((8, 6)), ("8.6", "sm_86"))

    def test_architecture_error_classification(self) -> None:
        error = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        self.assertTrue(is_architecture_failure(error))
        self.assertFalse(is_architecture_failure(RuntimeError("out of memory")))

    def test_missing_torch_is_reported_without_importing_it(self) -> None:
        def missing_import(_name: str):
            raise ModuleNotFoundError("No module named 'torch'")

        result = run_gpu_diagnostic(importer=missing_import)

        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "torch_import")
        self.assertIn("PyTorch could not be imported", format_result(result))

    def test_cuda_wheel_without_gpu_stops_before_extension_import(self) -> None:
        imports: list[str] = []

        def fake_import(name: str):
            imports.append(name)
            if name == "torch":
                return _CudaEnabledTorchWithoutGpu()
            raise AssertionError(f"unexpected import: {name}")

        result = run_gpu_diagnostic(importer=fake_import)

        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "cuda_runtime")
        self.assertEqual(imports, ["torch"])
        self.assertIn("no usable GPU", result.summary)

    def test_result_is_json_serializable(self) -> None:
        result = GpuDiagnosticResult(ok=True, stage="complete", summary="ok", details=("detail",))
        self.assertIn('"ok": true', json.dumps(result.to_dict()))

    def test_architecture_failure_includes_device_specific_rebuild_hint(self) -> None:
        def fake_import(name: str):
            return _CudaEnabledTorchWithGpu() if name == "torch" else _FakeOps()

        kernel_error = RuntimeError("CUDA error: invalid device function")
        with patch("camlabel3d.diagnostics.gpu._run_vis4d_kernel", side_effect=kernel_error):
            result = run_gpu_diagnostic(importer=fake_import)

        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "kernel_execution")
        self.assertEqual(result.compute_capability, "8.6")
        self.assertIn("sm_86", result.summary)
        self.assertTrue(any('TORCH_CUDA_ARCH_LIST="8.6"' in item for item in result.recommendations))

    def test_isolated_diagnostic_parses_failed_probe_and_forwards_device(self) -> None:
        payload = GpuDiagnosticResult(
            ok=False,
            stage="kernel_execution",
            summary="missing sm_86 kernel",
            recommendations=('TORCH_CUDA_ARCH_LIST="8.6"',),
        ).to_dict()
        completed = SimpleNamespace(returncode=1, stdout=json.dumps(payload), stderr="")

        with patch("camlabel3d.diagnostics.gpu.subprocess.run", return_value=completed) as runner:
            result = run_gpu_diagnostic_isolated(1)

        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "kernel_execution")
        command = runner.call_args.args[0]
        self.assertEqual(command[-2:], ["--device", "1"])
        self.assertIn("PYTHONPATH", runner.call_args.kwargs["env"])

    def test_isolated_diagnostic_tolerates_native_stdout_before_json(self) -> None:
        payload = GpuDiagnosticResult(ok=True, stage="complete", summary="ok").to_dict()
        completed = SimpleNamespace(
            returncode=0,
            stdout=f"native extension notice\n{json.dumps(payload)}\n",
            stderr="",
        )

        with patch("camlabel3d.diagnostics.gpu.subprocess.run", return_value=completed):
            result = run_gpu_diagnostic_isolated()

        self.assertTrue(result.ok)
        self.assertEqual(result.stage, "complete")


if __name__ == "__main__":
    unittest.main()
