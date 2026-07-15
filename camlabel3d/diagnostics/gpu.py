"""Verify PyTorch CUDA and execute a real ``vis4d_cuda_ops`` CUDA kernel.

Run from the repository root with::

    python -m camlabel3d.diagnostics.gpu
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ARCHITECTURE_ERROR_MARKERS = (
    "no kernel image is available",
    "invalid device function",
)
CPU_ONLY_EXTENSION_MARKERS = (
    "not compiled with gpu support",
    "not compiled with cuda support",
)


@dataclass(frozen=True, slots=True)
class GpuDiagnosticResult:
    """Structured result suitable for both console and JSON output."""

    ok: bool
    stage: str
    summary: str
    device_name: str | None = None
    compute_capability: str | None = None
    details: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capability_labels(capability: Sequence[int]) -> tuple[str, str]:
    """Return PyTorch architecture and NVIDIA SM labels for a capability."""
    if len(capability) != 2:
        raise ValueError(f"CUDA compute capability must contain two integers, got {capability!r}")
    major, minor = (int(part) for part in capability)
    if major < 1 or minor < 0:
        raise ValueError(f"Invalid CUDA compute capability: {(major, minor)!r}")
    return f"{major}.{minor}", f"sm_{major}{minor}"


def is_architecture_failure(error: BaseException | str) -> bool:
    """Identify the CUDA errors most commonly caused by a missing SM target."""
    message = str(error).lower()
    return any(marker in message for marker in ARCHITECTURE_ERROR_MARKERS)


def is_cpu_only_extension_failure(error: BaseException | str) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in CPU_ONLY_EXTENSION_MARKERS)


def _exception_message(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _rebuild_recommendations(architecture: str) -> tuple[str, ...]:
    return (
        f'Set TORCH_CUDA_ARCH_LIST="{architecture}" (this device is sm_{architecture.replace(".", "")}).',
        'Set VIS4D_CUDA_OPS_BUILD_CUDA="1" and choose a conservative MAX_JOBS value such as "2".',
        "After this diagnostic process exits, rebuild with: python -m pip install --force-reinstall "
        "--no-deps --no-cache-dir --no-build-isolation .\\workers\\WildDet3D\\vis4d_cuda_ops",
    )


def _run_vis4d_kernel(torch: Any, ops: Any, device_index: int) -> float:
    """Run the minimal box-IoU CUDA kernel used by WildDet3D postprocessing."""
    device = torch.device("cuda", device_index)
    unit_box = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    intersection, iou = ops.iou_box3d(unit_box, unit_box)
    # Kernel launch failures can be asynchronous; synchronizing here ensures the
    # diagnostic catches the real extension error instead of reporting a false pass.
    torch.cuda.synchronize(device)
    observed_intersection = float(intersection.reshape(-1)[0].item())
    observed_iou = float(iou.reshape(-1)[0].item())
    if not math.isclose(observed_intersection, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError(
            f"box-IoU kernel returned intersection {observed_intersection!r}; expected 1.0"
        )
    if not math.isclose(observed_iou, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError(f"box-IoU kernel returned IoU {observed_iou!r}; expected 1.0")
    return observed_iou


def run_gpu_diagnostic(
    device_index: int | None = None,
    *,
    importer: Callable[[str], Any] = importlib.import_module,
) -> GpuDiagnosticResult:
    """Inspect CUDA and prove that the installed extension can launch a kernel."""
    base_details = [f"Python: {platform.python_version()}"]
    try:
        torch = importer("torch")
    except Exception as error:
        return GpuDiagnosticResult(
            ok=False,
            stage="torch_import",
            summary="PyTorch could not be imported.",
            details=(*base_details, _exception_message(error)),
            recommendations=("Install the CUDA-enabled PyTorch build documented for CamLabel3D, then retry.",),
        )

    torch_version = str(getattr(torch, "__version__", "unknown"))
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    base_details.extend((f"PyTorch: {torch_version}", f"PyTorch CUDA runtime: {torch_cuda_version or 'none'}"))
    if not torch_cuda_version:
        return GpuDiagnosticResult(
            ok=False,
            stage="torch_cuda_build",
            summary="The installed PyTorch build has no CUDA support.",
            details=tuple(base_details),
            recommendations=(
                "Replace the CPU-only PyTorch package with a CUDA-enabled wheel, then rebuild vis4d_cuda_ops.",
            ),
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as error:
        return GpuDiagnosticResult(
            ok=False,
            stage="cuda_runtime",
            summary="PyTorch failed while initializing CUDA.",
            details=(*base_details, _exception_message(error)),
            recommendations=("Check the NVIDIA driver and ensure it supports the CUDA runtime bundled with PyTorch.",),
        )

    if not cuda_available:
        return GpuDiagnosticResult(
            ok=False,
            stage="cuda_runtime",
            summary="PyTorch has CUDA support, but no usable GPU is visible.",
            details=tuple(base_details),
            recommendations=(
                "Check the NVIDIA driver and CUDA_VISIBLE_DEVICES, then rerun the diagnostic.",
                "If compiling on a machine without a visible GPU, set TORCH_CUDA_ARCH_LIST explicitly.",
            ),
        )

    try:
        selected_device = int(torch.cuda.current_device() if device_index is None else device_index)
        device_count = int(torch.cuda.device_count())
        if selected_device < 0 or selected_device >= device_count:
            raise ValueError(f"CUDA device {selected_device} is outside the available range 0..{device_count - 1}")
        device_name = str(torch.cuda.get_device_name(selected_device))
        architecture, sm_label = capability_labels(torch.cuda.get_device_capability(selected_device))
    except Exception as error:
        return GpuDiagnosticResult(
            ok=False,
            stage="device_query",
            summary="A CUDA device is visible, but its properties could not be queried.",
            details=(*base_details, _exception_message(error)),
            recommendations=("Check the requested --device index and the NVIDIA driver state.",),
        )

    device_details = [
        *base_details,
        f"CUDA device: {selected_device} / {device_name}",
        f"Compute capability: {architecture} ({sm_label})",
    ]
    try:
        ops = importer("vis4d_cuda_ops")
    except Exception as error:
        return GpuDiagnosticResult(
            ok=False,
            stage="extension_import",
            summary="vis4d_cuda_ops could not be imported.",
            device_name=device_name,
            compute_capability=architecture,
            details=(*device_details, _exception_message(error)),
            recommendations=_rebuild_recommendations(architecture),
        )

    extension_path = getattr(ops, "__file__", None)
    if extension_path:
        device_details.append(f"vis4d_cuda_ops: {extension_path}")
    if not callable(getattr(ops, "iou_box3d", None)):
        return GpuDiagnosticResult(
            ok=False,
            stage="extension_api",
            summary="vis4d_cuda_ops is importable but the box-IoU function required by WildDet3D is missing.",
            device_name=device_name,
            compute_capability=architecture,
            details=tuple(device_details),
            recommendations=_rebuild_recommendations(architecture),
        )

    try:
        observed = _run_vis4d_kernel(torch, ops, selected_device)
    except Exception as error:
        if is_architecture_failure(error):
            summary = f"The CUDA extension does not contain a usable kernel for {sm_label}."
        elif is_cpu_only_extension_failure(error):
            summary = "vis4d_cuda_ops was built without CUDA support."
        else:
            summary = "The vis4d_cuda_ops CUDA kernel failed."
        return GpuDiagnosticResult(
            ok=False,
            stage="kernel_execution",
            summary=summary,
            device_name=device_name,
            compute_capability=architecture,
            details=(*device_details, _exception_message(error)),
            recommendations=_rebuild_recommendations(architecture),
        )

    return GpuDiagnosticResult(
        ok=True,
        stage="complete",
        summary=f"PyTorch CUDA and the vis4d_cuda_ops kernel are working on {sm_label}.",
        device_name=device_name,
        compute_capability=architecture,
        details=(*device_details, f"Box-IoU kernel output: {observed}"),
    )


def _bounded_process_output(value: str | bytes | None, limit: int = 2000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = value.strip()
    return value if len(value) <= limit else f"...{value[-limit:]}"


def _result_from_json_output(output: str) -> GpuDiagnosticResult:
    """Decode the CLI payload, tolerating harmless native-library chatter."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        start = output.find("{")
        end = output.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(output[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("GPU diagnostic JSON payload is not an object")
    return GpuDiagnosticResult(
        ok=bool(payload["ok"]),
        stage=str(payload["stage"]),
        summary=str(payload["summary"]),
        device_name=(str(payload["device_name"]) if payload.get("device_name") is not None else None),
        compute_capability=(
            str(payload["compute_capability"])
            if payload.get("compute_capability") is not None
            else None
        ),
        details=tuple(str(item) for item in payload.get("details", ())),
        recommendations=tuple(str(item) for item in payload.get("recommendations", ())),
    )


def run_gpu_diagnostic_isolated(
    device_index: int | None = None,
    *,
    timeout_seconds: float = 120.0,
) -> GpuDiagnosticResult:
    """Run the real-kernel probe outside the application CUDA context.

    An invalid CUDA kernel launch can leave process-local CUDA state unusable.
    The detector therefore uses this isolated form before loading a model and
    only consumes its structured result in the application process.
    """
    command = [sys.executable, str(Path(__file__).resolve()), "--json"]
    if device_index is not None:
        command.extend(("--device", str(int(device_index))))

    # ``ensure_wilddet3d_on_path`` may have amended sys.path immediately before
    # this call. Propagate it so the child can import an extension that has not
    # previously been imported in the application process.
    environment = os.environ.copy()
    path_entries = [str(entry) for entry in sys.path if entry]
    if path_entries:
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path_entries))

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=max(1.0, float(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as error:
        return GpuDiagnosticResult(
            ok=False,
            stage="diagnostic_timeout",
            summary="The isolated CUDA compatibility check timed out.",
            details=tuple(
                item
                for item in (
                    _bounded_process_output(error.stdout),
                    _bounded_process_output(error.stderr),
                )
                if item
            ),
            recommendations=(
                "Run python -m camlabel3d.diagnostics.gpu manually and inspect the NVIDIA driver state.",
            ),
        )
    except OSError as error:
        return GpuDiagnosticResult(
            ok=False,
            stage="diagnostic_process",
            summary="The isolated CUDA compatibility check could not be started.",
            details=(_exception_message(error),),
            recommendations=(
                "Run python -m camlabel3d.diagnostics.gpu from the active CamLabel3D environment.",
            ),
        )

    try:
        result = _result_from_json_output(completed.stdout)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        details = [_exception_message(error)]
        stdout = _bounded_process_output(completed.stdout)
        stderr = _bounded_process_output(completed.stderr)
        if stdout:
            details.append(f"stdout: {stdout}")
        if stderr:
            details.append(f"stderr: {stderr}")
        return GpuDiagnosticResult(
            ok=False,
            stage="diagnostic_protocol",
            summary="The isolated CUDA compatibility check returned an invalid result.",
            details=tuple(details),
            recommendations=(
                "Run python -m camlabel3d.diagnostics.gpu manually to see the complete diagnostic output.",
            ),
        )

    if completed.returncode != 0 and result.ok:
        return GpuDiagnosticResult(
            ok=False,
            stage="diagnostic_process",
            summary="The CUDA probe reported success but its process exited unexpectedly.",
            details=(f"Exit code: {completed.returncode}",),
            recommendations=(
                "Run python -m camlabel3d.diagnostics.gpu manually to verify the environment.",
            ),
        )
    return result


def format_result(result: GpuDiagnosticResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    lines = [f"[{status}] {result.summary}", f"Stage: {result.stage}"]
    lines.extend(f"  - {detail}" for detail in result.details)
    if result.recommendations:
        lines.append("Recommended actions:")
        lines.extend(f"  - {recommendation}" for recommendation in result.recommendations)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify PyTorch CUDA and execute WildDet3D's real vis4d_cuda_ops box-IoU kernel."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device index; defaults to PyTorch's current device",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_gpu_diagnostic(args.device)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_result(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
