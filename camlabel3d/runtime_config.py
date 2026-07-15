"""Process-wide compute and responsiveness settings.

The module intentionally depends only on the Python standard library so the
thread budget can be applied before NumPy, OpenCV, Qt, or PyTorch are imported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Bounded resource policy shared by UI, media, and inference services."""

    device: str = "auto"
    cpu_workers: int = 1
    torch_threads: int = 1
    torch_interop_threads: int = 1
    frame_cache_bytes: int = 2048 * 1024 * 1024
    preload_video_frames: bool = True
    preview_debounce_ms: int = 35
    autosave_debounce_ms: int = 350
    keep_model_loaded: bool = True
    enable_amp: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Build a safe policy from ``CAMLABEL3D_*`` environment variables."""

        logical_cpus = max(1, os.cpu_count() or 1)
        # Keep one logical CPU available to the Qt event loop and OS. A modest
        # cap also prevents NumPy/OpenMP + our executors from oversubscribing.
        default_workers = max(1, min(8, logical_cpus - 1 if logical_cpus > 1 else 1))
        cpu_workers = _bounded_int("CAMLABEL3D_CPU_WORKERS", default_workers, 1, logical_cpus)
        torch_threads = _bounded_int("CAMLABEL3D_TORCH_THREADS", cpu_workers, 1, logical_cpus)
        interop_default = min(2, cpu_workers)
        torch_interop_threads = _bounded_int(
            "CAMLABEL3D_TORCH_INTEROP_THREADS",
            interop_default,
            1,
            max(1, min(4, logical_cpus)),
        )
        cache_mb = _bounded_int("CAMLABEL3D_FRAME_CACHE_MB", 2048, 0, 8192)
        device = os.environ.get("CAMLABEL3D_DEVICE", "auto").strip().lower() or "auto"

        return cls(
            device=device,
            cpu_workers=cpu_workers,
            torch_threads=torch_threads,
            torch_interop_threads=torch_interop_threads,
            frame_cache_bytes=cache_mb * 1024 * 1024,
            preload_video_frames=_env_flag("CAMLABEL3D_PRELOAD_VIDEO_FRAMES", True),
            preview_debounce_ms=_bounded_int("CAMLABEL3D_PREVIEW_DEBOUNCE_MS", 35, 0, 1000),
            autosave_debounce_ms=_bounded_int("CAMLABEL3D_AUTOSAVE_DEBOUNCE_MS", 350, 0, 5000),
            keep_model_loaded=_env_flag("CAMLABEL3D_KEEP_MODEL_LOADED", True),
            enable_amp=_env_flag("CAMLABEL3D_ENABLE_AMP", False),
        )


def configure_process_environment(config: RuntimeConfig) -> None:
    """Apply conservative native-library thread limits before heavy imports."""

    thread_count = str(max(1, int(config.torch_threads)))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, thread_count)
    # Hugging Face tokenizers can otherwise create another process-wide pool.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
