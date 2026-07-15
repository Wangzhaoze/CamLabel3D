"""Build configuration for the Vis4D C++/CUDA operators.

Modified from Deformable DETR:
https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension


THIS_DIR = Path(__file__).resolve().parent
CUDA_BUILD_ENV = "VIS4D_CUDA_OPS_BUILD_CUDA"
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _environment_flag(name: str) -> bool | None:
    """Read an optional boolean environment variable with useful errors."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None

    value = raw_value.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise RuntimeError(f"{name} must be one of: {accepted}; got {raw_value!r}")


def _available_cpu_count() -> int:
    """Return the CPUs available to this process, respecting affinity if possible."""
    try:
        import psutil

        affinity = psutil.Process().cpu_affinity()
        if affinity:
            return len(affinity)
    except Exception:
        pass
    return max(1, os.cpu_count() or 1)


def _configure_build_parallelism() -> str:
    """Apply a conservative default while preserving an explicit MAX_JOBS."""
    if "MAX_JOBS" not in os.environ:
        cpu_count = _available_cpu_count()
        # C++/CUDA compiler processes are memory-heavy. Keep one CPU free for an
        # interactive desktop and cap the implicit fan-out; users can override it.
        default_jobs = min(4, max(1, cpu_count - 1))
        os.environ["MAX_JOBS"] = str(default_jobs)
    configured_jobs = os.environ["MAX_JOBS"]
    try:
        if int(configured_jobs) < 1:
            raise ValueError
    except ValueError as error:
        raise RuntimeError(f"MAX_JOBS must be a positive integer; got {configured_jobs!r}") from error
    return configured_jobs


def _cuda_runtime_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _should_build_cuda() -> bool:
    """Select CUDA without treating runtime device visibility as the toolchain."""
    requested = _environment_flag(CUDA_BUILD_ENV)
    if requested is None:
        # FORCE_CUDA is retained as a compatibility alias commonly used by
        # PyTorch extensions. The package-specific variable takes precedence.
        requested = _environment_flag("FORCE_CUDA")

    architecture_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    runtime_available = _cuda_runtime_available()
    if requested is None:
        requested = bool(architecture_list) or CUDA_HOME is not None or runtime_available

    if not requested:
        return False

    if torch.version.cuda is None:
        raise RuntimeError(
            "A CUDA build of vis4d_cuda_ops was requested, but this PyTorch build has no CUDA support. "
            "Install a CUDA-enabled PyTorch wheel or set VIS4D_CUDA_OPS_BUILD_CUDA=0 for a CPU-only build."
        )
    if CUDA_HOME is None:
        raise RuntimeError(
            "A CUDA build of vis4d_cuda_ops was requested, but the CUDA toolkit (nvcc) was not found. "
            "Install a toolkit compatible with PyTorch and set CUDA_HOME, or set "
            "VIS4D_CUDA_OPS_BUILD_CUDA=0 for a CPU-only build."
        )
    if not architecture_list and not runtime_available:
        raise RuntimeError(
            "CUDA compilation is enabled but no GPU is visible at build time. Set TORCH_CUDA_ARCH_LIST "
            "explicitly (for example '8.6' for an RTX 3060), then rebuild vis4d_cuda_ops."
        )
    return True


def get_extensions():
    max_jobs = _configure_build_parallelism()
    build_cuda = _should_build_cuda()

    extensions_dir = THIS_DIR / "src"
    main_source = extensions_dir / "vision.cpp"
    source_cpu = sorted(path for path in extensions_dir.rglob("*.cpp") if path != main_source)
    source_cuda = sorted(extensions_dir.rglob("*.cu"))

    sources = [str(main_source), *(str(path) for path in source_cpu)]
    extension = CppExtension
    extra_compile_args = {"cxx": ["/O2" if os.name == "nt" else "-O2"]}
    define_macros: list[tuple[str, str | None]] = []

    if build_cuda:
        extension = CUDAExtension
        sources.extend(str(path) for path in source_cuda)
        define_macros.append(("WITH_CUDA", None))
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-O2",
        ]

    architecture = os.environ.get("TORCH_CUDA_ARCH_LIST", "<visible GPU auto-detection>")
    print(
        f"[vis4d_cuda_ops] backend={'CUDA' if build_cuda else 'CPU'}, "
        f"MAX_JOBS={max_jobs}, TORCH_CUDA_ARCH_LIST={architecture}"
    )

    return [
        extension(
            "vis4d_cuda_ops",
            sources,
            include_dirs=[str(extensions_dir)],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]


requirements = (THIS_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()


setup(
    name="vis4d_cuda_ops",
    version="0.0.0",
    author="VIS @ ETH",
    author_email="i@yf.io",
    url="https://github.com/syscv/vis4d_cuda_ops",
    description="PyTorch Wrapper for CUDA Functions of Vis4D",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
