from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

_APP_TITLE = "CamLabel3D"
_RUNTIME_MARKER = ".camlabel3d_runtime_path"


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    return app_root() / "runtime"


def python_executable(windowed: bool) -> Path:
    runtime = runtime_root()
    candidate = runtime / ("pythonw.exe" if windowed else "python.exe")
    if candidate.exists():
        return candidate
    return runtime / "python.exe"


def _show_error(message: str, *, windowed: bool) -> None:
    if windowed:
        ctypes.windll.user32.MessageBoxW(None, message, _APP_TITLE, 0x10)
    else:
        print(message, file=sys.stderr)


def _runtime_env() -> dict[str, str]:
    root = app_root()
    runtime = runtime_root()
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"):
        env.pop(key, None)

    path_parts = [
        str(runtime),
        str(runtime / "Scripts"),
        str(runtime / "Library" / "bin"),
        str(runtime / "Library" / "usr" / "bin"),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join(part for part in path_parts if part)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(root)
    env["CAMLABEL3D_CONFIG_ROOT"] = str(root / "configs")
    env["CAMLABEL3D_WORKERS_ROOT"] = str(root / "workers")
    env["CAMLABEL3D_WILDDET3D_ROOT"] = str(root / "workers" / "WildDet3D")
    env["CAMLABEL3D_CHECKPOINT_ROOT"] = str(root / "ckpts")
    return env


def _conda_unpack_command() -> list[str] | None:
    runtime = runtime_root()
    candidates = [
        [str(runtime / "Scripts" / "conda-unpack.exe")],
        [str(runtime / "Scripts" / "conda-unpack-script.py")],
    ]
    for candidate in candidates:
        path = Path(candidate[0])
        if not path.exists():
            continue
        if path.suffix.lower() == ".py":
            return [str(python_executable(windowed=False)), str(path)]
        return candidate
    return None


def ensure_runtime_ready(*, windowed: bool) -> bool:
    runtime = runtime_root()
    python_exe = python_executable(windowed=False)
    if not runtime.exists() or not python_exe.exists():
        _show_error(
            "Bundled runtime is missing. Make sure the full release folder was extracted.",
            windowed=windowed,
        )
        return False

    marker = runtime / _RUNTIME_MARKER
    expected_path = str(runtime.resolve())
    current_path = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    if current_path == expected_path:
        return True

    unpack_command = _conda_unpack_command()
    if unpack_command is None:
        _show_error(
            "Bundled runtime is incomplete: conda-unpack was not found.",
            windowed=windowed,
        )
        return False

    try:
        subprocess.run(
            unpack_command,
            check=True,
            cwd=runtime,
            env=_runtime_env(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.CalledProcessError as exc:
        _show_error(
            f"Failed to finalize the bundled runtime (exit code {exc.returncode}).",
            windowed=windowed,
        )
        return False

    marker.write_text(expected_path + "\n", encoding="utf-8")
    return True


def run_python(
    python_args: list[str],
    *,
    windowed: bool,
    app_title: str = _APP_TITLE,
) -> int:
    global _APP_TITLE
    _APP_TITLE = app_title

    if not ensure_runtime_ready(windowed=windowed):
        return 1

    root = app_root()
    command = [str(python_executable(windowed=windowed)), *python_args]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=_runtime_env(),
            check=False,
        )
        return int(completed.returncode)
    except OSError as exc:
        _show_error(f"Failed to start bundled Python runtime: {exc}", windowed=windowed)
        return 1
