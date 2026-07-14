"""Runtime helpers for locating project-local configs, checkpoints, and workers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent


def config_root() -> Path:
    """Return the repository config directory."""
    env_override = os.environ.get("CAMLABEL3D_CONFIG_ROOT")
    if env_override:
        return Path(env_override).resolve()
    return (project_root() / "configs").resolve()


def workers_root() -> Path:
    """Return the repository workers directory."""
    env_override = os.environ.get("CAMLABEL3D_WORKERS_ROOT")
    if env_override:
        return Path(env_override).resolve()
    return (project_root() / "workers").resolve()


def wilddet3d_root() -> Path:
    """Return the local WildDet3D checkout path."""
    env_override = os.environ.get("CAMLABEL3D_WILDDET3D_ROOT") or os.environ.get("WILDDET3D_SOURCE_ROOT")
    if env_override:
        return Path(env_override).resolve()

    candidates = [
        workers_root() / "WildDet3D",
        project_root() / "WildDet3D",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def checkpoint_root() -> Path:
    """Return the repository checkpoint directory."""
    env_override = os.environ.get("CAMLABEL3D_CHECKPOINT_ROOT") or os.environ.get("WILDDET3D_CKPT_ROOT")
    if env_override:
        return Path(env_override).resolve()

    candidates = [
        project_root() / "ckpts",
        wilddet3d_root() / "ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def default_dataset_config_path() -> Path:
    """Return the default dataset config path."""
    return config_root() / "dataset_sources.json"


def ensure_wilddet3d_on_path() -> Path:
    """Make the local WildDet3D package importable."""
    root = wilddet3d_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ.setdefault("WILDDET3D_SOURCE_ROOT", root_str)
    lingbot_checkpoint = default_lingbot_depth_path()
    if lingbot_checkpoint.exists():
        os.environ.setdefault("WILDDET3D_LINGBOT_DEPTH_MODEL", str(lingbot_checkpoint))
    return root


def default_checkpoint_path() -> Path:
    """Return the preferred local WildDet3D checkpoint path."""
    env_override = os.environ.get("CAMLABEL3D_WILDDET3D_CHECKPOINT")
    if env_override:
        return Path(env_override).resolve()
    return (checkpoint_root() / "wilddet3d_alldata_all_prompt_v1.0.pt").resolve()


def default_lingbot_depth_path() -> Path:
    """Return the preferred local LingBot-Depth checkpoint path."""
    env_override = os.environ.get("CAMLABEL3D_LINGBOT_DEPTH_CHECKPOINT") or os.environ.get(
        "WILDDET3D_LINGBOT_DEPTH_MODEL"
    )
    if env_override:
        return Path(env_override).resolve()
    return (checkpoint_root() / "lingbot_depth_model.pt").resolve()
