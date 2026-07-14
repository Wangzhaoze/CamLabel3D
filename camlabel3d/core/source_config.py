"""Dataset-driven source configuration and result-path helpers."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from camlabel3d.runtime import default_dataset_config_path, project_root

from .models import SourceMode, natural_sort_key

CONFIG_FILENAME = "dataset_sources.json"
DEFAULT_RESULT_ROOT = project_root() / "annotation_results"


@dataclass(frozen=True)
class IntrinsicsPreset:
    """Camera intrinsics preset stored in dataset config."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class DatasetSourceConfig:
    """Single dataset entry from the config file."""

    id: str
    display_name: str
    root_path: str
    media_subdir: str
    default_intrinsics: IntrinsicsPreset

    @property
    def root(self) -> Path:
        return Path(self.root_path).resolve()


@dataclass(frozen=True)
class DatasetSourcesConfig:
    """In-memory representation of dataset_sources.json."""

    result_root: str
    datasets: list[DatasetSourceConfig]

    @property
    def result_root_path(self) -> Path:
        return Path(self.result_root).resolve()

    def dataset_ids(self) -> list[str]:
        return [item.id for item in self.datasets]

    def get_dataset(self, dataset_id: str) -> DatasetSourceConfig | None:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        return None


class DatasetConfigStore:
    """Loads dataset config, discovers recordings, and resolves output paths."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or default_dataset_config_path()).resolve()

    def ensure_exists(self) -> Path:
        if not self.path.exists():
            self.write_default()
        return self.path

    def write_default(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.default_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load(self) -> DatasetSourcesConfig:
        self.ensure_exists()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        datasets: list[DatasetSourceConfig] = []
        for raw in payload.get("datasets", []):
            intrinsics = raw.get("default_intrinsics", {})
            datasets.append(
                DatasetSourceConfig(
                    id=str(raw["id"]),
                    display_name=str(raw.get("display_name", raw["id"])),
                    root_path=str(raw["root_path"]),
                    media_subdir=str(raw.get("media_subdir", "images_0")),
                    default_intrinsics=IntrinsicsPreset(
                        fx=float(intrinsics["fx"]),
                        fy=float(intrinsics["fy"]),
                        cx=float(intrinsics["cx"]),
                        cy=float(intrinsics["cy"]),
                    ),
                )
            )
        return DatasetSourcesConfig(
            result_root=str(payload.get("result_root", DEFAULT_RESULT_ROOT)),
            datasets=datasets,
        )

    def discover_recordings(
        self,
        config: DatasetSourcesConfig,
        dataset_id: str,
    ) -> list[str]:
        dataset = config.get_dataset(dataset_id)
        if dataset is None or not dataset.root.exists() or not dataset.root.is_dir():
            return []
        recordings = [
            child.name
            for child in dataset.root.iterdir()
            if child.is_dir() and (child / dataset.media_subdir).is_dir()
        ]
        return sorted(recordings, key=natural_sort_key)

    def resolve_media_path(
        self,
        config: DatasetSourcesConfig,
        dataset_id: str,
        recording_id: str,
    ) -> Path:
        dataset = config.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Unknown dataset id: {dataset_id}")
        return (dataset.root / recording_id / dataset.media_subdir).resolve()

    def ensure_result_dir(
        self,
        config: DatasetSourcesConfig,
        on_date: date | None = None,
    ) -> Path:
        day = on_date or date.today()
        result_dir = config.result_root_path / day.strftime("%Y-%m-%d")
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def resolve_output_csv_path(
        self,
        config: DatasetSourcesConfig,
        source_mode: SourceMode,
        manual_source_path: str | Path | None = None,
        dataset_id: str = "",
        recording_id: str = "",
        on_date: date | None = None,
    ) -> Path:
        result_dir = self.ensure_result_dir(config, on_date=on_date)
        filename = self.build_output_filename(
            source_mode=source_mode,
            manual_source_path=manual_source_path,
            dataset_id=dataset_id,
            recording_id=recording_id,
        )
        return result_dir / filename

    def build_output_filename(
        self,
        source_mode: SourceMode,
        manual_source_path: str | Path | None = None,
        dataset_id: str = "",
        recording_id: str = "",
    ) -> str:
        if source_mode == SourceMode.DATASET:
            if not dataset_id or not recording_id:
                raise ValueError("Dataset mode requires dataset_id and recording_id.")
            return f"{self._safe_name(dataset_id)}__{self._safe_name(recording_id)}.camlabel3d.csv"
        else:
            if not manual_source_path:
                raise ValueError("Manual source output path requires a manual source path.")
            source_path = Path(manual_source_path).resolve()
            if source_mode == SourceMode.VIDEO:
                stem = source_path.stem
                return f"video__{self._safe_name(stem)}.camlabel3d.csv"
            elif source_mode == SourceMode.IMAGE_FOLDER:
                return f"folder__{self._safe_name(source_path.name)}.camlabel3d.csv"
            else:
                raise ValueError(f"Unsupported source mode: {source_mode}")

    def seed_output_session_files(
        self,
        config: DatasetSourcesConfig,
        source_mode: SourceMode,
        manual_source_path: str | Path | None = None,
        dataset_id: str = "",
        recording_id: str = "",
        on_date: date | None = None,
    ) -> Path:
        raw_path = self.resolve_output_csv_path(
            config=config,
            source_mode=source_mode,
            manual_source_path=manual_source_path,
            dataset_id=dataset_id,
            recording_id=recording_id,
            on_date=on_date,
        )
        latest_path = self._latest_path_for(raw_path)
        if raw_path.exists() or latest_path.exists():
            return raw_path

        existing_raw, existing_latest = self.find_existing_session_files(
            config=config,
            source_mode=source_mode,
            manual_source_path=manual_source_path,
            dataset_id=dataset_id,
            recording_id=recording_id,
        )

        if existing_raw is not None and existing_raw.exists():
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_raw, raw_path)
        if existing_latest is not None and existing_latest.exists():
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_latest, latest_path)
        return raw_path

    def find_existing_session_files(
        self,
        config: DatasetSourcesConfig,
        source_mode: SourceMode,
        manual_source_path: str | Path | None = None,
        dataset_id: str = "",
        recording_id: str = "",
    ) -> tuple[Path | None, Path | None]:
        filename = self.build_output_filename(
            source_mode=source_mode,
            manual_source_path=manual_source_path,
            dataset_id=dataset_id,
            recording_id=recording_id,
        )
        raw_match = self._latest_existing_match(config.result_root_path, filename)
        latest_match = self._latest_existing_match(config.result_root_path, self._latest_name_for(filename))
        return raw_match, latest_match

    @staticmethod
    def default_payload() -> dict:
        return {
            "result_root": str(DEFAULT_RESULT_ROOT),
            "datasets": [
                {
                    "id": "RAMPCNN",
                    "display_name": "RAMPCNN",
                    "root_path": "D:/Datasets/RAMPCNN",
                    "media_subdir": "images_0",
                    "default_intrinsics": {
                        "fx": 849.177455,
                        "fy": 854.207389,
                        "cx": 712.166787,
                        "cy": 543.445028,
                    },
                }
            ],
        }

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        return cleaned.strip("._") or "unnamed"

    @staticmethod
    def _latest_name_for(filename: str) -> str:
        suffix = ".camlabel3d.csv"
        if filename.endswith(suffix):
            return f"{filename[:-len(suffix)]}.latest{suffix}"
        path = Path(filename)
        return f"{path.stem}.latest{path.suffix}"

    @classmethod
    def _latest_path_for(cls, raw_path: Path) -> Path:
        return raw_path.with_name(cls._latest_name_for(raw_path.name))

    @staticmethod
    def _latest_existing_match(root: Path, filename: str) -> Path | None:
        if not root.exists():
            return None
        matches = [
            candidate.resolve()
            for candidate in root.glob(f"*/{filename}")
            if candidate.is_file()
        ]
        if not matches:
            return None
        return max(matches, key=lambda path: (path.parent.name, str(path)))
