"""Blocking source-opening use cases, designed to run on an I/O worker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from camlabel3d.core.frame_provider import FrameProvider, ImageFolderFrameProvider, VideoFrameProvider
from camlabel3d.core.models import SourceContext, SourceMode
from camlabel3d.core.source_config import (
    DatasetConfigStore,
    DatasetSourcesConfig,
    IntrinsicsPreset,
)
from camlabel3d.io.csv_store import CSVStore
from camlabel3d.runtime_config import RuntimeConfig


@dataclass(slots=True)
class LoadedSource:
    provider: FrameProvider
    source_context: SourceContext
    output_path: Path
    active_source_path: Path
    dataset_id: str = ""
    recording_id: str = ""
    intrinsics: IntrinsicsPreset | None = None
    manual_source_path: Path | None = None


class SourceService:
    """Own media/provider construction and session-path discovery."""

    def __init__(
        self,
        config_store: DatasetConfigStore,
        runtime_config: RuntimeConfig,
    ) -> None:
        self.config_store = config_store
        self.runtime_config = runtime_config

    def load_dataset(
        self,
        config: DatasetSourcesConfig,
        dataset_id: str,
        recording_id: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LoadedSource:
        dataset = config.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        provider: FrameProvider | None = None
        try:
            self._check_cancel(should_cancel)
            media_path = self.config_store.resolve_media_path(config, dataset_id, recording_id)
            provider = ImageFolderFrameProvider(
                media_path,
                cache_bytes=self.runtime_config.frame_cache_bytes,
                prefetch_workers=min(2, self.runtime_config.cpu_workers),
                should_cancel=should_cancel,
            )
            self._check_cancel(should_cancel)
            output_path = self.config_store.seed_output_session_files(
                config=config,
                source_mode=SourceMode.DATASET,
                dataset_id=dataset_id,
                recording_id=recording_id,
                on_date=date.today(),
            )
            self._ensure_session_file(output_path)
            self._check_cancel(should_cancel)
            return LoadedSource(
                provider=provider,
                source_context=SourceContext(
                    source_mode=SourceMode.DATASET,
                    source_type="dataset_image_folder",
                    dataset_id=dataset_id,
                    recording_id=recording_id,
                ),
                output_path=output_path,
                active_source_path=media_path,
                dataset_id=dataset_id,
                recording_id=recording_id,
                intrinsics=dataset.default_intrinsics,
            )
        except Exception:
            if provider is not None:
                provider.close()
            raise

    def load_manual(
        self,
        config: DatasetSourcesConfig,
        source_path: str | Path,
        mode: SourceMode,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LoadedSource:
        resolved = Path(source_path).resolve()
        provider: FrameProvider | None = None
        try:
            self._check_cancel(should_cancel)
            if mode == SourceMode.VIDEO:
                provider = VideoFrameProvider(
                    resolved,
                    cache_bytes=self.runtime_config.frame_cache_bytes,
                    preload_all_frames=self.runtime_config.preload_video_frames,
                )
            elif mode == SourceMode.IMAGE_FOLDER:
                provider = ImageFolderFrameProvider(
                    resolved,
                    cache_bytes=self.runtime_config.frame_cache_bytes,
                    prefetch_workers=min(2, self.runtime_config.cpu_workers),
                    should_cancel=should_cancel,
                )
            else:
                raise ValueError(f"Unsupported manual source mode: {mode}")
            self._check_cancel(should_cancel)
            output_path = self.config_store.seed_output_session_files(
                config=config,
                source_mode=mode,
                manual_source_path=resolved,
                on_date=date.today(),
            )
            self._ensure_session_file(output_path)
            self._check_cancel(should_cancel)
            return LoadedSource(
                provider=provider,
                source_context=SourceContext(source_mode=mode, source_type=provider.source_type),
                output_path=output_path,
                active_source_path=resolved,
                manual_source_path=resolved,
            )
        except Exception:
            if provider is not None:
                provider.close()
            raise

    @staticmethod
    def _ensure_session_file(path: Path) -> None:
        """Create the initial CSV while still on the source-loading worker."""

        resolved = Path(path).resolve()
        if not resolved.exists():
            CSVStore(resolved, backup_enabled=False).save_records([])

    @staticmethod
    def _check_cancel(should_cancel: Callable[[], bool] | None) -> None:
        if should_cancel is not None and should_cancel():
            raise RuntimeError("Source loading was canceled.")
