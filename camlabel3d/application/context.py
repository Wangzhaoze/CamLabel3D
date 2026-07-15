"""Application composition root independent of Qt widgets."""

from __future__ import annotations

from dataclasses import dataclass

from camlabel3d.core.detector import DetectorAdapter
from camlabel3d.core.postprocess import PostprocessSession
from camlabel3d.core.processing import (
    BulkOperationRegistry,
    OutlierRuleRegistry,
    ProcessingEngine,
    build_default_bulk_operation_registry,
    build_default_outlier_registry,
)
from camlabel3d.core.source_config import DatasetConfigStore
from camlabel3d.runtime_config import RuntimeConfig

from .source_service import SourceService


@dataclass(slots=True)
class ApplicationContext:
    runtime_config: RuntimeConfig
    detector: DetectorAdapter
    dataset_config_store: DatasetConfigStore
    postprocess_session: PostprocessSession
    outlier_registry: OutlierRuleRegistry
    bulk_operation_registry: BulkOperationRegistry
    processing_engine: ProcessingEngine
    source_service: SourceService

    @classmethod
    def create(cls, runtime_config: RuntimeConfig | None = None) -> "ApplicationContext":
        runtime = runtime_config or RuntimeConfig.from_env()
        config_store = DatasetConfigStore()
        outlier_registry = build_default_outlier_registry()
        bulk_registry = build_default_bulk_operation_registry()
        return cls(
            runtime_config=runtime,
            detector=DetectorAdapter(runtime_config=runtime),
            dataset_config_store=config_store,
            postprocess_session=PostprocessSession(),
            outlier_registry=outlier_registry,
            bulk_operation_registry=bulk_registry,
            processing_engine=ProcessingEngine(
                outlier_registry=outlier_registry,
                bulk_operation_registry=bulk_registry,
                max_workers=runtime.cpu_workers,
            ),
            source_service=SourceService(config_store, runtime),
        )
