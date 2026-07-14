"""Core domain models and runtime services for CamLabel3D."""

from .detector import DetectorAdapter
from .frame_provider import (
    FrameProvider,
    ImageFolderFrameProvider,
    VideoFrameProvider,
    open_media_source,
)
from .models import (
    DetectionConfig,
    DetectionRecord,
    PointPrompt,
    PromptMode,
    PromptSpec,
    SourceContext,
    SourceMode,
)
from .postprocess import FilterConfig, PostprocessSession, TrackSummary, WorkflowStage, clone_records
from .processing import (
    BulkOperation,
    BulkOperationRegistry,
    OperationResult,
    OperationScope,
    OutlierHit,
    OutlierRule,
    OutlierRuleRegistry,
    OutlierScope,
    ParameterSpec,
    ProcessingContext,
    ProcessingEngine,
    ProcessingScope,
    build_default_bulk_operation_registry,
    build_default_outlier_registry,
    hits_to_report_json,
)
from .source_config import DatasetConfigStore, DatasetSourceConfig, DatasetSourcesConfig, IntrinsicsPreset
from .tracking import TrackingConfig, TrackingEngine

__all__ = [
    "DatasetConfigStore",
    "DatasetSourceConfig",
    "DatasetSourcesConfig",
    "DetectionConfig",
    "DetectionRecord",
    "DetectorAdapter",
    "BulkOperation",
    "BulkOperationRegistry",
    "FrameProvider",
    "FilterConfig",
    "ImageFolderFrameProvider",
    "IntrinsicsPreset",
    "OperationResult",
    "OperationScope",
    "OutlierHit",
    "OutlierRule",
    "OutlierRuleRegistry",
    "OutlierScope",
    "ParameterSpec",
    "PointPrompt",
    "PostprocessSession",
    "ProcessingContext",
    "ProcessingEngine",
    "ProcessingScope",
    "PromptMode",
    "PromptSpec",
    "SourceContext",
    "SourceMode",
    "TrackSummary",
    "TrackingConfig",
    "TrackingEngine",
    "VideoFrameProvider",
    "WorkflowStage",
    "build_default_bulk_operation_registry",
    "build_default_outlier_registry",
    "clone_records",
    "hits_to_report_json",
    "open_media_source",
]
