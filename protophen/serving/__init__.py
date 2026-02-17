"""
ProToPhen Serving Infrastructure.

This subpackage provides production-ready deployment components:
- InferencePipeline: End-to-end sequence → prediction
- REST API: FastAPI service for real-time and batch inference
- Monitoring: Latency, throughput, and drift detection
- Registry: Model versioning and lifecycle management

- Trainer-checkpoint compatibility (auto-detection and state-dict inference)
- Prediction quality tracking via feedback loop
- Drift detector integration with Trainer.predict() output
- Registry registration from Trainer/CheckpointCallback checkpoints

Install extras with:

    pip install 'protophen[serving]'
"""

# --- Pipeline and checkpoint utils ---
from protophen.serving.pipeline import (
    InferencePipeline, PipelineConfig, load_checkpoint, build_model_from_checkpoint,
)

# --- Pydantic schemas ---
from protophen.serving.schemas import (
    ProteinInput,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModelInfoResponse,
    HealthResponse,
    UncertaintyOutput,
    TaskPrediction,
)

# --- Monitoring ---
from protophen.serving.monitoring import (
    PredictionMonitor,
    MonitoringConfig,
    DriftDetector,
    PredictionQualityTracker,
)

# --- Registry ---
from protophen.serving.registry import (
    ModelRegistry,
    ModelVersion,
    RegistryConfig,
)

# --- API factory (lazy - FastAPI may not be installed) ---
# We import the factory name, but don't call it at import time.
try:
    from protophen.serving.api import create_app, FeedbackStore
except ImportError:
    # FastAPI not installed - create_app and FeedbackStore unavailable
    create_app = None   # type: ignore[assignment]
    FeedbackStore = None   # type: ignore[assignment,misc]

__all__ = [
    # Pipeline
    "InferencePipeline",
    "PipelineConfig",
    "load_checkpoint",
    "build_model_from_checkpoint",
    # Schemas
    "ProteinInput",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "ModelInfoResponse",
    "HealthResponse",
    "UncertaintyOutput",
    "TaskPrediction",
    # Monitoring
    "PredictionMonitor",
    "MonitoringConfig",
    "DriftDetector",
    "PredictionQualityTracker",
    # Registry
    "ModelRegistry",
    "ModelVersion",
    "RegistryConfig",
    # API
    "create_app",
    "FeedbackStore",
]