"""
Training infrastructure for ProToPhen.

This package provides tools for training protein-to-phenotype
prediction models, including:
- Training loops with mixed precision and gradient accumulation
- Callbacks for checkpointing, early stopping, and logging
- Evaluation metrics for regression tasks
"""

from protophen.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingState,
)
from protophen.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    LearningRateCallback,
    TensorBoardCallback,
    ProgressCallback,
)
from protophen.training.metrics import (
    MetricCollection,
    MSEMetric,
    MAEMetric,
    R2Metric,
    PearsonCorrelationMetric,
    SpearmanCorrelationMetric,
    CosineSimilarityMetric,
    compute_regression_metrics,
    compute_per_feature_metrics,
    MultiTaskMetricCollection,
    create_multitask_metrics,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    # Callbacks
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "LearningRateCallback",
    "TensorBoardCallback",
    "ProgressCallback",
    # Metrics
    "MetricCollection",
    "MSEMetric",
    "MAEMetric",
    "R2Metric",
    "PearsonCorrelationMetric",
    "SpearmanCorrelationMetric",
    "CosineSimilarityMetric",
    "compute_regression_metrics",
    "compute_per_feature_metrics",
]