"""
Evaluation metrics for ProToPhen.

This module provides metrics for evaluating protein-to-phenotype
prediction models, focusing on regression and similarity metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats

from protophen.utils.logging import logger


# =============================================================================
# Base Metric Class
# =============================================================================

class Metric(ABC):
    """Abstract base class for metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass
    
    @abstractmethod
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update metric with new predictions and targets."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute the metric value."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# =============================================================================
# Regression Metrics
# =============================================================================

class MSEMetric(Metric):
    """Mean Squared Error metric."""
    
    def __init__(self, name: str = "mse"):
        super().__init__(name)
    
    def reset(self) -> None:
        self._sum_squared_error = 0.0
        self._count = 0
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        predictions = predictions.detach().float()
        targets = targets.detach().float()
        
        squared_error = (predictions - targets).pow(2).sum().item()
        self._sum_squared_error += squared_error
        self._count += predictions.numel()
    
    def compute(self) -> float:
        if self._count == 0:
            return 0.0
        return self._sum_squared_error / self._count


class MAEMetric(Metric):
    """Mean Absolute Error metric."""
    
    def __init__(self, name: str = "mae"):
        super().__init__(name)
    
    def reset(self) -> None:
        self._sum_absolute_error = 0.0
        self._count = 0
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        predictions = predictions.detach().float()
        targets = targets.detach().float()
        
        absolute_error = (predictions - targets).abs().sum().item()
        self._sum_absolute_error += absolute_error
        self._count += predictions.numel()
    
    def compute(self) -> float:
        if self._count == 0:
            return 0.0
        return self._sum_absolute_error / self._count


class RMSEMetric(Metric):
    """Root Mean Squared Error metric."""
    
    def __init__(self, name: str = "rmse"):
        self._mse = MSEMetric()
        super().__init__(name)
    
    def reset(self) -> None:
        self._mse.reset()
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._mse.update(predictions, targets)
    
    def compute(self) -> float:
        return np.sqrt(self._mse.compute())


class R2Metric(Metric):
    """
    R² (Coefficient of Determination) metric.
    
    R² = 1 - SS_res / SS_tot
    where SS_res = sum((y - y_pred)²) and SS_tot = sum((y - y_mean)²)
    """
    
    def __init__(self, name: str = "r2"):
        super().__init__(name)
    
    def reset(self) -> None:
        self._predictions: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._predictions.append(predictions.detach().cpu())
        self._targets.append(targets.detach().cpu())
    
    def compute(self) -> float:
        if not self._predictions:
            return 0.0
        
        predictions = torch.cat(self._predictions, dim=0).flatten()
        targets = torch.cat(self._targets, dim=0).flatten()
        
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        
        if ss_tot < 1e-8:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()


class PearsonCorrelationMetric(Metric):
    """
    Pearson correlation coefficient metric.
    
    Computes correlation either:
    - Per-sample: Correlation across features for each sample, then averaged
    - Per-feature: Correlation across samples for each feature, then averaged
    - Global: Flattened correlation
    """
    
    def __init__(
        self,
        name: str = "pearson",
        mode: Literal["per_sample", "per_feature", "global"] = "per_sample",
    ):
        self.mode = mode
        super().__init__(name)
    
    def reset(self) -> None:
        self._predictions: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._predictions.append(predictions.detach().cpu())
        self._targets.append(targets.detach().cpu())
    
    def compute(self) -> float:
        if not self._predictions:
            return 0.0
        
        predictions = torch.cat(self._predictions, dim=0)
        targets = torch.cat(self._targets, dim=0)
        
        if self.mode == "per_sample":
            # Correlation across features for each sample
            return self._compute_per_sample(predictions, targets)
        elif self.mode == "per_feature":
            # Correlation across samples for each feature
            return self._compute_per_feature(predictions, targets)
        else:
            # Global correlation
            return self._compute_global(predictions, targets)
    
    def _compute_per_sample(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute correlation per sample, then average."""
        if predictions.dim() == 1:
            return self._compute_global(predictions, targets)
        
        correlations = []
        for pred, tgt in zip(predictions, targets):
            corr = self._pearson(pred, tgt)
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_per_feature(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute correlation per feature, then average."""
        if predictions.dim() == 1:
            return self._compute_global(predictions, targets)
        
        correlations = []
        n_features = predictions.shape[1]
        
        for i in range(n_features):
            corr = self._pearson(predictions[:, i], targets[:, i])
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_global(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute global correlation on flattened arrays."""
        return self._pearson(predictions.flatten(), targets.flatten())
    
    @staticmethod
    def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson correlation between two 1D tensors."""
        x = x.numpy()
        y = y.numpy()
        
        if len(x) < 2:
            return 0.0
        
        # Handle constant arrays
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        
        corr, _ = stats.pearsonr(x, y)
        return corr if not np.isnan(corr) else 0.0


class SpearmanCorrelationMetric(Metric):
    """
    Spearman rank correlation coefficient metric.
    
    More robust to outliers than Pearson correlation.
    """
    
    def __init__(
        self,
        name: str = "spearman",
        mode: Literal["per_sample", "per_feature", "global"] = "per_sample",
    ):
        self.mode = mode
        super().__init__(name)
    
    def reset(self) -> None:
        self._predictions: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._predictions.append(predictions.detach().cpu())
        self._targets.append(targets.detach().cpu())
    
    def compute(self) -> float:
        if not self._predictions:
            return 0.0
        
        predictions = torch.cat(self._predictions, dim=0)
        targets = torch.cat(self._targets, dim=0)
        
        if self.mode == "per_sample":
            return self._compute_per_sample(predictions, targets)
        elif self.mode == "per_feature":
            return self._compute_per_feature(predictions, targets)
        else:
            return self._compute_global(predictions, targets)
    
    def _compute_per_sample(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        if predictions.dim() == 1:
            return self._compute_global(predictions, targets)
        
        correlations = []
        for pred, tgt in zip(predictions, targets):
            corr = self._spearman(pred, tgt)
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_per_feature(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        if predictions.dim() == 1:
            return self._compute_global(predictions, targets)
        
        correlations = []
        n_features = predictions.shape[1]
        
        for i in range(n_features):
            corr = self._spearman(predictions[:, i], targets[:, i])
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_global(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        return self._spearman(predictions.flatten(), targets.flatten())
    
    @staticmethod
    def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Spearman correlation between two 1D tensors."""
        x = x.numpy()
        y = y.numpy()
        
        if len(x) < 2:
            return 0.0
        
        corr, _ = stats.spearmanr(x, y)
        return corr if not np.isnan(corr) else 0.0


class CosineSimilarityMetric(Metric):
    """
    Cosine similarity metric.
    
    Measures the cosine of the angle between prediction and target vectors.
    """
    
    def __init__(
        self,
        name: str = "cosine_similarity",
        mode: Literal["per_sample", "global"] = "per_sample",
    ):
        self.mode = mode
        super().__init__(name)
    
    def reset(self) -> None:
        self._similarities: List[float] = []
    
    def update(
    self,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ) -> None:
        predictions = predictions.detach().float()
        targets = targets.detach().float()
        
        if self.mode == "per_sample":
            # Compute cosine similarity per sample
            sim = torch.nn.functional.cosine_similarity(
                predictions, targets, dim=-1
            )
            # Handle scalar output for 1D inputs
            if sim.dim() == 0:
                self._similarities.append(sim.item())
            else:
                self._similarities.extend(sim.cpu().tolist())
        else:
            # Global similarity
            pred_flat = predictions.flatten()
            tgt_flat = targets.flatten()
            sim = torch.nn.functional.cosine_similarity(
                pred_flat.unsqueeze(0), tgt_flat.unsqueeze(0)
            )
            self._similarities.append(sim.item())
    
    def compute(self) -> float:
        if not self._similarities:
            return 0.0
        return np.mean(self._similarities)


# =============================================================================
# Metric Collection
# =============================================================================

class MetricCollection:
    """
    Collection of metrics for easy management.
    
    Example:
        >>> metrics = MetricCollection([
        ...     MSEMetric(),
        ...     R2Metric(),
        ...     PearsonCorrelationMetric(),
        ... ])
        >>> 
        >>> for batch in dataloader:
        ...     predictions = model(batch)
        ...     metrics.update(predictions, targets)
        >>> 
        >>> results = metrics.compute()
        >>> print(results)
        {'mse': 0.05, 'r2': 0.85, 'pearson': 0.92}
    """
    
    def __init__(
        self,
        metrics: Optional[List[Metric]] = None,
        prefix: str = "",
    ):
        """
        Initialise metric collection.
        
        Args:
            metrics: List of Metric objects
            prefix: Prefix to add to metric names
        """
        self.metrics = metrics or []
        self.prefix = prefix
    
    def add(self, metric: Metric) -> None:
        """Add a metric to the collection."""
        self.metrics.append(metric)
    
    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics:
            metric.reset()
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update all metrics with new data."""
        for metric in self.metrics:
            metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}
        for metric in self.metrics:
            name = f"{self.prefix}{metric.name}" if self.prefix else metric.name
            results[name] = metric.compute()
        return results
    
    def __len__(self) -> int:
        return len(self.metrics)
    
    def __repr__(self) -> str:
        metric_names = [m.name for m in self.metrics]
        return f"MetricCollection(metrics={metric_names})"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_metrics(
    prefix: str = "",
    include_correlation: bool = True,
) -> MetricCollection:
    """
    Create a default set of metrics for regression tasks.
    
    Args:
        prefix: Prefix for metric names
        include_correlation: Whether to include correlation metrics
        
    Returns:
        MetricCollection with standard metrics
    """
    metrics = [
        MSEMetric(),
        MAEMetric(),
        RMSEMetric(),
        R2Metric(),
    ]
    
    if include_correlation:
        metrics.extend([
            PearsonCorrelationMetric(mode="per_sample"),
            CosineSimilarityMetric(mode="per_sample"),
        ])
    
    return MetricCollection(metrics=metrics, prefix=prefix)


def compute_regression_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """
    Compute standard regression metrics in one call.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of metric values
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    metrics = create_default_metrics()
    metrics.update(predictions, targets)
    return metrics.compute()


def compute_per_feature_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each feature separately.
    
    Args:
        predictions: Model predictions of shape (n_samples, n_features)
        targets: Ground truth targets of shape (n_samples, n_features)
        feature_names: Optional names for features
        
    Returns:
        Dictionary mapping feature names to their metrics
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    n_features = predictions.shape[1]
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    results = {}
    for i, name in enumerate(feature_names):
        pred_i = predictions[:, i]
        tgt_i = targets[:, i]
        
        # MSE
        mse = ((pred_i - tgt_i) ** 2).mean().item()
        
        # R²
        ss_res = ((tgt_i - pred_i) ** 2).sum()
        ss_tot = ((tgt_i - tgt_i.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        # Pearson correlation
        pred_np = pred_i.numpy()
        tgt_np = tgt_i.numpy()
        if np.std(pred_np) > 1e-8 and np.std(tgt_np) > 1e-8:
            corr, _ = stats.pearsonr(pred_np, tgt_np)
        else:
            corr = 0.0
        
        results[name] = {
            "mse": mse,
            "r2": r2,
            "pearson": corr if not np.isnan(corr) else 0.0,
        }
    
    return results


def summarise_per_feature_metrics(
    per_feature_results: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Summarise per-feature metrics into aggregate statistics.
    
    Args:
        per_feature_results: Output from compute_per_feature_metrics
        
    Returns:
        Summary statistics (mean, median, std for each metric)
    """
    all_metrics = {}
    
    for feature_name, metrics in per_feature_results.items():
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)
    
    summary = {}
    for metric_name, values in all_metrics.items():
        values = np.array(values)
        summary[f"{metric_name}_mean"] = float(np.mean(values))
        summary[f"{metric_name}_median"] = float(np.median(values))
        summary[f"{metric_name}_std"] = float(np.std(values))
        summary[f"{metric_name}_min"] = float(np.min(values))
        summary[f"{metric_name}_max"] = float(np.max(values))
    
    return summary

# =============================================================================
# Multi-Task Metrics
# =============================================================================

class MultiTaskMetricCollection:
    """
    Collection of metrics for multiple tasks.
    
    Maintains separate MetricCollection instances for each task to handle
    different output dimensions (e.g., Cell Painting has 1500 features,
    viability has 1).
    
    Example:
        >>> metrics = MultiTaskMetricCollection(
        ...     tasks=["cell_painting", "viability"],
        ...     include_correlation=True,
        ... )
        >>> 
        >>> # Update with predictions and targets per task
        >>> metrics.update("cell_painting", cp_pred, cp_target)
        >>> metrics.update("viability", viab_pred, viab_target)
        >>> 
        >>> # Compute all metrics
        >>> results = metrics.compute()
        >>> # {'cell_painting_mse': 0.05, 'cell_painting_r2': 0.85, 
        >>> #  'viability_mse': 0.02, 'viability_r2': 0.91, ...}
    """
    
    def __init__(
        self,
        tasks: List[str],
        include_correlation: bool = True,
        custom_metrics: Optional[Dict[str, List[Metric]]] = None,
    ):
        """
        Initialise multi-task metrics collection.
        
        Args:
            tasks: List of task names
            include_correlation: Whether to include correlation metrics
            custom_metrics: Optional dict mapping task names to custom metric lists
        """
        self.tasks = tasks
        self.collections: Dict[str, MetricCollection] = {}
        
        for task in tasks:
            if custom_metrics and task in custom_metrics:
                # Use custom metrics for this task
                self.collections[task] = MetricCollection(
                    metrics=custom_metrics[task],
                    prefix=f"{task}_",
                )
            else:
                # Use default metrics
                self.collections[task] = create_default_metrics(
                    prefix=f"{task}_",
                    include_correlation=include_correlation,
                )
    
    def reset(self, task: Optional[str] = None) -> None:
        """
        Reset metrics.
        
        Args:
            task: Specific task to reset (None = reset all)
        """
        if task is not None:
            if task in self.collections:
                self.collections[task].reset()
        else:
            for collection in self.collections.values():
                collection.reset()
    
    def update(
        self,
        task: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Update metrics for a specific task.
        
        Args:
            task: Task name
            predictions: Model predictions
            targets: Ground truth targets
        """
        if task not in self.collections:
            logger.warning(f"Unknown task '{task}' - skipping metrics update")
            return
        
        self.collections[task].update(predictions, targets)
    
    def compute(self, task: Optional[str] = None) -> Dict[str, float]:
        """
        Compute metrics.
        
        Args:
            task: Specific task to compute (None = compute all)
            
        Returns:
            Dictionary of metric values
        """
        if task is not None:
            if task in self.collections:
                return self.collections[task].compute()
            return {}
        
        # Compute all tasks
        results = {}
        for task_name, collection in self.collections.items():
            task_results = collection.compute()
            results.update(task_results)
        
        return results
    
    def get_task_metrics(self, task: str) -> Optional[MetricCollection]:
        """Get the MetricCollection for a specific task."""
        return self.collections.get(task)
    
    def add_task(
        self,
        task: str,
        metrics: Optional[List[Metric]] = None,
        include_correlation: bool = True,
    ) -> None:
        """
        Add a new task.
        
        Args:
            task: Task name
            metrics: Optional custom metrics
            include_correlation: Whether to include correlation metrics
        """
        if metrics:
            self.collections[task] = MetricCollection(
                metrics=metrics,
                prefix=f"{task}_",
            )
        else:
            self.collections[task] = create_default_metrics(
                prefix=f"{task}_",
                include_correlation=include_correlation,
            )
        
        if task not in self.tasks:
            self.tasks.append(task)
    
    def __len__(self) -> int:
        return len(self.collections)
    
    def __repr__(self) -> str:
        task_info = ", ".join(
            f"{task}: {len(coll)} metrics" 
            for task, coll in self.collections.items()
        )
        return f"MultiTaskMetricCollection({task_info})"

def create_multitask_metrics(
    tasks: List[str],
    include_correlation: bool = True,
) -> MultiTaskMetricCollection:
    """
    Factory function to create multi-task metrics.
    
    Args:
        tasks: List of task names
        include_correlation: Whether to include correlation metrics
        
    Returns:
        MultiTaskMetricCollection instance
    """
    return MultiTaskMetricCollection(
        tasks=tasks,
        include_correlation=include_correlation,
    )