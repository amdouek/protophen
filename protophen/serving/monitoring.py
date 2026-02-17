"""
Monitoring, metrics, and drift detection for ProToPhen serving.

Provides:

- **PredictionMonitor**: Lightweight, in-process collector for latency,
  throughput, and prediction-distribution statistics.  Optionally computes
  regression metrics from ``training.metrics`` when ground-truth feedback
  is available.
- **DriftDetector**: Detects distribution shift between recent predictions
  and a reference (training) distribution using the KS test.

If ``prometheus_client`` is installed the monitor also exposes Prometheus
gauge / histogram / counter objects; otherwise it operates in
*internal-only* mode with no external dependency.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from protophen.utils.logging import logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for the prediction monitor."""

    # Rolling window for latency / throughput stats
    window_size: int = 1000

    # Drift detection
    enable_drift_detection: bool = True
    drift_window_size: int = 500
    drift_significance: float = 0.01  # KS-test p-value threshold

    # Logging
    log_predictions: bool = True
    log_every_n: int = 100

    # Prometheus (optional)
    enable_prometheus: bool = False
    prometheus_port: int = 9090

    # Regression-metric tracking (when ground truth is supplied via feedback)
    track_regression_metrics: bool = True


# =============================================================================
# Prometheus (optional)
# =============================================================================

def _try_prometheus():
    """Return the ``prometheus_client`` module or ``None``."""
    try:
        import prometheus_client  # type: ignore[import-untyped]
        return prometheus_client
    except ImportError:
        return None


# =============================================================================
# Prediction-vs-Observation Tracker
# =============================================================================

class PredictionQualityTracker:
    """
    Track prediction quality when ground-truth observations arrive
    via the feedback endpoint.

    Uses metric classes from ``protophen.training.metrics`` (Session 6)
    so that the same R², Pearson, MSE definitions used during training
    are applied to production predictions.

    The tracker maintains a rolling window of (prediction, observation)
    pairs and recomputes metrics on demand.
    """

    def __init__(self, window_size: int = 500):
        self._window_size = window_size
        self._predictions: Deque[np.ndarray] = collections.deque(maxlen=window_size)
        self._observations: Deque[np.ndarray] = collections.deque(maxlen=window_size)
        self._protein_ids: Deque[str] = collections.deque(maxlen=window_size)

    def add(
        self,
        protein_id: str,
        prediction: np.ndarray,
        observation: np.ndarray,
    ) -> None:
        """Store a matched (prediction, observation) pair."""
        self._protein_ids.append(protein_id)
        self._predictions.append(np.asarray(prediction, dtype=np.float32))
        self._observations.append(np.asarray(observation, dtype=np.float32))

    @property
    def n_pairs(self) -> int:
        return len(self._predictions)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute regression metrics over the stored window.

        Returns an empty dict if fewer than 2 pairs are stored.
        """
        if self.n_pairs < 2:
            return {}

        try:
            import torch
            from protophen.training.metrics import create_default_metrics

            preds = torch.from_numpy(np.stack(list(self._predictions)))
            obs = torch.from_numpy(np.stack(list(self._observations)))

            metrics = create_default_metrics(prefix="quality_")
            metrics.update(preds, obs)
            return metrics.compute()

        except Exception as exc:
            logger.debug(f"Could not compute quality metrics: {exc}")
            return {}

    def reset(self) -> None:
        self._predictions.clear()
        self._observations.clear()
        self._protein_ids.clear()


# =============================================================================
# Prediction Monitor
# =============================================================================

class PredictionMonitor:
    """
    Tracks latency, throughput, and prediction-distribution statistics.

    The monitor operates in a fixed-size rolling window so that memory
    usage is bounded.  Optionally exposes Prometheus metrics and, when
    feedback arrives, computes regression-quality metrics via
    :class:`PredictionQualityTracker`.

    Example::

        monitor = PredictionMonitor()

        # Record each request
        monitor.record_request(
            latency_ms=42.5,
            sequence_length=150,
            predictions={"cell_painting": np.random.randn(1500)},
        )

        # After feedback arrives
        monitor.record_feedback(
            protein_id="prot_001",
            prediction=np.random.randn(1500),
            observation=np.random.randn(1500),
        )

        # Periodic summary
        print(monitor.summary())
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()

        # Rolling buffers
        self._latencies: Deque[float] = collections.deque(
            maxlen=self.config.window_size
        )
        self._sequence_lengths: Deque[int] = collections.deque(
            maxlen=self.config.window_size
        )
        self._prediction_norms: Deque[float] = collections.deque(
            maxlen=self.config.window_size
        )

        # Per-task prediction means (for drift detection)
        self._task_means: Dict[str, Deque[float]] = {}

        # Counters
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_feedback: int = 0
        self._start_time = time.monotonic()

        # Drift detector
        self._drift_detector: Optional[DriftDetector] = None
        if self.config.enable_drift_detection:
            self._drift_detector = DriftDetector(
                window_size=self.config.drift_window_size,
                significance=self.config.drift_significance,
            )

        # Quality tracker (prediction vs observation)
        self._quality_tracker: Optional[PredictionQualityTracker] = None
        if self.config.track_regression_metrics:
            self._quality_tracker = PredictionQualityTracker(
                window_size=self.config.window_size
            )

        # Prediction cache: protein_id → most recent prediction (for matching)
        self._recent_predictions: Dict[str, np.ndarray] = {}
        self._recent_predictions_maxsize = self.config.window_size

        # Optional Prometheus
        self._prom = None
        if self.config.enable_prometheus:
            prom = _try_prometheus()
            if prom is not None:
                self._prom = prom
                self._prom_latency = prom.Histogram(
                    "protophen_inference_latency_ms",
                    "Inference latency in milliseconds",
                    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
                )
                self._prom_requests = prom.Counter(
                    "protophen_requests_total",
                    "Total prediction requests",
                )
                self._prom_errors = prom.Counter(
                    "protophen_errors_total",
                    "Total prediction errors",
                )
                self._prom_feedback = prom.Counter(
                    "protophen_feedback_total",
                    "Total feedback entries received",
                )
            else:
                logger.warning(
                    "prometheus_client not installed; Prometheus metrics disabled."
                )

        logger.info("PredictionMonitor initialised")

    # =========================================================================
    # Recording
    # =========================================================================

    def record_request(
        self,
        latency_ms: float,
        sequence_length: int,
        predictions: Optional[Dict[str, np.ndarray]] = None,
        protein_id: Optional[str] = None,
    ) -> None:
        """
        Record a successful prediction request.

        Args:
            latency_ms: Wall-clock inference time in ms.
            sequence_length: Length of the input sequence.
            predictions: Dict mapping task name → predicted array.
            protein_id: Optional protein hash/ID for later feedback matching.
        """
        self._total_requests += 1
        self._latencies.append(latency_ms)
        self._sequence_lengths.append(sequence_length)

        if predictions is not None:
            for task, arr in predictions.items():
                norm = float(np.linalg.norm(arr))
                self._prediction_norms.append(norm)

                if task not in self._task_means:
                    self._task_means[task] = collections.deque(
                        maxlen=self.config.drift_window_size
                    )
                self._task_means[task].append(float(np.mean(arr)))

                if self._drift_detector is not None:
                    self._drift_detector.add_observation(task, arr)

            # Cache prediction for later feedback matching
            if protein_id is not None and predictions:
                primary_task = next(iter(predictions))
                self._cache_prediction(protein_id, predictions[primary_task])

        # Prometheus
        if self._prom is not None:
            self._prom_latency.observe(latency_ms)
            self._prom_requests.inc()

        # Periodic log
        if (
            self.config.log_predictions
            and self._total_requests % self.config.log_every_n == 0
        ):
            logger.info(
                f"Monitor: {self._total_requests} requests, "
                f"p50={self.latency_p50:.1f}ms, p99={self.latency_p99:.1f}ms"
            )

    def record_error(self) -> None:
        """Record a failed prediction request."""
        self._total_errors += 1
        if self._prom is not None:
            self._prom_errors.inc()

    def record_feedback(
        self,
        protein_id: str,
        observation: np.ndarray,
        prediction: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record ground-truth feedback for quality tracking.

        If *prediction* is not supplied, the monitor looks up the most
        recent cached prediction for *protein_id*.

        Args:
            protein_id: Protein identifier.
            observation: Observed phenotype feature vector.
            prediction: Predicted feature vector (optional).
        """
        self._total_feedback += 1

        if self._prom is not None:
            self._prom_feedback.inc()

        if prediction is None:
            prediction = self._recent_predictions.get(protein_id)

        if prediction is not None and self._quality_tracker is not None:
            self._quality_tracker.add(protein_id, prediction, observation)
            logger.debug(
                f"Quality pair recorded for '{protein_id}' "
                f"(total: {self._quality_tracker.n_pairs})"
            )
        else:
            logger.debug(
                f"Feedback for '{protein_id}' stored but no prediction "
                f"available for quality matching."
            )

    def _cache_prediction(self, protein_id: str, prediction: np.ndarray) -> None:
        """Cache a prediction for later feedback matching (bounded size)."""
        if len(self._recent_predictions) >= self._recent_predictions_maxsize:
            # Evict oldest entry (dict preserves insertion order in Python 3.7+)
            oldest_key = next(iter(self._recent_predictions))
            del self._recent_predictions[oldest_key]
        self._recent_predictions[protein_id] = prediction.copy()

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def latency_p50(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.percentile(list(self._latencies), 50))

    @property
    def latency_p99(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.percentile(list(self._latencies), 99))

    @property
    def throughput_rps(self) -> float:
        """Requests per second over the monitoring lifetime."""
        elapsed = time.monotonic() - self._start_time
        return self._total_requests / elapsed if elapsed > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        """Return a snapshot of current monitoring statistics."""
        latencies = list(self._latencies)
        result: Dict[str, Any] = {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "total_feedback": self._total_feedback,
            "error_rate": self._total_errors / max(self._total_requests, 1),
            "throughput_rps": round(self.throughput_rps, 2),
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
        }

        if latencies:
            result["latency_ms"] = {
                "mean": round(float(np.mean(latencies)), 2),
                "p50": round(float(np.percentile(latencies, 50)), 2),
                "p95": round(float(np.percentile(latencies, 95)), 2),
                "p99": round(float(np.percentile(latencies, 99)), 2),
                "max": round(float(np.max(latencies)), 2),
            }

        if self._prediction_norms:
            norms = list(self._prediction_norms)
            result["prediction_norm"] = {
                "mean": round(float(np.mean(norms)), 4),
                "std": round(float(np.std(norms)), 4),
            }

        # Drift
        if self._drift_detector is not None:
            result["drift"] = self._drift_detector.report()

        # Regression quality (when feedback is available)
        if self._quality_tracker is not None and self._quality_tracker.n_pairs >= 2:
            result["prediction_quality"] = self._quality_tracker.compute_metrics()
            result["prediction_quality"]["n_pairs"] = self._quality_tracker.n_pairs

        return result

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._latencies.clear()
        self._sequence_lengths.clear()
        self._prediction_norms.clear()
        self._task_means.clear()
        self._recent_predictions.clear()
        self._total_requests = 0
        self._total_errors = 0
        self._total_feedback = 0
        self._start_time = time.monotonic()
        if self._drift_detector is not None:
            self._drift_detector.reset()
        if self._quality_tracker is not None:
            self._quality_tracker.reset()
        logger.info("PredictionMonitor reset")


# =============================================================================
# Drift Detector
# =============================================================================

class DriftDetector:
    """
    Detect prediction distribution drift via the Kolmogorov–Smirnov test.

    A reference distribution is established by the first ``window_size``
    observations.  Subsequent windows are compared to the reference and
    a drift flag is raised when *p < significance*.

    The reference can also be set explicitly from training-set prediction
    statistics using :meth:`set_reference`, which accepts any 1-D array
    (e.g., per-sample mean predictions from the validation set computed
    during ``Trainer.evaluate()``).

    Attributes:
        window_size: Number of observations per comparison window.
        significance: KS-test p-value threshold for drift detection.
    """

    def __init__(self, window_size: int = 500, significance: float = 0.01):
        self.window_size = window_size
        self.significance = significance

        # Per-task buffers
        self._reference: Dict[str, np.ndarray] = {}
        self._current: Dict[str, List[float]] = {}
        self._is_reference_set: Dict[str, bool] = {}
        self._drift_flags: Dict[str, bool] = {}
        self._last_p_values: Dict[str, float] = {}

    def set_reference(
        self, task: str, reference_values: np.ndarray
    ) -> None:
        """
        Explicitly set the reference distribution for a task.

        This is typically called at startup with prediction-mean values
        from the training or validation set.

        Args:
            task: Task name.
            reference_values: 1-D array of reference prediction means or norms.
        """
        self._reference[task] = np.asarray(reference_values).ravel()
        self._is_reference_set[task] = True
        self._current.setdefault(task, [])
        self._drift_flags[task] = False
        logger.debug(
            f"Drift reference set for '{task}' "
            f"(n={len(self._reference[task])})"
        )

    def set_reference_from_trainer(
        self,
        trainer_predictions: Dict[str, np.ndarray],
    ) -> None:
        """
        Set reference distributions from ``Trainer.predict()`` output.

        The ``Trainer.predict()`` method (Session 6) returns a dict with
        keys like ``cell_painting_predictions``.  This convenience method
        extracts the per-sample means and registers them as references.

        Args:
            trainer_predictions: Output of ``Trainer.predict()``.
        """
        for key, arr in trainer_predictions.items():
            if key.endswith("_predictions"):
                task = key.replace("_predictions", "")
                # Per-sample mean across features
                if arr.ndim > 1:
                    means = arr.mean(axis=1)
                else:
                    means = arr
                self.set_reference(task, means)

    def add_observation(self, task: str, prediction: np.ndarray) -> None:
        """
        Add a single prediction observation.

        If no explicit reference has been set, the first ``window_size``
        observations become the reference automatically.
        """
        mean_val = float(np.mean(prediction))
        self._current.setdefault(task, [])
        self._current[task].append(mean_val)
        self._is_reference_set.setdefault(task, False)

        if not self._is_reference_set[task]:
            if len(self._current[task]) >= self.window_size:
                self._reference[task] = np.array(
                    self._current[task][:self.window_size]
                )
                self._is_reference_set[task] = True
                self._current[task] = self._current[task][self.window_size:]
                logger.info(
                    f"Drift reference auto-set for '{task}' "
                    f"from first {self.window_size} observations"
                )
            return

        # Enough new data to test?
        if len(self._current[task]) >= self.window_size:
            current_arr = np.array(self._current[task][-self.window_size:])
            stat, p_value = sp_stats.ks_2samp(
                self._reference[task], current_arr
            )
            self._last_p_values[task] = p_value
            self._drift_flags[task] = p_value < self.significance

            if self._drift_flags[task]:
                logger.warning(
                    f"Drift detected for task '{task}': "
                    f"KS p={p_value:.4e} < {self.significance}"
                )

            # Slide the window
            self._current[task] = self._current[task][-self.window_size:]

    def report(self) -> Dict[str, Any]:
        """Return drift status for all monitored tasks."""
        return {
            task: {
                "drift_detected": self._drift_flags.get(task, False),
                "p_value": round(self._last_p_values.get(task, 1.0), 6),
                "reference_set": self._is_reference_set.get(task, False),
                "current_observations": len(self._current.get(task, [])),
            }
            for task in set(
                list(self._reference.keys()) + list(self._current.keys())
            )
        }

    def reset(self) -> None:
        """Clear all state."""
        self._reference.clear()
        self._current.clear()
        self._is_reference_set.clear()
        self._drift_flags.clear()
        self._last_p_values.clear()