"""
Acquisition functions for active learning.

This module provides acquisition functions for scoring and selecting
the most informative samples for labeling.

Strategies:
- Uncertainty sampling: Select most uncertain samples
- Expected improvement: Balance exploration and exploitation
- Diversity sampling: Select diverse samples (DPP, k-means)
- Hybrid: Combine uncertainty and diversity
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

from protophen.active_learning.uncertainty import UncertaintyEstimate, UncertaintyType
from protophen.utils.logging import logger


# =============================================================================
# Base Acquisition Function
# =============================================================================

class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    @abstractmethod
    def score(
        self,
        uncertainty: UncertaintyEstimate,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute acquisition scores for samples.
        
        Args:
            uncertainty: Uncertainty estimates for candidate samples
            **kwargs: Additional arguments
            
        Returns:
            Acquisition scores of shape (n_samples,)
            Higher scores = higher priority for selection
        """
        pass
    
    def select(
        self,
        uncertainty: UncertaintyEstimate,
        n_select: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Select top samples by acquisition score.
        
        Args:
            uncertainty: Uncertainty estimates
            n_select: Number of samples to select
            **kwargs: Additional arguments
            
        Returns:
            Indices of selected samples
        """
        scores = self.score(uncertainty, **kwargs)
        return np.argsort(scores)[-n_select:][::-1]


# =============================================================================
# Uncertainty-Based Acquisition
# =============================================================================

class UncertaintySampling(AcquisitionFunction):
    """
    Select samples with highest uncertainty.
    
    This is the simplest active learning strategy - just pick the
    samples where the model is most uncertain.
    
    Example:
        >>> acq = UncertaintySampling(uncertainty_type="epistemic")
        >>> scores = acq.score(uncertainty_estimate)
        >>> top_indices = acq.select(uncertainty_estimate, n_select=10)
    """
    
    def __init__(
        self,
        uncertainty_type: UncertaintyType = UncertaintyType.TOTAL,
        reduction: Literal["mean", "sum", "max"] = "mean",
    ):
        """
        Initialise uncertainty sampling.
        
        Args:
            uncertainty_type: Which uncertainty to use
            reduction: How to reduce uncertainty across features
        """
        self.uncertainty_type = uncertainty_type
        self.reduction = reduction
    
    def score(
        self,
        uncertainty: UncertaintyEstimate,
        **kwargs,
    ) -> np.ndarray:
        """Compute acquisition scores based on uncertainty."""
        return uncertainty.get_uncertainty(
            uncertainty_type=self.uncertainty_type,
            reduction=self.reduction,
        )


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement acquisition function.
    
    Balances exploration (high uncertainty) with exploitation
    (predictions close to best observed value).
    
    EI(x) = E[max(f(x) - f_best, 0)]
    
    For minimisation:
    EI(x) = (f_best - mu) * Φ((f_best - mu) / sigma) + sigma * φ((f_best - mu) / sigma)
    
    where:
    - mu: predicted mean
    - sigma: predicted uncertainty
    - Φ: CDF of standard normal
    - φ: PDF of standard normal
    """
    
    def __init__(
        self,
        target_feature_idx: Optional[int] = None,
        maximise: bool = False,
        xi: float = 0.01,
    ):
        """
        Initialise Expected Improvement.
        
        Args:
            target_feature_idx: Index of target feature (None = use mean)
            maximise: If True, maximise the target; if False, minimise
            xi: Exploration-exploitation trade-off parameter
        """
        self.target_feature_idx = target_feature_idx
        self.maximise = maximise
        self.xi = xi
        self.best_value: Optional[float] = None
    
    def set_best_value(self, value: float) -> None:
        """Set the best observed value."""
        self.best_value = value
    
    def score(
        self,
        uncertainty: UncertaintyEstimate,
        best_value: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute Expected Improvement scores.
        
        Args:
            uncertainty: Uncertainty estimates with mean and std
            best_value: Best observed value (overrides stored value)
            
        Returns:
            EI scores for each sample
        """
        # Get mean predictions
        if self.target_feature_idx is not None:
            mu = uncertainty.mean[:, self.target_feature_idx]
        else:
            mu = uncertainty.mean.mean(axis=-1) if uncertainty.mean.ndim > 1 else uncertainty.mean
        
        # Get uncertainty (std)
        sigma = uncertainty.get_uncertainty(
            uncertainty_type=UncertaintyType.TOTAL,
            reduction="mean",
        )
        
        # Get best value
        f_best = best_value or self.best_value
        if f_best is None:
            # Use current best prediction as default
            f_best = mu.min() if not self.maximise else mu.max()
        
        # Add exploration bonus
        f_best = f_best - self.xi if not self.maximise else f_best + self.xi
        
        # Compute EI
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.maximise:
                z = (mu - f_best) / (sigma + 1e-8)
                ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
            else:
                z = (f_best - mu) / (sigma + 1e-8)
                ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Handle edge cases
        ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
        
        return ei


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement acquisition function.
    
    Simpler than EI - just computes the probability that a sample
    will improve upon the best observed value.
    
    PI(x) = P(f(x) > f_best) = Φ((mu - f_best) / sigma)  [for maximisation]
    """
    
    def __init__(
        self,
        target_feature_idx: Optional[int] = None,
        maximise: bool = False,
        xi: float = 0.01,
    ):
        """
        Initialise Probability of Improvement.
        
        Args:
            target_feature_idx: Index of target feature
            maximise: If True, maximise the target
            xi: Exploration bonus
        """
        self.target_feature_idx = target_feature_idx
        self.maximise = maximise
        self.xi = xi
        self.best_value: Optional[float] = None
    
    def score(
        self,
        uncertainty: UncertaintyEstimate,
        best_value: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute Probability of Improvement scores."""
        if self.target_feature_idx is not None:
            mu = uncertainty.mean[:, self.target_feature_idx]
        else:
            mu = uncertainty.mean.mean(axis=-1) if uncertainty.mean.ndim > 1 else uncertainty.mean
        
        sigma = uncertainty.get_uncertainty(
            uncertainty_type=UncertaintyType.TOTAL,
            reduction="mean",
        )
        
        f_best = best_value or self.best_value
        if f_best is None:
            f_best = mu.min() if not self.maximise else mu.max()
        
        f_best = f_best - self.xi if not self.maximise else f_best + self.xi
        
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.maximise:
                z = (mu - f_best) / (sigma + 1e-8)
            else:
                z = (f_best - mu) / (sigma + 1e-8)
            pi = norm.cdf(z)
        
        return np.nan_to_num(pi, nan=0.0)


# =============================================================================
# Diversity-Based Acquisition
# =============================================================================

class DiversitySampling(AcquisitionFunction):
    """
    Select diverse samples using embedding distance.
    
    Uses k-means++ style selection or Determinantal Point Processes (DPP)
    to ensure selected samples are diverse in embedding space.
    
    Example:
        >>> acq = DiversitySampling(method="kmeans++")
        >>> indices = acq.select(uncertainty, n_select=10, embeddings=pool_embeddings)
    """
    
    def __init__(
        self,
        method: Literal["kmeans++", "maxmin", "dpp_approx"] = "kmeans++",
        metric: str = "euclidean",
    ):
        """
        Initialise diversity sampling.
        
        Args:
            method: Diversity selection method
            metric: Distance metric for embeddings
        """
        self.method = method
        self.metric = metric
    
    def score(
        self,
        uncertainty: UncertaintyEstimate,
        embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Diversity doesn't have simple scores - use select() instead.
        
        Returns uniform scores as placeholder.
        """
        return np.ones(uncertainty.n_samples)
    
    def select(
        self,
        uncertainty: UncertaintyEstimate,
        n_select: int,
        embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Select diverse samples.
        
        Args:
            uncertainty: Uncertainty estimates (used for embeddings if not provided)
            n_select: Number of samples to select
            embeddings: Sample embeddings for diversity calculation
            
        Returns:
            Indices of selected samples
        """
        # Use predictions as embeddings if not provided
        if embeddings is None:
            embeddings = uncertainty.mean
        
        if self.method == "kmeans++":
            return self._kmeans_pp_select(embeddings, n_select)
        elif self.method == "maxmin":
            return self._maxmin_select(embeddings, n_select)
        elif self.method == "dpp_approx":
            return self._dpp_approx_select(embeddings, n_select)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _kmeans_pp_select(
        self,
        embeddings: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """Select samples using k-means++ initialisation."""
        n_samples = embeddings.shape[0]
        n_select = min(n_select, n_samples)
        
        # Start with random sample
        selected = [np.random.randint(n_samples)]
        
        for _ in range(n_select - 1):
            # Compute distance to nearest selected sample
            selected_embeddings = embeddings[selected]
            distances = cdist(embeddings, selected_embeddings, metric=self.metric)
            min_distances = distances.min(axis=1)
            
            # Zero out already selected
            min_distances[selected] = 0
            
            # Sample proportional to squared distance
            probs = min_distances ** 2
            probs = probs / (probs.sum() + 1e-8)
            
            new_idx = np.random.choice(n_samples, p=probs)
            selected.append(new_idx)
        
        return np.array(selected)
    
    def _maxmin_select(
        self,
        embeddings: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """Select samples maximising minimum distance to selected set."""
        n_samples = embeddings.shape[0]
        n_select = min(n_select, n_samples)
        
        # Start with random sample
        selected = [np.random.randint(n_samples)]
        
        # Track minimum distance to selected set for each sample
        min_dist_to_selected = np.full(n_samples, np.inf)
        
        for _ in range(n_select - 1):
            # Update minimum distances with last selected sample
            last_selected = selected[-1]
            distances = cdist(
                embeddings,
                embeddings[last_selected:last_selected+1],
                metric=self.metric,
            ).squeeze()
            min_dist_to_selected = np.minimum(min_dist_to_selected, distances)
            
            # Zero out already selected
            min_dist_to_selected[selected] = -np.inf
            
            # Select sample with maximum minimum distance
            new_idx = np.argmax(min_dist_to_selected)
            selected.append(new_idx)
        
        return np.array(selected)
    
    def _dpp_approx_select(
        self,
        embeddings: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """
        Approximate DPP selection using greedy algorithm.
        
        DPP encourages diversity by penalising similar items.
        """
        n_samples = embeddings.shape[0]
        n_select = min(n_select, n_samples)
        
        # Compute kernel matrix (similarity)
        similarity = np.exp(-cdist(embeddings, embeddings, metric=self.metric) ** 2)
        
        selected = []
        remaining = list(range(n_samples))
        
        # Greedy selection
        for _ in range(n_select):
            if not remaining:
                break
            
            if not selected:
                # First sample: pick randomly or by some criterion
                idx = np.random.choice(remaining)
            else:
                # Compute log-det gain for each remaining sample
                best_gain = -np.inf
                best_idx = remaining[0]
                
                for idx in remaining:
                    # Compute marginal gain in diversity
                    selected_plus = selected + [idx]
                    L_selected = similarity[np.ix_(selected_plus, selected_plus)]
                    
                    try:
                        sign, logdet = np.linalg.slogdet(L_selected)
                        gain = sign * logdet
                    except np.linalg.LinAlgError:
                        gain = -np.inf
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                
                idx = best_idx
            
            selected.append(idx)
            remaining.remove(idx)
        
        return np.array(selected)


# =============================================================================
# Hybrid Acquisition
# =============================================================================

class HybridAcquisition(AcquisitionFunction):
    """
    Combine uncertainty and diversity for sample selection.
    
    Uses a weighted combination of uncertainty scores and diversity
    to select samples that are both informative and diverse.
    
    Example:
        >>> acq = HybridAcquisition(
        ...     uncertainty_weight=0.7,
        ...     diversity_weight=0.3,
        ... )
        >>> indices = acq.select(uncertainty, n_select=10, embeddings=embeddings)
    """
    
    def __init__(
        self,
        uncertainty_weight: float = 0.7,
        diversity_weight: float = 0.3,
        uncertainty_type: UncertaintyType = UncertaintyType.TOTAL,
        diversity_method: str = "kmeans++",
    ):
        """
        Initialise hybrid acquisition.
        
        Args:
            uncertainty_weight: Weight for uncertainty score
            diversity_weight: Weight for diversity
            uncertainty_type: Which uncertainty to use
            diversity_method: Diversity selection method
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.uncertainty_type = uncertainty_type
        
        self.uncertainty_acq = UncertaintySampling(uncertainty_type=uncertainty_type)
        self.diversity_acq = DiversitySampling(method=diversity_method)
    
    def score(
        self,
        uncertainty: UncertaintyEstimate,
        **kwargs,
    ) -> np.ndarray:
        """Compute uncertainty scores (diversity handled in select)."""
        return self.uncertainty_acq.score(uncertainty)
    
    def select(
        self,
        uncertainty: UncertaintyEstimate,
        n_select: int,
        embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Select samples balancing uncertainty and diversity.
        
        Uses iterative selection that alternates between high-uncertainty
        samples and diverse samples.
        
        Args:
            uncertainty: Uncertainty estimates
            n_select: Number of samples to select
            embeddings: Sample embeddings for diversity
            
        Returns:
            Indices of selected samples
        """
        n_samples = uncertainty.n_samples
        n_select = min(n_select, n_samples)
        
        # Get uncertainty scores
        unc_scores = self.uncertainty_acq.score(uncertainty)
        
        # Normalise scores to [0, 1]
        unc_scores = (unc_scores - unc_scores.min()) / (unc_scores.max() - unc_scores.min() + 1e-8)
        
        # Use predictions as embeddings if not provided
        if embeddings is None:
            embeddings = uncertainty.mean
        
        # Iterative selection
        selected = []
        remaining = set(range(n_samples))
        
        for i in range(n_select):
            if not remaining:
                break
            
            remaining_list = list(remaining)
            
            if not selected:
                # First sample: highest uncertainty
                remaining_scores = unc_scores[remaining_list]
                best_local_idx = np.argmax(remaining_scores)
                best_idx = remaining_list[best_local_idx]
            else:
                # Compute diversity scores (distance to selected set)
                selected_embeddings = embeddings[selected]
                remaining_embeddings = embeddings[remaining_list]
                
                distances = cdist(remaining_embeddings, selected_embeddings)
                div_scores = distances.min(axis=1)  # Distance to nearest selected
                
                # Normalise
                div_scores = (div_scores - div_scores.min()) / (div_scores.max() - div_scores.min() + 1e-8)
                
                # Combine scores
                remaining_unc = unc_scores[remaining_list]
                combined_scores = (
                    self.uncertainty_weight * remaining_unc +
                    self.diversity_weight * div_scores
                )
                
                best_local_idx = np.argmax(combined_scores)
                best_idx = remaining_list[best_local_idx]
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return np.array(selected)


# =============================================================================
# Batch Acquisition
# =============================================================================

class BatchAcquisition(AcquisitionFunction):
    """
    Batch-aware acquisition that considers interactions within batch.
    
    When selecting multiple samples at once, naive greedy selection
    can lead to redundant samples. This class implements batch-aware
    strategies.
    """
    
    def __init__(
        self,
        base_acquisition: AcquisitionFunction,
        batch_strategy: Literal["greedy", "batch_bald", "stochastic"] = "greedy",
    ):
        """
        Initialise batch acquisition.
        
        Args:
            base_acquisition: Base acquisition function
            batch_strategy: Strategy for batch selection
        """
        self.base_acquisition = base_acquisition
        self.batch_strategy = batch_strategy
    
    def score(
        self,
        uncertainty: UncertaintyEstimate,
        **kwargs,
    ) -> np.ndarray:
        """Compute base acquisition scores."""
        return self.base_acquisition.score(uncertainty, **kwargs)
    
    def select(
        self,
        uncertainty: UncertaintyEstimate,
        n_select: int,
        embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Select batch of samples.
        
        Args:
            uncertainty: Uncertainty estimates
            n_select: Batch size
            embeddings: Sample embeddings
            
        Returns:
            Indices of selected samples
        """
        if self.batch_strategy == "greedy":
            return self._greedy_select(uncertainty, n_select, embeddings, **kwargs)
        elif self.batch_strategy == "stochastic":
            return self._stochastic_select(uncertainty, n_select, **kwargs)
        else:
            # Default to base acquisition
            return self.base_acquisition.select(uncertainty, n_select, **kwargs)
    
    def _greedy_select(
        self,
        uncertainty: UncertaintyEstimate,
        n_select: int,
        embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Greedy selection with diminishing scores for similar samples."""
        scores = self.base_acquisition.score(uncertainty, **kwargs)
        
        if embeddings is None:
            embeddings = uncertainty.mean
        
        n_samples = uncertainty.n_samples
        selected = []
        current_scores = scores.copy()
        
        for _ in range(min(n_select, n_samples)):
            # Select highest scoring sample
            best_idx = np.argmax(current_scores)
            selected.append(best_idx)
            
            # Reduce scores of similar samples
            distances = cdist(
                embeddings,
                embeddings[best_idx:best_idx+1],
            ).squeeze()
            
            # Exponential decay based on distance
            similarity = np.exp(-distances)
            current_scores = current_scores * (1 - 0.5 * similarity)
            
            # Zero out selected
            current_scores[best_idx] = -np.inf
        
        return np.array(selected)
    
    def _stochastic_select(
        self,
        uncertainty: UncertaintyEstimate,
        n_select: int,
        temperature: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Stochastic selection proportional to scores."""
        scores = self.base_acquisition.score(uncertainty, **kwargs)
        
        # Softmax with temperature
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / exp_scores.sum()
        
        # Sample without replacement
        selected = np.random.choice(
            uncertainty.n_samples,
            size=min(n_select, uncertainty.n_samples),
            replace=False,
            p=probs,
        )
        
        return selected


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_acquisition_scores(
    uncertainty: UncertaintyEstimate,
    method: Literal["uncertainty", "ei", "pi", "random"] = "uncertainty",
    **kwargs,
) -> np.ndarray:
    """
    Compute acquisition scores for samples.
    
    Args:
        uncertainty: Uncertainty estimates
        method: Acquisition method
        **kwargs: Method-specific arguments
        
    Returns:
        Acquisition scores
    """
    if method == "uncertainty":
        acq = UncertaintySampling(**kwargs)
    elif method == "ei":
        acq = ExpectedImprovement(**kwargs)
    elif method == "pi":
        acq = ProbabilityOfImprovement(**kwargs)
    elif method == "random":
        return np.random.rand(uncertainty.n_samples)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return acq.score(uncertainty)