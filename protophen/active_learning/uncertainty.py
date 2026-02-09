"""
Uncertainty estimation for active learning.

This module provides methods for quantifying prediction uncertainty,
which is crucial for identifying informative samples to label next.

Methods:
- MC Dropout: Monte Carlo sampling with dropout at inference
- Deep Ensembles: Multiple models trained with different seeds
- Evidential: Direct uncertainty prediction (optional)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from protophen.utils.logging import logger


# =============================================================================
# Uncertainty Types
# =============================================================================

class UncertaintyType(str, Enum):
    """Types of uncertainty that can be estimated."""
    
    EPISTEMIC = "epistemic"  # Model uncertainty (reducible with more data)
    ALEATORIC = "aleatoric"  # Data uncertainty (irreducible noise)
    TOTAL = "total"  # Combined uncertainty


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates."""
    
    # Mean predictions
    mean: np.ndarray  # Shape: (n_samples, n_features)
    
    # Uncertainty measures
    epistemic: Optional[np.ndarray] = None  # Model uncertainty
    aleatoric: Optional[np.ndarray] = None  # Data uncertainty
    total: Optional[np.ndarray] = None  # Total uncertainty
    
    # Raw samples (if available)
    samples: Optional[np.ndarray] = None  # Shape: (n_mc_samples, n_samples, n_features)
    
    # Sample identifiers
    sample_ids: Optional[List[str]] = None
    
    @property
    def n_samples(self) -> int:
        """Number of data samples."""
        return self.mean.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of output features."""
        return self.mean.shape[1] if self.mean.ndim > 1 else 1
    
    def get_uncertainty(
        self,
        uncertainty_type: UncertaintyType = UncertaintyType.TOTAL,
        reduction: Literal["mean", "sum", "max", "none"] = "mean",
    ) -> np.ndarray:
        """
        Get uncertainty values with optional reduction.
        
        Args:
            uncertainty_type: Which uncertainty to return
            reduction: How to reduce across features
            
        Returns:
            Uncertainty values of shape (n_samples,) or (n_samples, n_features)
        """
        if uncertainty_type == UncertaintyType.EPISTEMIC:
            unc = self.epistemic
        elif uncertainty_type == UncertaintyType.ALEATORIC:
            unc = self.aleatoric
        else:
            unc = self.total
        
        if unc is None:
            raise ValueError(f"Uncertainty type '{uncertainty_type}' not available")
        
        if reduction == "none":
            return unc
        elif reduction == "mean":
            return unc.mean(axis=-1) if unc.ndim > 1 else unc
        elif reduction == "sum":
            return unc.sum(axis=-1) if unc.ndim > 1 else unc
        elif reduction == "max":
            return unc.max(axis=-1) if unc.ndim > 1 else unc
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "total": self.total,
            "sample_ids": self.sample_ids,
        }


# =============================================================================
# Base Uncertainty Estimator
# =============================================================================

class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimation."""
    
    @abstractmethod
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        **kwargs,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty for data samples.
        
        Args:
            model: Trained model
            dataloader: Data loader for samples
            **kwargs: Additional arguments
            
        Returns:
            UncertaintyEstimate with predictions and uncertainties
        """
        pass


# =============================================================================
# MC Dropout Estimator
# =============================================================================

class MCDropoutEstimator(UncertaintyEstimator):
    """
    Monte Carlo Dropout uncertainty estimation.
    
    Performs multiple forward passes with dropout enabled to estimate
    epistemic (model) uncertainty through the variance of predictions.
    
    Reference:
        Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation:
        Representing model uncertainty in deep learning. ICML.
    
    Example:
        >>> estimator = MCDropoutEstimator(n_samples=20)
        >>> uncertainty = estimator.estimate(model, dataloader)
        >>> 
        >>> # Get most uncertain samples
        >>> scores = uncertainty.get_uncertainty(reduction="mean")
        >>> top_indices = np.argsort(scores)[-10:]
    """
    
    def __init__(
        self,
        n_samples: int = 20,
        tasks: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialise MC Dropout estimator.
        
        Args:
            n_samples: Number of MC forward passes
            tasks: Tasks to estimate uncertainty for
            device: Device for computation
        """
        self.n_samples = n_samples
        self.tasks = tasks or ["cell_painting"]
        self.device = device
    
    def _enable_dropout(self, model: nn.Module) -> None:
        """Enable dropout layers during inference."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def _disable_dropout(self, model: nn.Module) -> None:
        """Disable dropout layers (restore eval mode)."""
        model.eval()
    
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        show_progress: bool = True,
        return_samples: bool = False,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using MC Dropout.
        
        Args:
            model: Trained model with dropout layers
            dataloader: Data loader for samples
            show_progress: Whether to show progress bar
            return_samples: Whether to return all MC samples
            
        Returns:
            UncertaintyEstimate with epistemic uncertainty
        """
        device = self.device or next(model.parameters()).device
        model = model.to(device)
        
        # Collect predictions across MC samples
        all_predictions: Dict[str, List[List[torch.Tensor]]] = {
            task: [] for task in self.tasks
        }
        all_sample_ids: List[str] = []
        
        # First pass to collect sample IDs
        sample_ids_collected = False
        
        for mc_idx in tqdm(
            range(self.n_samples),
            desc="MC Dropout",
            disable=not show_progress,
        ):
            # Enable dropout
            model.train()
            self._enable_dropout(model)
            
            batch_predictions: Dict[str, List[torch.Tensor]] = {
                task: [] for task in self.tasks
            }
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move to device
                    protein_embedding = batch["protein_embedding"].to(device)
                    
                    # Forward pass
                    outputs = model(protein_embedding, tasks=self.tasks)
                    
                    # Collect predictions
                    for task in self.tasks:
                        if task in outputs:
                            batch_predictions[task].append(outputs[task].cpu())
                    
                    # Collect sample IDs (only on first MC sample)
                    if not sample_ids_collected and "protein_id" in batch:
                        all_sample_ids.extend(batch["protein_id"])
            
            sample_ids_collected = True
            
            # Stack batch predictions
            for task in self.tasks:
                if batch_predictions[task]:
                    task_preds = torch.cat(batch_predictions[task], dim=0)
                    all_predictions[task].append(task_preds)
        
        # Restore eval mode
        model.eval()
        
        # Process predictions for each task
        # For simplicity, we'll focus on the first task for the main output
        primary_task = self.tasks[0]
        
        if not all_predictions[primary_task]:
            raise ValueError("No predictions collected")
        
        # Stack MC samples: (n_mc_samples, n_data_samples, n_features)
        mc_predictions = torch.stack(all_predictions[primary_task], dim=0).numpy()
        
        # Compute statistics
        mean = mc_predictions.mean(axis=0)  # (n_samples, n_features)
        epistemic = mc_predictions.std(axis=0)  # Epistemic uncertainty
        
        # Total uncertainty (for MC Dropout, this equals epistemic)
        total = epistemic.copy()
        
        return UncertaintyEstimate(
            mean=mean,
            epistemic=epistemic,
            aleatoric=None,  # MC Dropout doesn't estimate aleatoric uncertainty
            total=total,
            samples=mc_predictions if return_samples else None,
            sample_ids=all_sample_ids if all_sample_ids else None,
        )


# =============================================================================
# Ensemble Estimator
# =============================================================================

class EnsembleEstimator(UncertaintyEstimator):
    """
    Deep Ensemble uncertainty estimation.
    
    Uses multiple models trained with different random seeds to estimate
    uncertainty through prediction disagreement.
    
    Reference:
        Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).
        Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.
    
    Example:
        >>> # Train ensemble
        >>> models = [train_model(seed=i) for i in range(5)]
        >>> 
        >>> # Estimate uncertainty
        >>> estimator = EnsembleEstimator()
        >>> uncertainty = estimator.estimate(models, dataloader)
    """
    
    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialise ensemble estimator.
        
        Args:
            tasks: Tasks to estimate uncertainty for
            device: Device for computation
        """
        self.tasks = tasks or ["cell_painting"]
        self.device = device
    
    def estimate(
        self,
        model: Union[nn.Module, List[nn.Module]],
        dataloader: DataLoader,
        show_progress: bool = True,
        return_samples: bool = False,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using ensemble.
        
        Args:
            model: List of trained models (ensemble members)
            dataloader: Data loader for samples
            show_progress: Whether to show progress bar
            return_samples: Whether to return all model predictions
            
        Returns:
            UncertaintyEstimate with epistemic uncertainty
        """
        if isinstance(model, nn.Module):
            raise ValueError(
                "EnsembleEstimator requires a list of models. "
                "Use MCDropoutEstimator for single-model uncertainty."
            )
        
        models = model
        n_models = len(models)
        
        if n_models < 2:
            raise ValueError("Ensemble requires at least 2 models")
        
        device = self.device or next(models[0].parameters()).device
        
        # Move all models to device and set to eval mode
        for m in models:
            m.to(device)
            m.eval()
        
        # Collect predictions from each model
        all_predictions: Dict[str, List[List[torch.Tensor]]] = {
            task: [] for task in self.tasks
        }
        all_sample_ids: List[str] = []
        
        sample_ids_collected = False
        
        for model_idx, m in enumerate(
            tqdm(models, desc="Ensemble", disable=not show_progress)
        ):
            batch_predictions: Dict[str, List[torch.Tensor]] = {
                task: [] for task in self.tasks
            }
            
            with torch.no_grad():
                for batch in dataloader:
                    protein_embedding = batch["protein_embedding"].to(device)
                    outputs = m(protein_embedding, tasks=self.tasks)
                    
                    for task in self.tasks:
                        if task in outputs:
                            batch_predictions[task].append(outputs[task].cpu())
                    
                    if not sample_ids_collected and "protein_id" in batch:
                        all_sample_ids.extend(batch["protein_id"])
            
            sample_ids_collected = True
            
            for task in self.tasks:
                if batch_predictions[task]:
                    task_preds = torch.cat(batch_predictions[task], dim=0)
                    all_predictions[task].append(task_preds)
        
        # Process predictions
        primary_task = self.tasks[0]
        
        if not all_predictions[primary_task]:
            raise ValueError("No predictions collected")
        
        # Stack ensemble predictions: (n_models, n_samples, n_features)
        ensemble_predictions = torch.stack(
            all_predictions[primary_task], dim=0
        ).numpy()
        
        # Compute statistics
        mean = ensemble_predictions.mean(axis=0)
        epistemic = ensemble_predictions.std(axis=0)
        total = epistemic.copy()
        
        return UncertaintyEstimate(
            mean=mean,
            epistemic=epistemic,
            aleatoric=None,
            total=total,
            samples=ensemble_predictions if return_samples else None,
            sample_ids=all_sample_ids if all_sample_ids else None,
        )


# =============================================================================
# Heteroscedastic Uncertainty (Aleatoric)
# =============================================================================

class HeteroscedasticEstimator(UncertaintyEstimator):
    """
    Estimate aleatoric uncertainty from models that predict variance.
    
    Requires a model that outputs both mean and log-variance predictions.
    Can be combined with MC Dropout for both uncertainty types.
    """
    
    def __init__(
        self,
        n_mc_samples: int = 20,
        use_mc_dropout: bool = True,
        tasks: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialise heteroscedastic estimator.
        
        Args:
            n_mc_samples: Number of MC samples (if using MC Dropout)
            use_mc_dropout: Whether to also estimate epistemic uncertainty
            tasks: Tasks to estimate uncertainty for
            device: Device for computation
        """
        self.n_mc_samples = n_mc_samples
        self.use_mc_dropout = use_mc_dropout
        self.tasks = tasks or ["cell_painting"]
        self.device = device
    
    def estimate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        show_progress: bool = True,
    ) -> UncertaintyEstimate:
        """
        Estimate both aleatoric and epistemic uncertainty.
        
        Args:
            model: Model that predicts mean and log-variance
            dataloader: Data loader for samples
            show_progress: Whether to show progress bar
            
        Returns:
            UncertaintyEstimate with both uncertainty types
        """
        device = self.device or next(model.parameters()).device
        model = model.to(device)
        
        n_passes = self.n_mc_samples if self.use_mc_dropout else 1
        
        all_means: List[torch.Tensor] = []
        all_log_vars: List[torch.Tensor] = []
        all_sample_ids: List[str] = []
        
        sample_ids_collected = False
        
        for mc_idx in tqdm(
            range(n_passes),
            desc="Heteroscedastic",
            disable=not show_progress,
        ):
            if self.use_mc_dropout:
                model.train()
            else:
                model.eval()
            
            batch_means: List[torch.Tensor] = []
            batch_log_vars: List[torch.Tensor] = []
            
            with torch.no_grad():
                for batch in dataloader:
                    protein_embedding = batch["protein_embedding"].to(device)
                    
                    # Forward pass expecting (mean, log_var) output
                    outputs = model(
                        protein_embedding,
                        tasks=self.tasks,
                        return_uncertainty=True,
                    )
                    
                    primary_task = self.tasks[0]
                    
                    if primary_task in outputs:
                        batch_means.append(outputs[primary_task].cpu())
                    
                    log_var_key = f"{primary_task}_log_var"
                    if log_var_key in outputs:
                        batch_log_vars.append(outputs[log_var_key].cpu())
                    
                    if not sample_ids_collected and "protein_id" in batch:
                        all_sample_ids.extend(batch["protein_id"])
            
            sample_ids_collected = True
            
            if batch_means:
                all_means.append(torch.cat(batch_means, dim=0))
            if batch_log_vars:
                all_log_vars.append(torch.cat(batch_log_vars, dim=0))
        
        model.eval()
        
        # Stack predictions: (n_passes, n_samples, n_features)
        means = torch.stack(all_means, dim=0).numpy()
        
        # Mean prediction
        mean = means.mean(axis=0)
        
        # Epistemic uncertainty (from MC Dropout variance)
        epistemic = means.std(axis=0) if self.use_mc_dropout else np.zeros_like(mean)
        
        # Aleatoric uncertainty (from predicted variance)
        if all_log_vars:
            log_vars = torch.stack(all_log_vars, dim=0).numpy()
            # Average predicted variance across MC samples
            aleatoric = np.sqrt(np.exp(log_vars.mean(axis=0)))
        else:
            aleatoric = None
        
        # Total uncertainty
        if aleatoric is not None:
            total = np.sqrt(epistemic ** 2 + aleatoric ** 2)
        else:
            total = epistemic
        
        return UncertaintyEstimate(
            mean=mean,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            sample_ids=all_sample_ids if all_sample_ids else None,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def estimate_uncertainty(
    model: Union[nn.Module, List[nn.Module]],
    dataloader: DataLoader,
    method: Literal["mc_dropout", "ensemble", "heteroscedastic"] = "mc_dropout",
    n_samples: int = 20,
    tasks: Optional[List[str]] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> UncertaintyEstimate:
    """
    Convenience function to estimate uncertainty.
    
    Args:
        model: Trained model(s)
        dataloader: Data loader for samples
        method: Uncertainty estimation method
        n_samples: Number of MC samples (for mc_dropout/heteroscedastic)
        tasks: Tasks to estimate uncertainty for
        device: Device for computation
        show_progress: Whether to show progress bar
        
    Returns:
        UncertaintyEstimate with predictions and uncertainties
        
    Example:
        >>> uncertainty = estimate_uncertainty(
        ...     model=trained_model,
        ...     dataloader=pool_loader,
        ...     method="mc_dropout",
        ...     n_samples=20,
        ... )
        >>> 
        >>> # Get uncertainty scores
        >>> scores = uncertainty.get_uncertainty(reduction="mean")
    """
    if method == "mc_dropout":
        estimator = MCDropoutEstimator(
            n_samples=n_samples,
            tasks=tasks,
            device=device,
        )
    elif method == "ensemble":
        estimator = EnsembleEstimator(
            tasks=tasks,
            device=device,
        )
    elif method == "heteroscedastic":
        estimator = HeteroscedasticEstimator(
            n_mc_samples=n_samples,
            use_mc_dropout=True,
            tasks=tasks,
            device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return estimator.estimate(
        model=model,
        dataloader=dataloader,
        show_progress=show_progress,
    )


def get_uncertainty_ranking(
    uncertainty: UncertaintyEstimate,
    uncertainty_type: UncertaintyType = UncertaintyType.TOTAL,
    reduction: str = "mean",
    ascending: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank samples by uncertainty.
    
    Args:
        uncertainty: UncertaintyEstimate object
        uncertainty_type: Which uncertainty to use for ranking
        reduction: How to reduce uncertainty across features
        ascending: If True, rank from lowest to highest uncertainty
        
    Returns:
        Tuple of (indices, scores) sorted by uncertainty
    """
    scores = uncertainty.get_uncertainty(
        uncertainty_type=uncertainty_type,
        reduction=reduction,
    )
    
    if ascending:
        indices = np.argsort(scores)
    else:
        indices = np.argsort(scores)[::-1]
    
    return indices, scores[indices]