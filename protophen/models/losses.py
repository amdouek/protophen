"""
Loss functions for ProToPhen.

This module provides loss functions for training the protein-to-phenotype
prediction model, including multi-task losses and uncertainty-aware losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Basic Loss Components
# =============================================================================

class MSELoss(nn.Module):
    """Mean squared error loss with optional masking."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Args:
            pred: Predictions of shape (batch, features)
            target: Targets of shape (batch, features)
            mask: Optional mask of shape (batch,) or (batch, features)
            
        Returns:
            Loss value
        """
        loss = F.mse_loss(pred, target, reduction="none")
        
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                return loss.sum()
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HuberLoss(nn.Module):
    """Huber loss (smooth L1) - robust to outliers."""
    
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Huber loss."""
        loss = F.smooth_l1_loss(pred, target, beta=self.delta, reduction="none")
        
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == "mean":
            return loss.mean()
        return loss


# =============================================================================
# Correlation-Based Losses
# =============================================================================

class CorrelationLoss(nn.Module):
    """
    Negative Pearson correlation loss.
    
    Encourages predictions to correlate with targets.
    """
    
    def __init__(
        self,
        dim: int = -1,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        """
        Initialise correlation loss.
        
        Args:
            dim: Dimension to compute correlation over
            reduction: Reduction method
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative correlation loss.
        
        Args:
            pred: Predictions of shape (batch, features)
            target: Targets of shape (batch, features)
            
        Returns:
            Loss value (negative correlation, so minimising increases correlation)
        """
        # Center the vectors
        pred_centered = pred - pred.mean(dim=self.dim, keepdim=True)
        target_centered = target - target.mean(dim=self.dim, keepdim=True)
        
        # Compute correlation
        numerator = (pred_centered * target_centered).sum(dim=self.dim)
        denominator = (
            pred_centered.pow(2).sum(dim=self.dim).sqrt() *
            target_centered.pow(2).sum(dim=self.dim).sqrt()
        )
        
        correlation = numerator / (denominator + self.eps)
        
        # Negative correlation as loss (we want to maximise correlation)
        loss = 1 - correlation
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CosineDistanceLoss(nn.Module):
    """
    Cosine distance loss.
    
    1 - cosine_similarity, encouraging similar direction in feature space.
    """
    
    def __init__(self, dim: int = -1, reduction: str = "mean"):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine distance loss."""
        cos_sim = F.cosine_similarity(pred, target, dim=self.dim)
        loss = 1 - cos_sim
        
        if self.reduction == "mean":
            return loss.mean()
        return loss


# =============================================================================
# Cell Painting Specific Losses
# =============================================================================

class CellPaintingLoss(nn.Module):
    """
    Combined loss for Cell Painting feature prediction.
    
    Combines MSE for accurate reconstruction with correlation
    for maintaining feature relationships.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        correlation_weight: float = 0.1,
        cosine_weight: float = 0.0,
        feature_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialise Cell Painting loss.
        
        Args:
            mse_weight: Weight for MSE component
            correlation_weight: Weight for correlation component
            cosine_weight: Weight for cosine similarity component
            feature_weights: Optional per-feature weights
        """
        super().__init__()
        
        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.cosine_weight = cosine_weight
        
        self.mse_loss = MSELoss(reduction="none")
        self.corr_loss = CorrelationLoss(dim=-1)
        self.cosine_loss = CosineDistanceLoss(dim=-1)
        
        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights)
        else:
            self.feature_weights = None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Cell Painting loss.
        
        Args:
            pred: Predictions of shape (batch, n_features)
            target: Targets of shape (batch, n_features)
            mask: Optional sample mask of shape (batch,)
            
        Returns:
            Dictionary with total loss and components
        """
        losses = {}
        
        # MSE loss
        mse = self.mse_loss(pred, target, mask=None)
        if self.feature_weights is not None:
            mse = mse * self.feature_weights
        mse = mse.mean()
        losses["mse"] = mse
        
        # Correlation loss (per-sample)
        if self.correlation_weight > 0:
            corr = self.corr_loss(pred, target)
            losses["correlation"] = corr
        
        # Cosine loss
        if self.cosine_weight > 0:
            cosine = self.cosine_loss(pred, target)
            losses["cosine"] = cosine
        
        # Total loss
        total = self.mse_weight * mse
        if self.correlation_weight > 0:
            total = total + self.correlation_weight * losses["correlation"]
        if self.cosine_weight > 0:
            total = total + self.cosine_weight * losses["cosine"]
        
        losses["total"] = total
        
        return losses


# =============================================================================
# Uncertainty-Aware Losses
# =============================================================================

class GaussianNLLLoss(nn.Module):
    """
    Gaussian negative log-likelihood loss for aleatoric uncertainty.
    
    Assumes predictions include both mean and log variance.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.
        
        Args:
            mean: Predicted mean of shape (batch, features)
            log_var: Predicted log variance of shape (batch, features)
            target: Target values of shape (batch, features)
            
        Returns:
            Loss value
        """
        # Clamp log_var for stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # NLL = 0.5 * (log(var) + (y - mu)^2 / var)
        var = torch.exp(log_var) + self.eps
        nll = 0.5 * (log_var + (target - mean).pow(2) / var)
        
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll


# =============================================================================
# Multi-Task Losses
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with configurable task weights.
    
    Combines losses from multiple prediction tasks with optional
    learned or fixed task weights.
    """
    
    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        loss_types: Optional[Dict[str, str]] = None,
    ):
        """
        Initialise multi-task loss.
        
        Args:
            task_weights: Dictionary mapping task names to weights
            loss_types: Dictionary mapping task names to loss types
                Options: "mse", "huber", "cell_painting", "bce"
        """
        super().__init__()
        
        self.task_weights = task_weights or {
            "cell_painting": 1.0,
            "viability": 0.5,
        }
        
        loss_types = loss_types or {
            "cell_painting": "cell_painting",
            "viability": "mse",
        }
        
        # Create loss functions for each task
        self.loss_fns = nn.ModuleDict()
        for task_name, loss_type in loss_types.items():
            if loss_type == "mse":
                self.loss_fns[task_name] = MSELoss()
            elif loss_type == "huber":
                self.loss_fns[task_name] = HuberLoss()
            elif loss_type == "cell_painting":
                self.loss_fns[task_name] = CellPaintingLoss()
            elif loss_type == "bce":
                self.loss_fns[task_name] = nn.BCELoss()
            else:
                self.loss_fns[task_name] = MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary of predictions per task
            targets: Dictionary of targets per task
            masks: Optional dictionary of masks per task
            
        Returns:
            Dictionary with total loss and per-task losses
        """
        masks = masks or {}
        losses = {}
        total_loss = 0.0
        
        for task_name, pred in predictions.items():
            if task_name not in targets:
                continue
            
            target = targets[task_name]
            mask = masks.get(task_name)
            
            # Skip if all samples masked
            if mask is not None and mask.sum() == 0:
                continue
            
            # Compute task loss
            if task_name in self.loss_fns:
                loss_fn = self.loss_fns[task_name]
                
                if isinstance(loss_fn, CellPaintingLoss):
                    task_losses = loss_fn(pred, target, mask)
                    task_loss = task_losses["total"]
                    losses[f"{task_name}_mse"] = task_losses["mse"]
                else:
                    if mask is not None:
                        task_loss = loss_fn(pred[mask], target[mask])
                    else:
                        task_loss = loss_fn(pred, target)
            else:
                # Default to MSE
                task_loss = F.mse_loss(pred, target)
            
            losses[task_name] = task_loss
            
            # Add weighted loss to total
            weight = self.task_weights.get(task_name, 1.0)
            total_loss = total_loss + weight * task_loss
        
        losses["total"] = total_loss
        
        return losses


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learned uncertainty-based weights.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al., 2018).
    
    The loss for each task is weighted by learned uncertainty parameters:
    L = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)
    """
    
    def __init__(
        self,
        task_names: List[str],
        initial_log_vars: Optional[Dict[str, float]] = None,
    ):
        """
        Initialise uncertainty-weighted loss.
        
        Args:
            task_names: List of task names
            initial_log_vars: Initial log variance for each task
        """
        super().__init__()
        
        self.task_names = task_names
        
        # Learnable log variances
        initial_log_vars = initial_log_vars or {}
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.tensor(initial_log_vars.get(task, 0.0)))
            for task in task_names
        })
        
        # Base loss functions
        self.loss_fns = nn.ModuleDict({
            task: MSELoss() for task in task_names
        })
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-weighted multi-task loss.
        
        Args:
            predictions: Dictionary of predictions per task
            targets: Dictionary of targets per task
            masks: Optional dictionary of masks per task
            
        Returns:
            Dictionary with total loss, per-task losses, and weights
        """
        masks = masks or {}
        losses = {}
        total_loss = 0.0
        
        for task_name in self.task_names:
            if task_name not in predictions or task_name not in targets:
                continue
            
            pred = predictions[task_name]
            target = targets[task_name]
            mask = masks.get(task_name)
            
            # Compute base loss
            if mask is not None and mask.sum() > 0:
                task_loss = self.loss_fns[task_name](pred[mask], target[mask])
            else:
                task_loss = self.loss_fns[task_name](pred, target)
            
            losses[task_name] = task_loss
            
            # Apply uncertainty weighting
            log_var = self.log_vars[task_name]
            precision = torch.exp(-log_var)
            weighted_loss = precision * task_loss + log_var
            
            total_loss = total_loss + weighted_loss
            
            # Store effective weight for logging
            losses[f"{task_name}_weight"] = precision.detach()
        
        losses["total"] = total_loss
        
        return losses
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current effective task weights."""
        return {
            task: torch.exp(-self.log_vars[task]).item()
            for task in self.task_names
        }

# =============================================================================
# Combined Loss
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Flexible combined loss supporting multiple configurations.
    
    Example:
        >>> loss_fn = CombinedLoss(
        ...     tasks=["cell_painting", "viability"],
        ...     task_weights={"cell_painting": 1.0, "viability": 0.5},
        ...     use_uncertainty_weighting=False,
        ...     cell_painting_config={
        ...         "mse_weight": 1.0,
        ...         "correlation_weight": 0.1,
        ...     },
        ... )
        >>> 
        >>> losses = loss_fn(predictions, targets)
    """
    
    def __init__(
        self,
        tasks: List[str],
        task_weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = False,
        cell_painting_config: Optional[Dict] = None,
        predict_aleatoric: bool = False,
    ):
        """
        Initialise combined loss.
        
        Args:
            tasks: List of task names
            task_weights: Fixed task weights (ignored if use_uncertainty_weighting)
            use_uncertainty_weighting: Use learned uncertainty weights
            cell_painting_config: Configuration for Cell Painting loss
            predict_aleatoric: Whether model predicts aleatoric uncertainty
        """
        super().__init__()
        
        self.tasks = tasks
        self.predict_aleatoric = predict_aleatoric
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Default task weights
        task_weights = task_weights or {task: 1.0 for task in tasks}
        self.task_weights = task_weights
        
        # Create task-specific loss functions
        self.loss_fns = nn.ModuleDict()
        
        for task in tasks:
            if task == "cell_painting":
                config = cell_painting_config or {}
                self.loss_fns[task] = CellPaintingLoss(**config)
            elif task == "viability":
                self.loss_fns[task] = MSELoss()
            elif task == "transcriptomics":
                self.loss_fns[task] = MSELoss()
            else:
                self.loss_fns[task] = MSELoss()
        
        # Aleatoric uncertainty loss
        if predict_aleatoric:
            self.nll_loss = GaussianNLLLoss()
        
        # Uncertainty weighting (learned task weights)
        if use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.tensor(0.0))
                for task in tasks
            })
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
        log_vars: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dictionary of predictions per task
            targets: Dictionary of targets per task
            masks: Optional dictionary of sample masks per task
            log_vars: Optional predicted log variances for aleatoric uncertainty
            
        Returns:
            Dictionary with total loss and per-task losses
        """
        masks = masks or {}
        log_vars = log_vars or {}
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for task in self.tasks:
            if task not in predictions or task not in targets:
                continue
            
            pred = predictions[task]
            target = targets[task]
            mask = masks.get(task)
            
            # Handle masking
            if mask is not None:
                # Ensure mask is boolean
                if mask.dtype != torch.bool:
                    mask = mask.bool()
                
                # Skip if all masked
                if mask.sum() == 0:
                    continue
                
                pred_masked = pred[mask]
                target_masked = target[mask]
            else:
                pred_masked = pred
                target_masked = target
            
            # Compute task loss
            if self.predict_aleatoric and task in log_vars:
                # Use Gaussian NLL for aleatoric uncertainty
                task_log_var = log_vars[task]
                if mask is not None:
                    task_log_var = task_log_var[mask]
                task_loss = self.nll_loss(pred_masked, task_log_var, target_masked)
                losses[f"{task}_nll"] = task_loss
            else:
                # Use standard loss
                loss_fn = self.loss_fns[task]
                
                if isinstance(loss_fn, CellPaintingLoss):
                    task_losses = loss_fn(pred_masked, target_masked)
                    task_loss = task_losses["total"]
                    # Store component losses
                    for key, value in task_losses.items():
                        if key != "total":
                            losses[f"{task}_{key}"] = value
                else:
                    task_loss = loss_fn(pred_masked, target_masked)
            
            losses[task] = task_loss
            
            # Apply task weighting
            if self.use_uncertainty_weighting:
                # Learned uncertainty weighting
                log_var = self.log_vars[task]
                precision = torch.exp(-log_var)
                weighted_loss = precision * task_loss + log_var
                losses[f"{task}_weight"] = precision.detach()
            else:
                # Fixed weighting
                weight = self.task_weights.get(task, 1.0)
                weighted_loss = weight * task_loss
            
            total_loss = total_loss + weighted_loss
        
        losses["total"] = total_loss
        
        return losses
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        if self.use_uncertainty_weighting:
            return {
                task: torch.exp(-self.log_vars[task]).item()
                for task in self.tasks
            }
        return self.task_weights.copy()



# =============================================================================
# Regularisation Losses
# =============================================================================

class LatentRegularisationLoss(nn.Module):
    """
    Regularisation loss for latent representations.
    
    Encourages well-structured latent space through various constraints.
    """
    
    def __init__(
        self,
        l2_weight: float = 0.0,
        l1_weight: float = 0.0,
        variance_weight: float = 0.0,
        covariance_weight: float = 0.0,
    ):
        """
        Initialise regularisation loss.
        
        Args:
            l2_weight: Weight for L2 regularisation (small latent values)
            l1_weight: Weight for L1 regularisation (sparse latent)
            variance_weight: Weight for variance regularisation (unit variance)
            covariance_weight: Weight for covariance regularisation (decorrelation)
        """
        super().__init__()
        
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
    
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute regularisation loss on latent representations.
        
        Args:
            latent: Latent representations of shape (batch, latent_dim)
            
        Returns:
            Dictionary with total loss and components
        """
        losses = {}
        total = torch.tensor(0.0, device=latent.device)
        
        # L2 regularisation
        if self.l2_weight > 0:
            l2_loss = latent.pow(2).mean()
            losses["l2"] = l2_loss
            total = total + self.l2_weight * l2_loss
        
        # L1 regularisation (sparsity)
        if self.l1_weight > 0:
            l1_loss = latent.abs().mean()
            losses["l1"] = l1_loss
            total = total + self.l1_weight * l1_loss
        
        # Variance regularisation (encourage unit variance per dimension)
        if self.variance_weight > 0:
            var = latent.var(dim=0)
            var_loss = (var - 1.0).pow(2).mean()
            losses["variance"] = var_loss
            total = total + self.variance_weight * var_loss
        
        # Covariance regularisation (encourage decorrelated dimensions)
        if self.covariance_weight > 0:
            latent_centered = latent - latent.mean(dim=0, keepdim=True)
            cov = (latent_centered.T @ latent_centered) / (latent.size(0) - 1)
            # Zero out diagonal (we only penalise off-diagonal covariances)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_loss = off_diag.pow(2).mean()
            losses["covariance"] = cov_loss
            total = total + self.covariance_weight * cov_loss
        
        losses["total"] = total
        
        return losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative representations.
    
    Encourages similar proteins (same phenotype) to have similar
    latent representations, and dissimilar proteins to be separated.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialise contrastive loss.
        
        Args:
            temperature: Temperature for softmax scaling
            similarity_threshold: Phenotype correlation threshold for positive pairs
        """
        super().__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
    
    def forward(
        self,
        latent: torch.Tensor,
        phenotypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            latent: Latent representations of shape (batch, latent_dim)
            phenotypes: Phenotype vectors of shape (batch, phenotype_dim)
            
        Returns:
            Contrastive loss value
        """
        batch_size = latent.size(0)
        
        # Compute phenotype similarities (correlation)
        phenotypes_norm = F.normalize(phenotypes, p=2, dim=-1)
        phenotype_sim = phenotypes_norm @ phenotypes_norm.T
        
        # Create positive/negative masks based on phenotype similarity
        positive_mask = phenotype_sim > self.similarity_threshold
        # Exclude self-comparisons
        positive_mask.fill_diagonal_(False)
        
        # Compute latent similarities
        latent_norm = F.normalize(latent, p=2, dim=-1)
        latent_sim = latent_norm @ latent_norm.T / self.temperature
        
        # InfoNCE-style loss
        # For each anchor, pull together positives and push apart negatives
        exp_sim = torch.exp(latent_sim)
        
        # Mask out self-comparisons
        mask = torch.eye(batch_size, device=latent.device).bool()
        exp_sim = exp_sim.masked_fill(mask, 0)
        
        # Compute loss
        pos_sum = (exp_sim * positive_mask.float()).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        
        # Avoid division by zero
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        
        # Only compute for samples that have positives
        has_positives = positive_mask.sum(dim=1) > 0
        if has_positives.sum() > 0:
            return loss[has_positives].mean()
        
        return torch.tensor(0.0, device=latent.device)


# =============================================================================
# Loss Factory
# =============================================================================

def create_loss_function(
    tasks: List[str],
    task_weights: Optional[Dict[str, float]] = None,
    use_uncertainty_weighting: bool = False,
    predict_aleatoric: bool = False,
    cell_painting_config: Optional[Dict] = None,
    latent_regularisation: Optional[Dict[str, float]] = None,
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        tasks: List of prediction tasks
        task_weights: Task weights for multi-task loss
        use_uncertainty_weighting: Use learned uncertainty weights
        predict_aleatoric: Whether model predicts aleatoric uncertainty
        cell_painting_config: Configuration for Cell Painting loss
        latent_regularisation: Configuration for latent regularisation
        
    Returns:
        Loss function module
        
    Example:
        >>> loss_fn = create_loss_function(
        ...     tasks=["cell_painting", "viability"],
        ...     task_weights={"cell_painting": 1.0, "viability": 0.5},
        ...     cell_painting_config={
        ...         "mse_weight": 1.0,
        ...         "correlation_weight": 0.1,
        ...     },
        ... )
    """
    loss = CombinedLoss(
        tasks=tasks,
        task_weights=task_weights,
        use_uncertainty_weighting=use_uncertainty_weighting,
        predict_aleatoric=predict_aleatoric,
        cell_painting_config=cell_painting_config,
    )
    
    return loss


# =============================================================================
# Utility Functions
# =============================================================================

def compute_metrics_from_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute additional metrics beyond loss values.
    
    Args:
        predictions: Dictionary of predictions per task
        targets: Dictionary of targets per task
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    for task in predictions:
        if task not in targets:
            continue
        
        pred = predictions[task].detach()
        target = targets[task].detach()
        
        # MSE
        mse = F.mse_loss(pred, target).item()
        metrics[f"{task}_mse"] = mse
        
        # MAE
        mae = F.l1_loss(pred, target).item()
        metrics[f"{task}_mae"] = mae
        
        # R² (coefficient of determination)
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
        metrics[f"{task}_r2"] = r2
        
        # Pearson correlation (per-sample, then averaged)
        if pred.dim() > 1 and pred.size(-1) > 1:
            pred_centered = pred - pred.mean(dim=-1, keepdim=True)
            target_centered = target - target.mean(dim=-1, keepdim=True)
            
            num = (pred_centered * target_centered).sum(dim=-1)
            denom = (
                pred_centered.pow(2).sum(dim=-1).sqrt() *
                target_centered.pow(2).sum(dim=-1).sqrt()
            )
            corr = (num / (denom + 1e-8)).mean().item()
            metrics[f"{task}_pearson"] = corr
    
    return metrics