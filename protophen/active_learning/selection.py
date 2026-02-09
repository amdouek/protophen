"""
Experiment selection for active learning.

This module provides high-level utilities for selecting the next
experiments to run in an active learning loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from protophen.active_learning.acquisition import (
    AcquisitionFunction,
    BatchAcquisition,
    DiversitySampling,
    ExpectedImprovement,
    HybridAcquisition,
    UncertaintySampling,
)
from protophen.active_learning.uncertainty import (
    MCDropoutEstimator,
    UncertaintyEstimate,
    UncertaintyType,
    estimate_uncertainty,
)
from protophen.utils.logging import logger


# =============================================================================
# Selection Configuration
# =============================================================================

@dataclass
class SelectionConfig:
    """Configuration for experiment selection."""
    
    # Batch size
    n_select: int = 10
    
    # Uncertainty estimation
    uncertainty_method: Literal["mc_dropout", "ensemble"] = "mc_dropout"
    n_mc_samples: int = 20
    
    # Acquisition function
    acquisition_method: Literal["uncertainty", "ei", "diversity", "hybrid"] = "hybrid"
    
    # Hybrid acquisition settings
    uncertainty_weight: float = 0.7
    diversity_weight: float = 0.3
    
    # Uncertainty type
    uncertainty_type: UncertaintyType = UncertaintyType.TOTAL
    
    # Constraints
    exclude_ids: List[str] = field(default_factory=list)
    
    # Tasks
    tasks: List[str] = field(default_factory=lambda: ["cell_painting"])


@dataclass
class SelectionResult:
    """Result of experiment selection."""
    
    # Selected sample information
    selected_indices: np.ndarray
    selected_ids: List[str]
    acquisition_scores: np.ndarray
    
    # Uncertainty information
    uncertainty_estimates: UncertaintyEstimate
    
    # Full rankings
    all_indices_ranked: np.ndarray
    all_scores: np.ndarray
    
    # Metadata
    config: SelectionConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "selected_indices": self.selected_indices.tolist(),
            "selected_ids": self.selected_ids,
            "acquisition_scores": self.acquisition_scores.tolist(),
            "all_scores": self.all_scores.tolist(),
            "config": {
                "n_select": self.config.n_select,
                "uncertainty_method": self.config.uncertainty_method,
                "acquisition_method": self.config.acquisition_method,
            },
        }
    
    def get_selected_proteins(self) -> List[Dict[str, Any]]:
        """Get information about selected proteins."""
        proteins = []
        for idx, (sample_idx, sample_id, score) in enumerate(zip(
            self.selected_indices,
            self.selected_ids,
            self.acquisition_scores,
        )):
            proteins.append({
                "rank": idx + 1,
                "index": int(sample_idx),
                "id": sample_id,
                "acquisition_score": float(score),
                "uncertainty": float(
                    self.uncertainty_estimates.get_uncertainty(
                        self.config.uncertainty_type,
                        reduction="mean",
                    )[sample_idx]
                ),
            })
        return proteins
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Selection Result: {len(self.selected_ids)} samples selected",
            f"  Method: {self.config.acquisition_method}",
            f"  Top scores: {self.acquisition_scores[:5]}",
            f"  Top IDs: {self.selected_ids[:5]}",
        ]
        return "\n".join(lines)


# =============================================================================
# Experiment Selector
# =============================================================================

class ExperimentSelector:
    """
    High-level class for selecting experiments in active learning.
    
    Coordinates uncertainty estimation, acquisition scoring, and
    sample selection.
    
    Example:
        >>> selector = ExperimentSelector(
        ...     model=trained_model,
        ...     config=SelectionConfig(n_select=10),
        ... )
        >>> 
        >>> # Select from pool of candidate proteins
        >>> result = selector.select(pool_dataloader)
        >>> 
        >>> # Get selected protein IDs
        >>> print(result.selected_ids)
        >>> 
        >>> # Get detailed information
        >>> for protein in result.get_selected_proteins():
        ...     print(f"{protein['id']}: score={protein['acquisition_score']:.4f}")
    """
    
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        config: Optional[SelectionConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialise experiment selector.
        
        Args:
            model: Trained model(s) for uncertainty estimation
            config: Selection configuration
            device: Device for computation
        """
        self.model = model
        self.config = config or SelectionConfig()
        self.device = device
        
        # Create uncertainty estimator
        self.uncertainty_estimator = self._create_uncertainty_estimator()
        
        # Create acquisition function
        self.acquisition_fn = self._create_acquisition_fn()
        
        # Track selection history
        self.selection_history: List[SelectionResult] = []
    
    def _create_uncertainty_estimator(self):
        """Create uncertainty estimator based on config."""
        if self.config.uncertainty_method == "mc_dropout":
            return MCDropoutEstimator(
                n_samples=self.config.n_mc_samples,
                tasks=self.config.tasks,
                device=self.device,
            )
        elif self.config.uncertainty_method == "ensemble":
            from protophen.active_learning.uncertainty import EnsembleEstimator
            return EnsembleEstimator(
                tasks=self.config.tasks,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown uncertainty method: {self.config.uncertainty_method}")
    
    def _create_acquisition_fn(self) -> AcquisitionFunction:
        """Create acquisition function based on config."""
        if self.config.acquisition_method == "uncertainty":
            return UncertaintySampling(
                uncertainty_type=self.config.uncertainty_type,
            )
        elif self.config.acquisition_method == "ei":
            return ExpectedImprovement()
        elif self.config.acquisition_method == "diversity":
            return DiversitySampling(method="kmeans++")
        elif self.config.acquisition_method == "hybrid":
            return HybridAcquisition(
                uncertainty_weight=self.config.uncertainty_weight,
                diversity_weight=self.config.diversity_weight,
                uncertainty_type=self.config.uncertainty_type,
            )
        else:
            raise ValueError(f"Unknown acquisition method: {self.config.acquisition_method}")
    
    def select(
        self,
        dataloader: DataLoader,
        embeddings: Optional[np.ndarray] = None,
        n_select: Optional[int] = None,
        show_progress: bool = True,
    ) -> SelectionResult:
        """
        Select next experiments from candidate pool.
        
        Args:
            dataloader: DataLoader for candidate samples
            embeddings: Optional embeddings for diversity (uses predictions if None)
            n_select: Number to select (overrides config if provided)
            show_progress: Whether to show progress bar
            
        Returns:
            SelectionResult with selected samples
        """
        n_select = n_select or self.config.n_select
        
        logger.info(f"Starting experiment selection (n_select={n_select})")
        
        # Step 1: Estimate uncertainty
        logger.info("Estimating uncertainty...")
        uncertainty = self.uncertainty_estimator.estimate(
            model=self.model,
            dataloader=dataloader,
            show_progress=show_progress,
        )
        
        logger.info(
            f"Uncertainty estimated for {uncertainty.n_samples} samples. "
            f"Mean uncertainty: {uncertainty.get_uncertainty(reduction='mean').mean():.4f}"
        )
        
        # Step 2: Filter out excluded samples
        if self.config.exclude_ids:
            valid_mask = self._get_valid_mask(uncertainty)
            logger.info(f"Excluding {(~valid_mask).sum()} previously selected samples")
        else:
            valid_mask = np.ones(uncertainty.n_samples, dtype=bool)
        
        # Step 3: Compute acquisition scores and select
        logger.info(f"Computing acquisition scores using {self.config.acquisition_method}...")
        
        selected_indices = self.acquisition_fn.select(
            uncertainty=uncertainty,
            n_select=n_select,
            embeddings=embeddings,
        )
        
        # Filter to valid samples only
        valid_indices = np.where(valid_mask)[0]
        selected_indices = np.array([
            idx for idx in selected_indices if valid_mask[idx]
        ])[:n_select]
        
        # Get scores for selected
        all_scores = self.acquisition_fn.score(uncertainty)
        selected_scores = all_scores[selected_indices]
        
        # Get selected IDs
        if uncertainty.sample_ids:
            selected_ids = [uncertainty.sample_ids[i] for i in selected_indices]
        else:
            selected_ids = [f"sample_{i}" for i in selected_indices]
        
        # Create result
        result = SelectionResult(
            selected_indices=selected_indices,
            selected_ids=selected_ids,
            acquisition_scores=selected_scores,
            uncertainty_estimates=uncertainty,
            all_indices_ranked=np.argsort(all_scores)[::-1],
            all_scores=all_scores,
            config=self.config,
        )
        
        # Update history
        self.selection_history.append(result)
        
        # Update exclude list
        self.config.exclude_ids.extend(selected_ids)
        
        logger.info(f"Selected {len(selected_ids)} experiments")
        logger.info(result.summary())
        
        return result
    
    def _get_valid_mask(self, uncertainty: UncertaintyEstimate) -> np.ndarray:
        """Get mask of samples that aren't excluded."""
        valid_mask = np.ones(uncertainty.n_samples, dtype=bool)
        
        if uncertainty.sample_ids and self.config.exclude_ids:
            exclude_set = set(self.config.exclude_ids)
            for i, sample_id in enumerate(uncertainty.sample_ids):
                if sample_id in exclude_set:
                    valid_mask[i] = False
        
        return valid_mask
    
    def select_iterative(
        self,
        dataloader: DataLoader,
        n_iterations: int,
        n_per_iteration: int,
        embeddings: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> List[SelectionResult]:
        """
        Perform multiple selection iterations.
        
        Args:
            dataloader: DataLoader for candidate samples
            n_iterations: Number of selection rounds
            n_per_iteration: Samples per round
            embeddings: Optional embeddings for diversity
            show_progress: Whether to show progress
            
        Returns:
            List of SelectionResult for each iteration
        """
        results = []
        
        for i in range(n_iterations):
            logger.info(f"Selection iteration {i + 1}/{n_iterations}")
            
            result = self.select(
                dataloader=dataloader,
                embeddings=embeddings,
                n_select=n_per_iteration,
                show_progress=show_progress,
            )
            results.append(result)
        
        return results
    
    def reset_exclusions(self) -> None:
        """Reset the exclusion list."""
        self.config.exclude_ids = []
        logger.info("Reset exclusion list")
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of all selections."""
        if not self.selection_history:
            return {"n_selections": 0}
        
        all_selected_ids = []
        all_scores = []
        
        for result in self.selection_history:
            all_selected_ids.extend(result.selected_ids)
            all_scores.extend(result.acquisition_scores.tolist())
        
        return {
            "n_selections": len(self.selection_history),
            "total_selected": len(all_selected_ids),
            "mean_acquisition_score": np.mean(all_scores),
            "all_selected_ids": all_selected_ids,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def select_next_experiments(
    model: nn.Module,
    dataloader: DataLoader,
    n_select: int = 10,
    method: str = "hybrid",
    uncertainty_method: str = "mc_dropout",
    n_mc_samples: int = 20,
    exclude_ids: Optional[List[str]] = None,
    show_progress: bool = True,
) -> SelectionResult:
    """
    Convenience function to select next experiments.
    
    Args:
        model: Trained model
        dataloader: Candidate pool dataloader
        n_select: Number of experiments to select
        method: Acquisition method
        uncertainty_method: Uncertainty estimation method
        n_mc_samples: Number of MC samples
        exclude_ids: IDs to exclude from selection
        show_progress: Whether to show progress
        
    Returns:
        SelectionResult with selected experiments
        
    Example:
        >>> result = select_next_experiments(
        ...     model=trained_model,
        ...     dataloader=pool_loader,
        ...     n_select=10,
        ...     method="hybrid",
        ... )
        >>> 
        >>> print("Selected proteins:", result.selected_ids)
    """
    config = SelectionConfig(
        n_select=n_select,
        uncertainty_method=uncertainty_method,
        n_mc_samples=n_mc_samples,
        acquisition_method=method,
        exclude_ids=exclude_ids or [],
    )
    
    selector = ExperimentSelector(model=model, config=config)
    
    return selector.select(dataloader, show_progress=show_progress)


def rank_by_uncertainty(
    model: nn.Module,
    dataloader: DataLoader,
    n_mc_samples: int = 20,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Rank all samples by uncertainty.
    
    Args:
        model: Trained model
        dataloader: DataLoader for samples
        n_mc_samples: Number of MC samples
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (ranked_indices, uncertainty_scores, sample_ids)
        
    Example:
        >>> indices, scores, ids = rank_by_uncertainty(model, pool_loader)
        >>> 
        >>> # Get top 10 most uncertain
        >>> top_10_ids = [ids[i] for i in indices[:10]]
        >>> top_10_scores = scores[:10]
    """
    # Estimate uncertainty
    uncertainty = estimate_uncertainty(
        model=model,
        dataloader=dataloader,
        method="mc_dropout",
        n_samples=n_mc_samples,
        show_progress=show_progress,
    )
    
    # Get uncertainty scores
    scores = uncertainty.get_uncertainty(
        uncertainty_type=UncertaintyType.TOTAL,
        reduction="mean",
    )
    
    # Rank by uncertainty (highest first)
    ranked_indices = np.argsort(scores)[::-1]
    ranked_scores = scores[ranked_indices]
    
    # Get sample IDs
    if uncertainty.sample_ids:
        ranked_ids = [uncertainty.sample_ids[i] for i in ranked_indices]
    else:
        ranked_ids = [f"sample_{i}" for i in ranked_indices]
    
    return ranked_indices, ranked_scores, ranked_ids


def compute_diversity_matrix(
    embeddings: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute pairwise diversity (distance) matrix.
    
    Args:
        embeddings: Sample embeddings of shape (n_samples, n_features)
        metric: Distance metric
        
    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    from scipy.spatial.distance import cdist
    return cdist(embeddings, embeddings, metric=metric)


def select_diverse_subset(
    embeddings: np.ndarray,
    n_select: int,
    method: str = "kmeans++",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Select a diverse subset of samples based on embeddings.
    
    Args:
        embeddings: Sample embeddings
        n_select: Number of samples to select
        method: Selection method ("kmeans++", "maxmin")
        seed: Random seed
        
    Returns:
        Indices of selected samples
        
    Example:
        >>> # Select diverse subset for initial training
        >>> initial_indices = select_diverse_subset(
        ...     embeddings=protein_embeddings,
        ...     n_select=100,
        ... )
    """
    if seed is not None:
        np.random.seed(seed)
    
    diversity_acq = DiversitySampling(method=method)
    
    # Create dummy uncertainty estimate
    dummy_uncertainty = UncertaintyEstimate(
        mean=embeddings,
        total=np.ones(len(embeddings)),
    )
    
    return diversity_acq.select(
        uncertainty=dummy_uncertainty,
        n_select=n_select,
        embeddings=embeddings,
    )


# =============================================================================
# Active Learning Loop
# =============================================================================

class ActiveLearningLoop:
    """
    Complete active learning loop manager.
    
    Manages the iterative process of:
    1. Training model on current labeled data
    2. Selecting new samples to label
    3. Updating training set with new labels
    
    Example:
        >>> al_loop = ActiveLearningLoop(
        ...     model_factory=lambda: ProToPhenModel(config),
        ...     initial_train_data=initial_dataset,
        ...     pool_data=pool_dataset,
        ...     n_iterations=10,
        ...     n_samples_per_iteration=10,
        ... )
        >>> 
        >>> # Run with simulated labels (for benchmarking)
        >>> history = al_loop.run(label_fn=oracle.get_labels)
    """
    
    def __init__(
        self,
        model_factory: callable,
        initial_train_data,
        pool_data,
        val_data=None,
        n_iterations: int = 10,
        n_samples_per_iteration: int = 10,
        selection_config: Optional[SelectionConfig] = None,
        trainer_config: Optional[Any] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialise active learning loop.
        
        Args:
            model_factory: Callable that returns a new model instance
            initial_train_data: Initial labeled training dataset
            pool_data: Unlabeled pool dataset
            val_data: Validation dataset
            n_iterations: Number of AL iterations
            n_samples_per_iteration: Samples to select per iteration
            selection_config: Configuration for sample selection
            trainer_config: Configuration for model training
            output_dir: Directory for saving results
        """
        self.model_factory = model_factory
        self.train_data = initial_train_data
        self.pool_data = pool_data
        self.val_data = val_data
        self.n_iterations = n_iterations
        self.n_samples_per_iteration = n_samples_per_iteration
        self.selection_config = selection_config or SelectionConfig(
            n_select=n_samples_per_iteration
        )
        self.trainer_config = trainer_config
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Track history
        self.history: List[Dict[str, Any]] = []
        self.current_model: Optional[nn.Module] = None
        self.iteration = 0
        
        # Track selected samples
        self.selected_indices_history: List[np.ndarray] = []
        self.selected_ids_history: List[List[str]] = []
    
    def train_model(self) -> nn.Module:
        """Train model on current training data."""
        from protophen.data.loaders import create_dataloader
        from protophen.training.trainer import Trainer, TrainerConfig
        
        # Create new model
        model = self.model_factory()
        
        # Create data loaders
        train_loader = create_dataloader(
            self.train_data,
            batch_size=32,
            shuffle=True,
        )
        
        val_loader = None
        if self.val_data is not None:
            val_loader = create_dataloader(
                self.val_data,
                batch_size=32,
                shuffle=False,
            )
        
        # Create trainer
        config = self.trainer_config or TrainerConfig(epochs=50)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        
        # Train
        trainer.train()
        
        self.current_model = model
        return model
    
    def select_samples(self) -> SelectionResult:
        """Select next samples from pool."""
        from protophen.data.loaders import create_dataloader
        
        if self.current_model is None:
            raise RuntimeError("Must train model before selecting samples")
        
        # Create pool loader
        pool_loader = create_dataloader(
            self.pool_data,
            batch_size=32,
            shuffle=False,
        )
        
        # Create selector
        selector = ExperimentSelector(
            model=self.current_model,
            config=self.selection_config,
        )
        
        # Select
        result = selector.select(pool_loader)
        
        return result
    
    def update_datasets(
        self,
        selected_indices: np.ndarray,
        labels: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Update training and pool datasets.
        
        Args:
            selected_indices: Indices of selected samples in pool
            labels: Labels for selected samples (if available)
        """
        # This is a simplified implementation
        # In practice, you'd need to properly handle the dataset updates
        # based on your specific dataset implementation
        
        logger.info(f"Updating datasets with {len(selected_indices)} new samples")
        
        # Track selections
        self.selected_indices_history.append(selected_indices)
    
    def run_iteration(
        self,
        label_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run single AL iteration.
        
        Args:
            label_fn: Function to get labels for selected samples
                      Signature: label_fn(sample_ids) -> Dict[str, np.ndarray]
                      
        Returns:
            Iteration results
        """
        self.iteration += 1
        logger.info(f"=== Active Learning Iteration {self.iteration} ===")
        
        iteration_result = {
            "iteration": self.iteration,
            "train_size": len(self.train_data),
        }
        
        # Step 1: Train model
        logger.info("Training model...")
        model = self.train_model()
        
        # Step 2: Evaluate if validation data available
        if self.val_data is not None:
            from protophen.data.loaders import create_dataloader
            from protophen.training.metrics import compute_regression_metrics
            
            val_loader = create_dataloader(self.val_data, batch_size=32, shuffle=False)
            
            # Get predictions
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    pred = model(batch["protein_embedding"].to(next(model.parameters()).device))
                    all_preds.append(pred["cell_painting"].cpu())
                    if "cell_painting" in batch:
                        all_targets.append(batch["cell_painting"])
            
            if all_targets:
                preds = torch.cat(all_preds, dim=0)
                targets = torch.cat(all_targets, dim=0)
                metrics = compute_regression_metrics(preds, targets)
                iteration_result["val_metrics"] = metrics
                logger.info(f"Validation metrics: {metrics}")
        
        # Step 3: Select new samples
        logger.info("Selecting samples...")
        selection_result = self.select_samples()
        
        iteration_result["selected_ids"] = selection_result.selected_ids
        iteration_result["acquisition_scores"] = selection_result.acquisition_scores.tolist()
        
        # Step 4: Get labels (if label function provided)
        if label_fn is not None:
            labels = label_fn(selection_result.selected_ids)
            self.update_datasets(selection_result.selected_indices, labels)
        else:
            self.update_datasets(selection_result.selected_indices)
        
        self.selected_ids_history.append(selection_result.selected_ids)
        
        # Save results
        self.history.append(iteration_result)
        
        if self.output_dir is not None:
            self._save_iteration_results(iteration_result)
        
        return iteration_result
    
    def run(
        self,
        label_fn: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run full active learning loop.
        
        Args:
            label_fn: Function to get labels for selected samples
            
        Returns:
            History of all iterations
        """
        logger.info(f"Starting active learning loop: {self.n_iterations} iterations")
        
        for _ in range(self.n_iterations):
            self.run_iteration(label_fn=label_fn)
        
        logger.info("Active learning loop completed")
        
        return self.history
    
    def _save_iteration_results(self, result: Dict[str, Any]) -> None:
        """Save iteration results to disk."""
        import json
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = self.output_dir / f"iteration_{result['iteration']:03d}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            else:
                serializable_result[key] = value
        
        with open(filepath, "w") as f:
            json.dump(serializable_result, f, indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of AL loop."""
        if not self.history:
            return {"status": "not started"}
        
        all_selected = []
        for ids in self.selected_ids_history:
            all_selected.extend(ids)
        
        summary = {
            "iterations_completed": len(self.history),
            "total_samples_selected": len(all_selected),
            "final_train_size": self.history[-1].get("train_size", 0),
        }
        
        # Track validation performance over iterations
        if "val_metrics" in self.history[0]:
            val_metrics_history = [
                h.get("val_metrics", {}) for h in self.history
            ]
            summary["val_metrics_history"] = val_metrics_history
        
        return summary