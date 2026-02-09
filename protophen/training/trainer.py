"""
Training loop for ProToPhen.

This module provides the Trainer class for training protein-to-phenotype
prediction models with support for mixed precision, gradient accumulation,
and various learning rate schedules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from protophen.models.losses import CombinedLoss, create_loss_function
from protophen.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    ProgressCallback,
)
from protophen.training.metrics import (
    MetricCollection,
    MultiTaskMetricCollection,
    create_default_metrics,
    create_multitask_metrics,
)
from protophen.utils.logging import logger


def _to_python_types(obj):
    """
    Recursively convert numpy types to Python native types.
    
    This ensures checkpoints can be loaded with weights_only=True.
    """
    if obj is None:
        return None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_python_types(v) for v in obj)
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return obj


# =============================================================================
# Training State
# =============================================================================

@dataclass
class TrainingState:
    """Mutable state during training."""
    
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    should_stop: bool = False
    
    # Epoch-level tracking
    train_loss: float = 0.0
    val_loss: float = 0.0
    
    # History
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# Trainer Configuration
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""
    
    # Training loop
    epochs: int = 100
    
    # Optimisation
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimiser: str = "adamw"
    
    # Learning rate schedule
    scheduler: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: float = 0.0
    min_lr: float = 1e-6
    
    # Gradient handling
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Evaluation
    eval_every_n_epochs: int = 1
    
    # Tasks
    tasks: List[str] = field(default_factory=lambda: ["cell_painting"])
    task_weights: Dict[str, float] = field(default_factory=lambda: {"cell_painting": 1.0})
    
    # Device
    device: str = "cuda"
    
    # Reproducibility
    seed: int = 42


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Trainer for ProToPhen models.
    
    Handles the complete training loop including:
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Validation and metric computation
    - Callback system for extensibility
    
    Example:
        >>> from protophen.models import ProToPhenModel
        >>> from protophen.data import create_dataloaders
        >>> 
        >>> model = ProToPhenModel(config)
        >>> loaders = create_dataloaders(train_data, val_data)
        >>> 
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=loaders["train"],
        ...     val_loader=loaders["val"],
        ...     config=TrainerConfig(epochs=100, learning_rate=1e-4),
        ... )
        >>> 
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        optimiser: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[Union[MetricCollection, MultiTaskMetricCollection]] = None,
    ):
        """
        Initialise trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            loss_fn: Loss function (created from config if None)
            optimiser: Optimiser (created from config if None)
            scheduler: LR scheduler (created from config if None)
            callbacks: List of callbacks
            metrics: Metrics collection for evaluation
        """
        self.config = config or TrainerConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device
        self.device = torch.device(self.config.device)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.loss_fn = loss_fn or self._create_loss_fn()
        
        # Setup optimiser
        self.optimiser = optimiser or self._create_optimiser()
        
        # Setup scheduler
        self.scheduler = scheduler or self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler('cuda') if self.config.use_amp and self.device.type == "cuda" else None
        
        # Setup metrics - use multi-task metrics for proper handling of different output shapes
        self.metrics = self._setup_metrics(metrics)
        
        # Setup callbacks
        self.callbacks = CallbackList(callbacks or [])
        self.callbacks.set_trainer(self)
        
        # Training state
        self.state = TrainingState()
        
        # Set random seed
        self._set_seed(self.config.seed)
        
        logger.info(
            f"Trainer initialised: "
            f"device={self.device}, "
            f"epochs={self.config.epochs}, "
            f"lr={self.config.learning_rate}, "
            f"amp={self.config.use_amp and self.device.type == 'cuda'}"
        )
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_metrics(
        self,
        metrics: Optional[Union[MetricCollection, MultiTaskMetricCollection]] = None,
    ) -> MultiTaskMetricCollection:
        """
        Setup metrics collection for all tasks.
        
        Args:
            metrics: Optional provided metrics (single or multi-task)
            
        Returns:
            MultiTaskMetricCollection instance
        """
        if metrics is None:
            # Create default multi-task metrics
            return create_multitask_metrics(
                tasks=self.config.tasks,
                include_correlation=True,
            )
        elif isinstance(metrics, MultiTaskMetricCollection):
            # Already multi-task, use as-is
            return metrics
        elif isinstance(metrics, MetricCollection):
            # Single MetricCollection provided - wrap it for the primary task
            # and create defaults for other tasks
            multi_metrics = create_multitask_metrics(
                tasks=self.config.tasks,
                include_correlation=True,
            )
            # Replace primary task metrics with provided collection
            if self.config.tasks:
                primary_task = self.config.tasks[0]
                multi_metrics.collections[primary_task] = metrics
            return multi_metrics
        else:
            # Unknown type, create defaults
            logger.warning(f"Unknown metrics type: {type(metrics)}, creating defaults")
            return create_multitask_metrics(
                tasks=self.config.tasks,
                include_correlation=True,
            )
    
    def _create_loss_fn(self) -> nn.Module:
        """Create loss function from config."""
        return create_loss_function(
            tasks=self.config.tasks,
            task_weights=self.config.task_weights,
        )
    
    def _create_optimiser(self) -> Optimizer:
        """Create optimiser from config."""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Don't apply weight decay to biases and layer norm
            if "bias" in name or "norm" in name or "LayerNorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        optimiser_name = self.config.optimiser.lower()
        
        if optimiser_name == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif optimiser_name == "adam":
            return torch.optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif optimiser_name == "sgd":
            return torch.optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimiser: {optimiser_name}")
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler from config."""
        scheduler_name = self.config.scheduler.lower()
        
        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.epochs
        
        # Calculate warmup steps
        if self.config.warmup_ratio > 0:
            warmup_steps = int(total_steps * self.config.warmup_ratio)
        else:
            warmup_steps = self.config.warmup_steps
        
        if scheduler_name == "cosine":
            return self._create_cosine_scheduler(total_steps, warmup_steps)
        elif scheduler_name == "linear":
            return self._create_linear_scheduler(total_steps, warmup_steps)
        elif scheduler_name == "constant":
            return self._create_constant_scheduler(warmup_steps)
        elif scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimiser,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr,
            )
        elif scheduler_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _create_cosine_scheduler(
        self,
        total_steps: int,
        warmup_steps: int,
    ) -> _LRScheduler:
        """Create cosine annealing scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            
            min_lr_ratio = self.config.min_lr / self.config.learning_rate
            return max(min_lr_ratio, cosine_decay)
        
        return LambdaLR(self.optimiser, lr_lambda)
    
    def _create_linear_scheduler(
        self,
        total_steps: int,
        warmup_steps: int,
    ) -> _LRScheduler:
        """Create linear decay scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            min_lr_ratio = self.config.min_lr / self.config.learning_rate
            return max(min_lr_ratio, 1.0 - progress)
        
        return LambdaLR(self.optimiser, lr_lambda)
    
    def _create_constant_scheduler(self, warmup_steps: int) -> _LRScheduler:
        """Create constant scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        return LambdaLR(self.optimiser, lr_lambda)
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _extract_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract target tensors from batch."""
        targets = {}
        
        for task in self.config.tasks:
            if task in batch:
                targets[task] = batch[task]
        
        return targets
    
    def _extract_masks(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract task masks from batch."""
        masks = {}
        
        for task in self.config.tasks:
            mask_key = f"mask_{task}"
            if mask_key in batch:
                masks[task] = batch[mask_key]
        
        return masks
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        # Move batch to device
        batch = self._move_batch_to_device(batch)
        
        # Extract inputs and targets
        protein_embedding = batch["protein_embedding"]
        targets = self._extract_targets(batch)
        masks = self._extract_masks(batch)
        
        # Forward pass with optional mixed precision
        if self.scaler is not None:
            with autocast(device_type=self.device.type):
                predictions = self.model(protein_embedding, tasks=self.config.tasks)
                losses = self.loss_fn(predictions, targets, masks)
                loss = losses["total"] / self.config.gradient_accumulation_steps
        else:
            predictions = self.model(protein_embedding, tasks=self.config.tasks)
            losses = self.loss_fn(predictions, targets, masks)
            loss = losses["total"] / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Return loss values
        return {
            "loss": losses["total"].item(),
            **{k: v.item() for k, v in losses.items() if k != "total"},
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimiser.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Callback: batch begin
            self.callbacks.on_batch_begin(self.state, batch)
            
            # Training step
            outputs = self.train_step(batch)
            total_loss += outputs["loss"]
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimiser)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                
                # Optimiser step
                if self.scaler is not None:
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    self.optimiser.step()
                
                self.optimiser.zero_grad()
                
                # Scheduler step (per step, not per epoch)
                if self.scheduler is not None and not isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step()
                
                self.state.global_step += 1
            
            # Callback: batch end
            self.callbacks.on_batch_end(self.state, batch, outputs)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Run validation.
        
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        if self.val_loader is None:
            return 0.0, {}
        
        self.model.eval()
        
        # Reset all task metrics
        self.metrics.reset()
        
        self.callbacks.on_validation_begin(self.state)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Extract inputs and targets
            protein_embedding = batch["protein_embedding"]
            targets = self._extract_targets(batch)
            masks = self._extract_masks(batch)
            
            # Forward pass
            if self.scaler is not None:
                with autocast(device_type=self.device.type):
                    predictions = self.model(protein_embedding, tasks=self.config.tasks)
                    losses = self.loss_fn(predictions, targets, masks)
            else:
                predictions = self.model(protein_embedding, tasks=self.config.tasks)
                losses = self.loss_fn(predictions, targets, masks)
            
            total_loss += losses["total"].item()
            num_batches += 1
            
            # Update metrics for each task separately
            for task in self.config.tasks:
                if task in predictions and task in targets:
                    # Apply mask if present
                    mask = masks.get(task)
                    if mask is not None:
                        pred = predictions[task][mask]
                        tgt = targets[task][mask]
                    else:
                        pred = predictions[task]
                        tgt = targets[task]
                    
                    if pred.numel() > 0:
                        self.metrics.update(task, pred, tgt)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute all task metrics
        all_metrics = self.metrics.compute()
        all_metrics["val_loss"] = avg_loss
        
        # Scheduler step for ReduceLROnPlateau
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        
        self.callbacks.on_validation_end(self.state, all_metrics)
        
        return avg_loss, all_metrics
    
    def train(
        self,
        epochs: Optional[int] = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Args:
            epochs: Number of epochs (overrides config if provided)
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Dictionary with training history
        """
        epochs = epochs or self.config.epochs
        
        # Resume from checkpoint if specified
        if resume_from is not None:
            self._load_checkpoint(resume_from)
        
        # Callback: training begin
        self.callbacks.on_train_begin(self.state)
        
        logger.info(f"Starting training for {epochs} epochs")
        
        try:
            for epoch in range(self.state.epoch + 1, epochs + 1):
                self.state.epoch = epoch
                
                # Check for early stopping
                if self.state.should_stop:
                    logger.info("Early stopping triggered")
                    break
                
                # Callback: epoch begin
                self.callbacks.on_epoch_begin(self.state)
                
                # Train for one epoch
                train_loss = self.train_epoch()
                self.state.train_loss = train_loss
                self.state.train_losses.append(train_loss)
                
                # Validation
                if self.val_loader is not None and epoch % self.config.eval_every_n_epochs == 0:
                    val_loss, val_metrics = self.validate()
                    self.state.val_loss = val_loss
                    self.state.val_losses.append(val_loss)
                    self.state.val_metrics.append(val_metrics)
                    
                    # Track best validation loss
                    if val_loss < self.state.best_val_loss:
                        self.state.best_val_loss = val_loss
                
                # Callback: epoch end
                self.callbacks.on_epoch_end(self.state)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        # Callback: training end
        self.callbacks.on_train_end(self.state)
        
        return {
            "train_losses": self.state.train_losses,
            "val_losses": self.state.val_losses,
            "val_metrics": self.state.val_metrics,
            "best_val_loss": self.state.best_val_loss,
            "final_epoch": self.state.epoch,
        }
    
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load training state from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimiser state
        if "optimiser_state_dict" in checkpoint:
            self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Restore training state
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.global_step = checkpoint.get("global_step", 0)
        self.state.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        logger.info(f"Resumed from checkpoint: {checkpoint_path} (epoch {self.state.epoch})")
    
    def save_checkpoint(self, path: Path) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "config": asdict(self.config),
            "best_val_loss": self.state.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        checkpoint = _to_python_types(checkpoint)
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        return_targets: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a dataloader.
        
        Args:
            dataloader: Data loader for prediction
            return_targets: Whether to also return targets
            
        Returns:
            Dictionary with predictions (and optionally targets)
        """
        self.model.eval()
        
        all_predictions: Dict[str, List[torch.Tensor]] = {
            task: [] for task in self.config.tasks
        }
        all_targets: Dict[str, List[torch.Tensor]] = {
            task: [] for task in self.config.tasks
        }
        all_protein_ids: List[str] = []
        
        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            protein_embedding = batch["protein_embedding"]
            
            # Forward pass
            predictions = self.model(protein_embedding, tasks=self.config.tasks)
            
            # Collect predictions
            for task in self.config.tasks:
                if task in predictions:
                    all_predictions[task].append(predictions[task].cpu())
                
                if return_targets and task in batch:
                    all_targets[task].append(batch[task].cpu())
            
            # Collect protein IDs
            if "protein_id" in batch:
                all_protein_ids.extend(batch["protein_id"])
        
        # Concatenate predictions
        results = {
            "protein_ids": all_protein_ids,
        }
        
        for task in self.config.tasks:
            if all_predictions[task]:
                results[f"{task}_predictions"] = torch.cat(
                    all_predictions[task], dim=0
                ).numpy()
            
            if return_targets and all_targets[task]:
                results[f"{task}_targets"] = torch.cat(
                    all_targets[task], dim=0
                ).numpy()
        
        return results
    
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader (uses val_loader if None)
            
        Returns:
            Dictionary of metrics
        """
        dataloader = dataloader or self.val_loader
        
        if dataloader is None:
            raise ValueError("No dataloader provided for evaluation")
        
        # Get predictions
        results = self.predict(dataloader, return_targets=True)
        
        # Create fresh metrics for evaluation
        eval_metrics = create_multitask_metrics(
            tasks=self.config.tasks,
            include_correlation=True,
        )
        
        # Compute metrics for each task
        for task in self.config.tasks:
            pred_key = f"{task}_predictions"
            target_key = f"{task}_targets"
            
            if pred_key in results and target_key in results:
                predictions = torch.from_numpy(results[pred_key])
                targets = torch.from_numpy(results[target_key])
                eval_metrics.update(task, predictions, targets)
        
        return eval_metrics.compute()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    learning_rate: float = 1e-4,
    epochs: int = 100,
    device: str = "cuda",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    early_stopping_patience: Optional[int] = None,
    use_tensorboard: bool = False,
    tensorboard_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Trainer:
    """
    Create a trainer with common configurations.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate
        epochs: Number of epochs
        device: Device to use
        checkpoint_dir: Directory for checkpoints
        early_stopping_patience: Patience for early stopping
        use_tensorboard: Whether to use TensorBoard logging
        tensorboard_dir: Directory for TensorBoard logs
        **kwargs: Additional TrainerConfig parameters
        
    Returns:
        Configured Trainer instance
    """
    # Create config
    config = TrainerConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        **kwargs,
    )
    
    # Setup callbacks
    callbacks = [
        LoggingCallback(log_every_n_steps=10),
        ProgressCallback(),
    ]
    
    if checkpoint_dir is not None:
        callbacks.append(
            CheckpointCallback(
                checkpoint_dir=checkpoint_dir,
                save_best=True,
                monitor="val_loss",
                mode="min",
            )
        )
    
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                monitor="val_loss",
                patience=early_stopping_patience,
                mode="min",
            )
        )
    
    if use_tensorboard:
        from protophen.training.callbacks import TensorBoardCallback
        
        tb_dir = tensorboard_dir or (Path(checkpoint_dir) / "tensorboard" if checkpoint_dir else "runs")
        callbacks.append(TensorBoardCallback(log_dir=tb_dir))
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        callbacks=callbacks,
    )