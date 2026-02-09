"""
Training callbacks for ProToPhen.

This module provides callback classes for monitoring and controlling
the training process, including checkpointing, early stopping, and logging.
"""

from __future__ import annotations

import json
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch

from protophen.utils.logging import logger

if TYPE_CHECKING:
    from protophen.training.trainer import Trainer, TrainingState


# =============================================================================
# Base Callback
# =============================================================================

class Callback(ABC):
    """
    Base class for training callbacks.
    
    Callbacks can hook into various points of the training loop:
    - on_train_begin/end: Start/end of training
    - on_epoch_begin/end: Start/end of each epoch
    - on_batch_begin/end: Start/end of each batch
    - on_validation_begin/end: Start/end of validation
    """
    
    def set_trainer(self, trainer: "Trainer") -> None:
        """Set reference to trainer."""
        self.trainer = trainer
    
    def on_train_begin(self, state: "TrainingState") -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, state: "TrainingState") -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, state: "TrainingState") -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, state: "TrainingState") -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, state: "TrainingState", batch: Dict[str, Any]) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(
        self,
        state: "TrainingState",
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_validation_begin(self, state: "TrainingState") -> None:
        """Called at the beginning of validation."""
        pass
    
    def on_validation_end(
        self,
        state: "TrainingState",
        metrics: Dict[str, float],
    ) -> None:
        """Called at the end of validation."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialise callback list.
        
        Args:
            callbacks: List of Callback objects
        """
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def set_trainer(self, trainer: "Trainer") -> None:
        """Set trainer reference for all callbacks."""
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def on_train_begin(self, state: "TrainingState") -> None:
        for callback in self.callbacks:
            callback.on_train_begin(state)
    
    def on_train_end(self, state: "TrainingState") -> None:
        for callback in self.callbacks:
            callback.on_train_end(state)
    
    def on_epoch_begin(self, state: "TrainingState") -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(state)
    
    def on_epoch_end(self, state: "TrainingState") -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(state)
    
    def on_batch_begin(self, state: "TrainingState", batch: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(state, batch)
    
    def on_batch_end(
        self,
        state: "TrainingState",
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(state, batch, outputs)
    
    def on_validation_begin(self, state: "TrainingState") -> None:
        for callback in self.callbacks:
            callback.on_validation_begin(state)
    
    def on_validation_end(
        self,
        state: "TrainingState",
        metrics: Dict[str, float],
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(state, metrics)
    
    def __len__(self) -> int:
        return len(self.callbacks)
    
    def __iter__(self):
        return iter(self.callbacks)


# =============================================================================
# Checkpointing
# =============================================================================

class CheckpointCallback(Callback):
    """
    Save model checkpoints during training.
    
    Features:
    - Save best model based on validation metric
    - Save periodic checkpoints
    - Keep only N most recent checkpoints
    - Save training state for resumption
    
    Example:
        >>> callback = CheckpointCallback(
        ...     checkpoint_dir="checkpoints",
        ...     save_best=True,
        ...     monitor="val_loss",
        ...     mode="min",
        ...     save_every_n_epochs=5,
        ... )
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_every_n_epochs: Optional[int] = None,
        keep_n_checkpoints: int = 3,
        save_weights_only: bool = False,
    ):
        """
        Initialise checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save the best model
            monitor: Metric to monitor for best model
            mode: "min" or "max" - whether lower or higher is better
            save_every_n_epochs: Save checkpoint every N epochs
            keep_n_checkpoints: Number of recent checkpoints to keep
            save_weights_only: Only save model weights (not optimiser state)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_n_checkpoints = keep_n_checkpoints
        self.save_weights_only = save_weights_only
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.checkpoint_files: List[Path] = []
    
    def on_train_begin(self, state: "TrainingState") -> None:
        """Create checkpoint directory."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to {self.checkpoint_dir}")
    
    def on_epoch_end(self, state: "TrainingState") -> None:
        """Save checkpoint if conditions are met."""
        epoch = state.epoch
        
        # Save periodic checkpoint
        if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
            self._save_checkpoint(state, f"checkpoint_epoch_{epoch:04d}.pt")
    
    def on_validation_end(
        self,
        state: "TrainingState",
        metrics: Dict[str, float],
    ) -> None:
        """Save best model based on validation metrics."""
        if not self.save_best:
            return
        
        if self.monitor not in metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return
        
        current_value = metrics[self.monitor]
        
        is_better = (
            (self.mode == "min" and current_value < self.best_value) or
            (self.mode == "max" and current_value > self.best_value)
        )
        
        if is_better:
            self.best_value = current_value
            self._save_checkpoint(state, "best_model.pt", is_best=True)
            logger.info(
                f"New best model saved: {self.monitor}={current_value:.6f}"
            )
    
    def _save_checkpoint(
        self,
        state: "TrainingState",
        filename: str,
        is_best: bool = False,
    ) -> None:
        """Save a checkpoint."""
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model_state_dict": self.trainer.model.state_dict(),
            "config": self.trainer.config.__dict__ if hasattr(self.trainer, 'config') else {},
            "best_value": self.best_value,
            "monitor": self.monitor,
        }
        
        if not self.save_weights_only:
            checkpoint["optimiser_state_dict"] = self.trainer.optimiser.state_dict()
            if self.trainer.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.trainer.scheduler.state_dict()
            if self.trainer.scaler is not None:
                checkpoint["scaler_state_dict"] = self.trainer.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.debug(f"Saved checkpoint: {filepath}")
        
        # Manage checkpoint history
        if not is_best:
            self.checkpoint_files.append(filepath)
            
            # Remove old checkpoints
            while len(self.checkpoint_files) > self.keep_n_checkpoints:
                old_checkpoint = self.checkpoint_files.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimiser: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimiser: Whether to load optimiser state
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.trainer.device)
        
        self.trainer.model.load_state_dict(checkpoint["model_state_dict"])
        
        if load_optimiser and "optimiser_state_dict" in checkpoint:
            self.trainer.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.trainer.scheduler is not None:
            self.trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "scaler_state_dict" in checkpoint and self.trainer.scaler is not None:
            self.trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.best_value = checkpoint.get("best_value", self.best_value)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStoppingCallback(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    
    Example:
        >>> callback = EarlyStoppingCallback(
        ...     monitor="val_loss",
        ...     patience=10,
        ...     min_delta=1e-4,
        ...     mode="min",
        ... )
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Initialise early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: "min" or "max"
            restore_best_weights: Restore model weights from best epoch
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, state: "TrainingState") -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_weights = None
    
    def on_validation_end(
        self,
        state: "TrainingState",
        metrics: Dict[str, float],
    ) -> None:
        """Check if training should stop."""
        if self.monitor not in metrics:
            logger.warning(f"Early stopping monitor '{self.monitor}' not in metrics")
            return
        
        current_value = metrics[self.monitor]
        
        if self.mode == "min":
            is_improvement = current_value < (self.best_value - self.min_delta)
        else:
            is_improvement = current_value > (self.best_value + self.min_delta)
        
        if is_improvement:
            self.best_value = current_value
            self.counter = 0
            
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() 
                    for k, v in self.trainer.model.state_dict().items()
                }
        else:
            self.counter += 1
            logger.debug(
                f"Early stopping counter: {self.counter}/{self.patience}"
            )
            
            if self.counter >= self.patience:
                state.should_stop = True
                self.stopped_epoch = state.epoch
                logger.info(
                    f"Early stopping triggered at epoch {state.epoch}. "
                    f"Best {self.monitor}: {self.best_value:.6f}"
                )
    
    def on_train_end(self, state: "TrainingState") -> None:
        """Restore best weights if requested."""
        if self.restore_best_weights and self.best_weights is not None:
            self.trainer.model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


# =============================================================================
# Logging Callbacks
# =============================================================================

class LoggingCallback(Callback):
    """
    Log training progress to console and/or file.
    
    Example:
        >>> callback = LoggingCallback(
        ...     log_every_n_steps=10,
        ...     log_file="training.log",
        ... )
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 10,
        log_file: Optional[Union[str, Path]] = None,
    ):
        """
        Initialise logging callback.
        
        Args:
            log_every_n_steps: Log training metrics every N steps
            log_file: Optional file to write logs
        """
        self.log_every_n_steps = log_every_n_steps
        self.log_file = Path(log_file) if log_file else None
        
        self._batch_losses: List[float] = []
        self._epoch_start_time: float = 0
        self._train_start_time: float = 0
    
    def on_train_begin(self, state: "TrainingState") -> None:
        """Record training start time."""
        self._train_start_time = time.time()
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                f.write(f"Training started at {datetime.now().isoformat()}\n")
                f.write(f"{'='*60}\n")
    
    def on_epoch_begin(self, state: "TrainingState") -> None:
        """Record epoch start time."""
        self._epoch_start_time = time.time()
        self._batch_losses = []
    
    def on_batch_end(
        self,
        state: "TrainingState",
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Log batch metrics."""
        if "loss" in outputs:
            self._batch_losses.append(outputs["loss"])
        
        if state.global_step % self.log_every_n_steps == 0:
            avg_loss = np.mean(self._batch_losses[-self.log_every_n_steps:])
            lr = self.trainer.optimiser.param_groups[0]["lr"]
            
            msg = (
                f"Step {state.global_step} | "
                f"Epoch {state.epoch} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e}"
            )
            logger.info(msg)
    
    def on_epoch_end(self, state: "TrainingState") -> None:
        """Log epoch summary."""
        epoch_time = time.time() - self._epoch_start_time
        avg_loss = np.mean(self._batch_losses) if self._batch_losses else 0
        
        msg = (
            f"Epoch {state.epoch} completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f}"
        )
        logger.info(msg)
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{msg}\n")
    
    def on_validation_end(
        self,
        state: "TrainingState",
        metrics: Dict[str, float],
    ) -> None:
        """Log validation metrics."""
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        msg = f"Validation | {metrics_str}"
        logger.info(msg)
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{msg}\n")
    
    def on_train_end(self, state: "TrainingState") -> None:
        """Log training completion."""
        total_time = time.time() - self._train_start_time
        msg = (
            f"Training completed in {total_time:.1f}s "
            f"({total_time/60:.1f} min)"
        )
        logger.info(msg)
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{'='*60}\n")
                f.write(f"{msg}\n")


class LearningRateCallback(Callback):
    """Track and log learning rate changes."""
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        self.lr_history: List[Tuple[int, float]] = []
    
    def on_batch_end(
        self,
        state: "TrainingState",
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Record learning rate."""
        lr = self.trainer.optimiser.param_groups[0]["lr"]
        self.lr_history.append((state.global_step, lr))
        
        if state.global_step % self.log_every_n_steps == 0:
            logger.debug(f"Step {state.global_step}: LR = {lr:.2e}")


class ProgressCallback(Callback):
    """Show progress bar during training using tqdm."""
    
    def __init__(self):
        self._epoch_pbar = None
        self._batch_pbar = None
    
    def on_train_begin(self, state: "TrainingState") -> None:
        """Initialise epoch progress bar."""
        from tqdm.auto import tqdm
        
        self._epoch_pbar = tqdm(
            total=self.trainer.config.epochs,
            desc="Training",
            unit="epoch",
        )
    
    def on_epoch_begin(self, state: "TrainingState") -> None:
        """Initialise batch progress bar."""
        from tqdm.auto import tqdm
        
        n_batches = len(self.trainer.train_loader)
        self._batch_pbar = tqdm(
            total=n_batches,
            desc=f"Epoch {state.epoch}",
            unit="batch",
            leave=False,
        )
    
    def on_batch_end(
        self,
        state: "TrainingState",
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Update batch progress bar."""
        if self._batch_pbar is not None:
            loss = outputs.get("loss", 0)
            self._batch_pbar.set_postfix({"loss": f"{loss:.4f}"})
            self._batch_pbar.update(1)
    
    def on_epoch_end(self, state: "TrainingState") -> None:
        """Close batch progress bar and update epoch progress."""
        if self._batch_pbar is not None:
            self._batch_pbar.close()
            self._batch_pbar = None
        
        if self._epoch_pbar is not None:
            self._epoch_pbar.update(1)
    
    def on_train_end(self, state: "TrainingState") -> None:
        """Close all progress bars."""
        if self._batch_pbar is not None:
            self._batch_pbar.close()
        if self._epoch_pbar is not None:
            self._epoch_pbar.close()


# =============================================================================
# TensorBoard Callback
# =============================================================================

class TensorBoardCallback(Callback):
    """
    Log metrics to TensorBoard.
    
    Example:
        >>> callback = TensorBoardCallback(log_dir="runs/experiment1")
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        log_every_n_steps: int = 10,
        log_histograms: bool = False,
        histogram_every_n_epochs: int = 5,
    ):
        """
        Initialise TensorBoard callback.
        
        Args:
            log_dir: Directory for TensorBoard logs
            log_every_n_steps: Log scalars every N steps
            log_histograms: Whether to log weight histograms
            histogram_every_n_epochs: Log histograms every N epochs
        """
        self.log_dir = Path(log_dir)
        self.log_every_n_steps = log_every_n_steps
        self.log_histograms = log_histograms
        self.histogram_every_n_epochs = histogram_every_n_epochs
        
        self._writer = None
    
    def on_train_begin(self, state: "TrainingState") -> None:
        """Initialise TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info(f"TensorBoard logging to {self.log_dir}")
        except ImportError:
            logger.warning(
                "TensorBoard not available. "
                "Install with: pip install tensorboard"
            )
            self._writer = None
    
    def on_batch_end(
        self,
        state: "TrainingState",
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Log batch metrics to TensorBoard."""
        if self._writer is None:
            return
        
        if state.global_step % self.log_every_n_steps != 0:
            return
        
        # Log loss
        if "loss" in outputs:
            self._writer.add_scalar(
                "train/loss",
                outputs["loss"],
                state.global_step,
            )
        
        # Log learning rate
        lr = self.trainer.optimiser.param_groups[0]["lr"]
        self._writer.add_scalar("train/lr", lr, state.global_step)
        
        # Log individual task losses
        for key, value in outputs.items():
            if key != "loss" and isinstance(value, (int, float)):
                self._writer.add_scalar(
                    f"train/{key}",
                    value,
                    state.global_step,
                )
    
    def on_epoch_end(self, state: "TrainingState") -> None:
        """Log histograms if enabled."""
        if self._writer is None or not self.log_histograms:
            return
        
        if state.epoch % self.histogram_every_n_epochs != 0:
            return
        
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad:
                self._writer.add_histogram(
                    f"weights/{name}",
                    param.data.cpu(),
                    state.epoch,
                )
                if param.grad is not None:
                    self._writer.add_histogram(
                        f"gradients/{name}",
                        param.grad.cpu(),
                        state.epoch,
                    )
    
    def on_validation_end(
        self,
        state: "TrainingState",
        metrics: Dict[str, float],
    ) -> None:
        """Log validation metrics to TensorBoard."""
        if self._writer is None:
            return
        
        for name, value in metrics.items():
            self._writer.add_scalar(f"val/{name}", value, state.epoch)
    
    def on_train_end(self, state: "TrainingState") -> None:
        """Close TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()