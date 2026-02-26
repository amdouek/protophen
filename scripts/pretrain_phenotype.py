#!/usr/bin/env python
"""
Phase 1 pre-training: PhenotypeAutoencoder on JUMP-CP data.

Trains a phenotype autoencoder on ALL JUMP-CP plates to learn the structure
of Cell Painting phenotype space.  The trained encoder and decoder are used
in Phase 2 to provide a biologically meaningful target space for the protein
encoder.

Usage
-----
    # Standard run
    python scripts/pretrain_phenotype.py \\
        --config configs/pretraining.yaml \\
        --data-path data/processed/pretraining/curated.parquet

    # Override training parameters
    python scripts/pretrain_phenotype.py \\
        --config configs/pretraining.yaml \\
        --data-path data/processed/pretraining/curated.parquet \\
        --epochs 200 --batch-size 512 --learning-rate 5e-4

    # Resume from checkpoint
    python scripts/pretrain_phenotype.py \\
        --config configs/pretraining.yaml \\
        --data-path data/processed/pretraining/curated.parquet \\
        --resume data/checkpoints/pretraining/phase1/phase1_epoch_0050.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from protophen.models.autoencoder import (
    AutoencoderLoss,
    PhenotypeAutoencoder,
    PretrainingConfig,
    PretrainingDataset,
    compute_latent_silhouette,
    compute_replicate_correlation,
    load_autoencoder_from_checkpoint,
    save_phase1_checkpoint,
)
from protophen.utils.logging import logger, setup_logging


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 pre-training: PhenotypeAutoencoder on JUMP-CP data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Config and data ----------------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretraining.yaml",
        help="Path to pretraining configuration YAML file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to curated Parquet file (output of curate_pretraining.py)",
    )
    parser.add_argument(
        "--feature-list",
        type=str,
        default=None,
        help="Path to feature column list (.txt, one name per line)",
    )
    parser.add_argument(
        "--treatment-column",
        type=str,
        default="treatment_label",
        help="Column name for treatment labels in the Parquet file",
    )
    parser.add_argument(
        "--plate-column",
        type=str,
        default="plate_id",
        help="Column name for plate identifiers in the Parquet file",
    )

    # -- Output -------------------------------------------------------------
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory from config",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override log directory from config",
    )

    # -- Training overrides -------------------------------------------------
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save periodic checkpoint every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in eval cycles (0 to disable)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Gaussian noise std for training augmentation",
    )

    # -- Resume -------------------------------------------------------------
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to Phase 1 checkpoint to resume training from",
    )

    # -- Hardware -----------------------------------------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )

    # -- Misc ---------------------------------------------------------------
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to ``torch.device`` with auto-detection."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return device


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimiser(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    optimiser_name: str = "adamw",
) -> torch.optim.Optimizer:
    """Create optimiser with separate weight-decay groups.

    Mirrors ``Trainer._create_optimiser()`` — biases and layer-norm
    parameters are excluded from weight decay.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    opt = optimiser_name.lower()
    if opt == "adamw":
        return torch.optim.AdamW(
            param_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
        )
    if opt == "adam":
        return torch.optim.Adam(
            param_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
        )
    if opt == "sgd":
        return torch.optim.SGD(param_groups, lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimiser: {optimiser_name}")


def create_cosine_scheduler(
    optimiser: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup.

    Mirrors ``Trainer._create_cosine_scheduler()``.
    """
    min_lr_ratio = min_lr / max(base_lr, 1e-12)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate that stacks tensors and gathers strings into lists.

    ``PretrainingDataset.__getitem__`` returns a mix of tensors
    (``phenotype_features``, ``treatment_label``, ``sample_weight``)
    and strings (``plate_id``).  The default collate would handle this
    correctly in recent PyTorch, but an explicit function is clearer and
    avoids version-dependent behaviour.
    """
    elem = batch[0]
    result: Dict[str, Any] = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result


# ---------------------------------------------------------------------------
# Validation and latent collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    autoencoder: PhenotypeAutoencoder,
    loss_fn: AutoencoderLoss,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Run validation; return average loss components.

    Returns
    -------
    dict
        Keys: ``total``, ``reconstruction``, ``contrastive``, ``kl``.
    """
    autoencoder.eval()
    totals: Dict[str, float] = defaultdict(float)
    n_batches = 0

    for batch in val_loader:
        features = batch["phenotype_features"].to(device, non_blocking=True)
        labels = batch["treatment_label"].to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp):
            outputs = autoencoder(features)
            losses = loss_fn(outputs, features, labels)

        for k, v in losses.items():
            totals[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def collect_latents(
    autoencoder: PhenotypeAutoencoder,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 10_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a loader and return ``(latents, labels)`` tensors on CPU.

    Collects at most *max_samples* for efficiency — latent quality
    metrics (silhouette, replicate correlation) are costly on large N.
    """
    autoencoder.eval()
    all_latents: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    n = 0

    for batch in loader:
        features = batch["phenotype_features"].to(device, non_blocking=True)
        enc_out = autoencoder.encode(features)
        all_latents.append(enc_out["latent"].cpu())
        all_labels.append(batch["treatment_label"])
        n += features.size(0)
        if n >= max_samples:
            break

    return torch.cat(all_latents, dim=0)[:max_samples], \
           torch.cat(all_labels, dim=0)[:max_samples]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_epoch(
    epoch: int,
    num_epochs: int,
    train: Dict[str, float],
    val: Optional[Dict[str, float]],
    lr: float,
    elapsed: float,
) -> None:
    """Print a single summary line for the epoch."""
    parts = [
        f"Epoch {epoch:>4d}/{num_epochs}",
        f"lr={lr:.2e}",
        f"train={train['total']:.4f}",
        f"recon={train.get('reconstruction', 0.0):.4f}",
        f"contr={train.get('contrastive', 0.0):.4f}",
    ]
    if train.get("kl", 0.0) > 0:
        parts.append(f"kl={train['kl']:.4f}")
    if val is not None:
        parts.append(f"val={val['total']:.4f}")
        if "replicate_corr" in val:
            parts.append(f"rep_corr={val['replicate_corr']:.4f}")
        if "silhouette" in val:
            parts.append(f"sil={val['silhouette']:.4f}")
    parts.append(f"{elapsed:.1f}s")
    logger.info(" | ".join(parts))


def load_feature_list(path: str) -> List[str]:
    """Load feature column names from a text file (one per line)."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


# =============================================================================
# Main
# =============================================================================


def main() -> None:  # noqa: C901 — intentionally long orchestration function
    args = parse_args()
    setup_logging(level=args.log_level)

    logger.info("=" * 70)
    logger.info("ProToPhen — Phase 1 Pre-training: PhenotypeAutoencoder")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Configuration                                                    #
    # ------------------------------------------------------------------ #
    config = PretrainingConfig.from_yaml(args.config)
    p1 = config.phase1
    train_cfg = p1.training
    eval_cfg = p1.evaluation

    # Apply CLI overrides
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.seed is not None:
        train_cfg.seed = args.seed

    checkpoint_dir = Path(
        args.checkpoint_dir or config.output.checkpoint_dir
    ) / "phase1"
    log_dir = Path(args.log_dir or config.output.log_dir) / "phase1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Persist effective config for reproducibility
    config.save(log_dir / "effective_config.yaml")

    device = resolve_device(args.device)
    use_amp = train_cfg.use_amp and device.type == "cuda"

    logger.info(f"Device: {device}  |  AMP: {use_amp}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Logs:        {log_dir}")

    set_seed(train_cfg.seed)

    # ------------------------------------------------------------------ #
    # 2. Data loading                                                     #
    # ------------------------------------------------------------------ #
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    feature_columns: Optional[List[str]] = None
    if args.feature_list is not None:
        feature_columns = load_feature_list(args.feature_list)
        logger.info(
            f"Using {len(feature_columns)} features from {args.feature_list}"
        )

    logger.info(f"Loading dataset from {data_path}")
    full_dataset = PretrainingDataset.from_parquet(
        parquet_path=data_path,
        feature_columns=feature_columns,
        treatment_column=args.treatment_column,
        plate_column=args.plate_column,
        augmentation_noise_std=args.noise_std,
    )

    # Verify input_dim matches config (auto-adjust if needed)
    actual_dim = full_dataset.n_features
    if actual_dim != config.autoencoder.input_dim:
        logger.warning(
            f"Config input_dim={config.autoencoder.input_dim} does not match "
            f"data features={actual_dim}. Updating config to match data."
        )
        config.autoencoder.input_dim = actual_dim

    # Split into train / validation
    train_dataset, val_dataset = full_dataset.split(
        train_frac=0.9,
        val_frac=0.1,
        seed=train_cfg.seed,
    )

    logger.info(
        f"Dataset: {len(full_dataset):,} total → "
        f"{len(train_dataset):,} train, {len(val_dataset):,} val  |  "
        f"{full_dataset.n_features} features, "
        f"{full_dataset.n_treatments:,} treatments"
    )

    # ------------------------------------------------------------------ #
    # 3. DataLoaders                                                      #
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size * 2,  # larger batch OK for eval
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )

    steps_per_epoch = max(
        len(train_loader) // train_cfg.gradient_accumulation_steps, 1,
    )
    total_steps = steps_per_epoch * train_cfg.epochs
    warmup_steps = train_cfg.warmup_epochs * steps_per_epoch

    logger.info(
        f"Loader: {len(train_loader)} train batches/epoch, "
        f"{steps_per_epoch} optimiser steps/epoch, "
        f"{total_steps} total steps, {warmup_steps} warmup steps"
    )

    # ------------------------------------------------------------------ #
    # 4. Model + loss                                                     #
    # ------------------------------------------------------------------ #
    autoencoder = PhenotypeAutoencoder(config.autoencoder).to(device)
    logger.info(f"Model: {autoencoder}")
    logger.info(
        f"Parameters: {autoencoder.n_parameters:,} total, "
        f"{autoencoder.n_trainable_parameters:,} trainable"
    )

    loss_fn = AutoencoderLoss(
        config=p1.loss,
        variational=config.autoencoder.variational,
    )

    # ------------------------------------------------------------------ #
    # 5. Optimiser + scheduler + scaler                                   #
    # ------------------------------------------------------------------ #
    optimiser = create_optimiser(
        model=autoencoder,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        optimiser_name=train_cfg.optimiser,
    )
    scheduler = create_cosine_scheduler(
        optimiser=optimiser,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=train_cfg.min_lr,
        base_lr=train_cfg.learning_rate,
    )
    scaler = GradScaler("cuda") if use_amp else None

    # ------------------------------------------------------------------ #
    # 6. Resume from checkpoint (optional)                                #
    # ------------------------------------------------------------------ #
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume checkpoint not found: {resume_path}")
            sys.exit(1)

        logger.info(f"Resuming from {resume_path}")
        autoencoder_resumed, ckpt = load_autoencoder_from_checkpoint(
            resume_path, device=device,
        )
        # Copy loaded weights into our model (in case config was updated)
        autoencoder.load_state_dict(autoencoder_resumed.state_dict())
        del autoencoder_resumed

        # Restore training state
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

        if "optimiser_state_dict" in ckpt:
            optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        logger.info(
            f"Resumed: epoch={start_epoch}, step={global_step}, "
            f"best_val_loss={best_val_loss:.6f}"
        )

    # ------------------------------------------------------------------ #
    # 7. Training loop                                                    #
    # ------------------------------------------------------------------ #
    num_epochs = train_cfg.epochs
    eval_every = eval_cfg.eval_every_n_epochs
    grad_accum = train_cfg.gradient_accumulation_steps
    max_grad_norm = train_cfg.max_grad_norm
    patience = args.patience
    save_every = args.save_every

    patience_counter = 0
    history: Dict[str, List[Any]] = {
        "train_loss": [],
        "val_loss": [],
        "replicate_corr": [],
        "silhouette": [],
        "learning_rate": [],
    }

    best_path = checkpoint_dir / "phase1_best.pt"

    logger.info(f"Starting Phase 1 training: epochs {start_epoch}→{num_epochs}")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # ---- train -------------------------------------------------------
        autoencoder.train()
        epoch_losses: Dict[str, float] = defaultdict(float)
        n_batches = 0
        optimiser.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            features = batch["phenotype_features"].to(device, non_blocking=True)
            labels = batch["treatment_label"].to(device, non_blocking=True)

            with autocast(device.type, enabled=use_amp):
                outputs = autoencoder(features)
                losses = loss_fn(outputs, features, labels)
                scaled_loss = losses["total"] / grad_accum

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimiser)
                if max_grad_norm > 0:
                    clip_grad_norm_(autoencoder.parameters(), max_grad_norm)
                if scaler is not None:
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                optimiser.zero_grad()
                scheduler.step()
                global_step += 1

            for k, v in losses.items():
                epoch_losses[k] += v.item()
            n_batches += 1

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        lr = optimiser.param_groups[0]["lr"]

        history["train_loss"].append(avg_train["total"])
        history["learning_rate"].append(lr)

        # ---- validate / evaluate -----------------------------------------
        val_metrics: Optional[Dict[str, float]] = None
        is_eval_epoch = (epoch % eval_every == 0) or (epoch == num_epochs)

        if is_eval_epoch:
            val_metrics = validate(
                autoencoder, loss_fn, val_loader, device, use_amp,
            )

            # Latent quality metrics
            if eval_cfg.compute_replicate_correlation or eval_cfg.compute_silhouette:
                latents, latent_labels = collect_latents(
                    autoencoder, val_loader, device,
                )
                if eval_cfg.compute_replicate_correlation:
                    val_metrics["replicate_corr"] = compute_replicate_correlation(
                        latents, latent_labels,
                    )
                if eval_cfg.compute_silhouette:
                    val_metrics["silhouette"] = compute_latent_silhouette(
                        latents, latent_labels,
                    )

            history["val_loss"].append(val_metrics["total"])
            history["replicate_corr"].append(
                val_metrics.get("replicate_corr", float("nan"))
            )
            history["silhouette"].append(
                val_metrics.get("silhouette", float("nan"))
            )

            # Best model tracking
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                patience_counter = 0
                save_phase1_checkpoint(
                    path=best_path,
                    autoencoder=autoencoder,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    optimiser_state=optimiser.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    scaler_state=scaler.state_dict() if scaler else None,
                    metrics=val_metrics,
                )
                logger.info(
                    f"  ★ New best model saved (val_loss={best_val_loss:.6f})"
                )
            else:
                patience_counter += 1
                if patience > 0:
                    logger.debug(
                        f"  Patience: {patience_counter}/{patience}"
                    )

            # Early stopping
            if patience > 0 and patience_counter >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} eval cycles)"
                )
                break

        # ---- periodic checkpoint -----------------------------------------
        if save_every > 0 and epoch % save_every == 0:
            epoch_path = checkpoint_dir / f"phase1_epoch_{epoch:04d}.pt"
            save_phase1_checkpoint(
                path=epoch_path,
                autoencoder=autoencoder,
                config=config,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                optimiser_state=optimiser.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict() if scaler else None,
            )

        # ---- epoch log ---------------------------------------------------
        elapsed = time.time() - epoch_start
        log_epoch(epoch, num_epochs, avg_train, val_metrics, lr, elapsed)

    # ------------------------------------------------------------------ #
    # 8. Final evaluation                                                 #
    # ------------------------------------------------------------------ #
    logger.info("-" * 70)
    logger.info("Phase 1 training complete — running final evaluation")

    # Reload best model
    if best_path.exists():
        best_ae, _ = load_autoencoder_from_checkpoint(best_path, device=device)
        autoencoder.load_state_dict(best_ae.state_dict())
        del best_ae
        logger.info(f"Loaded best checkpoint from {best_path}")

    final_metrics = validate(autoencoder, loss_fn, val_loader, device, use_amp)

    latents, latent_labels = collect_latents(autoencoder, val_loader, device)
    final_metrics["replicate_corr"] = compute_replicate_correlation(
        latents, latent_labels,
    )
    final_metrics["silhouette"] = compute_latent_silhouette(
        latents, latent_labels,
    )

    logger.info("Final validation metrics:")
    for k, v in final_metrics.items():
        logger.info(f"  {k:>25s}: {v:.6f}")

    # Save final checkpoint (distinct from best, includes final state)
    final_path = checkpoint_dir / "phase1_final.pt"
    save_phase1_checkpoint(
        path=final_path,
        autoencoder=autoencoder,
        config=config,
        epoch=epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
        metrics=final_metrics,
    )

    # ------------------------------------------------------------------ #
    # 9. Persist results                                                  #
    # ------------------------------------------------------------------ #
    results = {
        "phase": 1,
        "data_path": str(data_path),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_features": full_dataset.n_features,
        "n_treatments": full_dataset.n_treatments,
        "autoencoder": autoencoder.summary(),
        "training_config": {
            "epochs_completed": epoch,
            "total_steps": global_step,
            "batch_size": train_cfg.batch_size,
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "optimiser": train_cfg.optimiser,
            "scheduler": train_cfg.scheduler,
            "warmup_epochs": train_cfg.warmup_epochs,
            "seed": train_cfg.seed,
            "use_amp": use_amp,
            "noise_std": args.noise_std,
        },
        "loss_config": {
            "reconstruction_weight": p1.loss.reconstruction_weight,
            "contrastive_weight": p1.loss.contrastive_weight,
            "kl_weight": p1.loss.kl_weight,
            "reconstruction_type": p1.loss.reconstruction_type,
            "contrastive_temperature": p1.loss.contrastive_temperature,
        },
        "best_val_loss": best_val_loss,
        "final_metrics": {k: float(v) for k, v in final_metrics.items()},
        "history": {
            k: [float(x) if not np.isnan(x) else None for x in v]
            for k, v in history.items()
        },
        "checkpoints": {
            "best": str(best_path),
            "final": str(final_path),
        },
    }

    results_path = log_dir / "phase1_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 70)
    logger.info(
        f"Phase 1 complete | best_val_loss={best_val_loss:.6f} | "
        f"rep_corr={final_metrics.get('replicate_corr', 0.0):.4f} | "
        f"silhouette={final_metrics.get('silhouette', 0.0):.4f}"
    )
    logger.info(
        f"Best checkpoint: {best_path}\n"
        f"Use this as --phase1-checkpoint in pretrain_mapping.py"
    )


if __name__ == "__main__":
    main()