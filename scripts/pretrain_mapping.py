#!/usr/bin/env python
"""
Phase 2 pre-training: Protein→Phenotype mapping via frozen autoencoder decoder.

Loads a Phase 1 autoencoder checkpoint, freezes the decoder, and trains a
protein encoder (ESM-2 embeddings → hidden layers → phenotype latent) so that
the full pipeline predicts Cell Painting profiles from protein sequence alone.

The frozen autoencoder decoder constrains the protein encoder's output space
to be biologically meaningful — only latent vectors that reconstruct plausible
Cell Painting profiles will receive low loss.

Usage
-----
    # Standard run
    python scripts/pretrain_mapping.py \\
        --config configs/pretraining.yaml \\
        --phase1-checkpoint data/checkpoints/pretraining/phase1/phase1_best.pt \\
        --embeddings-path data/embeddings/esm2_orf_crispr.pkl \\
        --phenotypes-path data/processed/pretraining/phase2_curated.parquet

    # With gradual decoder unfreezing
    python scripts/pretrain_mapping.py \\
        --config configs/pretraining.yaml \\
        --phase1-checkpoint data/checkpoints/pretraining/phase1/phase1_best.pt \\
        --embeddings-path data/embeddings/esm2_orf_crispr.pkl \\
        --phenotypes-path data/processed/pretraining/phase2_curated.parquet \\
        --gradual-unfreeze

    # Resume from Phase 2 checkpoint
    python scripts/pretrain_mapping.py \\
        --config configs/pretraining.yaml \\
        --phase1-checkpoint data/checkpoints/pretraining/phase1/phase1_best.pt \\
        --embeddings-path data/embeddings/esm2_orf_crispr.pkl \\
        --phenotypes-path data/processed/pretraining/phase2_curated.parquet \\
        --resume data/checkpoints/pretraining/phase2/checkpoint_epoch_0020.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protophen.data.dataset import ProtoPhenDataset
from protophen.models.autoencoder import (
    PhenotypeAutoencoder,
    PretrainingConfig,
    load_autoencoder_from_checkpoint,
    save_phase2_checkpoint,
)
from protophen.models.protophen import ProToPhenConfig, ProToPhenModel
from protophen.training.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    ProgressCallback,
    TensorBoardCallback,
)
from protophen.training.trainer import Trainer, TrainerConfig, TrainingState
from protophen.utils.logging import logger, setup_logging


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 pre-training: train protein encoder to predict "
            "autoencoder latent via frozen decoder"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Required -----------------------------------------------------------
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 1 autoencoder checkpoint (from pretrain_phenotype.py)",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        required=True,
        help="Path to pre-computed protein embeddings (.pkl, .pkl.gz, .npz, or directory of .npy)",
    )
    parser.add_argument(
        "--phenotypes-path",
        type=str,
        required=True,
        help="Path to curated Phase 2 Parquet (ORF/CRISPR/TARGET plates)",
    )

    # -- Config and data options --------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretraining.yaml",
        help="Path to pretraining configuration YAML file",
    )
    parser.add_argument(
        "--gene-mapping",
        type=str,
        default=None,
        help="Path to gene→embedding_key mapping (JSON or CSV)",
    )
    parser.add_argument(
        "--treatment-column",
        type=str,
        default="treatment_label",
        help="Column name for treatment/gene labels in the Parquet file",
    )
    parser.add_argument(
        "--feature-list",
        type=str,
        default=None,
        help="Path to feature column list (.txt, one name per line)",
    )
    parser.add_argument(
        "--aggregate-replicates",
        action="store_true",
        help="Mean-aggregate replicates per gene (default: keep all wells)",
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
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--gradual-unfreeze",
        action="store_true",
        help="Gradually unfreeze decoder after N epochs (from config)",
    )

    # -- Resume -------------------------------------------------------------
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to Phase 2 checkpoint to resume training from",
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
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
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


def load_feature_list(path: str) -> List[str]:
    """Load feature column names from a text file (one per line)."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


# =============================================================================
# Gradual Unfreeze Callback
# =============================================================================


class GradualUnfreezeCallback(Callback):
    """Unfreeze the autoencoder decoder after a configurable number of epochs.

    Once unfrozen, the decoder parameters are added to the optimiser with a
    separate (typically lower) learning rate so that the pre-trained decoder
    is fine-tuned gently while the protein encoder continues training.

    Parameters
    ----------
    unfreeze_after_epochs : int
        Epoch at which to unfreeze the decoder.
    decoder_lr : float
        Learning rate for the decoder parameter group.
    """

    def __init__(
        self,
        unfreeze_after_epochs: int = 20,
        decoder_lr: float = 1e-5,
    ) -> None:
        self.unfreeze_after_epochs = unfreeze_after_epochs
        self.decoder_lr = decoder_lr
        self._unfrozen = False

    def on_epoch_begin(self, state: "TrainingState") -> None:
        if self._unfrozen or state.epoch < self.unfreeze_after_epochs:
            return

        decoder = self.trainer.model.decoders.get("cell_painting")
        if decoder is None:
            logger.warning("GradualUnfreezeCallback: no cell_painting decoder found")
            return

        # Unfreeze all decoder parameters
        decoder_params = []
        for p in decoder.parameters():
            p.requires_grad = True
            decoder_params.append(p)

        if not decoder_params:
            logger.warning("GradualUnfreezeCallback: decoder has no parameters")
            return

        # Add decoder params to optimiser as a new param group
        # The LambdaLR scheduler only applies to existing groups, so the
        # decoder group retains a constant LR — this is intentional.
        self.trainer.optimiser.add_param_group(
            {
                "params": decoder_params,
                "lr": self.decoder_lr,
                "weight_decay": 0.0,
            }
        )

        n_unfrozen = sum(p.numel() for p in decoder_params)
        self._unfrozen = True
        logger.info(
            f"Epoch {state.epoch}: decoder unfrozen — "
            f"{n_unfrozen:,} params added at lr={self.decoder_lr:.2e}"
        )


# =============================================================================
# Embedding loading
# =============================================================================


def _load_protein_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load protein embeddings from file or directory.

    Supported formats
    -----------------
    - ``.pkl`` / ``.pkl.gz`` / ``.pickle`` : pickled ``dict[str, ndarray]``
    - ``.npz`` : NumPy archive with named arrays
    - **directory** of ``.npy`` files : file stem → array

    Falls back to ``protophen.utils.io.load_embeddings`` if the format is
    not directly handled.

    Returns
    -------
    dict[str, ndarray]
        Mapping from protein / gene identifier to embedding vector.
    """
    path = Path(path)

    # Directory of per-gene .npy files
    if path.is_dir():
        emb: Dict[str, np.ndarray] = {}
        for f in sorted(path.glob("*.npy")):
            emb[f.stem] = np.load(f)
        if not emb:
            raise FileNotFoundError(f"No .npy files found in {path}")
        logger.info(f"Loaded {len(emb)} embeddings from directory {path}")
        return emb

    # NumPy archive
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        emb = {k: data[k] for k in data.files}
        logger.info(f"Loaded {len(emb)} embeddings from {path}")
        return emb

    # Pickle (optionally gzipped)
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(path, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict from pickle, got {type(data).__name__}"
            )
        logger.info(f"Loaded {len(data)} embeddings from {path}")
        return data
    except Exception as exc:
        # Last resort: try protophen utility
        try:
            from protophen.utils.io import load_embeddings

            data = load_embeddings(str(path))
            logger.info(f"Loaded {len(data)} embeddings via protophen.utils.io")
            return data
        except Exception:
            raise RuntimeError(
                f"Could not load embeddings from {path}: {exc}"
            ) from exc


def _load_gene_mapping(path: Path) -> Dict[str, str]:
    """Load a gene-symbol → embedding-key mapping from JSON or CSV.

    JSON format::

        {"BRCA1": "P38398", "TP53": "P04637"}

    CSV format (first two columns)::

        gene_symbol,embedding_key
        BRCA1,P38398
        TP53,P04637
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path) as fh:
            return json.load(fh)

    # CSV
    import csv

    mapping: Dict[str, str] = {}
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                mapping[row[0].strip()] = row[1].strip()
    logger.info(f"Loaded {len(mapping)} gene→key mappings from {path}")
    return mapping


# =============================================================================
# DataLoader collate
# =============================================================================


def phase2_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate ``ProtoPhenDataset`` samples for the ``Trainer``.

    Stacks tensors, gathers strings and dicts into lists.
    ``Trainer._move_batch_to_device`` handles non-tensor values gracefully.
    """
    elem = batch[0]
    result: Dict[str, Any] = {}
    for key in elem:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


# =============================================================================
# Data loading
# =============================================================================


def load_phase2_data(
    phenotypes_path: Path,
    embeddings_path: Path,
    treatment_column: str = "treatment_label",
    feature_columns: Optional[List[str]] = None,
    gene_mapping_path: Optional[Path] = None,
    aggregate_replicates: bool = False,
    seed: int = 42,
) -> Tuple[ProtoPhenDataset, ProtoPhenDataset]:
    """Join protein embeddings with phenotype profiles for Phase 2 training.

    Parameters
    ----------
    phenotypes_path
        Curated Parquet containing Cell Painting profiles with a
        treatment/gene column.
    embeddings_path
        Pre-computed protein embeddings keyed by gene or protein identifier.
    treatment_column
        Column in the Parquet that identifies the gene / protein.
    feature_columns
        Explicit list of feature columns.  If *None*, all non-metadata
        numeric columns are used.
    gene_mapping_path
        Optional JSON/CSV mapping ``treatment_label`` values to embedding
        keys.  If *None*, embedding keys are assumed to match treatment
        labels directly.
    aggregate_replicates
        If *True*, mean-aggregate all wells per gene into one sample.
        If *False* (default), each well is a separate sample paired with
        the same embedding — more training data but potential class
        imbalance.
    seed
        Random seed for train/val split.

    Returns
    -------
    tuple[ProtoPhenDataset, ProtoPhenDataset]
        Train and validation datasets.
    """
    import pandas as pd

    # ---- Load phenotype profiles -----------------------------------------
    logger.info(f"Loading phenotype profiles from {phenotypes_path}")
    df = pd.read_parquet(phenotypes_path)
    logger.info(f"  {len(df):,} rows, {len(df.columns)} columns")

    if treatment_column not in df.columns:
        raise ValueError(
            f"Treatment column '{treatment_column}' not found. "
            f"Available: {list(df.columns[:20])}"
        )

    # Identify feature columns
    meta_prefixes = ("Metadata_",)
    if feature_columns is None:
        meta_cols = {treatment_column, "plate_id"}
        feature_columns = [
            c for c in df.columns
            if c not in meta_cols
            and not any(c.startswith(p) for p in meta_prefixes)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    logger.info(f"  {len(feature_columns)} feature columns")

    # ---- Load protein embeddings -----------------------------------------
    embeddings = _load_protein_embeddings(embeddings_path)
    sample_key = next(iter(embeddings))
    embedding_dim = embeddings[sample_key].shape[-1]
    logger.info(
        f"  {len(embeddings)} embeddings, dim={embedding_dim}, "
        f"sample key='{sample_key}'"
    )

    # ---- Gene mapping (optional) -----------------------------------------
    gene_mapping: Optional[Dict[str, str]] = None
    if gene_mapping_path is not None:
        gene_mapping = _load_gene_mapping(gene_mapping_path)

    # ---- Join: match genes to embeddings ---------------------------------
    matched_embeddings: List[np.ndarray] = []
    matched_features: List[np.ndarray] = []
    matched_ids: List[str] = []
    skipped_genes: List[str] = []

    groups = df.groupby(treatment_column)
    for gene, group in groups:
        gene_str = str(gene)
        emb_key = (
            gene_mapping.get(gene_str, gene_str)
            if gene_mapping
            else gene_str
        )
        if emb_key not in embeddings:
            skipped_genes.append(gene_str)
            continue
            
        emb = embeddings[emb_key].astype(np.float32)

        if aggregate_replicates:
            # Mean-aggregate all wells for this gene into one sample
            features = (
                group[feature_columns]
                .values.astype(np.float32)
                .mean(axis=0)
            )
            matched_embeddings.append(emb)
            matched_features.append(features)
            matched_ids.append(gene_str)
        else:
            # Keep every well as a separate sample (same embedding)
            well_features = group[feature_columns].values.astype(np.float32)
            for i in range(len(well_features)):
                matched_embeddings.append(emb)
                matched_features.append(well_features[i])
                matched_ids.append(gene_str)

    if skipped_genes:
        logger.warning(
            f"Skipped {len(skipped_genes)} genes with no embedding match. "
            f"First 10: {skipped_genes[:10]}"
        )

    if not matched_embeddings:
        raise RuntimeError(
            "No genes matched between phenotype profiles and embeddings. "
            "Check that treatment labels and embedding keys use the same "
            "identifiers, or provide --gene-mapping."
        )

    protein_embeddings = np.stack(matched_embeddings)
    phenotype_features = np.stack(matched_features)

    logger.info(
        f"Matched {len(matched_ids):,} samples "
        f"({len(set(matched_ids)):,} unique genes) — "
        f"embedding_dim={protein_embeddings.shape[1]}, "
        f"phenotype_dim={phenotype_features.shape[1]}"
    )

    # ---- Build ProtoPhenDataset ------------------------------------------
    dataset = ProtoPhenDataset.from_arrays(
        protein_embeddings=protein_embeddings,
        phenotype_features=phenotype_features,
        protein_ids=matched_ids,
    )

    # ---- Train / val split -----------------------------------------------
    # 80/20 split; test comes from held-out experimental data, not here
    train_ds, val_ds, _ = dataset.split(
        train_frac=0.8,
        val_frac=0.2,
        test_frac=0.0,
        seed=seed,
    )

    logger.info(
        f"Split: {len(train_ds):,} train, {len(val_ds):,} val"
    )
    return train_ds, val_ds


# =============================================================================
# Model construction
# =============================================================================


def build_phase2_model(
    config: PretrainingConfig,
    autoencoder: PhenotypeAutoencoder,
    actual_embedding_dim: Optional[int] = None,
    actual_phenotype_dim: Optional[int] = None,
) -> ProToPhenModel:
    """Build a ``ProToPhenModel`` with frozen autoencoder decoder.

    The protein encoder maps ESM-2 embeddings → latent space.  The
    autoencoder decoder (frozen) maps latent → reconstructed Cell Painting
    features.  The model is trained end-to-end with MSE loss in feature
    space, but the frozen decoder constrains the latent space to be
    biologically meaningful.

    Parameters
    ----------
    config
        Full pretraining configuration.
    autoencoder
        Phase 1 trained autoencoder.
    actual_embedding_dim
        If provided, overrides ``config.phase2.protein_encoder.embedding_dim``
        to match the actual loaded embeddings.
    actual_phenotype_dim
        If provided, overrides ``config.autoencoder.input_dim`` to match the
        actual loaded phenotype features.

    Returns
    -------
    ProToPhenModel
        Model with ``AutoencoderDecoderHead`` as the cell_painting decoder.
    """
    p2_enc = config.phase2.protein_encoder

    # Use actual dimensions when available (data takes precedence over config)
    emb_dim = actual_embedding_dim or p2_enc.embedding_dim
    cp_dim = actual_phenotype_dim or config.autoencoder.input_dim

    if emb_dim != p2_enc.embedding_dim:
        logger.warning(
            f"Embedding dim from data ({emb_dim}) differs from config "
            f"({p2_enc.embedding_dim}). Using data dimension."
        )
    if cp_dim != config.autoencoder.input_dim:
        logger.warning(
            f"Phenotype dim from data ({cp_dim}) differs from config "
            f"({config.autoencoder.input_dim}). Using data dimension."
        )

    # Verify critical dimension match: encoder output == autoencoder latent
    if p2_enc.output_dim != autoencoder.latent_dim:
        raise ValueError(
            f"Protein encoder output_dim ({p2_enc.output_dim}) must match "
            f"autoencoder latent_dim ({autoencoder.latent_dim}). "
            f"Update phase2.protein_encoder.output_dim in the config."
        )

    # Build the standard ProToPhen model
    model_config = ProToPhenConfig(
        protein_embedding_dim=emb_dim,
        encoder_hidden_dims=p2_enc.hidden_dims,
        encoder_output_dim=p2_enc.output_dim,
        encoder_dropout=p2_enc.dropout,
        encoder_activation=p2_enc.activation,
        cell_painting_dim=cp_dim,
        predict_viability=False,
        predict_transcriptomics=False,
        predict_uncertainty=False,
    )
    model = ProToPhenModel(model_config)

    # Replace the cell_painting decoder with the frozen autoencoder decoder
    freeze_decoder = config.phase2.freeze.decoder
    decoder_head = autoencoder.get_decoder_head(freeze=freeze_decoder)
    model.decoders["cell_painting"] = decoder_head

    # Param summary
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_total - n_trainable
    logger.info(
        f"Phase 2 model: {n_total:,} total params — "
        f"{n_trainable:,} trainable (encoder), "
        f"{n_frozen:,} frozen (decoder)"
    )

    return model


# =============================================================================
# Main
# =============================================================================


def main() -> None:  # noqa: C901
    args = parse_args()
    setup_logging(level=args.log_level)

    logger.info("=" * 70)
    logger.info("ProToPhen — Phase 2 Pre-training: Protein→Phenotype Mapping")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Configuration                                                    #
    # ------------------------------------------------------------------ #
    config = PretrainingConfig.from_yaml(args.config)
    p2 = config.phase2
    train_cfg = p2.training

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
    ) / "phase2"
    log_dir = Path(args.log_dir or config.output.log_dir) / "phase2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config.save(log_dir / "effective_config.yaml")

    device = resolve_device(args.device)
    device_str = str(device)
    use_amp = train_cfg.use_amp and device.type == "cuda"

    logger.info(f"Device: {device}  |  AMP: {use_amp}")
    logger.info(f"Phase 1 checkpoint: {args.phase1_checkpoint}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Logs:        {log_dir}")

    set_seed(train_cfg.seed)

    # ------------------------------------------------------------------ #
    # 2. Load Phase 1 autoencoder                                         #
    # ------------------------------------------------------------------ #
    phase1_path = Path(args.phase1_checkpoint)
    if not phase1_path.exists():
        logger.error(f"Phase 1 checkpoint not found: {phase1_path}")
        sys.exit(1)

    autoencoder, phase1_ckpt = load_autoencoder_from_checkpoint(
        phase1_path, device=device,
    )
    phase1_epoch = phase1_ckpt.get("epoch", "?")
    phase1_loss = phase1_ckpt.get("best_val_loss", "?")
    logger.info(
        f"Phase 1 autoencoder loaded: epoch={phase1_epoch}, "
        f"val_loss={phase1_loss}, latent_dim={autoencoder.latent_dim}"
    )

    # ------------------------------------------------------------------ #
    # 3. Load Phase 2 data                                                #
    # ------------------------------------------------------------------ #
    feature_columns: Optional[List[str]] = None
    if args.feature_list is not None:
        feature_columns = load_feature_list(args.feature_list)

    train_dataset, val_dataset = load_phase2_data(
        phenotypes_path=Path(args.phenotypes_path),
        embeddings_path=Path(args.embeddings_path),
        treatment_column=args.treatment_column,
        feature_columns=feature_columns,
        gene_mapping_path=Path(args.gene_mapping) if args.gene_mapping else None,
        aggregate_replicates=args.aggregate_replicates,
        seed=train_cfg.seed,
    )

    # ------------------------------------------------------------------ #
    # 4. Build Phase 2 model                                              #
    # ------------------------------------------------------------------ #
    model = build_phase2_model(
        config=config,
        autoencoder=autoencoder,
        actual_embedding_dim=train_dataset.embedding_dim,
        actual_phenotype_dim=train_dataset.phenotype_dims.get("cell_painting"),
    )
    model = model.to(device)

    # Free autoencoder memory — weights are now inside the decoder head
    del autoencoder

    logger.info(f"Model: {model}")

    # ------------------------------------------------------------------ #
    # 5. DataLoaders                                                      #
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=phase2_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=phase2_collate_fn,
    )

    steps_per_epoch = max(
        len(train_loader) // train_cfg.gradient_accumulation_steps, 1,
    )
    warmup_steps = train_cfg.warmup_epochs * steps_per_epoch

    logger.info(
        f"Loader: {len(train_loader)} train batches, "
        f"{steps_per_epoch} steps/epoch, {warmup_steps} warmup steps"
    )

    # ------------------------------------------------------------------ #
    # 6. Trainer setup                                                    #
    # ------------------------------------------------------------------ #
    trainer_config = TrainerConfig(
        epochs=train_cfg.epochs,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        optimiser=train_cfg.optimiser,
        scheduler=train_cfg.scheduler,
        warmup_steps=warmup_steps,
        min_lr=train_cfg.min_lr,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        max_grad_norm=train_cfg.max_grad_norm,
        use_amp=use_amp,
        device=device_str,
        seed=train_cfg.seed,
        tasks=["cell_painting"],
        task_weights={"cell_painting": 1.0},
    )

    # -- Callbacks ---------------------------------------------------------
    callbacks = [
        LoggingCallback(
            log_every_n_steps=10,
            log_file=log_dir / "phase2_training.log",
        ),
        ProgressCallback(),
        CheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            save_best=True,
            monitor="val_loss",
            mode="min",
            save_every_n_epochs=10,
            keep_n_checkpoints=3,
        ),
    ]

    if args.patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                monitor="val_loss",
                patience=args.patience,
                mode="min",
            )
        )

    if args.gradual_unfreeze:
        freeze_cfg = config.phase2.freeze
        callbacks.append(
            GradualUnfreezeCallback(
                unfreeze_after_epochs=freeze_cfg.unfreeze_after_epochs,
                decoder_lr=train_cfg.learning_rate * 0.1,
            )
        )
        logger.info(
            f"Gradual unfreezing enabled: decoder will unfreeze at "
            f"epoch {freeze_cfg.unfreeze_after_epochs}"
        )

    if args.tensorboard:
        callbacks.append(
            TensorBoardCallback(log_dir=log_dir / "tensorboard")
        )

    # -- Create Trainer (reuses Session 6 infrastructure) ------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------ #
    # 7. Train                                                            #
    # ------------------------------------------------------------------ #
    logger.info(
        f"Starting Phase 2 training: {train_cfg.epochs} epochs, "
        f"lr={train_cfg.learning_rate}, batch_size={train_cfg.batch_size}"
    )

    t_start = time.time()
    history = trainer.train(resume_from=args.resume)
    t_elapsed = time.time() - t_start

    logger.info(
        f"Phase 2 training complete in {t_elapsed:.1f}s "
        f"({t_elapsed / 60:.1f} min)"
    )

    # ------------------------------------------------------------------ #
    # 8. Final evaluation                                                 #
    # ------------------------------------------------------------------ #
    logger.info("-" * 70)
    logger.info("Running final evaluation on validation set")

    final_metrics = trainer.evaluate(val_loader)
    logger.info("Final validation metrics:")
    for k, v in sorted(final_metrics.items()):
        logger.info(f"  {k:>35s}: {v:.6f}")

    # ------------------------------------------------------------------ #
    # 9. Save Phase 2 checkpoint with provenance                          #
    # ------------------------------------------------------------------ #
    phase2_best_path = checkpoint_dir / "phase2_best.pt"
    save_phase2_checkpoint(
        path=phase2_best_path,
        model=model,
        config=config,
        phase1_checkpoint_path=phase1_path,
        epoch=trainer.state.epoch,
        global_step=trainer.state.global_step,
        best_val_loss=trainer.state.best_val_loss,
        metrics=final_metrics,
    )

    # ------------------------------------------------------------------ #
    # 10. Persist results                                                 #
    # ------------------------------------------------------------------ #
    results = {
        "phase": 2,
        "phase1_checkpoint": str(phase1_path),
        "phase1_epoch": phase1_epoch,
        "phase1_val_loss": float(phase1_loss) if isinstance(phase1_loss, (int, float)) else phase1_loss,
        "embeddings_path": str(args.embeddings_path),
        "phenotypes_path": str(args.phenotypes_path),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "embedding_dim": train_dataset.embedding_dim,
        "phenotype_dim": train_dataset.phenotype_dims.get("cell_painting", 0),
        "model_summary": model.summary(),
        "training_config": {
            "epochs_completed": trainer.state.epoch,
            "total_steps": trainer.state.global_step,
            "batch_size": train_cfg.batch_size,
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "optimiser": train_cfg.optimiser,
            "scheduler": train_cfg.scheduler,
            "warmup_epochs": train_cfg.warmup_epochs,
            "seed": train_cfg.seed,
            "use_amp": use_amp,
            "gradual_unfreeze": args.gradual_unfreeze,
        },
        "freeze_config": {
            "decoder_frozen": config.phase2.freeze.decoder,
            "gradual_unfreeze": args.gradual_unfreeze,
            "unfreeze_after_epochs": config.phase2.freeze.unfreeze_after_epochs,
        },
        "best_val_loss": float(trainer.state.best_val_loss),
        "final_metrics": {k: float(v) for k, v in final_metrics.items()},
        "history": {
            "train_losses": [float(x) for x in history.get("train_losses", [])],
            "val_losses": [float(x) for x in history.get("val_losses", [])],
            "best_val_loss": float(history.get("best_val_loss", float("inf"))),
        },
        "checkpoints": {
            "phase2_best": str(phase2_best_path),
            "trainer_best": str(checkpoint_dir / "best_model.pt"),
        },
    }

    results_path = log_dir / "phase2_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 70)
    logger.info(
        f"Phase 2 complete | best_val_loss={trainer.state.best_val_loss:.6f}"
    )
    logger.info(f"Phase 2 checkpoint: {phase2_best_path}")
    logger.info(
        "This checkpoint is compatible with the serving pipeline:\n"
        "  from protophen.serving.pipeline import InferencePipeline\n"
        f'  pipeline = InferencePipeline.from_checkpoint("{phase2_best_path}")'
    )


if __name__ == "__main__":
    main()