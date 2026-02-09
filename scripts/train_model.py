#!/usr/bin/env python
"""
Training script for ProToPhen models.

Usage:
    python scripts/train_model.py --config configs/experiment.yaml
    python scripts/train_model.py --protein-embeddings data/embeddings.pkl --phenotypes data/phenotypes.csv
    
    # Using installed CLI entry point
    protophen-train --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from protophen.data.dataset import ProtoPhenDataset
from protophen.data.loaders import create_dataloaders, split_by_protein
from protophen.models.protophen import ProToPhenModel, ProToPhenConfig
from protophen.training.trainer import Trainer, TrainerConfig, create_trainer
from protophen.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    ProgressCallback,
    TensorBoardCallback,
)
from protophen.utils.config import ProtoPhenConfig as FullConfig, load_config
from protophen.utils.logging import logger, setup_logging
from protophen.utils.io import load_embeddings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ProToPhen model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    
    # Data arguments
    parser.add_argument(
        "--protein-embeddings",
        type=str,
        default=None,
        help="Path to protein embeddings file (.pkl or .pkl.gz)",
    )
    parser.add_argument(
        "--phenotypes",
        type=str,
        default=None,
        help="Path to phenotype data file (.csv)",
    )
    parser.add_argument(
        "--protein-library",
        type=str,
        default=None,
        help="Path to protein library JSON file",
    )
    
    # Model arguments
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1280,
        help="Dimension of protein embeddings",
    )
    parser.add_argument(
        "--cell-painting-dim",
        type=int,
        default=1500,
        help="Number of Cell Painting features",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[1024, 512],
        help="Hidden layer dimensions for encoder",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "plateau", "none"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Patience for early stopping (0 to disable)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="protophen",
        help="Name for this experiment",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    
    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    return parser.parse_args()


def load_data_from_files(
    embeddings_path: str,
    phenotypes_path: str,
    seed: int = 42,
) -> tuple:
    """
    Load data from embedding and phenotype files.
    
    Args:
        embeddings_path: Path to embeddings file
        phenotypes_path: Path to phenotypes CSV
        seed: Random seed for splitting
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    import pandas as pd
    import numpy as np
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = load_embeddings(embeddings_path)
    
    logger.info(f"Loading phenotypes from {phenotypes_path}")
    phenotypes_df = pd.read_csv(phenotypes_path)
    
    # Match embeddings with phenotypes
    # Assumes phenotypes_df has 'protein_id' column and feature columns
    feature_cols = [c for c in phenotypes_df.columns if c.startswith(("Cells_", "Cytoplasm_", "Nuclei_"))]
    
    if not feature_cols:
        # Fall back to all numeric columns except protein_id
        feature_cols = phenotypes_df.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Found {len(feature_cols)} feature columns")
    
    protein_ids = []
    protein_embeddings = []
    phenotype_features = []
    
    for _, row in phenotypes_df.iterrows():
        pid = str(row.get("protein_id", row.get("Metadata_Protein", "")))
        
        if pid in embeddings:
            protein_ids.append(pid)
            protein_embeddings.append(embeddings[pid])
            phenotype_features.append(row[feature_cols].values.astype(np.float32))
    
    logger.info(f"Matched {len(protein_ids)} proteins with phenotypes")
    
    protein_embeddings = np.stack(protein_embeddings)
    phenotype_features = np.stack(phenotype_features)
    
    # Create dataset
    dataset = ProtoPhenDataset.from_arrays(
        protein_embeddings=protein_embeddings,
        phenotype_features=phenotype_features,
        protein_ids=protein_ids,
    )
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=seed,
    )
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load config if provided
    if args.config is not None:
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
        
        # Override with command line arguments if provided
        if args.epochs != 100:
            config.training.epochs = args.epochs
        if args.learning_rate != 1e-4:
            config.training.learning_rate = args.learning_rate
    else:
        config = None
    
    # Load data
    if args.protein_embeddings and args.phenotypes:
        train_dataset, val_dataset, test_dataset = load_data_from_files(
            args.protein_embeddings,
            args.phenotypes,
            seed=args.seed,
        )
    elif args.protein_library:
        # Load from protein library
        from protophen.data.protein import ProteinLibrary
        from protophen.data.phenotype import PhenotypeDataset
        
        library = ProteinLibrary.from_json(args.protein_library)
        phenotypes = PhenotypeDataset()
        phenotypes.load_from_csv(args.phenotypes)
        
        dataset = ProtoPhenDataset.from_data(
            proteins=library,
            phenotypes=phenotypes,
        )
        train_dataset, val_dataset, test_dataset = dataset.split(seed=args.seed)
    else:
        logger.error("Must provide either --protein-embeddings and --phenotypes, or --protein-library")
        sys.exit(1)
    
    logger.info(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Get dimensions from data
    embedding_dim = train_dataset.embedding_dim
    phenotype_dim = train_dataset.phenotype_dims.get("cell_painting", args.cell_painting_dim)
    
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Phenotype dimension: {phenotype_dim}")
    
    # Create data loaders
    loaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Create model
    model_config = ProToPhenConfig(
        protein_embedding_dim=embedding_dim,
        encoder_hidden_dims=args.hidden_dims,
        cell_painting_dim=phenotype_dim,
        predict_viability=False,
    )
    model = ProToPhenModel(model_config)
    
    logger.info(f"Model: {model}")
    logger.info(f"Parameters: {model.n_parameters:,}")
    
    # Create trainer config
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        use_amp=not args.no_amp,
        device=args.device,
        seed=args.seed,
        tasks=["cell_painting"],
    )
    
    # Setup callbacks
    callbacks = [
        LoggingCallback(log_every_n_steps=10, log_file=output_dir / "training.log"),
        ProgressCallback(),
        CheckpointCallback(
            checkpoint_dir=output_dir / "checkpoints",
            save_best=True,
            monitor="val_loss",
            mode="min",
            save_every_n_epochs=10,
        ),
    ]
    
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                mode="min",
            )
        )
    
    if args.tensorboard:
        callbacks.append(
            TensorBoardCallback(log_dir=output_dir / "tensorboard")
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=trainer_config,
        callbacks=callbacks,
    )
    
    # Train
    history = trainer.train(resume_from=args.resume)
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(loaders["test"])
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final results
    import json
    
    results = {
        "config": trainer_config.__dict__,
        "model_config": model_config.__dict__,
        "history": {
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "best_val_loss": history["best_val_loss"],
        },
        "test_metrics": test_metrics,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()