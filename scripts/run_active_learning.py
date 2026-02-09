#!/usr/bin/env python
"""
Active learning script for ProToPhen.

Runs iterative active learning to select informative proteins for
experimental characterization.

Usage:
    python scripts/run_active_learning.py \
        --model checkpoints/best_model.pt \
        --pool-data data/candidate_proteins.json \
        --n-select 10 \
        --method hybrid \
        --output-dir outputs/active_learning
        
    # Using installed CLI entry point
    protophen-al --model checkpoints/best.pt --pool-data candidates.json --n-select 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch

from protophen.active_learning.selection import (
    ExperimentSelector,
    SelectionConfig,
    SelectionResult,
    select_next_experiments,
    rank_by_uncertainty,
)
from protophen.active_learning.uncertainty import (
    UncertaintyType,
    estimate_uncertainty,
)
from protophen.data.dataset import ProteinInferenceDataset
from protophen.data.loaders import create_dataloader
from protophen.data.protein import ProteinLibrary
from protophen.models.protophen import ProToPhenModel, ProToPhenConfig
from protophen.utils.io import load_embeddings
from protophen.utils.logging import logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run active learning for experiment selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config (if not in checkpoint)",
    )
    
    # Data arguments
    parser.add_argument(
        "--pool-data",
        type=str,
        required=True,
        help="Path to candidate protein data (JSON library or embeddings)",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to pre-computed embeddings (optional)",
    )
    parser.add_argument(
        "--exclude-file",
        type=str,
        default=None,
        help="File with protein IDs to exclude (one per line)",
    )
    
    # Selection arguments
    parser.add_argument(
        "--n-select",
        type=int,
        default=10,
        help="Number of proteins to select",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["uncertainty", "ei", "diversity", "hybrid"],
        help="Acquisition method",
    )
    parser.add_argument(
        "--uncertainty-weight",
        type=float,
        default=0.7,
        help="Weight for uncertainty in hybrid method",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.3,
        help="Weight for diversity in hybrid method",
    )
    
    # Uncertainty arguments
    parser.add_argument(
        "--uncertainty-method",
        type=str,
        default="mc_dropout",
        choices=["mc_dropout", "ensemble"],
        help="Uncertainty estimation method",
    )
    parser.add_argument(
        "--n-mc-samples",
        type=int,
        default=20,
        help="Number of MC dropout samples",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/active_learning",
        help="Output directory",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv", "both"],
        help="Output format for results",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for computation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
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
        "--save-all-rankings",
        action="store_true",
        help="Save rankings for all proteins (not just selected)",
    )
    
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    device: str,
    config_path: Optional[str] = None,
) -> ProToPhenModel:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    if "config" in checkpoint:
        model_config = ProToPhenConfig(**checkpoint["config"])
    elif config_path is not None:
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        model_config = ProToPhenConfig(**config_dict.get("model", {}))
    else:
        logger.warning("No model config found, using defaults")
        model_config = ProToPhenConfig()
    
    # Create and load model
    model = ProToPhenModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model parameters: {model.n_parameters:,}")
    
    return model


def load_pool_data(
    pool_path: str,
    embeddings_path: Optional[str] = None,
) -> ProteinInferenceDataset:
    """Load candidate protein pool."""
    pool_path = Path(pool_path)
    
    if pool_path.suffix == ".json":
        # Load from protein library
        library = ProteinLibrary.from_json(pool_path)
        logger.info(f"Loaded {len(library)} proteins from library")
        
        # Check if embeddings are in the library
        embedding_key = None
        if library[0].has_embeddings:
            available_keys = library[0].embedding_types
            for key in ["fused", "esm2", "physicochemical"]:
                if key in available_keys:
                    embedding_key = key
                    break
        
        if embedding_key is None and embeddings_path is None:
            raise ValueError(
                "No embeddings found in library. Please provide --embeddings"
            )
        
        dataset = ProteinInferenceDataset.from_library(
            proteins=library,
            embedding_key=embedding_key or "fused",
        )
    
    elif pool_path.suffix in [".pkl", ".gz"]:
        # Load embeddings directly
        embeddings_dict = load_embeddings(pool_path)
        
        protein_ids = list(embeddings_dict.keys())
        embeddings = np.stack([embeddings_dict[pid] for pid in protein_ids])
        
        dataset = ProteinInferenceDataset(
            protein_embeddings=embeddings,
            protein_ids=protein_ids,
            protein_names=protein_ids,
        )
        logger.info(f"Loaded {len(protein_ids)} embeddings")
    
    else:
        raise ValueError(f"Unsupported pool data format: {pool_path.suffix}")
    
    # Override with external embeddings if provided
    if embeddings_path is not None:
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings_dict = load_embeddings(embeddings_path)
        
        # Match with existing protein IDs
        matched_embeddings = []
        matched_ids = []
        matched_names = []
        
        for i, pid in enumerate(dataset.protein_ids):
            if pid in embeddings_dict:
                matched_embeddings.append(embeddings_dict[pid])
                matched_ids.append(pid)
                matched_names.append(dataset.protein_names[i])
        
        if len(matched_embeddings) < len(dataset):
            logger.warning(
                f"Only {len(matched_embeddings)}/{len(dataset)} proteins "
                f"have matching embeddings"
            )
        
        dataset = ProteinInferenceDataset(
            protein_embeddings=np.stack(matched_embeddings),
            protein_ids=matched_ids,
            protein_names=matched_names,
        )
    
    logger.info(f"Pool dataset: {len(dataset)} proteins, embedding dim={dataset.embedding_dim}")
    
    return dataset


def load_exclude_list(exclude_file: Optional[str]) -> List[str]:
    """Load list of protein IDs to exclude."""
    if exclude_file is None:
        return []
    
    exclude_path = Path(exclude_file)
    if not exclude_path.exists():
        logger.warning(f"Exclude file not found: {exclude_file}")
        return []
    
    with open(exclude_path) as f:
        exclude_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(exclude_ids)} IDs to exclude")
    return exclude_ids


def save_results(
    result: SelectionResult,
    output_dir: Path,
    output_format: str,
    save_all_rankings: bool = False,
) -> None:
    """Save selection results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get selected proteins info
    selected_proteins = result.get_selected_proteins()
    
    # Save as JSON
    if output_format in ["json", "both"]:
        json_path = output_dir / "selected_proteins.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "selected_proteins": selected_proteins,
                    "config": {
                        "n_select": result.config.n_select,
                        "method": result.config.acquisition_method,
                        "uncertainty_method": result.config.uncertainty_method,
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Saved JSON results to {json_path}")
    
    # Save as CSV
    if output_format in ["csv", "both"]:
        import pandas as pd
        
        csv_path = output_dir / "selected_proteins.csv"
        df = pd.DataFrame(selected_proteins)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV results to {csv_path}")
    
    # Save all rankings if requested
    if save_all_rankings:
        rankings_path = output_dir / "all_rankings.csv"
        
        import pandas as pd
        
        all_data = []
        for idx in result.all_indices_ranked:
            sample_id = (
                result.uncertainty_estimates.sample_ids[idx]
                if result.uncertainty_estimates.sample_ids
                else f"sample_{idx}"
            )
            
            all_data.append({
                "rank": len(all_data) + 1,
                "index": int(idx),
                "id": sample_id,
                "acquisition_score": float(result.all_scores[idx]),
                "uncertainty": float(
                    result.uncertainty_estimates.get_uncertainty(
                        result.config.uncertainty_type,
                        reduction="mean",
                    )[idx]
                ),
            })
        
        df = pd.DataFrame(all_data)
        df.to_csv(rankings_path, index=False)
        logger.info(f"Saved all rankings to {rankings_path}")
    
    # Save simple list of selected IDs
    ids_path = output_dir / "selected_ids.txt"
    with open(ids_path, "w") as f:
        for protein_id in result.selected_ids:
            f.write(f"{protein_id}\n")
    logger.info(f"Saved selected IDs to {ids_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.model, device, args.model_config)
    
    # Load pool data
    logger.info("Loading candidate pool...")
    pool_dataset = load_pool_data(args.pool_data, args.embeddings)
    
    # Load exclude list
    exclude_ids = load_exclude_list(args.exclude_file)
    
    # Create data loader
    pool_loader = create_dataloader(
        pool_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Create selection config
    config = SelectionConfig(
        n_select=args.n_select,
        uncertainty_method=args.uncertainty_method,
        n_mc_samples=args.n_mc_samples,
        acquisition_method=args.method,
        uncertainty_weight=args.uncertainty_weight,
        diversity_weight=args.diversity_weight,
        exclude_ids=exclude_ids,
    )
    
    # Run selection
    logger.info(f"Selecting {args.n_select} proteins using {args.method} acquisition...")
    
    selector = ExperimentSelector(
        model=model,
        config=config,
        device=device,
    )
    
    result = selector.select(
        dataloader=pool_loader,
        show_progress=True,
    )
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("SELECTION RESULTS")
    logger.info("=" * 60)
    
    for protein in result.get_selected_proteins():
        logger.info(
            f"  {protein['rank']:2d}. {protein['id']}: "
            f"score={protein['acquisition_score']:.4f}, "
            f"uncertainty={protein['uncertainty']:.4f}"
        )
    
    logger.info("=" * 60)
    
    # Save results
    save_results(
        result=result,
        output_dir=output_dir,
        output_format=args.output_format,
        save_all_rankings=args.save_all_rankings,
    )
    
    # Save configuration
    config_path = output_dir / "selection_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "n_select": args.n_select,
                "method": args.method,
                "uncertainty_method": args.uncertainty_method,
                "n_mc_samples": args.n_mc_samples,
                "uncertainty_weight": args.uncertainty_weight,
                "diversity_weight": args.diversity_weight,
                "seed": args.seed,
                "model_checkpoint": args.model,
                "pool_data": args.pool_data,
            },
            f,
            indent=2,
        )
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()