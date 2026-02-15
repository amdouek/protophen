#!/usr/bin/env python
"""
CLI for constructing a curated JUMP-CP pre-training dataset.

Usage examples::

    # Quick test (5 plates, ORF only)
    python scripts/curate_pretraining.py \\
        --perturbation orf --max-plates 5 --output-dir data/processed/pretraining

    # Full ORF curation with diversity sampling
    python scripts/curate_pretraining.py \\
        --perturbation orf \\
        --sampling diversity --target-genes 5000 \\
        --output-dir data/processed/pretraining

    # ORF + CRISPR combined
    python scripts/curate_pretraining.py \\
        --perturbation orf crispr \\
        --output-dir data/processed/pretraining \\
        --name pretraining_orf_crispr_v1

    # Using custom config
    python scripts/curate_pretraining.py \\
        --jumpcp-config configs/jumpcp.yaml \\
        --output-dir data/processed/pretraining
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protophen.data.jumpcp.access import JUMPCPConfig
from protophen.data.jumpcp.curation import CurationConfig, DataCurator
from protophen.utils.logging import logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate a JUMP-CP pre-training dataset for ProToPhen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config
    parser.add_argument(
        "--jumpcp-config",
        type=str,
        default=None,
        help="Path to jumpcp.yaml.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/pretraining",
        help="Output directory for curated dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pretraining_v1",
        help="Name for the curated dataset.",
    )

    # Perturbation
    parser.add_argument(
        "--perturbation",
        nargs="+",
        default=["orf"],
        choices=["orf", "crispr", "compound"],
        help="Perturbation types to include.",
    )

    # QC
    parser.add_argument(
        "--min-replicates",
        type=int,
        default=2,
        help="Minimum replicates per gene.",
    )
    parser.add_argument(
        "--min-cell-count",
        type=int,
        default=50,
        help="Minimum cell count per well.",
    )

    # Normalisation
    parser.add_argument(
        "--normalisation",
        type=str,
        default="robust_mad",
        choices=["robust_mad", "zscore", "minmax", "none"],
        help="Normalisation method.",
    )
    parser.add_argument(
        "--no-batch-correction",
        action="store_true",
        help="Disable batch correction.",
    )

    # Aggregation
    parser.add_argument(
        "--aggregation",
        type=str,
        default="median",
        choices=["mean", "median"],
        help="Well-to-treatment aggregation method.",
    )

    # Sampling
    parser.add_argument(
        "--sampling",
        type=str,
        default="all",
        choices=["all", "diversity", "coverage"],
        help="Gene sampling strategy.",
    )
    parser.add_argument(
        "--target-genes",
        type=int,
        default=None,
        help="Target number of genes for sampling.",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Hard cap on number of genes.",
    )

    # Size limit (for testing)
    parser.add_argument(
        "--max-plates",
        type=int,
        default=None,
        help="Maximum plates to load (for testing).",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Load JUMP-CP access config
    jumpcp_config = None
    if args.jumpcp_config and Path(args.jumpcp_config).exists():
        jumpcp_config = JUMPCPConfig.from_yaml(args.jumpcp_config)

    # Build curation config
    curation_config = CurationConfig(
        perturbation_types=args.perturbation,
        min_replicates=args.min_replicates,
        max_genes=args.max_genes,
        min_cell_count=args.min_cell_count,
        normalisation_method=args.normalisation,
        batch_correction=not args.no_batch_correction,
        aggregation_method=args.aggregation,
        sampling_strategy=args.sampling,
        target_n_genes=args.target_genes,
        max_plates=args.max_plates,
        output_name=args.name,
    )

    logger.info("=" * 60)
    logger.info("JUMP-CP Pre-training Data Curation")
    logger.info("=" * 60)
    logger.info(f"Perturbation types: {curation_config.perturbation_types}")
    logger.info(f"Sampling strategy:  {curation_config.sampling_strategy}")
    logger.info(f"Output directory:   {args.output_dir}")
    logger.info(f"Dataset name:       {args.name}")
    logger.info("=" * 60)

    # Run curation
    curator = DataCurator(
        config=curation_config,
        jumpcp_config=jumpcp_config,
    )

    curated_df = curator.build_pretraining_set()

    # Save
    paths = curator.save(curated_df, args.output_dir)

    # Print report
    curator.print_report()

    print(f"\nOutput files:")
    for name, path in paths.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {name}: {path} ({size_mb:.1f} MB)")

    # Convert to PhenotypeDataset as a validation step
    dataset = curator.to_phenotype_dataset(curated_df)
    summary = dataset.summary()
    print(f"\nPhenotypeDataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()