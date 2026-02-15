#!/usr/bin/env python
"""
CLI for downloading and caching JUMP-CP data.

Usage examples::

    # Check connectivity
    python scripts/download_jumpcp.py --check

    # Download all ORF metadata
    python scripts/download_jumpcp.py --metadata orf plate well

    # Download profiles for a small test set (5 plates)
    python scripts/download_jumpcp.py --profiles --perturbation orf --max-plates 5

    # Download gene sequences from UniProt
    python scripts/download_jumpcp.py --sequences --perturbation orf

    # Full download with custom cache directory
    python scripts/download_jumpcp.py --metadata orf crispr plate well \\
        --profiles --perturbation orf --cache-dir /data/jumpcp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure protophen is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protophen.data.jumpcp.access import JUMPCPAccess, JUMPCPConfig
from protophen.data.jumpcp.metadata import JUMPCPMetadata
from protophen.data.jumpcp.profiles import ProfileLoader
from protophen.utils.logging import logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache JUMP-CP data for ProToPhen pre-training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to jumpcp.yaml config file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache directory.",
    )

    # Action flags
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check S3 and HTTPS connectivity, then exit.",
    )
    parser.add_argument(
        "--metadata",
        nargs="*",
        default=None,
        metavar="TABLE",
        help="Metadata tables to download (e.g. orf plate well crispr compound).",
    )
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="Download plate profiles.",
    )
    parser.add_argument(
        "--sequences",
        action="store_true",
        help="Fetch protein sequences from UniProt for resolved genes.",
    )

    # Filtering
    parser.add_argument(
        "--perturbation",
        type=str,
        default="orf",
        choices=["orf", "crispr", "compound"],
        help="Perturbation type for profile downloads.",
    )
    parser.add_argument(
        "--max-plates",
        type=int,
        default=None,
        help="Maximum number of plates to download (for testing).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download even if cached.",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Load config
    if args.config and Path(args.config).exists():
        config = JUMPCPConfig.from_yaml(args.config)
    else:
        config = JUMPCPConfig()

    if args.cache_dir:
        config.cache_dir = args.cache_dir

    access = JUMPCPAccess(config=config)
    metadata = JUMPCPMetadata(access=access)

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------
    if args.check:
        result = access.check_connectivity()
        print(f"\nConnectivity check:")
        print(f"  S3:    {'✓' if result['s3'] else '✗'}")
        print(f"  HTTPS: {'✓' if result['https'] else '✗'}")
        print(f"\nCache: {access.cache.get_cache_info()}")
        return

    # ------------------------------------------------------------------
    # Metadata download
    # ------------------------------------------------------------------
    if args.metadata is not None:
        tables = args.metadata if args.metadata else ["orf", "plate", "well"]
        for table_name in tables:
            logger.info(f"Downloading metadata table: {table_name}")
            try:
                df = access.fetch_metadata_table(
                    table_name, force_refresh=args.force_refresh
                )
                logger.info(
                    f"  → {table_name}: {len(df)} rows, "
                    f"{len(df.columns)} columns"
                )
            except Exception as exc:
                logger.error(f"  → Failed to download '{table_name}': {exc}")

    # ------------------------------------------------------------------
    # Profile download
    # ------------------------------------------------------------------
    if args.profiles:
        logger.info(
            f"Downloading profiles for perturbation type: {args.perturbation}"
        )
        plate_map = metadata.get_plate_batch_source_map(args.perturbation)
        total = len(plate_map)
        max_plates = args.max_plates or total
        logger.info(f"Downloading up to {max_plates} of {total} plates")

        loader = ProfileLoader(access=access, metadata=metadata)
        loaded = 0

        for idx, (_, row) in enumerate(plate_map.iterrows()):
            if idx >= max_plates:
                break
            try:
                df = loader.load_plate_profiles(
                    source=row["source"],
                    batch=row["batch"],
                    plate=row["plate"],
                    force_refresh=args.force_refresh,
                )
                loaded += 1
                logger.info(
                    f"  [{loaded}/{max_plates}] {row['plate']}: "
                    f"{len(df)} wells"
                )
            except Exception as exc:
                logger.error(f"  Failed to load '{row['plate']}': {exc}")

        logger.info(f"Downloaded {loaded} plates")

    # ------------------------------------------------------------------
    # Sequence download
    # ------------------------------------------------------------------
    if args.sequences:
        logger.info(
            f"Fetching UniProt sequences for {args.perturbation} genes"
        )
        if args.perturbation == "orf":
            genes = metadata.get_orf_genes()
        elif args.perturbation == "crispr":
            genes = metadata.get_crispr_genes()
        else:
            logger.error("Sequence fetching only supported for orf/crispr")
            return

        logger.info(f"Resolving {len(genes)} gene symbols via UniProt")
        seq_map = access.fetch_uniprot_sequences(genes)
        logger.info(
            f"Resolved {len(seq_map)}/{len(genes)} genes to sequences"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    cache_info = access.cache.get_cache_info()
    print(f"\nCache summary:")
    print(f"  Directory:  {cache_info['cache_dir']}")
    print(f"  Entries:    {cache_info['total_entries']}")
    print(f"  Total size: {cache_info['total_size_gb']:.2f} GB")
    print(f"  Profiles:   {cache_info['n_profiles']}")
    print(f"  Metadata:   {cache_info['n_metadata']}")


if __name__ == "__main__":
    main()