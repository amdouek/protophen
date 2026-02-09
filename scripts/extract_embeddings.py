#!/usr/bin/env python
"""
Extract ESM-2 embeddings for a protein library.

Usage:
    python scripts/extract_embeddings.py --input proteins.fasta --output embeddings.json
    python scripts/extract_embeddings.py --input library.json --output library_embedded.json --model esm2_t33_650M_UR50D

Examples:
    # Basic usage with FASTA input
    python scripts/extract_embeddings.py \\
        --input data/proteins.fasta \\
        --output outputs/embeddings.pkl.gz
    
    # With specific model and caching
    python scripts/extract_embeddings.py \\
        --input data/library.json \\
        --output outputs/library_embedded.json \\
        --model esm2_t33_650M_UR50D \\
        --cache-dir cache/esm2 \\
        --batch-size 16
        
    # Using installed CLI entry point
    protophen-embed --input proteins.fasta --output enbeddings.json
"""

import argparse
import sys
from pathlib import Path

from protophen import ProteinLibrary, setup_logging, logger
from protophen.embeddings import ESMEmbedder, list_esm_models
from protophen.utils.io import save_embeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract ESM-2 embeddings for a protein library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file (FASTA or JSON library)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file (JSON library with embeddings, or .pkl.gz for embeddings only)",
    )
    
    # Model settings
    parser.add_argument(
        "--model", "-m",
        default="esm2_t33_650M_UR50D",
        choices=list(list_esm_models().keys()),
        help="ESM-2 model to use (default: esm2_t33_650M_UR50D)",
    )
    parser.add_argument(
        "--pooling", "-p",
        default="mean",
        choices=["mean", "cls", "max", "mean_cls"],
        help="Pooling strategy (default: mean)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Which layer to extract (-1 = last, default: -1)",
    )
    
    # Computation settings
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for embedding extraction (default: 8)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 (half precision) computation",
    )
    
    # Caching
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for caching embeddings",
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO")
    setup_logging(level=log_level)
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Load protein library
    logger.info(f"Loading proteins from {input_path}")
    
    if input_path.suffix.lower() in (".fasta", ".fa", ".faa"):
        library = ProteinLibrary(name=input_path.stem)
        n_loaded = library.add_from_fasta(input_path)
        logger.info(f"Loaded {n_loaded} proteins from FASTA")
    elif input_path.suffix.lower() == ".json":
        library = ProteinLibrary.from_json(input_path)
        logger.info(f"Loaded {len(library)} proteins from JSON")
    else:
        logger.error(f"Unsupported input format: {input_path.suffix}")
        sys.exit(1)
    
    if len(library) == 0:
        logger.error("No proteins loaded from input file")
        sys.exit(1)
    
    # Print library summary
    summary = library.summary()
    logger.info(f"Library summary: {summary['count']} proteins, "
                f"length range: {summary['length_stats']['min']}-{summary['length_stats']['max']}")
    
    # Initialize embedder
    logger.info(f"Initialising ESM-2 embedder: {args.model}")
    
    embedder = ESMEmbedder(
        model_name=args.model,
        layer=args.layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
        device=args.device,
        use_fp16=not args.no_fp16,
        cache_dir=args.cache_dir,
    )
    
    logger.info(f"Embedder: {embedder}")
    
    # Extract embeddings
    logger.info("Extracting embeddings...")
    
    embeddings = embedder.embed_library(
        library,
        embedding_key="esm2",
        show_progress=not args.quiet,
    )
    
    logger.info(f"Extracted embeddings with shape: {embeddings.shape}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == ".json":
        # Save full library with embeddings
        library.to_json(output_path, include_embeddings=True)
        logger.info(f"Saved library with embeddings to {output_path}")
    else:
        # Save embeddings only
        embedding_dict = {p.hash: p.embeddings["esm2"] for p in library}
        save_embeddings(embedding_dict, output_path)
        logger.info(f"Saved embeddings to {output_path}")
    
    # Print final summary
    logger.info("Done!")
    logger.info(f"  Proteins processed: {len(library)}")
    logger.info(f"  Embedding dimension: {embedder.output_dim}")
    logger.info(f"  Output file: {output_path}")


if __name__ == "__main__":
    main()