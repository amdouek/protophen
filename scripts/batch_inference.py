#!/usr/bin/env python
"""
Batch inference CLI for ProToPhen.

Process a FASTA file or CSV of protein sequences and write predictions
to a Parquet / CSV output file.

Usage::

    # From FASTA
    python scripts/batch_inference.py \\
        --input proteins.fasta \\
        --checkpoint checkpoints/best.pt \\
        --output predictions.parquet

    # From CSV (expects a 'sequence' column)
    python scripts/batch_inference.py \\
        --input proteins.csv \\
        --checkpoint checkpoints/best.pt \\
        --output predictions.csv \\
        --uncertainty

    # With intermediate checkpointing (resumes on crash)
    python scripts/batch_inference.py \\
        --input proteins.fasta \\
        --checkpoint checkpoints/best.pt \\
        --output predictions.parquet \\
        --resume-dir ./batch_progress

    # With deployment config
    python scripts/batch_inference.py \\
        --input proteins.fasta \\
        --checkpoint checkpoints/best.pt \\
        --output predictions.parquet \\
        --config configs/deployment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch phenotype predictions with ProToPhen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", required=True, type=str,
        help="Input file: FASTA (.fasta/.fa) or CSV (.csv) with a 'sequence' column.",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output file (.parquet or .csv).",
    )

    # Options
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to deployment.yaml (optional).",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Tasks to predict (default: all).",
    )
    parser.add_argument(
        "--uncertainty", action="store_true",
        help="Include MC-Dropout uncertainty estimates.",
    )
    parser.add_argument(
        "--mc-samples", type=int, default=20,
        help="Number of MC-Dropout forward passes.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Proteins per processing chunk.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (cuda, cpu, mps).",
    )
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help="Directory for intermediate checkpointing (enables resume).",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level override.",
    )

    return parser.parse_args()


# =========================================================================
# Input Loaders
# =========================================================================

def load_sequences(path: str) -> pd.DataFrame:
    """
    Load protein sequences from FASTA or CSV.

    Returns a DataFrame with at least columns: ``name``, ``sequence``.
    """
    path_obj = Path(path)

    if path_obj.suffix in (".fasta", ".fa", ".faa"):
        from Bio import SeqIO

        records = []
        for rec in SeqIO.parse(path_obj, "fasta"):
            records.append({"name": rec.id, "sequence": str(rec.seq)})
        if not records:
            raise ValueError(f"No sequences found in {path_obj}")
        return pd.DataFrame(records)

    elif path_obj.suffix == ".csv":
        df = pd.read_csv(path_obj)
        if "sequence" not in df.columns:
            raise ValueError("CSV must contain a 'sequence' column.")
        if "name" not in df.columns:
            df["name"] = [f"protein_{i}" for i in range(len(df))]
        return df

    else:
        raise ValueError(f"Unsupported input format: {path_obj.suffix}")


# =========================================================================
# Resume / Checkpointing
# =========================================================================

class BatchCheckpointer:
    """Simple line-level progress tracker for resumable batch runs."""

    def __init__(self, resume_dir: Optional[str] = None):
        self._dir = Path(resume_dir) if resume_dir else None
        self._completed: set[str] = set()
        self._results_buffer: List[dict] = []

        if self._dir is not None:
            from protophen.utils.io import ensure_dir

            ensure_dir(self._dir)
            progress_file = self._dir / "completed.json"
            if progress_file.exists():
                with open(progress_file) as f:
                    self._completed = set(json.load(f))

    @property
    def n_completed(self) -> int:
        return len(self._completed)

    def is_done(self, name: str) -> bool:
        return name in self._completed

    def mark_done(self, name: str, result: dict) -> None:
        self._completed.add(name)
        self._results_buffer.append(result)

    def flush(self) -> None:
        """Persist progress to disk."""
        if self._dir is None:
            return
        with open(self._dir / "completed.json", "w") as f:
            json.dump(sorted(self._completed), f)
        # Also write partial results
        if self._results_buffer:
            part_path = self._dir / "partial_results.jsonl"
            with open(part_path, "a") as f:
                for r in self._results_buffer:
                    f.write(json.dumps(r, default=str) + "\n")
            self._results_buffer.clear()


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()

    from protophen.serving.pipeline import InferencePipeline, PipelineConfig
    from protophen.utils.io import ensure_dir
    from protophen.utils.logging import logger, setup_logging

    # ---- Configure logging -------------------------------------------------
    log_level = "INFO"
    log_file: str | None = None

    if args.config is not None:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "logging" in cfg:
            log_level = cfg["logging"].get("level", log_level)
            log_file = cfg["logging"].get("log_file", log_file)

    if args.log_level is not None:
        log_level = args.log_level

    setup_logging(level=log_level, log_file=log_file)

    # ---- Load pipeline config from YAML ------------------------------------
    pipeline_kwargs: dict = {}
    if args.config is not None:
        if "pipeline" in cfg:
            pipeline_kwargs.update(cfg["pipeline"])

    if args.device is not None:
        pipeline_kwargs["device"] = args.device
    pipeline_kwargs["max_batch_size"] = args.batch_size

    # ---- Load sequences ----------------------------------------------------
    logger.info(f"Loading sequences from {args.input}")
    seq_df = load_sequences(args.input)
    logger.info(f"Loaded {len(seq_df)} sequences")

    # ---- Build pipeline ----------------------------------------------------
    config = PipelineConfig(**pipeline_kwargs)
    pipeline = InferencePipeline(checkpoint_path=args.checkpoint, config=config)

    # ---- Resume support ----------------------------------------------------
    ckpt = BatchCheckpointer(args.resume_dir)
    if ckpt.n_completed > 0:
        logger.info(f"Resuming: {ckpt.n_completed} already completed")

    # ---- Ensure output directory exists ------------------------------------
    out_path = Path(args.output)
    ensure_dir(out_path.parent)

    # ---- Run inference -----------------------------------------------------
    all_results: List[dict] = []
    n_errors = 0
    t_start = time.perf_counter()

    for idx, row in seq_df.iterrows():
        name = row["name"]
        sequence = row["sequence"]

        if ckpt.is_done(name):
            continue

        try:
            resp = pipeline.predict(
                sequence=sequence,
                tasks=args.tasks,
                return_uncertainty=args.uncertainty,
                n_mc_samples=args.mc_samples,
                protein_name=name,
            )

            record: dict = {
                "name": name,
                "sequence": sequence,
                "protein_hash": resp.protein_hash,
                "sequence_length": resp.sequence_length,
                "model_version": resp.model_version,
                "inference_time_ms": resp.inference_time_ms,
            }

            # Flatten predictions
            for tp in resp.predictions:
                if len(tp.values) == 1:
                    record[tp.task_name] = tp.values[0]
                else:
                    for i, v in enumerate(tp.values):
                        record[f"{tp.task_name}_{i}"] = v

            # Flatten uncertainty
            if resp.uncertainty:
                for uo in resp.uncertainty:
                    for i, v in enumerate(uo.std):
                        record[f"{uo.task_name}_std_{i}"] = v

            all_results.append(record)
            ckpt.mark_done(name, record)

        except Exception as exc:
            n_errors += 1
            logger.error(f"Failed for '{name}': {exc}")
            error_record = {
                "name": name,
                "sequence": sequence,
                "error": str(exc),
            }
            all_results.append(error_record)
            ckpt.mark_done(name, error_record)

        # Periodic flush and progress log
        processed = idx + 1
        if processed % 100 == 0:
            ckpt.flush()
            elapsed_so_far = time.perf_counter() - t_start
            rate = processed / elapsed_so_far if elapsed_so_far > 0 else 0
            logger.info(
                f"Processed {processed}/{len(seq_df)} "
                f"({rate:.1f} proteins/s, {n_errors} errors)"
            )

    ckpt.flush()
    elapsed = time.perf_counter() - t_start

    # ---- Write output ------------------------------------------------------
    out_df = pd.DataFrame(all_results)

    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False)
    else:
        # Default to parquet
        out_path = out_path.with_suffix(".parquet")
        out_df.to_parquet(out_path, index=False)

    logger.info(
        f"Batch inference complete: {len(all_results)} proteins in {elapsed:.1f}s "
        f"({len(all_results) / elapsed:.1f} proteins/s), "
        f"{n_errors} errors, output → {out_path}"
    )


if __name__ == "__main__":
    main()