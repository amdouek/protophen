"""
ProToPhen command-line scripts.

This subpackage contains CLI scripts for common ProToPhen workflows.

Entry points (available after `pip install -e .`):
    - protophen-embed: Extract protein embeddings
    - protophen-train: Train ProToPhen models  
    - protophen-al: Run active learning selection

Direct usage:
    python -m protophen.scripts.extract_embeddings --help
    python -m protophen.scripts.train_model --help
    python -m protophen.scripts.run_active_learning --help
"""

__all__ = [
    "extract_embeddings",
    "train_model",
    "run_active_learning",
]