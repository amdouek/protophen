"""
Data structures and loading utilities for ProToPhen.
"""

from protophen.data.protein import (
    AMINO_ACIDS,
    Protein,
    ProteinLibrary,
    compute_sequence_hash,
    validate_sequence,
)
from protophen.data.phenotype import (
    Phenotype,
    PhenotypeDataset,
)
from protophen.data.dataset import (
    ProtoPhenDataset,
    ProtoPhenSample,
    ProteinInferenceDataset,
    DatasetConfig,
)
from protophen.data.loaders import (
    create_dataloader,
    create_dataloaders,
    create_balanced_sampler,
    split_by_protein,
    split_by_plate,
    protophen_collate_fn,
)

__all__ = [
    # Protein
    "Protein",
    "ProteinLibrary",
    "AMINO_ACIDS",
    "validate_sequence",
    "compute_sequence_hash",
    # Phenotype
    "Phenotype",
    "PhenotypeDataset",
    # Dataset
    "ProtoPhenDataset",
    "ProtoPhenSample",
    "ProteinInferenceDataset",
    "DatasetConfig",
    # Loaders
    "create_dataloader",
    "create_dataloaders",
    "create_balanced_sampler",
    "split_by_protein",
    "split_by_plate",
    "protophen_collate_fn",
]