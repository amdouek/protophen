"""
Phenotype processing modules for ProToPhen.

This package provides tools for processing Cell Painting and other
phenotypic data:
- Loading and processing CellProfiler output
- Plate-level normalisation
- Feature selection
- Batch effect correction
- Dimensionality reduction
"""

from protophen.phenotype.cellpainting import (
    CellPaintingProcessor,
    CellPaintingConfig,
    load_cell_painting_data,
)
from protophen.phenotype.normalisation import (
    Normaliser,
    BatchCorrector,
    normalise_plate,
    robust_mad_normalise,
)
from protophen.phenotype.embedding import (
    PhenotypeEmbedder,
    reduce_dimensions,
)

__all__ = [
    # Cell Painting
    "CellPaintingProcessor",
    "CellPaintingConfig",
    "load_cell_painting_data",
    # Normalisation
    "Normaliser",
    "BatchCorrector",
    "normalise_plate",
    "robust_mad_normalise",
    # Embedding
    "PhenotypeEmbedder",
    "reduce_dimensions",
]