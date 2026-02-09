"""
Neural network models for ProToPhen.

This package provides the model architecture for predicting cellular
phenotypes from protein embeddings.

Main components:
- ProteinEncoder: Encodes protein embeddings
- PhenotypeDecoder: Predicts phenotypic features
- ProToPhenModel: Complete model combining encoder and decoder
"""

from protophen.models.encoders import (
    ProteinEncoder,
    ProteinEncoderConfig,
    MLPBlock,
)
from protophen.models.decoders import (
    PhenotypeDecoder,
    CellPaintingHead,
    ViabilityHead,
    TranscriptomicsHead,
    MultiTaskHead,
)
from protophen.models.protophen import (
    ProToPhenModel,
    ProToPhenConfig,
)
from protophen.models.losses import (
    MultiTaskLoss,
    UncertaintyWeightedLoss,
    CellPaintingLoss,
    CorrelationLoss,
    CombinedLoss,
)

__all__ = [
    # Encoders
    "ProteinEncoder",
    "ProteinEncoderConfig",
    "MLPBlock",
    # Decoders
    "PhenotypeDecoder",
    "CellPaintingHead",
    "ViabilityHead",
    "TranscriptomicsHead",
    "MultiTaskHead",
    # Main model
    "ProToPhenModel",
    "ProToPhenConfig",
    # Losses
    "MultiTaskLoss",
    "UncertaintyWeightedLoss",
    "CellPaintingLoss",
    "CorrelationLoss",
    "CombinedLoss",
]