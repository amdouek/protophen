"""
Neural network models for ProToPhen.

This package provides the model architecture for predicting cellular
phenotypes from protein embeddings.

Main components:
- ProteinEncoder: Encodes protein embeddings
- PhenotypeDecoder: Predicts phenotypic features
- ProToPhenModel: Complete model combining encoder and decoder
- PhenotypeAutoencoder: Autoencoder for phenotype space pre-training
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
from protophen.models.autoencoder import (
    PhenotypeAutoencoder,
    PhenotypeAutoencoderConfig,
    AutoencoderDecoderHead,
    AutoencoderLoss,
    NTXentLoss,
    NTXentLossVectorised,
    PretrainingDataset,
    PretrainingConfig,
    Phase1Config,
    Phase2Config,
    save_phase1_checkpoint,
    save_phase2_checkpoint,
    load_autoencoder_from_checkpoint,
    compute_replicate_correlation,
    compute_latent_silhouette,
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
    # Autoencoder
    "PhenotypeAutoencoder",
    "PhenotypeAutoencoderConfig",
    "AutoencoderDecoderHead",
    "AutoencoderLoss",
    "NTXentLoss",
    "NTXentLossVectorised",
    "PretrainingDataset",
    "PretrainingConfig",
    "Phase1Config",
    "Phase2Config",
    "save_phase1_checkpoint",
    "save_phase2_checkpoint",
    "load_autoencoder_from_checkpoint",
    "compute_replicate_correlation",
    "compute_latent_silhouette",
]