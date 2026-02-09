"""
Protein embedding extraction modules for ProToPhen.

This package provides tools for extracting various types of protein embeddings:
- ESM-2 embeddings (protein language model)
- Physicochemical features (sequence-derived)
- Structure-based features (ESMFold predictions)
- Embedding fusion (combining multiple embedding types)
"""

from protophen.embeddings.esm import (
    ESM2_MODELS,
    ESMEmbedder,
    ESMEmbedderConfig,
    get_esm_embedder,
    list_esm_models,
)
from protophen.embeddings.physicochemical import (
    PhysicochemicalCalculator,
    PhysicochemicalConfig,
    calculate_all_features,
    get_physicochemical_calculator,
    # Individual calculation functions
    calculate_aa_composition,
    calculate_dipeptide_composition,
    calculate_molecular_weight,
    calculate_isoelectric_point,
    calculate_gravy,
    calculate_instability_index,
    calculate_aromaticity,
    calculate_aliphatic_index,
    calculate_charge_at_ph,
    calculate_sequence_entropy,
    calculate_sequence_complexity,
    calculate_hydrophobic_moment,
    calculate_secondary_structure_fractions,
)
from protophen.embeddings.esmfold import (
    ESMFoldPredictor,
    ESMFoldConfig,
    predict_structure,
    check_esmfold_available,
)
from protophen.embeddings.fusion import (
    EmbeddingFusion,
    FusionConfig,
    FusionMethod,
    fuse_embeddings,
    get_fusion_module,
    # PyTorch modules
    ConcatFusion,
    WeightedFusion,
    AttentionFusion,
    GatedFusion,
)

__all__ = [
    # ESM-2
    "ESMEmbedder",
    "ESMEmbedderConfig",
    "ESM2_MODELS",
    "get_esm_embedder",
    "list_esm_models",
    # Physicochemical
    "PhysicochemicalCalculator",
    "PhysicochemicalConfig",
    "calculate_all_features",
    "get_physicochemical_calculator",
    # Individual functions
    "calculate_aa_composition",
    "calculate_dipeptide_composition",
    "calculate_molecular_weight",
    "calculate_isoelectric_point",
    "calculate_gravy",
    "calculate_instability_index",
    "calculate_aromaticity",
    "calculate_aliphatic_index",
    "calculate_charge_at_ph",
    "calculate_sequence_entropy",
    "calculate_sequence_complexity",
    "calculate_hydrophobic_moment",
    "calculate_secondary_structure_fractions",
    # ESMFold
    "ESMFoldPredictor",
    "ESMFoldConfig",
    "predict_structure",
    "check_esmfold_available",
    # Fusion
    "EmbeddingFusion",
    "FusionConfig",
    "FusionMethod",
    "fuse_embeddings",
    "get_fusion_module",
    "ConcatFusion",
    "WeightedFusion",
    "AttentionFusion",
    "GatedFusion",
]