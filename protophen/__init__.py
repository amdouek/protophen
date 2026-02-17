"""
ProToPhen: A foundation model for predicting cellular responses to de novo-designed proteins.

This package provides tools for:
- Extracting protein embeddings (ESM-2, physicochemical features, structure)
- Processing Cell Painting phenotypic data
- Training protein-to-phenotype prediction models
- Active learning for experiment selection
- Serving infrastructure for deployment (optional, requires ``pip install "protophen[serving]"``)
"""

__version__ = "0.1.0"
__author__ = "Alon M Douek"

from protophen.data.protein import Protein, ProteinLibrary
from protophen.embeddings.esm import ESMEmbedder, get_esm_embedder, list_esm_models
from protophen.embeddings.physicochemical import (
    PhysicochemicalCalculator,
    get_physicochemical_calculator,
    calculate_all_features,
)
from protophen.embeddings.esmfold import (
    ESMFoldPredictor,
    check_esmfold_available,
)
from protophen.utils.config import ProtoPhenConfig, load_config
from protophen.utils.logging import logger, setup_logging

__all__ = [
    # Data
    "Protein",
    "ProteinLibrary",
    # Embeddings - ESM-2
    "ESMEmbedder",
    "get_esm_embedder",
    "list_esm_models",
    # Embeddings - Physicochemical
    "PhysicochemicalCalculator",
    "get_physicochemical_calculator",
    "calculate_all_features",
    # Embeddings - Structure
    "ESMFoldPredictor",
    "check_esmfold_available",
    # Config
    "ProtoPhenConfig",
    "load_config",
    # Logging
    "logger",
    "setup_logging",
    # Version
    "__version__",
]

# =========================================================================
# Lazy imports for optional serving subpackage
# =========================================================================
# Serving components require ``pip install protophen[serving]`` (FastAPI, uvicorn, etc.).
# They are exposed at the top level for convenience but we guard the import so the core package works without serving extras.

def __getattr__(name: str):
    """Lazy-load serving exports on first access."""
    _SERVING_EXPORTS = {
        "InferencePipeline",
        "PipelineConfig",
        "ModelRegistry",
        "RegistryConfig",
        "ModelVersion",
        "PredictionMonitor",
        "MonitoringConfig",
        "DriftDetector",
        "PredictionQualityTracker",
        "create_app",
    }

    if name in _SERVING_EXPORTS:
        try:
            import protophen.serving as _serving
            return getattr(_serving, name)
        except ImportError as exc:
            raise ImportError(
                f"'{name}' requires the serving extras. "
                f"Install with: pip install 'protophen[serving]'"
            ) from exc

    raise AttributeError(f"module 'protophen' has no attribute '{name}'")