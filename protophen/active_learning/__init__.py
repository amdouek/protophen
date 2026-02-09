"""
Active learning module for ProToPhen.

This package provides tools for intelligent experiment selection using
uncertainty quantification and acquisition functions:
- Uncertainty estimation (MC Dropout, Deep Ensembles)
- Acquisition functions (uncertainty sampling, expected improvement, diversity)
- Experiment selection and ranking
"""

from protophen.active_learning.uncertainty import (
    UncertaintyEstimator,
    MCDropoutEstimator,
    EnsembleEstimator,
    UncertaintyType,
    estimate_uncertainty,
)
from protophen.active_learning.acquisition import (
    AcquisitionFunction,
    UncertaintySampling,
    ExpectedImprovement,
    DiversitySampling,
    HybridAcquisition,
    BatchAcquisition,
    compute_acquisition_scores,
)
from protophen.active_learning.selection import (
    ExperimentSelector,
    SelectionConfig,
    SelectionResult,
    select_next_experiments,
)

__all__ = [
    # Uncertainty
    "UncertaintyEstimator",
    "MCDropoutEstimator",
    "EnsembleEstimator",
    "UncertaintyType",
    "estimate_uncertainty",
    # Acquisition
    "AcquisitionFunction",
    "UncertaintySampling",
    "ExpectedImprovement",
    "DiversitySampling",
    "HybridAcquisition",
    "BatchAcquisition",
    "compute_acquisition_scores",
    # Selection
    "ExperimentSelector",
    "SelectionConfig",
    "SelectionResult",
    "select_next_experiments",
]