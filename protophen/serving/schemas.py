"""
Pydantic request/response models for the ProToPhen REST API.

This module defines the data contracts for all API endpoints, ensuring
strict validation on inputs and consistent, well-documented outputs.
All schemas use Pydantic v2 for compatibility with the rest of the project.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Shared / Nested Models
# =============================================================================

class ProteinInput(BaseModel):
    """A single protein input for prediction."""

    model_config = ConfigDict(extra="forbid")

    sequence: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Amino acid sequence (standard 20 letter alphabet).",
    )
    name: Optional[str] = Field(
        None,
        max_length=256,
        description="Optional protein identifier.",
    )
    source: Optional[str] = Field(
        None,
        description="Origin of the protein (e.g., 'de_novo', 'pdb').",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata carried through to the response.",
    )

    @field_validator("sequence")
    @classmethod
    def validate_amino_acids(cls, v: str) -> str:
        """Normalise and validate the amino acid sequence."""
        import re

        v = v.upper().strip()
        v = re.sub(r"[\s\-\.\*]", "", v)
        if not v:
            raise ValueError("Sequence cannot be empty after normalisation.")
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(v) - valid
        if invalid:
            raise ValueError(
                f"Sequence contains invalid characters: {invalid}. "
                f"Allowed: {''.join(sorted(valid))}"
            )
        return v


class TaskPrediction(BaseModel):
    """Prediction output for a single task."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str = Field(..., description="Name of the prediction task.")
    values: List[float] = Field(..., description="Predicted values.")
    dimension: int = Field(..., description="Output dimensionality.")


class UncertaintyOutput(BaseModel):
    """Uncertainty estimates for a single task."""

    task_name: str
    mean: List[float] = Field(..., description="Mean prediction across MC samples.")
    std: List[float] = Field(..., description="Standard deviation (epistemic uncertainty).")
    n_samples: int = Field(..., description="Number of MC-Dropout forward passes.")


# =============================================================================
# Request Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Request body for single-protein prediction."""

    model_config = ConfigDict(extra="forbid")

    protein: ProteinInput
    tasks: Optional[List[str]] = Field(
        None,
        description="Tasks to predict. None = all available tasks.",
    )
    return_latent: bool = Field(
        False,
        description="Whether to include the latent representation.",
    )
    return_uncertainty: bool = Field(
        False,
        description="Whether to return MC-Dropout uncertainty estimates.",
    )
    n_uncertainty_samples: int = Field(
        20,
        ge=2,
        le=200,
        description="Number of MC-Dropout forward passes for uncertainty.",
    )
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use. None = latest.",
    )


class BatchPredictionRequest(BaseModel):
    """Request body for batch protein prediction."""

    model_config = ConfigDict(extra="forbid")

    proteins: List[ProteinInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of proteins to predict (max 1000 per request).",
    )
    tasks: Optional[List[str]] = None
    return_latent: bool = False
    return_uncertainty: bool = False
    n_uncertainty_samples: int = Field(20, ge=2, le=200)
    model_version: Optional[str] = None


class FeedbackRequest(BaseModel):
    """
    Feedback payload for the active-learning loop.

    After wet-lab experiments are run, observed phenotypes can be fed
    back through this endpoint to update the training set and optionally
    trigger re-selection of the next experimental batch.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    protein_id: str = Field(..., description="Identifier of the tested protein.")
    sequence: str = Field(..., description="Amino acid sequence.")
    observed_features: List[float] = Field(
        ...,
        description="Observed Cell Painting feature vector.",
    )
    plate_id: Optional[str] = None
    well_id: Optional[str] = None
    cell_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trigger_reselection: bool = Field(
        False,
        description="If True, trigger active learning re-selection after ingestion.",
    )


# =============================================================================
# Response Models
# =============================================================================

class PredictionResponse(BaseModel):
    """Response for a single-protein prediction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    protein_name: Optional[str] = None
    protein_hash: str
    sequence_length: int
    predictions: List[TaskPrediction]
    uncertainty: Optional[List[UncertaintyOutput]] = None
    latent: Optional[List[float]] = None
    model_version: str
    inference_time_ms: float = Field(
        ..., description="Wall-clock inference time in milliseconds."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictionResponse(BaseModel):
    """Response for batch protein prediction."""

    results: List[PredictionResponse]
    n_proteins: int
    total_inference_time_ms: float
    model_version: str


class FeedbackResponse(BaseModel):
    """Response after feedback ingestion."""

    status: Literal["accepted", "rejected", "error"]
    protein_id: str
    message: str = ""
    reselection_triggered: bool = False
    next_candidates: Optional[List[str]] = Field(
        None,
        description="If reselection was triggered, the next suggested protein IDs.",
    )


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model."""

    model_version: str
    model_name: str = "ProToPhen"
    tasks: Dict[str, int] = Field(
        ..., description="Mapping of task name → output dimension."
    )
    latent_dim: int
    protein_embedding_dim: int
    n_parameters: int
    n_trainable_parameters: int
    encoder_hidden_dims: List[int]
    decoder_hidden_dims: List[int]
    esm_model: str
    fusion_method: str
    device: str
    loaded_at: str = Field(..., description="ISO-8601 timestamp of model load.")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    esm_loaded: bool
    uptime_seconds: float
    version: str
    device: str
    checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Individual component health checks.",
    )