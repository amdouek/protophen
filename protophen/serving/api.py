"""
FastAPI REST service for ProToPhen predictions.

This module defines the API application, mounts all endpoints, and
manages the shared application state (pipeline, monitor, registry).

Endpoints:

    POST /predict              Single-protein prediction
    POST /predict/batch        Batch prediction (up to 1000 proteins)
    POST /feedback             Active-learning feedback ingestion
    GET  /health               Readiness / liveness probe
    GET  /model/info           Model metadata
    GET  /metrics              Monitoring summary (JSON)
    GET  /metrics/prometheus   Prometheus-formatted metrics (if enabled)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from protophen.serving.pipeline import InferencePipeline, PipelineConfig
from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig
from protophen.serving.registry import ModelRegistry, RegistryConfig
from protophen.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from protophen.utils.logging import logger

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def _require_fastapi() -> None:
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the serving API. "
            "Install with: pip install 'protophen[serving]'"
        )


# =============================================================================
# Feedback Store
# =============================================================================

class FeedbackStore:
    """
    In-process feedback store for the active-learning loop.

    In production this would be backed by a database; here we keep
    an in-memory list that can be flushed to disk.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self._entries: List[Dict[str, Any]] = []
        self._persist_dir = persist_dir

    def add(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)
        if self._persist_dir is not None:
            self._flush_to_disk(entry)

    def _flush_to_disk(self, entry: Dict[str, Any]) -> None:
        import json

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        path = self._persist_dir / "feedback.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# =============================================================================
# Application State
# =============================================================================

class _AppState:
    """Mutable container shared across request handlers via ``app.state``."""

    def __init__(self) -> None:
        self.pipeline: Optional[InferencePipeline] = None
        self.monitor: Optional[PredictionMonitor] = None
        self.registry: Optional[ModelRegistry] = None
        self.feedback_store: FeedbackStore = FeedbackStore()
        self.start_time: float = time.monotonic()


# =============================================================================
# Application Factory
# =============================================================================

def create_app(
    checkpoint_path: Optional[str] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    monitoring_config: Optional[MonitoringConfig] = None,
    registry_dir: Optional[str] = None,
    feedback_dir: Optional[str] = None,
    cors_origins: Optional[list[str]] = None,
) -> "FastAPI":
    """
    Create and configure the FastAPI application.

    Args:
        checkpoint_path: Path to the model checkpoint.
        pipeline_config: Full pipeline configuration.
        monitoring_config: Monitoring configuration.
        registry_dir: Filesystem path for the model registry.
        feedback_dir: Filesystem path for persisting feedback entries.
        cors_origins: Allowed CORS origins (default: ``["*"]``).

    Returns:
        Configured ``FastAPI`` application instance.
    """
    _require_fastapi()

    state = _AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("ProToPhen API starting up")

        cfg = pipeline_config or PipelineConfig()
        if checkpoint_path is not None:
            cfg.checkpoint_path = checkpoint_path
        state.pipeline = InferencePipeline(config=cfg)

        if cfg.checkpoint_path is not None:
            try:
                state.pipeline.load_model(cfg.checkpoint_path)
            except Exception as exc:
                logger.error(f"Failed to load model at startup: {exc}")

        state.monitor = PredictionMonitor(
            config=monitoring_config or MonitoringConfig()
        )

        if registry_dir is not None:
            state.registry = ModelRegistry(
                config=RegistryConfig(registry_dir=registry_dir)
            )

        if feedback_dir is not None:
            state.feedback_store = FeedbackStore(persist_dir=Path(feedback_dir))

        state.start_time = time.monotonic()
        logger.info("ProToPhen API ready")
        yield
        logger.info("ProToPhen API shutting down")

    app = FastAPI(
        title="ProToPhen API",
        description=(
            "REST service for predicting cellular phenotypes from protein "
            "sequences using the ProToPhen foundation model."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state._protophen = state  # type: ignore[attr-defined]

    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- helpers -------------------------------------------------------------
    def _state(request: Request) -> _AppState:
        return request.app.state._protophen  # type: ignore[attr-defined]

    def _require_pipeline(request: Request) -> InferencePipeline:
        s = _state(request)
        if s.pipeline is None or not s.pipeline.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. The service is not ready.",
            )
        return s.pipeline

    # =========================================================================
    # Endpoints
    # =========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        s = _state(request)
        if s.pipeline is not None:
            info = s.pipeline.health_check()
        else:
            info = {
                "status": "unhealthy",
                "model_loaded": False,
                "esm_loaded": False,
                "uptime_seconds": round(time.monotonic() - s.start_time, 1),
                "version": "unknown",
                "device": "unknown",
                "checks": {},
            }
        return HealthResponse(**info)

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
    async def model_info(request: Request) -> ModelInfoResponse:
        pipeline = _require_pipeline(request)
        info = pipeline.get_model_info()
        return ModelInfoResponse(**info)

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    async def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
        pipeline = _require_pipeline(request)
        s = _state(request)

        try:
            response = pipeline.predict(
                sequence=body.protein.sequence,
                tasks=body.tasks,
                return_latent=body.return_latent,
                return_uncertainty=body.return_uncertainty,
                n_mc_samples=body.n_uncertainty_samples,
                protein_name=body.protein.name,
            )

            if s.monitor is not None:
                pred_arrays = {
                    tp.task_name: np.array(tp.values) for tp in response.predictions
                }
                s.monitor.record_request(
                    latency_ms=response.inference_time_ms,
                    sequence_length=response.sequence_length,
                    predictions=pred_arrays,
                )

            return response

        except ValueError as exc:
            if s.monitor is not None:
                s.monitor.record_error()
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            if s.monitor is not None:
                s.monitor.record_error()
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["prediction"],
    )
    async def predict_batch(
        request: Request, body: BatchPredictionRequest
    ) -> BatchPredictionResponse:
        pipeline = _require_pipeline(request)
        s = _state(request)
        t0 = time.perf_counter()

        try:
            sequences = [p.sequence for p in body.proteins]
            names = [p.name for p in body.proteins]

            results = pipeline.predict_batch(
                sequences=sequences,
                tasks=body.tasks,
                return_latent=body.return_latent,
                return_uncertainty=body.return_uncertainty,
                n_mc_samples=body.n_uncertainty_samples,
                protein_names=names,
            )

            total_ms = (time.perf_counter() - t0) * 1000.0

            if s.monitor is not None:
                for resp in results:
                    pred_arrays = {
                        tp.task_name: np.array(tp.values)
                        for tp in resp.predictions
                    }
                    s.monitor.record_request(
                        latency_ms=resp.inference_time_ms,
                        sequence_length=resp.sequence_length,
                        predictions=pred_arrays,
                    )

            return BatchPredictionResponse(
                results=results,
                n_proteins=len(results),
                total_inference_time_ms=round(total_ms, 2),
                model_version=pipeline.model_version,
            )

        except ValueError as exc:
            if s.monitor is not None:
                s.monitor.record_error()
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            if s.monitor is not None:
                s.monitor.record_error()
            logger.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/feedback", response_model=FeedbackResponse, tags=["active_learning"])
    async def feedback(request: Request, body: FeedbackRequest) -> FeedbackResponse:
        """
        Ingest experimental feedback for the active-learning loop.

        Stores the observed phenotype, and optionally triggers
        re-selection of the next experimental batch via
        ``ExperimentSelector`` (Session 7).
        """
        s = _state(request)

        try:
            # 1. Persist feedback
            entry = {
                "protein_id": body.protein_id,
                "sequence": body.sequence,
                "observed_features": body.observed_features,
                "plate_id": body.plate_id,
                "well_id": body.well_id,
                "cell_count": body.cell_count,
                "metadata": body.metadata,
                "received_at": time.time(),
                "model_version": (
                    s.pipeline.model_version if s.pipeline and s.pipeline.is_ready
                    else "unknown"
                ),
            }
            s.feedback_store.add(entry)

            logger.info(
                f"Feedback received for protein '{body.protein_id}': "
                f"{len(body.observed_features)} features "
                f"(total stored: {len(s.feedback_store)})"
            )

            # 2. Optionally trigger active-learning re-selection
            next_candidates: Optional[List[str]] = None

            if body.trigger_reselection:
                next_candidates = _run_reselection(s)

            return FeedbackResponse(
                status="accepted",
                protein_id=body.protein_id,
                message=(
                    f"Feedback stored successfully "
                    f"({len(s.feedback_store)} total entries)."
                ),
                reselection_triggered=body.trigger_reselection,
                next_candidates=next_candidates,
            )

        except Exception as exc:
            logger.exception("Feedback ingestion failed")
            return FeedbackResponse(
                status="error",
                protein_id=body.protein_id,
                message=str(exc),
            )

    def _run_reselection(s: _AppState) -> Optional[List[str]]:
        """
        Run active-learning re-selection using ``ExperimentSelector``.

        This is a best-effort operation; if the selector cannot be
        constructed (e.g. no pool dataloader is available) it returns
        an empty list and logs a warning.
        """
        if s.pipeline is None or not s.pipeline.is_ready:
            logger.warning("Reselection requested but model not loaded")
            return []

        try:
            from protophen.active_learning.selection import (
                ExperimentSelector,
                SelectionConfig,
            )
            from protophen.active_learning.uncertainty import UncertaintyType

            # Build exclude list from all previously fed-back protein IDs
            already_tested = [
                e["protein_id"] for e in s.feedback_store.entries
            ]

            config = SelectionConfig(
                n_select=10,
                uncertainty_method="mc_dropout",
                n_mc_samples=s.pipeline.config.default_mc_samples,
                acquisition_method="hybrid",
                exclude_ids=already_tested,
                tasks=list(s.pipeline.model.task_names),
            )

            selector = ExperimentSelector(
                model=s.pipeline.model,
                config=config,
                device=s.pipeline.device,
            )

            # NOTE: In a full deployment, a pool DataLoader would be
            # constructed from the candidate protein database.  Here
            # we log the intent and return the exclude list length so
            # the caller knows re-selection was attempted.
            logger.info(
                f"Active-learning re-selection triggered "
                f"(excluding {len(already_tested)} tested proteins). "
                f"Full pool DataLoader not available in API context; "
                f"returning empty candidate list."
            )
            return []

        except ImportError:
            logger.warning(
                "Active learning modules not available for re-selection"
            )
            return []
        except Exception as exc:
            logger.error(f"Re-selection failed: {exc}")
            return []

    # -- Monitoring ----------------------------------------------------------
    @app.get("/metrics", tags=["monitoring"])
    async def metrics(request: Request) -> JSONResponse:
        s = _state(request)
        if s.monitor is not None:
            return JSONResponse(content=s.monitor.summary())
        return JSONResponse(content={"message": "Monitoring not enabled"})

    @app.get("/metrics/prometheus", tags=["monitoring"])
    async def prometheus_metrics(request: Request) -> PlainTextResponse:
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

            return PlainTextResponse(
                content=generate_latest().decode("utf-8"),
                media_type=CONTENT_TYPE_LATEST,
            )
        except ImportError:
            raise HTTPException(
                status_code=501, detail="prometheus_client not installed."
            )

    return app