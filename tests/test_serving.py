"""
Tests for the ProToPhen serving infrastructure (Session 10).

These tests validate the inference pipeline, API schemas, monitoring,
and model registry without requiring a real ESM-2 model or GPU.
Heavy components are mocked to keep tests fast.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="protophen_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def dummy_checkpoint(tmp_dir) -> Path:
    """
    Create a minimal ProToPhen checkpoint on disk.

    Uses a tiny model config so that no real ESM-2 weights are needed.
    """
    from protophen.models.protophen import ProToPhenConfig, ProToPhenModel

    config = ProToPhenConfig(
        protein_embedding_dim=32,
        encoder_hidden_dims=[16],
        encoder_output_dim=8,
        decoder_hidden_dims=[16],
        cell_painting_dim=10,
        predict_viability=True,
        predict_transcriptomics=False,
        mc_dropout=True,
    )
    model = ProToPhenModel(config)

    ckpt_path = tmp_dir / "test_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "protein_embedding_dim": 32,
                "encoder_hidden_dims": [16],
                "encoder_output_dim": 8,
                "decoder_hidden_dims": [16],
                "cell_painting_dim": 10,
                "predict_viability": True,
                "predict_transcriptomics": False,
                "mc_dropout": True,
            },
            "epoch": 5,
            "version": "test_v1",
            "metrics": {"val_r2": 0.65},
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def mock_esm_embedder():
    """
    Return a mock ESMEmbedder that produces deterministic 32-dim embeddings.
    """
    mock = MagicMock()
    mock.embedding_dim = 32
    mock.output_dim = 32
    mock._model_loaded = True

    def _embed_sequence(seq):
        np.random.seed(hash(seq) % 2**31)
        return np.random.randn(32).astype(np.float32)

    def _embed_sequences(seqs, **kwargs):
        return np.stack([_embed_sequence(s) for s in seqs])

    mock.embed_sequence.side_effect = _embed_sequence
    mock.embed_sequences.side_effect = _embed_sequences
    return mock


# =========================================================================
# Schema Tests
# =========================================================================

class TestSchemas:
    """Tests for Pydantic request / response models."""

    def test_protein_input_valid(self):
        from protophen.serving.schemas import ProteinInput

        pi = ProteinInput(sequence="MKFLIL")
        assert pi.sequence == "MKFLIL"
        assert pi.name is None

    def test_protein_input_normalisation(self):
        from protophen.serving.schemas import ProteinInput

        pi = ProteinInput(sequence="  mk fl-il  ")
        assert pi.sequence == "MKFLIL"

    def test_protein_input_invalid_chars(self):
        from protophen.serving.schemas import ProteinInput

        with pytest.raises(Exception):  # Pydantic ValidationError
            ProteinInput(sequence="MKFLIL123")

    def test_protein_input_empty(self):
        from protophen.serving.schemas import ProteinInput

        with pytest.raises(Exception):
            ProteinInput(sequence="   ")

    def test_prediction_request_defaults(self):
        from protophen.serving.schemas import PredictionRequest, ProteinInput

        req = PredictionRequest(protein=ProteinInput(sequence="ACDEFG"))
        assert req.tasks is None
        assert req.return_latent is False
        assert req.return_uncertainty is False
        assert req.n_uncertainty_samples == 20

    def test_batch_request_size_limit(self):
        from protophen.serving.schemas import BatchPredictionRequest, ProteinInput

        with pytest.raises(Exception):
            BatchPredictionRequest(proteins=[])  # min_length=1

    def test_feedback_request(self):
        from protophen.serving.schemas import FeedbackRequest

        fb = FeedbackRequest(
            protein_id="prot_001",
            sequence="ACDEFG",
            observed_features=[0.1, 0.2, 0.3],
        )
        assert fb.protein_id == "prot_001"
        assert len(fb.observed_features) == 3
        assert fb.trigger_reselection is False

    def test_task_prediction(self):
        from protophen.serving.schemas import TaskPrediction

        tp = TaskPrediction(
            task_name="cell_painting",
            values=[0.1, 0.2, 0.3],
            dimension=3,
        )
        assert tp.dimension == 3

    def test_health_response(self):
        from protophen.serving.schemas import HealthResponse

        hr = HealthResponse(
            status="healthy",
            model_loaded=True,
            esm_loaded=True,
            uptime_seconds=42.0,
            version="v1",
            device="cpu",
        )
        assert hr.status == "healthy"


# =========================================================================
# Pipeline Tests
# =========================================================================

class TestPipeline:
    """Tests for InferencePipeline (with mocked ESM)."""

    def test_pipeline_init_no_checkpoint(self):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", checkpoint_path=None)
        pipe = InferencePipeline(config=config)
        assert not pipe.is_ready

    def test_pipeline_load_checkpoint(self, dummy_checkpoint, mock_esm_embedder):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(
            device="cpu",
            use_fp16=False,
            include_physicochemical=False,
        )
        pipe = InferencePipeline(config=config)
        pipe.load_model(dummy_checkpoint)
        assert pipe.is_ready
        assert pipe.model_version == "test_v1"

    def test_pipeline_predict(self, dummy_checkpoint, mock_esm_embedder):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(
            device="cpu",
            use_fp16=False,
            include_physicochemical=False,
        )
        pipe = InferencePipeline(config=config)
        pipe.load_model(dummy_checkpoint)

        # Patch the ESM embedder
        pipe._esm_embedder = mock_esm_embedder

        resp = pipe.predict("ACDEFGHIKL")
        assert resp.sequence_length == 10
        assert len(resp.predictions) >= 1
        assert resp.inference_time_ms > 0

    def test_pipeline_predict_with_uncertainty(self, dummy_checkpoint, mock_esm_embedder):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(
            device="cpu",
            use_fp16=False,
            include_physicochemical=False,
        )
        pipe = InferencePipeline(config=config)
        pipe.load_model(dummy_checkpoint)
        pipe._esm_embedder = mock_esm_embedder

        resp = pipe.predict("ACDEFGHIKL", return_uncertainty=True, n_mc_samples=5)
        assert resp.uncertainty is not None
        assert len(resp.uncertainty) >= 1
        assert resp.uncertainty[0].n_samples == 5

    def test_pipeline_predict_with_latent(self, dummy_checkpoint, mock_esm_embedder):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(
            device="cpu",
            use_fp16=False,
            include_physicochemical=False,
        )
        pipe = InferencePipeline(config=config)
        pipe.load_model(dummy_checkpoint)
        pipe._esm_embedder = mock_esm_embedder

        resp = pipe.predict("ACDEFGHIKL", return_latent=True)
        assert resp.latent is not None
        assert len(resp.latent) == 8  # encoder_output_dim

    def test_pipeline_predict_batch(self, dummy_checkpoint, mock_esm_embedder):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(
            device="cpu",
            use_fp16=False,
            include_physicochemical=False,
        )
        pipe = InferencePipeline(config=config)
        pipe.load_model(dummy_checkpoint)
        pipe._esm_embedder = mock_esm_embedder

        results = pipe.predict_batch(
            ["ACDEFGHIKL", "FGHIKLMNPQ", "STWYACDEFG"],
            protein_names=["p1", "p2", "p3"],
        )
        assert len(results) == 3
        assert results[0].protein_name == "p1"

    def test_pipeline_health_check(self, dummy_checkpoint):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(config=config)

        health = pipe.health_check()
        assert health["status"] == "unhealthy"

        pipe.load_model(dummy_checkpoint)
        health = pipe.health_check()
        assert health["model_loaded"] is True

    def test_pipeline_model_info(self, dummy_checkpoint):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(checkpoint_path=dummy_checkpoint, config=config)
        info = pipe.get_model_info()
        assert info["model_name"] == "ProToPhen"
        assert "cell_painting" in info["tasks"]

    def test_pipeline_missing_checkpoint_raises(self):
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", checkpoint_path="/nonexistent/model.pt")
        pipe = InferencePipeline(config=config)
        with pytest.raises(FileNotFoundError):
            pipe.predict("ACDEFG")


# =========================================================================
# Monitoring Tests
# =========================================================================

class TestMonitoring:
    """Tests for PredictionMonitor and DriftDetector."""

    def test_monitor_record_and_summary(self):
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                window_size=100,
                enable_drift_detection=False,
                log_predictions=False,
            )
        )

        for i in range(50):
            monitor.record_request(
                latency_ms=10.0 + i * 0.5,
                sequence_length=100 + i,
                predictions={"cell_painting": np.random.randn(10)},
            )

        summary = monitor.summary()
        assert summary["total_requests"] == 50
        assert summary["total_errors"] == 0
        assert "latency_ms" in summary
        assert summary["latency_ms"]["p50"] > 0

    def test_monitor_error_tracking(self):
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(log_predictions=False, enable_drift_detection=False)
        )
        monitor.record_error()
        monitor.record_error()
        assert monitor.summary()["total_errors"] == 2

    def test_monitor_reset(self):
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(log_predictions=False, enable_drift_detection=False)
        )
        monitor.record_request(latency_ms=10, sequence_length=50)
        monitor.reset()
        assert monitor.summary()["total_requests"] == 0

    def test_drift_detector_no_drift(self):
        from protophen.serving.monitoring import DriftDetector

        det = DriftDetector(window_size=100, significance=0.01)

        # Reference: N(0,1)
        rng = np.random.default_rng(42)
        for _ in range(200):
            det.add_observation("task_a", rng.standard_normal(10))

        report = det.report()
        assert "task_a" in report
        assert report["task_a"]["reference_set"] == True
        # Same distribution → no drift expected
        assert report["task_a"]["drift_detected"] == False

    def test_drift_detector_detects_shift(self):
        from protophen.serving.monitoring import DriftDetector

        det = DriftDetector(window_size=50, significance=0.05)

        # Reference: N(0, 1)
        rng = np.random.default_rng(42)
        for _ in range(50):
            det.add_observation("task_a", rng.standard_normal(10))

        # Shifted: N(5, 1) — large shift, should be detected
        for _ in range(50):
            det.add_observation("task_a", rng.standard_normal(10) + 5.0)

        report = det.report()
        assert report["task_a"]["drift_detected"] == True

    def test_drift_detector_explicit_reference(self):
        from protophen.serving.monitoring import DriftDetector

        det = DriftDetector(window_size=50, significance=0.05)
        det.set_reference("task_a", np.random.randn(100))

        assert det.report()["task_a"]["reference_set"] is True


# =========================================================================
# Registry Tests
# =========================================================================

class TestRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_list(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register(
            checkpoint_path=dummy_checkpoint,
            version="v1",
            description="Test model",
            metrics={"val_r2": 0.72},
            tags=["baseline"],
        )

        assert mv.version == "v1"
        assert mv.stage == "staging"

        versions = reg.list_versions()
        assert len(versions) == 1
        assert versions[0].version == "v1"

    def test_auto_version(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv1 = reg.register(dummy_checkpoint)
        mv2 = reg.register(dummy_checkpoint)
        assert mv1.version == "v1"
        assert mv2.version == "v2"

    def test_promote_to_production(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        reg.register(dummy_checkpoint, version="v1")
        reg.set_stage("v1", "production")

        prod = reg.get_latest(stage="production")
        assert prod is not None
        assert prod.version == "v1"

        ckpt = reg.get_production_checkpoint()
        assert ckpt is not None
        assert Path(ckpt).exists()

    def test_production_replacement(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        reg.register(dummy_checkpoint, version="v1")
        reg.register(dummy_checkpoint, version="v2")

        reg.set_stage("v1", "production")
        reg.set_stage("v2", "production")

        # v1 should now be archived
        v1 = reg.get_version("v1")
        assert v1.stage == "archived"

        v2 = reg.get_version("v2")
        assert v2.stage == "production"

    def test_rollback(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        reg.register(dummy_checkpoint, version="v1")
        reg.register(dummy_checkpoint, version="v2")

        reg.set_stage("v1", "production")
        reg.set_stage("v2", "production")  # v1 → archived

        rolled = reg.rollback()
        assert rolled is not None
        assert rolled.version == "v1"
        assert rolled.stage == "production"

    def test_compare_versions(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        reg.register(dummy_checkpoint, version="v1", metrics={"r2": 0.60, "mse": 0.15})
        reg.register(dummy_checkpoint, version="v2", metrics={"r2": 0.72, "mse": 0.10})

        cmp = reg.compare_versions("v1", "v2")
        assert cmp["metrics"]["r2"]["delta"] == pytest.approx(0.12)
        assert cmp["metrics"]["mse"]["delta"] == pytest.approx(-0.05)

    def test_get_best_version(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        reg.register(dummy_checkpoint, version="v1", metrics={"r2": 0.60})
        reg.register(dummy_checkpoint, version="v2", metrics={"r2": 0.72})
        reg.register(dummy_checkpoint, version="v3", metrics={"r2": 0.68})

        best = reg.get_best_version("r2", higher_is_better=True)
        assert best.version == "v2"

    def test_delete_version(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )
        reg.register(dummy_checkpoint, version="v1")
        reg.delete_version("v1")

        with pytest.raises(KeyError):
            reg.get_version("v1")

    def test_persistence(self, dummy_checkpoint, tmp_dir):
        """Registry state survives re-instantiation."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg_dir = str(tmp_dir / "registry")

        reg1 = ModelRegistry(config=RegistryConfig(registry_dir=reg_dir))
        reg1.register(dummy_checkpoint, version="v1", metrics={"r2": 0.7})
        reg1.set_stage("v1", "production")

        # Re-open
        reg2 = ModelRegistry(config=RegistryConfig(registry_dir=reg_dir))
        assert len(reg2.list_versions()) == 1
        assert reg2.get_production_checkpoint() is not None

    def test_max_versions_eviction(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(
                registry_dir=str(tmp_dir / "registry"),
                max_versions=3,
            )
        )

        for i in range(5):
            reg.register(
                dummy_checkpoint,
                version=f"v{i+1}",
                stage="archived" if i < 3 else "staging",
            )

        # Should have evicted oldest archived to stay at max_versions
        assert len(reg.list_versions()) <= 3

    def test_duplicate_version_raises(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )
        reg.register(dummy_checkpoint, version="v1")

        with pytest.raises(ValueError, match="already exists"):
            reg.register(dummy_checkpoint, version="v1")

    def test_summary(self, dummy_checkpoint, tmp_dir):
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )
        reg.register(dummy_checkpoint, version="v1")
        reg.set_stage("v1", "production")

        s = reg.summary()
        assert s["total_versions"] == 1
        assert s["production_version"] == "v1"


# =========================================================================
# Checkpoint Utility Tests
# =========================================================================

class TestCheckpointUtils:
    """Tests for checkpoint loading helpers."""

    def test_load_checkpoint(self, dummy_checkpoint):
        from protophen.serving.pipeline import load_checkpoint

        ckpt = load_checkpoint(dummy_checkpoint, device="cpu")
        assert "model_state_dict" in ckpt
        assert ckpt["epoch"] == 5

    def test_load_checkpoint_missing(self):
        from protophen.serving.pipeline import load_checkpoint

        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.pt")

    def test_build_model_from_checkpoint(self, dummy_checkpoint):
        from protophen.serving.pipeline import load_checkpoint, build_model_from_checkpoint

        ckpt = load_checkpoint(dummy_checkpoint, device="cpu")
        model = build_model_from_checkpoint(ckpt, device="cpu")
        assert model is not None
        assert not model.training  # eval mode

        # Smoke test forward pass
        x = torch.randn(2, 32)
        outputs = model(x)
        assert "cell_painting" in outputs
        assert outputs["cell_painting"].shape == (2, 10)
        
# =========================================================================
# Trainer-Checkpoint Compatibility Tests (Session 10.2)
# =========================================================================

class TestTrainerCheckpointCompatibility:
    """
    Tests that the serving pipeline correctly loads checkpoints produced
    by Trainer.save_checkpoint() and CheckpointCallback (Session 6).

    The key challenge: Trainer saves TrainerConfig (not ProToPhenConfig)
    under the 'config' key, so the pipeline must detect this and infer
    model architecture from state_dict tensor shapes instead.
    """

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny ProToPhenModel for checkpoint generation."""
        from protophen.models.protophen import ProToPhenConfig, ProToPhenModel

        config = ProToPhenConfig(
            protein_embedding_dim=32,
            encoder_hidden_dims=[16],
            encoder_output_dim=8,
            decoder_hidden_dims=[16],
            cell_painting_dim=10,
            predict_viability=True,
            predict_transcriptomics=False,
            mc_dropout=True,
        )
        return ProToPhenModel(config)

    @pytest.fixture
    def trainer_checkpoint(self, tiny_model, tmp_dir) -> Path:
        """
        Create a checkpoint in the exact format produced by
        ``Trainer.save_checkpoint()`` (Session 6).

        Key difference from pipeline checkpoints: ``config`` contains
        TrainerConfig fields, NOT ProToPhenConfig fields.
        """
        from dataclasses import asdict

        from protophen.training.trainer import TrainerConfig

        trainer_config = TrainerConfig(
            epochs=50,
            learning_rate=1e-4,
            weight_decay=0.01,
            optimiser="adamw",
            scheduler="cosine",
            warmup_steps=100,
            tasks=["cell_painting", "viability"],
            task_weights={"cell_painting": 1.0, "viability": 0.5},
            seed=42,
        )

        ckpt_path = tmp_dir / "trainer_checkpoint.pt"
        torch.save(
            {
                "epoch": 25,
                "global_step": 1000,
                "model_state_dict": tiny_model.state_dict(),
                "optimiser_state_dict": {},  # empty for test brevity
                "config": asdict(trainer_config),
                "best_val_loss": 0.0423,
            },
            ckpt_path,
        )
        return ckpt_path

    @pytest.fixture
    def callback_checkpoint(self, tiny_model, tmp_dir) -> Path:
        """
        Create a checkpoint in the format produced by
        ``CheckpointCallback._save_checkpoint()`` (Session 6).

        Distinctive keys: ``best_value`` and ``monitor`` instead of
        ``best_val_loss``; ``config`` is ``trainer.config.__dict__``.
        """
        from protophen.training.trainer import TrainerConfig

        trainer_config = TrainerConfig(
            epochs=100,
            learning_rate=5e-5,
            tasks=["cell_painting"],
        )

        ckpt_path = tmp_dir / "callback_checkpoint.pt"
        torch.save(
            {
                "epoch": 42,
                "global_step": 2100,
                "model_state_dict": tiny_model.state_dict(),
                "optimiser_state_dict": {},
                "config": trainer_config.__dict__,
                "best_value": 0.0312,
                "monitor": "val_loss",
            },
            ckpt_path,
        )
        return ckpt_path

    @pytest.fixture
    def raw_state_dict_checkpoint(self, tiny_model, tmp_dir) -> Path:
        """Create a checkpoint that is just a raw ``OrderedDict`` of tensors."""
        ckpt_path = tmp_dir / "raw_state_dict.pt"
        torch.save(tiny_model.state_dict(), ckpt_path)
        return ckpt_path

    # -- _is_trainer_config heuristic --

    def test_is_trainer_config_true(self):
        """_is_trainer_config returns True for TrainerConfig-shaped dicts."""
        from protophen.serving.pipeline import _is_trainer_config

        trainer_cfg = {
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "optimiser": "adamw",
            "scheduler": "cosine",
            "seed": 42,
        }
        assert _is_trainer_config(trainer_cfg) is True

    def test_is_trainer_config_false(self):
        """_is_trainer_config returns False for ProToPhenConfig-shaped dicts."""
        from protophen.serving.pipeline import _is_trainer_config

        model_cfg = {
            "protein_embedding_dim": 32,
            "encoder_hidden_dims": [16],
            "encoder_output_dim": 8,
            "cell_painting_dim": 10,
        }
        assert _is_trainer_config(model_cfg) is False

    # -- _infer_model_config_from_state_dict --

    def test_infer_config_embedding_dim(self, tiny_model):
        """State-dict inference recovers protein_embedding_dim."""
        from protophen.serving.pipeline import _infer_model_config_from_state_dict

        config = _infer_model_config_from_state_dict(tiny_model.state_dict())
        assert config.protein_embedding_dim == 32

    def test_infer_config_encoder_output_dim(self, tiny_model):
        """State-dict inference recovers encoder_output_dim."""
        from protophen.serving.pipeline import _infer_model_config_from_state_dict

        config = _infer_model_config_from_state_dict(tiny_model.state_dict())
        assert config.encoder_output_dim == 8

    def test_infer_config_cell_painting_dim(self, tiny_model):
        """State-dict inference recovers cell_painting_dim."""
        from protophen.serving.pipeline import _infer_model_config_from_state_dict

        config = _infer_model_config_from_state_dict(tiny_model.state_dict())
        assert config.cell_painting_dim == 10

    def test_infer_config_viability(self, tiny_model):
        """State-dict inference detects predict_viability from decoder keys."""
        from protophen.serving.pipeline import _infer_model_config_from_state_dict

        config = _infer_model_config_from_state_dict(tiny_model.state_dict())
        assert config.predict_viability is True

    def test_infer_config_no_transcriptomics(self, tiny_model):
        """State-dict inference detects absence of transcriptomics decoder."""
        from protophen.serving.pipeline import _infer_model_config_from_state_dict

        config = _infer_model_config_from_state_dict(tiny_model.state_dict())
        assert config.predict_transcriptomics is False

    def test_infer_config_encoder_hidden_dims(self, tiny_model):
        """State-dict inference recovers encoder hidden dimensions."""
        from protophen.serving.pipeline import _infer_model_config_from_state_dict

        config = _infer_model_config_from_state_dict(tiny_model.state_dict())
        assert config.encoder_hidden_dims == [16]

    # -- load_checkpoint normalisation --

    def test_load_trainer_checkpoint_normalises(self, trainer_checkpoint):
        """load_checkpoint normalises a Trainer checkpoint."""
        from protophen.serving.pipeline import load_checkpoint

        ckpt = load_checkpoint(trainer_checkpoint, device="cpu")
        assert "model_state_dict" in ckpt
        assert ckpt["epoch"] == 25
        # TrainerConfig should be stashed in _trainer_config
        assert "_trainer_config" in ckpt
        assert "learning_rate" in ckpt["_trainer_config"]

    def test_load_callback_checkpoint_normalises(self, callback_checkpoint):
        """load_checkpoint normalises a CheckpointCallback checkpoint."""
        from protophen.serving.pipeline import load_checkpoint

        ckpt = load_checkpoint(callback_checkpoint, device="cpu")
        assert "model_state_dict" in ckpt
        assert ckpt["epoch"] == 42

    def test_load_raw_state_dict_normalises(self, raw_state_dict_checkpoint):
        """load_checkpoint wraps a raw state dict into the standard format."""
        from protophen.serving.pipeline import load_checkpoint

        ckpt = load_checkpoint(raw_state_dict_checkpoint, device="cpu")
        assert "model_state_dict" in ckpt
        assert "config" in ckpt

    def test_version_from_trainer_epoch(self, trainer_checkpoint):
        """Trainer checkpoint version is derived from epoch."""
        from protophen.serving.pipeline import load_checkpoint

        ckpt = load_checkpoint(trainer_checkpoint, device="cpu")
        assert "version" in ckpt
        assert "25" in ckpt["version"]  # epoch_25

    def test_version_from_callback_epoch(self, callback_checkpoint):
        """Callback checkpoint version is derived from epoch."""
        from protophen.serving.pipeline import load_checkpoint

        ckpt = load_checkpoint(callback_checkpoint, device="cpu")
        assert "42" in ckpt["version"]

    def test_metrics_normalised_from_best_val_loss(self, trainer_checkpoint):
        """best_val_loss is lifted into the normalised metrics dict."""
        from protophen.serving.pipeline import load_checkpoint

        ckpt = load_checkpoint(trainer_checkpoint, device="cpu")
        assert "metrics" in ckpt
        assert "best_val_loss" in ckpt["metrics"]
        assert ckpt["metrics"]["best_val_loss"] == pytest.approx(0.0423)

    def test_metrics_normalised_from_best_value_monitor(self, callback_checkpoint):
        """best_value + monitor are combined into the normalised metrics dict."""
        from protophen.serving.pipeline import load_checkpoint

        ckpt = load_checkpoint(callback_checkpoint, device="cpu")
        metrics = ckpt.get("metrics", {})
        assert "best_val_loss" in metrics
        assert metrics["best_val_loss"] == pytest.approx(0.0312)

    # -- build_model_from_checkpoint --

    def test_build_from_trainer_checkpoint(self, trainer_checkpoint):
        """build_model_from_checkpoint works with Trainer format."""
        from protophen.serving.pipeline import build_model_from_checkpoint, load_checkpoint

        ckpt = load_checkpoint(trainer_checkpoint, device="cpu")
        model = build_model_from_checkpoint(ckpt, device="cpu")

        assert model is not None
        assert not model.training
        x = torch.randn(2, 32)
        outputs = model(x)
        assert "cell_painting" in outputs
        assert outputs["cell_painting"].shape == (2, 10)
        # Viability head should also be present
        assert "viability" in outputs

    def test_build_from_callback_checkpoint(self, callback_checkpoint):
        """build_model_from_checkpoint works with CheckpointCallback format."""
        from protophen.serving.pipeline import build_model_from_checkpoint, load_checkpoint

        ckpt = load_checkpoint(callback_checkpoint, device="cpu")
        model = build_model_from_checkpoint(ckpt, device="cpu")

        assert model is not None
        x = torch.randn(2, 32)
        outputs = model(x)
        assert "cell_painting" in outputs

    def test_build_from_raw_state_dict(self, raw_state_dict_checkpoint):
        """build_model_from_checkpoint works with a raw state dict."""
        from protophen.serving.pipeline import build_model_from_checkpoint, load_checkpoint

        ckpt = load_checkpoint(raw_state_dict_checkpoint, device="cpu")
        model = build_model_from_checkpoint(ckpt, device="cpu")

        assert model is not None
        x = torch.randn(2, 32)
        outputs = model(x)
        assert "cell_painting" in outputs

        # -- Full InferencePipeline with Trainer checkpoints --

    def test_pipeline_loads_trainer_checkpoint(self, trainer_checkpoint):
        """InferencePipeline.load_model works with a Trainer checkpoint."""
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(config=config)
        pipe.load_model(trainer_checkpoint)

        assert pipe.is_ready
        assert "25" in pipe.model_version  # epoch_25

    def test_pipeline_loads_callback_checkpoint(self, callback_checkpoint):
        """InferencePipeline.load_model works with a CheckpointCallback checkpoint."""
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(config=config)
        pipe.load_model(callback_checkpoint)

        assert pipe.is_ready
        assert "42" in pipe.model_version

    def test_pipeline_loads_raw_state_dict(self, raw_state_dict_checkpoint):
        """InferencePipeline.load_model works with a bare state dict."""
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(config=config)
        pipe.load_model(raw_state_dict_checkpoint)

        assert pipe.is_ready

    def test_pipeline_preserves_trainer_config(self, trainer_checkpoint):
        """Pipeline stores the original TrainerConfig for reproducibility."""
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(config=config)
        pipe.load_model(trainer_checkpoint)

        tc = pipe.trainer_config
        assert tc is not None
        assert tc["learning_rate"] == 1e-4
        assert tc["optimiser"] == "adamw"
        assert "cell_painting" in tc["tasks"]

    def test_pipeline_checkpoint_metrics_from_trainer(self, trainer_checkpoint):
        """Pipeline exposes normalised metrics from Trainer checkpoint."""
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(config=config)
        pipe.load_model(trainer_checkpoint)

        metrics = pipe.checkpoint_metrics
        assert "best_val_loss" in metrics
        assert metrics["best_val_loss"] == pytest.approx(0.0423)

    def test_pipeline_predict_with_trainer_checkpoint(
        self, trainer_checkpoint, mock_esm_embedder
    ):
        """End-to-end prediction works after loading a Trainer checkpoint."""
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(
            device="cpu",
            use_fp16=False,
            include_physicochemical=False,
        )
        pipe = InferencePipeline(config=config)
        pipe.load_model(trainer_checkpoint)
        pipe._esm_embedder = mock_esm_embedder

        resp = pipe.predict("ACDEFGHIKL")
        assert resp.sequence_length == 10
        assert len(resp.predictions) >= 1

        # Should have cell_painting and viability since the model was
        # created with predict_viability=True
        task_names = {p.task_name for p in resp.predictions}
        assert "cell_painting" in task_names

    def test_pipeline_model_info_after_trainer_checkpoint(self, trainer_checkpoint):
        """get_model_info works after loading a Trainer checkpoint."""
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(config=config)
        pipe.load_model(trainer_checkpoint)

        info = pipe.get_model_info()
        assert info["model_name"] == "ProToPhen"
        assert info["protein_embedding_dim"] == 32
        assert info["latent_dim"] == 8
        assert "cell_painting" in info["tasks"]

    def test_reload_different_checkpoint_formats(
        self, trainer_checkpoint, callback_checkpoint, mock_esm_embedder
    ):
        """Pipeline can reload from one checkpoint format to another."""
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig

        config = PipelineConfig(
            device="cpu",
            use_fp16=False,
            include_physicochemical=False,
        )
        pipe = InferencePipeline(config=config)

        # Load Trainer checkpoint
        pipe.load_model(trainer_checkpoint)
        pipe._esm_embedder = mock_esm_embedder
        assert "25" in pipe.model_version

        resp1 = pipe.predict("ACDEFGHIKL")
        assert resp1.predictions

        # Reload with Callback checkpoint
        pipe.load_model(callback_checkpoint)
        assert "42" in pipe.model_version

        resp2 = pipe.predict("ACDEFGHIKL")
        assert resp2.predictions


# =========================================================================
# Feedback Quality Tracking Tests (Session 10.2)
# =========================================================================

class TestFeedbackQualityTracking:
    """
    Tests for the PredictionQualityTracker and the feedback-related
    monitoring integration.

    When ground-truth observations arrive through the /feedback endpoint,
    the monitor matches them against cached predictions and computes
    regression metrics using the same metric classes from
    ``protophen.training.metrics`` (Session 6).
    """

    # -- PredictionQualityTracker unit tests --

    def test_tracker_empty(self):
        """Fresh tracker has no pairs and returns empty metrics."""
        from protophen.serving.monitoring import PredictionQualityTracker

        tracker = PredictionQualityTracker(window_size=100)
        assert tracker.n_pairs == 0
        assert tracker.compute_metrics() == {}

    def test_tracker_single_pair_returns_empty(self):
        """A single pair is insufficient for regression metrics."""
        from protophen.serving.monitoring import PredictionQualityTracker

        tracker = PredictionQualityTracker(window_size=100)
        tracker.add("prot_1", np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        assert tracker.n_pairs == 1
        assert tracker.compute_metrics() == {}

    def test_tracker_two_pairs_computes_metrics(self):
        """Two or more pairs produce regression metrics."""
        from protophen.serving.monitoring import PredictionQualityTracker

        tracker = PredictionQualityTracker(window_size=100)
        rng = np.random.default_rng(42)
        for i in range(5):
            pred = rng.standard_normal(10).astype(np.float32)
            obs = pred + rng.standard_normal(10).astype(np.float32) * 0.1
            tracker.add(f"prot_{i}", pred, obs)

        assert tracker.n_pairs == 5
        metrics = tracker.compute_metrics()
        assert len(metrics) > 0
        # Should contain default metric names from create_default_metrics
        metric_keys = set(metrics.keys())
        # Check for at least MSE and R² (prefixed with quality_)
        assert any("mse" in k for k in metric_keys)
        assert any("r2" in k for k in metric_keys)

    def test_tracker_uses_training_metrics(self):
        """Verify tracker delegates to protophen.training.metrics."""
        from protophen.serving.monitoring import PredictionQualityTracker

        tracker = PredictionQualityTracker(window_size=100)
        rng = np.random.default_rng(99)

        # Perfect predictions → R² ≈ 1, MSE ≈ 0
        for i in range(10):
            values = rng.standard_normal(5).astype(np.float32)
            tracker.add(f"prot_{i}", values, values)

        metrics = tracker.compute_metrics()
        # R² should be very close to 1 for perfect predictions
        r2_key = [k for k in metrics if "r2" in k]
        assert len(r2_key) > 0
        assert metrics[r2_key[0]] == pytest.approx(1.0, abs=1e-4)

        # MSE should be very close to 0
        mse_key = [k for k in metrics if "mse" in k]
        assert len(mse_key) > 0
        assert metrics[mse_key[0]] == pytest.approx(0.0, abs=1e-6)

    def test_tracker_window_eviction(self):
        """Tracker respects its rolling window size."""
        from protophen.serving.monitoring import PredictionQualityTracker

        tracker = PredictionQualityTracker(window_size=5)
        rng = np.random.default_rng(42)

        for i in range(10):
            tracker.add(
                f"prot_{i}",
                rng.standard_normal(3).astype(np.float32),
                rng.standard_normal(3).astype(np.float32),
            )

        assert tracker.n_pairs == 5  # window_size=5

    def test_tracker_reset(self):
        """Tracker reset clears all stored pairs."""
        from protophen.serving.monitoring import PredictionQualityTracker

        tracker = PredictionQualityTracker(window_size=100)
        tracker.add("p1", np.array([1.0]), np.array([1.0]))
        tracker.add("p2", np.array([2.0]), np.array([2.0]))
        assert tracker.n_pairs == 2

        tracker.reset()
        assert tracker.n_pairs == 0
        assert tracker.compute_metrics() == {}

    # -- PredictionMonitor feedback integration --

    def test_monitor_record_feedback_with_cached_prediction(self):
        """Monitor matches cached prediction with incoming observation."""
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                enable_drift_detection=False,
                log_predictions=False,
                track_regression_metrics=True,
            )
        )

        pred_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Record a prediction with protein_id
        monitor.record_request(
            latency_ms=10.0,
            sequence_length=100,
            predictions={"cell_painting": pred_array},
            protein_id="prot_001",
        )

        # Feed back observation
        observation = np.array([1.1, 2.1, 2.9], dtype=np.float32)
        monitor.record_feedback(protein_id="prot_001", observation=observation)

        assert monitor._total_feedback == 1
        assert monitor._quality_tracker is not None
        assert monitor._quality_tracker.n_pairs == 1

    def test_monitor_record_feedback_with_explicit_prediction(self):
        """Monitor accepts an explicit prediction in the feedback call."""
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                enable_drift_detection=False,
                log_predictions=False,
                track_regression_metrics=True,
            )
        )

        prediction = np.array([1.0, 2.0], dtype=np.float32)
        observation = np.array([1.1, 2.2], dtype=np.float32)

        monitor.record_feedback(
            protein_id="prot_explicit",
            observation=observation,
            prediction=prediction,
        )

        assert monitor._quality_tracker.n_pairs == 1

    def test_monitor_feedback_no_matching_prediction(self):
        """Feedback with unknown protein_id and no explicit prediction: no pair stored."""
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                enable_drift_detection=False,
                log_predictions=False,
                track_regression_metrics=True,
            )
        )

        monitor.record_feedback(
            protein_id="unknown_prot",
            observation=np.array([1.0, 2.0]),
        )

        assert monitor._total_feedback == 1
        # No pair stored because no matching prediction was cached
        assert monitor._quality_tracker.n_pairs == 0

    def test_monitor_summary_includes_quality(self):
        """summary() includes prediction_quality when enough feedback is available."""
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                enable_drift_detection=False,
                log_predictions=False,
                track_regression_metrics=True,
            )
        )

        rng = np.random.default_rng(42)
        for i in range(5):
            pred = rng.standard_normal(10).astype(np.float32)
            obs = pred + rng.standard_normal(10).astype(np.float32) * 0.05

            monitor.record_request(
                latency_ms=10.0,
                sequence_length=100,
                predictions={"cell_painting": pred},
                protein_id=f"prot_{i}",
            )
            monitor.record_feedback(protein_id=f"prot_{i}", observation=obs)

        summary = monitor.summary()
        assert "prediction_quality" in summary
        assert "n_pairs" in summary["prediction_quality"]
        assert summary["prediction_quality"]["n_pairs"] == 5

    def test_monitor_summary_no_quality_without_feedback(self):
        """summary() omits prediction_quality when no feedback is available."""
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                enable_drift_detection=False,
                log_predictions=False,
                track_regression_metrics=True,
            )
        )

        # Record requests but no feedback
        for i in range(5):
            monitor.record_request(
                latency_ms=10.0,
                sequence_length=100,
                predictions={"cell_painting": np.random.randn(10)},
            )

        summary = monitor.summary()
        assert "prediction_quality" not in summary

    def test_monitor_prediction_cache_eviction(self):
        """Prediction cache does not grow unboundedly."""
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                window_size=5,  # small window → small cache
                enable_drift_detection=False,
                log_predictions=False,
                track_regression_metrics=True,
            )
        )

        for i in range(20):
            monitor.record_request(
                latency_ms=1.0,
                sequence_length=50,
                predictions={"cell_painting": np.random.randn(3)},
                protein_id=f"prot_{i}",
            )

        # Cache should not exceed window_size
        assert len(monitor._recent_predictions) <= 5

    def test_monitor_feedback_counter_in_summary(self):
        """summary() reports total_feedback count."""
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                enable_drift_detection=False,
                log_predictions=False,
                track_regression_metrics=False,
            )
        )

        monitor.record_feedback("p1", np.array([1.0]))
        monitor.record_feedback("p2", np.array([2.0]))
        monitor.record_feedback("p3", np.array([3.0]))

        summary = monitor.summary()
        assert summary["total_feedback"] == 3

    def test_monitor_reset_clears_feedback(self):
        """reset() clears feedback state including quality tracker."""
        from protophen.serving.monitoring import PredictionMonitor, MonitoringConfig

        monitor = PredictionMonitor(
            config=MonitoringConfig(
                enable_drift_detection=False,
                log_predictions=False,
                track_regression_metrics=True,
            )
        )

        monitor.record_feedback(
            "p1",
            observation=np.array([1.0]),
            prediction=np.array([1.1]),
        )
        monitor.record_feedback(
            "p2",
            observation=np.array([2.0]),
            prediction=np.array([2.1]),
        )
        assert monitor._quality_tracker.n_pairs == 2

        monitor.reset()
        assert monitor._total_feedback == 0
        assert monitor._quality_tracker.n_pairs == 0
        assert len(monitor._recent_predictions) == 0


# =========================================================================
# Drift Detector Trainer Integration Tests (Session 10.2)
# =========================================================================

class TestDriftDetectorTrainerIntegration:
    """
    Tests for DriftDetector integration with Trainer outputs.

    The DriftDetector can accept reference distributions either:
    - Automatically from the first N observations
    - Explicitly via set_reference() with 1-D arrays
    - From Trainer.predict() output via set_reference_from_trainer()
    """

    def test_set_reference_from_trainer_predictions(self):
        """set_reference_from_trainer correctly parses Trainer.predict() output."""
        from protophen.serving.monitoring import DriftDetector

        det = DriftDetector(window_size=50, significance=0.05)

        # Simulate Trainer.predict() output format
        trainer_output = {
            "protein_ids": [f"p{i}" for i in range(100)],
            "cell_painting_predictions": np.random.randn(100, 10),
            "viability_predictions": np.random.randn(100, 1),
        }

        det.set_reference_from_trainer(trainer_output)

        report = det.report()
        assert "cell_painting" in report
        assert "viability" in report
        assert report["cell_painting"]["reference_set"] is True
        assert report["viability"]["reference_set"] is True

    def test_set_reference_from_trainer_skips_non_predictions(self):
        """set_reference_from_trainer ignores keys not ending in _predictions."""
        from protophen.serving.monitoring import DriftDetector

        det = DriftDetector(window_size=50)

        trainer_output = {
            "protein_ids": [f"p{i}" for i in range(50)],
            "cell_painting_predictions": np.random.randn(50, 10),
            "cell_painting_targets": np.random.randn(50, 10),
        }

        det.set_reference_from_trainer(trainer_output)

        report = det.report()
        assert "cell_painting" in report
        # _targets key should NOT produce a reference
        assert "cell_painting_targets" not in report

    def test_set_reference_from_trainer_2d_to_per_sample_mean(self):
        """2-D prediction arrays are reduced to per-sample means for reference."""
        from protophen.serving.monitoring import DriftDetector

        det = DriftDetector(window_size=50)

        predictions_2d = np.random.randn(100, 50)
        trainer_output = {
            "cell_painting_predictions": predictions_2d,
        }

        det.set_reference_from_trainer(trainer_output)

        # The stored reference should be 1-D with length 100
        ref = det._reference.get("cell_painting")
        assert ref is not None
        assert ref.ndim == 1
        assert len(ref) == 100
        np.testing.assert_allclose(ref, predictions_2d.mean(axis=1), atol=1e-6)

    def test_set_reference_from_trainer_1d_passthrough(self):
        """1-D prediction arrays are stored directly."""
        from protophen.serving.monitoring import DriftDetector

        det = DriftDetector(window_size=50)

        predictions_1d = np.random.randn(80)
        trainer_output = {
            "viability_predictions": predictions_1d,
        }

        det.set_reference_from_trainer(trainer_output)

        ref = det._reference.get("viability")
        assert ref is not None
        assert ref.ndim == 1
        assert len(ref) == 80

    def test_drift_with_trainer_reference_no_drift(self):
        """No drift when new observations match the trainer reference."""
        from protophen.serving.monitoring import DriftDetector

        rng = np.random.default_rng(42)
        det = DriftDetector(window_size=50, significance=0.05)

        # Set reference from "training" predictions
        ref_preds = rng.standard_normal((200, 10))
        det.set_reference_from_trainer({
            "cell_painting_predictions": ref_preds,
        })

        # Feed observations from the same distribution
        for _ in range(60):
            obs = rng.standard_normal(10)
            det.add_observation("cell_painting", obs)

        report = det.report()
        assert report["cell_painting"]["drift_detected"] == False

    def test_drift_with_trainer_reference_detects_shift(self):
        """Drift is detected when new observations shift from trainer reference."""
        from protophen.serving.monitoring import DriftDetector

        rng = np.random.default_rng(42)
        det = DriftDetector(window_size=50, significance=0.05)

        # Set reference from "training" predictions: N(0, 1)
        ref_preds = rng.standard_normal((200, 10))
        det.set_reference_from_trainer({
            "cell_painting_predictions": ref_preds,
        })

        # Feed observations from a shifted distribution: N(5, 1)
        for _ in range(60):
            obs = rng.standard_normal(10) + 5.0
            det.add_observation("cell_painting", obs)

        report = det.report()
        assert report["cell_painting"]["drift_detected"] == True

    def test_drift_reset_clears_trainer_reference(self):
        """reset() clears all state including trainer-set references."""
        from protophen.serving.monitoring import DriftDetector

        det = DriftDetector(window_size=50)

        det.set_reference_from_trainer({
            "cell_painting_predictions": np.random.randn(100, 10),
        })
        assert det._is_reference_set.get("cell_painting") is True

        det.reset()
        assert len(det._reference) == 0
        assert len(det._is_reference_set) == 0


# =========================================================================
# Registry Trainer Integration Tests (Session 10.2)
# =========================================================================

class TestRegistryTrainerIntegration:
    """
    Tests for ``ModelRegistry.register_from_trainer_checkpoint()``.

    This method automatically extracts epoch, metrics, config, and
    TrainerConfig from checkpoints produced by the Trainer (Session 6).
    """

    @pytest.fixture
    def tiny_model(self):
        from protophen.models.protophen import ProToPhenConfig, ProToPhenModel

        config = ProToPhenConfig(
            protein_embedding_dim=32,
            encoder_hidden_dims=[16],
            encoder_output_dim=8,
            decoder_hidden_dims=[16],
            cell_painting_dim=10,
            predict_viability=False,
            predict_transcriptomics=False,
            mc_dropout=True,
        )
        return ProToPhenModel(config)

    @pytest.fixture
    def trainer_checkpoint(self, tiny_model, tmp_dir) -> Path:
        """Trainer.save_checkpoint() format."""
        from protophen.training.trainer import TrainerConfig
        from dataclasses import asdict

        ckpt_path = tmp_dir / "trainer_ckpt.pt"
        torch.save(
            {
                "epoch": 30,
                "global_step": 1500,
                "model_state_dict": tiny_model.state_dict(),
                "optimiser_state_dict": {},
                "config": asdict(TrainerConfig(
                    epochs=100,
                    learning_rate=3e-4,
                    tasks=["cell_painting"],
                )),
                "best_val_loss": 0.0567,
            },
            ckpt_path,
        )
        return ckpt_path

    @pytest.fixture
    def callback_best_checkpoint(self, tiny_model, tmp_dir) -> Path:
        """CheckpointCallback best_model.pt format."""
        from protophen.training.trainer import TrainerConfig

        ckpt_path = tmp_dir / "best_model.pt"
        torch.save(
            {
                "epoch": 45,
                "global_step": 2250,
                "model_state_dict": tiny_model.state_dict(),
                "optimiser_state_dict": {},
                "config": TrainerConfig(
                    epochs=100,
                    learning_rate=1e-4,
                ).__dict__,
                "best_value": 0.0289,
                "monitor": "val_loss",
            },
            ckpt_path,
        )
        return ckpt_path

    def test_register_from_trainer_basic(self, trainer_checkpoint, tmp_dir):
        """register_from_trainer_checkpoint creates a valid ModelVersion."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register_from_trainer_checkpoint(trainer_checkpoint)

        assert mv.version == "v1"
        assert mv.stage == "staging"
        assert mv.epoch == 30
        assert "trainer_checkpoint" in mv.tags

    def test_register_from_trainer_extracts_metrics(self, trainer_checkpoint, tmp_dir):
        """Metrics are extracted from best_val_loss."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register_from_trainer_checkpoint(trainer_checkpoint)

        assert "best_val_loss" in mv.metrics
        assert mv.metrics["best_val_loss"] == pytest.approx(0.0567)

    def test_register_from_callback_extracts_metrics(
        self, callback_best_checkpoint, tmp_dir
    ):
        """Metrics from CheckpointCallback (best_value + monitor) are extracted."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register_from_trainer_checkpoint(callback_best_checkpoint)

        assert "best_val_loss" in mv.metrics
        assert mv.metrics["best_val_loss"] == pytest.approx(0.0289)
        assert mv.epoch == 45

    def test_register_from_trainer_extracts_trainer_config(
        self, trainer_checkpoint, tmp_dir
    ):
        """TrainerConfig dict is stored in the ModelVersion."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register_from_trainer_checkpoint(trainer_checkpoint)

        assert mv.trainer_config is not None
        assert mv.trainer_config["learning_rate"] == 3e-4
        assert "cell_painting" in mv.trainer_config["tasks"]

    def test_register_from_trainer_auto_description(
        self, trainer_checkpoint, tmp_dir
    ):
        """Auto-generated description includes epoch and best metric."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register_from_trainer_checkpoint(trainer_checkpoint)

        assert "epoch 30" in mv.description.lower()
        assert "0.0567" in mv.description

    def test_register_from_trainer_custom_version(
        self, trainer_checkpoint, tmp_dir
    ):
        """Custom version label is respected."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register_from_trainer_checkpoint(
            trainer_checkpoint, version="experiment_1"
        )

        assert mv.version == "experiment_1"

    def test_register_from_trainer_copies_checkpoint(
        self, trainer_checkpoint, tmp_dir
    ):
        """Checkpoint is copied into the registry directory."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        registry_dir = tmp_dir / "registry"
        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(registry_dir))
        )

        mv = reg.register_from_trainer_checkpoint(trainer_checkpoint)

        stored_path = Path(mv.checkpoint_path)
        assert stored_path.exists()
        assert str(registry_dir) in str(stored_path)

    def test_register_from_trainer_then_promote(
        self, trainer_checkpoint, tmp_dir
    ):
        """Full workflow: register from trainer → promote to production."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register_from_trainer_checkpoint(trainer_checkpoint)
        reg.set_stage(mv.version, "production")

        prod = reg.get_production_checkpoint()
        assert prod is not None
        assert Path(prod).exists()

    def test_register_from_trainer_then_load_in_pipeline(
        self, trainer_checkpoint, tmp_dir
    ):
        """
        End-to-end: register trainer checkpoint → retrieve from registry
        → load in InferencePipeline.
        """
        from protophen.serving.pipeline import InferencePipeline, PipelineConfig
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(tmp_dir / "registry"))
        )

        mv = reg.register_from_trainer_checkpoint(trainer_checkpoint)
        reg.set_stage(mv.version, "production")

        # Retrieve the production checkpoint path
        ckpt_path = reg.get_production_checkpoint()

        # Load into pipeline
        config = PipelineConfig(device="cpu", use_fp16=False)
        pipe = InferencePipeline(config=config)
        pipe.load_model(ckpt_path)

        assert pipe.is_ready
        info = pipe.get_model_info()
        assert info["protein_embedding_dim"] == 32

    def test_register_multiple_trainer_checkpoints_compare(
        self, tiny_model, tmp_dir
    ):
        """Register multiple trainer checkpoints and compare their metrics."""
        from dataclasses import asdict

        from protophen.serving.registry import ModelRegistry, RegistryConfig
        from protophen.training.trainer import TrainerConfig

        registry_dir = tmp_dir / "registry"
        reg = ModelRegistry(
            config=RegistryConfig(registry_dir=str(registry_dir))
        )

        # Create two trainer checkpoints with different metrics
        for epoch, val_loss in [(20, 0.08), (50, 0.04)]:
            ckpt_path = tmp_dir / f"ckpt_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": epoch * 50,
                    "model_state_dict": tiny_model.state_dict(),
                    "optimiser_state_dict": {},
                    "config": asdict(TrainerConfig(epochs=100)),
                    "best_val_loss": val_loss,
                },
                ckpt_path,
            )
            reg.register_from_trainer_checkpoint(ckpt_path)

        # Compare
        cmp = reg.compare_versions("v1", "v2")
        assert cmp["metrics"]["best_val_loss"]["delta"] == pytest.approx(-0.04)

        # Best version by val_loss (lower is better)
        best = reg.get_best_version("best_val_loss", higher_is_better=False)
        assert best.version == "v2"
        assert best.epoch == 50

    def test_register_from_trainer_persists(self, trainer_checkpoint, tmp_dir):
        """Trainer-registered version survives registry re-instantiation."""
        from protophen.serving.registry import ModelRegistry, RegistryConfig

        registry_dir = str(tmp_dir / "registry")

        reg1 = ModelRegistry(config=RegistryConfig(registry_dir=registry_dir))
        reg1.register_from_trainer_checkpoint(
            trainer_checkpoint, version="trainer_v1"
        )

        # Re-open
        reg2 = ModelRegistry(config=RegistryConfig(registry_dir=registry_dir))
        mv = reg2.get_version("trainer_v1")

        assert mv.epoch == 30
        assert mv.metrics["best_val_loss"] == pytest.approx(0.0567)
        assert mv.trainer_config is not None
        assert "trainer_checkpoint" in mv.tags