"""
Unit tests for ProToPhen model architecture.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from protophen.models.encoders import (
    AttentionEncoder,
    MLPBlock,
    ProteinEncoder,
    ProteinEncoderConfig,
)
from protophen.models.decoders import (
    CellPaintingHead,
    MultiTaskHead,
    PhenotypeDecoder,
    TranscriptomicsHead,
    ViabilityHead,
)
from protophen.models.protophen import (
    ProToPhenConfig,
    ProToPhenModel,
    create_lightweight_model,
    create_protophen_model,
)
from protophen.models.losses import (
    CellPaintingLoss,
    CombinedLoss,
    CorrelationLoss,
    GaussianNLLLoss,
    HuberLoss,
    MSELoss,
    MultiTaskLoss,
    UncertaintyWeightedLoss,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def batch_embeddings():
    """Create batch of protein embeddings."""
    return torch.randn(8, 1280)


@pytest.fixture
def batch_fused_embeddings():
    """Create batch of fused embeddings (ESM-2 + physicochemical)."""
    return torch.randn(8, 1719)


@pytest.fixture
def batch_phenotypes():
    """Create batch of phenotype targets."""
    return {
        "cell_painting": torch.randn(8, 1500),
        "viability": torch.randn(8, 1),
    }


# =============================================================================
# MLP Block Tests
# =============================================================================

class TestMLPBlock:
    """Tests for MLPBlock."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        block = MLPBlock(
            in_features=256,
            out_features=128,
            activation="gelu",
            dropout=0.1,
        )
        
        x = torch.randn(8, 256)
        out = block(x)
        
        assert out.shape == (8, 128)

    def test_residual_connection(self):
        """Test residual connection when dimensions match."""
        block = MLPBlock(
            in_features=256,
            out_features=256,
            use_residual=True,
        )
        
        x = torch.randn(8, 256)
        out = block(x)
        
        assert out.shape == (8, 256)

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "gelu", "silu", "tanh"]:
            block = MLPBlock(
                in_features=128,
                out_features=64,
                activation=activation,
            )
            
            x = torch.randn(4, 128)
            out = block(x)
            
            assert out.shape == (4, 64)


# =============================================================================
# Protein Encoder Tests
# =============================================================================

class TestProteinEncoder:
    """Tests for ProteinEncoder."""

    def test_default_encoder(self, batch_embeddings):
        """Test encoder with default configuration."""
        config = ProteinEncoderConfig(input_dim=1280)
        encoder = ProteinEncoder(config)
        
        out = encoder(batch_embeddings)
        
        assert out.shape == (8, config.output_dim)

    def test_custom_encoder(self, batch_fused_embeddings):
        """Test encoder with custom configuration."""
        config = ProteinEncoderConfig(
            input_dim=1719,
            hidden_dims=[1024, 512],
            output_dim=256,
            dropout=0.2,
        )
        encoder = ProteinEncoder(config)
        
        out = encoder(batch_fused_embeddings)
        
        assert out.shape == (8, 256)

    def test_encoder_with_hidden_states(self, batch_embeddings):
        """Test encoder returning hidden states."""
        config = ProteinEncoderConfig(
            input_dim=1280,
            hidden_dims=[512, 256],
        )
        encoder = ProteinEncoder(config)
        
        out, hidden = encoder(batch_embeddings, return_hidden=True)
        
        assert len(hidden) == 2  # Two hidden layers
        assert hidden[0].shape == (8, 512)
        assert hidden[1].shape == (8, 256)

    def test_encoder_output_dim_property(self):
        """Test output_dim property."""
        config = ProteinEncoderConfig(output_dim=128)
        encoder = ProteinEncoder(config)
        
        assert encoder.output_dim == 128


class TestAttentionEncoder:
    """Tests for AttentionEncoder."""

    def test_attention_encoder(self, batch_embeddings):
        """Test attention-based encoder."""
        encoder = AttentionEncoder(
            input_dim=1280,
            hidden_dim=512,
            output_dim=256,
            n_heads=8,
            n_layers=2,
        )
        
        out = encoder(batch_embeddings)
        
        assert out.shape == (8, 256)

    def test_attention_encoder_with_sequence(self):
        """Test attention encoder with sequence input."""
        encoder = AttentionEncoder(
            input_dim=320,
            hidden_dim=256,
            output_dim=128,
        )
        
        # Sequence of embeddings
        x = torch.randn(4, 10, 320)  # (batch, seq_len, features)
        out = encoder(x)
        
        assert out.shape == (4, 128)


# =============================================================================
# Decoder Tests
# =============================================================================

class TestPhenotypeDecoder:
    """Tests for PhenotypeDecoder."""

    def test_basic_decoder(self):
        """Test basic decoder."""
        decoder = PhenotypeDecoder(
            input_dim=256,
            output_dim=1500,
            hidden_dims=[512, 1024],
        )
        
        x = torch.randn(8, 256)
        out = decoder(x)
        
        assert out.shape == (8, 1500)


class TestCellPaintingHead:
    """Tests for CellPaintingHead."""

    def test_cell_painting_head(self):
        """Test Cell Painting prediction head."""
        head = CellPaintingHead(
            input_dim=256,
            output_dim=1500,
        )
        
        x = torch.randn(8, 256)
        out = head(x)
        
        assert out.shape == (8, 1500)

    def test_cell_painting_with_uncertainty(self):
        """Test Cell Painting head with uncertainty prediction."""
        head = CellPaintingHead(
            input_dim=256,
            output_dim=1500,
            predict_uncertainty=True,
        )
        
        x = torch.randn(8, 256)
        mean, log_var = head(x)
        
        assert mean.shape == (8, 1500)
        assert log_var.shape == (8, 1500)


class TestViabilityHead:
    """Tests for ViabilityHead."""

    def test_viability_sigmoid(self):
        """Test viability head with sigmoid output."""
        head = ViabilityHead(
            input_dim=256,
            output_type="sigmoid",
        )
        
        x = torch.randn(8, 256)
        out = head(x)
        
        assert out.shape == (8, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_viability_beta(self):
        """Test viability head with beta distribution output."""
        head = ViabilityHead(
            input_dim=256,
            output_type="beta",
        )
        
        x = torch.randn(8, 256)
        alpha, beta = head(x)
        
        assert alpha.shape == (8, 1)
        assert beta.shape == (8, 1)
        assert (alpha > 0).all()
        assert (beta > 0).all()


class TestMultiTaskHead:
    """Tests for MultiTaskHead."""

    def test_multi_task_head(self):
        """Test multi-task prediction head."""
        head = MultiTaskHead(
            input_dim=256,
            task_configs={
                "cell_painting": {"type": "cell_painting", "output_dim": 1500},
                "viability": {"type": "viability"},
            },
        )
        
        x = torch.randn(8, 256)
        outputs = head(x)
        
        assert "cell_painting" in outputs
        assert "viability" in outputs
        assert outputs["cell_painting"].shape == (8, 1500)

    def test_multi_task_selective(self):
        """Test computing only selected tasks."""
        head = MultiTaskHead(
            input_dim=256,
            task_configs={
                "cell_painting": {"type": "cell_painting", "output_dim": 1500},
                "viability": {"type": "viability"},
            },
        )
        
        x = torch.randn(8, 256)
        outputs = head(x, tasks=["cell_painting"])
        
        assert "cell_painting" in outputs
        assert "viability" not in outputs


# =============================================================================
# ProToPhen Model Tests
# =============================================================================

class TestProToPhenModel:
    """Tests for the main ProToPhen model."""

    def test_model_creation(self):
        """Test model creation with default config."""
        model = ProToPhenModel()
        
        assert model is not None
        assert "cell_painting" in model.task_names

    def test_model_forward(self, batch_fused_embeddings):
        """Test model forward pass."""
        config = ProToPhenConfig(
            protein_embedding_dim=1719,
            cell_painting_dim=1500,
        )
        model = ProToPhenModel(config)
        
        outputs = model(batch_fused_embeddings)
        
        assert "cell_painting" in outputs
        assert outputs["cell_painting"].shape == (8, 1500)

    def test_model_with_viability(self, batch_fused_embeddings):
        """Test model with viability prediction."""
        config = ProToPhenConfig(
            protein_embedding_dim=1719,
            predict_viability=True,
        )
        model = ProToPhenModel(config)
        
        outputs = model(batch_fused_embeddings)
        
        assert "viability" in outputs
        assert outputs["viability"].shape == (8, 1)

    def test_model_return_latent(self, batch_fused_embeddings):
        """Test returning latent representation."""
        model = ProToPhenModel(
            protein_embedding_dim=1719,
            encoder_output_dim=256,
        )
        
        outputs = model(batch_fused_embeddings, return_latent=True)
        
        assert "latent" in outputs
        assert outputs["latent"].shape == (8, 256)

    def test_model_predict(self, batch_fused_embeddings):
        """Test prediction mode."""
        model = ProToPhenModel(protein_embedding_dim=1719)
        
        predictions = model.predict(batch_fused_embeddings)
        
        assert "cell_painting" in predictions

    def test_model_freeze_encoder(self, batch_fused_embeddings):
        """Test freezing encoder."""
        model = ProToPhenModel(protein_embedding_dim=1719)
        
        model.freeze_encoder()
        
        for param in model.encoder.parameters():
            assert not param.requires_grad
        
        model.unfreeze_encoder()
        
        for param in model.encoder.parameters():
            assert param.requires_grad

    def test_model_add_task(self):
        """Test adding a new task."""
        model = ProToPhenModel(protein_embedding_dim=1719)
        
        model.add_task("new_task", output_dim=100)
        
        assert "new_task" in model.task_names

    def test_model_summary(self):
        """Test model summary."""
        model = ProToPhenModel(protein_embedding_dim=1719)
        
        summary = model.summary()
        
        assert "n_parameters" in summary
        assert "tasks" in summary
        assert "latent_dim" in summary

    def test_create_lightweight_model(self, batch_fused_embeddings):
        """Test creating lightweight model."""
        model = create_lightweight_model(
            protein_embedding_dim=1719,
            cell_painting_dim=1500,
        )
        
        outputs = model(batch_fused_embeddings)
        
        assert outputs["cell_painting"].shape == (8, 1500)

    def test_mc_dropout_uncertainty(self, batch_fused_embeddings):
        """Test MC Dropout uncertainty estimation."""
        config = ProToPhenConfig(
            protein_embedding_dim=1719,
            mc_dropout=True,
            encoder_dropout=0.1,
        )
        model = ProToPhenModel(config)
        
        results = model.predict_with_uncertainty(
            batch_fused_embeddings,
            n_samples=5,
        )
        
        assert "cell_painting" in results
        assert "mean" in results["cell_painting"]
        assert "std" in results["cell_painting"]


# =============================================================================
# Loss Function Tests
# =============================================================================

class TestLossFunctions:
    """Tests for loss functions."""

    def test_mse_loss(self):
        """Test MSE loss."""
        loss_fn = MSELoss()
        
        pred = torch.randn(8, 100)
        target = torch.randn(8, 100)
        
        loss = loss_fn(pred, target)
        
        assert loss.shape == ()
        assert loss >= 0

    def test_mse_loss_with_mask(self):
        """Test MSE loss with masking."""
        loss_fn = MSELoss()
        
        pred = torch.randn(8, 100)
        target = torch.randn(8, 100)
        mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
        
        loss = loss_fn(pred, target, mask=mask)
        
        assert loss.shape == ()

    def test_huber_loss(self):
        """Test Huber loss."""
        loss_fn = HuberLoss(delta=1.0)
        
        pred = torch.randn(8, 100)
        target = torch.randn(8, 100)
        
        loss = loss_fn(pred, target)
        
        assert loss >= 0

    def test_correlation_loss(self):
        """Test correlation loss."""
        loss_fn = CorrelationLoss()
        
        pred = torch.randn(8, 100)
        target = pred + torch.randn(8, 100) * 0.1  # Highly correlated
        
        loss = loss_fn(pred, target)
        
        assert loss.shape == ()
        assert loss < 0.5  # Should be low for correlated data

    def test_cell_painting_loss(self):
        """Test Cell Painting loss."""
        loss_fn = CellPaintingLoss(
            mse_weight=1.0,
            correlation_weight=0.1,
        )
        
        pred = torch.randn(8, 1500)
        target = torch.randn(8, 1500)
        
        losses = loss_fn(pred, target)
        
        assert "total" in losses
        assert "mse" in losses
        assert "correlation" in losses

    def test_gaussian_nll_loss(self):
        """Test Gaussian NLL loss."""
        loss_fn = GaussianNLLLoss()
        
        mean = torch.randn(8, 100)
        log_var = torch.randn(8, 100)
        target = torch.randn(8, 100)
        
        loss = loss_fn(mean, log_var, target)
        
        assert loss.shape == ()

    def test_multi_task_loss(self, batch_phenotypes):
        """Test multi-task loss."""
        loss_fn = MultiTaskLoss(
            task_weights={"cell_painting": 1.0, "viability": 0.5},
        )
        
        predictions = {
            "cell_painting": torch.randn(8, 1500),
            "viability": torch.randn(8, 1),
        }
        
        losses = loss_fn(predictions, batch_phenotypes)
        
        assert "total" in losses
        assert "cell_painting" in losses
        assert "viability" in losses

    def test_uncertainty_weighted_loss(self, batch_phenotypes):
        """Test uncertainty-weighted multi-task loss."""
        loss_fn = UncertaintyWeightedLoss(
            task_names=["cell_painting", "viability"],
        )
        
        predictions = {
            "cell_painting": torch.randn(8, 1500),
            "viability": torch.randn(8, 1),
        }
        
        losses = loss_fn(predictions, batch_phenotypes)
        
        assert "total" in losses
        
        # Check that weights are learnable
        weights = loss_fn.get_task_weights()
        assert "cell_painting" in weights
        assert "viability" in weights

    def test_combined_loss(self, batch_phenotypes):
        """Test combined loss."""
        loss_fn = CombinedLoss(
            tasks=["cell_painting", "viability"],
            task_weights={"cell_painting": 1.0, "viability": 0.5},
            cell_painting_config={"mse_weight": 1.0, "correlation_weight": 0.1},
        )
        
        predictions = {
            "cell_painting": torch.randn(8, 1500),
            "viability": torch.randn(8, 1),
        }
        
        losses = loss_fn(predictions, batch_phenotypes)
        
        assert "total" in losses