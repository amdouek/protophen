"""
Tests for ProToPhen PhenotypeAutoencoder and related components.

Covers:
- PhenotypeAutoencoder (deterministic and variational)
- AutoencoderDecoderHead (CellPaintingHead compatibility)
- NTXentLoss / NTXentLossVectorised
- AutoencoderLoss
- PretrainingDataset
- PretrainingConfig (YAML round-trip)
- Checkpoint save/load helpers
- Latent space quality utilities
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from protophen.models.autoencoder import (
    AutoencoderDecoderHead,
    AutoencoderLoss,
    NTXentLoss,
    NTXentLossVectorised,
    Phase1Config,
    Phase1LossConfig,
    Phase2Config,
    PhenotypeAutoencoder,
    PhenotypeAutoencoderConfig,
    PretrainingConfig,
    PretrainingDataset,
    compute_latent_silhouette,
    compute_replicate_correlation,
    load_autoencoder_from_checkpoint,
    save_phase1_checkpoint,
    save_phase2_checkpoint,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ae_config() -> PhenotypeAutoencoderConfig:
    """Small autoencoder config for fast tests."""
    return PhenotypeAutoencoderConfig(
        input_dim=100,
        latent_dim=16,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=None,  # symmetric
        activation="gelu",
        dropout=0.0,  # deterministic for tests
        use_layer_norm=True,
        use_residual=False,
        use_skip_connections=True,
        variational=False,
    )


@pytest.fixture
def vae_config() -> PhenotypeAutoencoderConfig:
    """Small VAE config for tests."""
    return PhenotypeAutoencoderConfig(
        input_dim=100,
        latent_dim=16,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=None,
        dropout=0.0,
        use_skip_connections=False,
        variational=True,
    )


@pytest.fixture
def ae(ae_config: PhenotypeAutoencoderConfig) -> PhenotypeAutoencoder:
    return PhenotypeAutoencoder(ae_config)


@pytest.fixture
def vae(vae_config: PhenotypeAutoencoderConfig) -> PhenotypeAutoencoder:
    return PhenotypeAutoencoder(vae_config)


@pytest.fixture
def batch_features() -> torch.Tensor:
    return torch.randn(8, 100)


@pytest.fixture
def treatment_labels() -> torch.Tensor:
    # 8 samples, 4 unique treatments (2 replicates each)
    return torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])


@pytest.fixture
def pretraining_config() -> PretrainingConfig:
    return PretrainingConfig()

@pytest.fixture
def matching_pretraining_config(ae_config) -> PretrainingConfig:
    """PretrainingConfig whose autoencoder section matches the ae fixture."""
    return PretrainingConfig(autoencoder=ae_config)


# ============================================================================
# PhenotypeAutoencoder — deterministic (AE)
# ============================================================================


class TestPhenotypeAutoencoder:
    """Tests for the deterministic autoencoder."""

    def test_init(self, ae: PhenotypeAutoencoder):
        assert ae.input_dim == 100
        assert ae.latent_dim == 16
        assert ae.output_dim == 100
        assert ae.n_parameters > 0
        assert not ae.config.variational

    def test_forward_shape(self, ae, batch_features):
        out = ae(batch_features)
        assert "reconstruction" in out
        assert "latent" in out
        assert out["reconstruction"].shape == (8, 100)
        assert out["latent"].shape == (8, 16)

    def test_forward_no_vae_keys(self, ae, batch_features):
        out = ae(batch_features)
        assert "mu" not in out
        assert "log_var" not in out

    def test_encode_returns_hiddens(self, ae, batch_features):
        enc_out = ae.encode(batch_features)
        assert "latent" in enc_out
        assert "encoder_hiddens" in enc_out
        # 2 encoder hidden layers
        assert len(enc_out["encoder_hiddens"]) == 2
        assert enc_out["encoder_hiddens"][0].shape == (8, 64)
        assert enc_out["encoder_hiddens"][1].shape == (8, 32)

    def test_decode_without_skips(self, ae, batch_features):
        enc_out = ae.encode(batch_features)
        recon = ae.decode(enc_out["latent"], encoder_hiddens=None)
        assert recon.shape == (8, 100)

    def test_decode_with_skips(self, ae, batch_features):
        enc_out = ae.encode(batch_features)
        recon = ae.decode(
            enc_out["latent"],
            encoder_hiddens=enc_out["encoder_hiddens"],
        )
        assert recon.shape == (8, 100)

    def test_symmetric_decoder_dims(self, ae_config):
        assert ae_config.effective_decoder_hidden_dims == [32, 64]

    def test_custom_decoder_dims(self):
        cfg = PhenotypeAutoencoderConfig(
            input_dim=100,
            latent_dim=16,
            encoder_hidden_dims=[64, 32],
            decoder_hidden_dims=[48, 80],
            use_skip_connections=False,
        )
        ae = PhenotypeAutoencoder(cfg)
        out = ae(torch.randn(4, 100))
        assert out["reconstruction"].shape == (4, 100)

    def test_no_skip_connections(self):
        cfg = PhenotypeAutoencoderConfig(
            input_dim=50,
            latent_dim=8,
            encoder_hidden_dims=[32],
            use_skip_connections=False,
        )
        ae = PhenotypeAutoencoder(cfg)
        out = ae(torch.randn(4, 50))
        assert out["reconstruction"].shape == (4, 50)

    def test_freeze_encoder(self, ae):
        ae.freeze_encoder()
        for layer in ae.encoder_layers:
            for p in layer.parameters():
                assert not p.requires_grad
        for p in ae.latent_proj.parameters():
            assert not p.requires_grad

    def test_unfreeze_encoder(self, ae):
        ae.freeze_encoder()
        ae.unfreeze_encoder()
        for layer in ae.encoder_layers:
            for p in layer.parameters():
                assert p.requires_grad

    def test_freeze_decoder(self, ae):
        ae.freeze_decoder()
        for p in ae.decoder_input_proj.parameters():
            assert not p.requires_grad
        for layer in ae.decoder_layers:
            for p in layer.parameters():
                assert not p.requires_grad
        for p in ae.output_proj.parameters():
            assert not p.requires_grad

    def test_unfreeze_decoder(self, ae):
        ae.freeze_decoder()
        ae.unfreeze_decoder()
        for p in ae.decoder_input_proj.parameters():
            assert p.requires_grad

    def test_n_trainable_after_freeze(self, ae):
        total = ae.n_parameters
        ae.freeze_encoder()
        trainable = ae.n_trainable_parameters
        assert trainable < total
        assert trainable > 0  # decoder still trainable

    def test_summary(self, ae):
        s = ae.summary()
        assert s["input_dim"] == 100
        assert s["latent_dim"] == 16
        assert s["variational"] is False
        assert s["skip_connections"] is True

    def test_repr(self, ae):
        r = repr(ae)
        assert "PhenotypeAutoencoder" in r
        assert "100" in r
        assert "16" in r

    def test_gradient_flows(self, ae, batch_features):
        out = ae(batch_features)
        loss = out["reconstruction"].mean()
        loss.backward()
        # Check that encoder and decoder gradients are non-None
        for p in ae.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_kwargs_override(self):
        ae = PhenotypeAutoencoder(
            input_dim=50, latent_dim=10,
            encoder_hidden_dims=[32],
            use_skip_connections=False,
        )
        assert ae.input_dim == 50
        assert ae.latent_dim == 10

    def test_config_override(self, ae_config):
        ae = PhenotypeAutoencoder(ae_config, latent_dim=32)
        assert ae.latent_dim == 32


# ============================================================================
# PhenotypeAutoencoder — variational (VAE)
# ============================================================================


class TestVariationalAutoencoder:
    """Tests specific to VAE mode."""

    def test_forward_vae_keys(self, vae, batch_features):
        out = vae(batch_features)
        assert "mu" in out
        assert "log_var" in out
        assert out["mu"].shape == (8, 16)
        assert out["log_var"].shape == (8, 16)

    def test_training_vs_eval_latent(self, vae, batch_features):
        """In eval mode the VAE should use mu directly (no sampling)."""
        vae.train()
        out_train1 = vae(batch_features)
        out_train2 = vae(batch_features)
        # Stochastic — latent should (almost certainly) differ
        # Note: extremely unlikely to be exactly equal with random sampling
        # but we test the structure, not stochasticity

        vae.eval()
        out_eval1 = vae(batch_features)
        out_eval2 = vae(batch_features)
        # Deterministic — latent should be identical
        torch.testing.assert_close(out_eval1["latent"], out_eval2["latent"])

    def test_log_var_clamped(self, vae, batch_features):
        out = vae(batch_features)
        assert out["log_var"].min() >= -10.0
        assert out["log_var"].max() <= 10.0

    def test_freeze_encoder_vae(self, vae):
        vae.freeze_encoder()
        for p in vae.latent_mu.parameters():
            assert not p.requires_grad
        for p in vae.latent_log_var.parameters():
            assert not p.requires_grad


# ============================================================================
# AutoencoderDecoderHead
# ============================================================================


class TestAutoencoderDecoderHead:
    """Tests for the CellPaintingHead-compatible decoder wrapper."""

    def test_construction(self, ae):
        head = ae.get_decoder_head(freeze=True)
        assert isinstance(head, AutoencoderDecoderHead)
        assert head.output_dim == 100
        assert head.input_dim == 16

    def test_forward_shape(self, ae):
        head = ae.get_decoder_head(freeze=True)
        latent = torch.randn(4, 16)
        out = head(latent)
        assert out.shape == (4, 100)

    def test_frozen_parameters(self, ae):
        head = ae.get_decoder_head(freeze=True)
        for p in head.parameters():
            assert not p.requires_grad

    def test_unfrozen_parameters(self, ae):
        head = ae.get_decoder_head(freeze=False)
        has_trainable = any(p.requires_grad for p in head.parameters())
        assert has_trainable

    def test_last_latent_cache(self, ae):
        head = ae.get_decoder_head(freeze=True)
        latent = torch.randn(4, 16)
        _ = head(latent)
        cached = head.get_last_latent()
        assert cached is not None
        torch.testing.assert_close(cached, latent)

    def test_drop_in_compatibility_with_protophen(self, ae):
        """Verify that the head can replace CellPaintingHead in a model dict."""
        from protophen.models.protophen import ProToPhenModel, ProToPhenConfig

        model_config = ProToPhenConfig(
            protein_embedding_dim=50,
            encoder_hidden_dims=[32],
            encoder_output_dim=16,  # matches ae latent_dim
            cell_painting_dim=100,
            predict_viability=False,
        )
        model = ProToPhenModel(model_config)

        # Replace the cell painting head
        model.decoders["cell_painting"] = ae.get_decoder_head(freeze=True)

        # Forward pass should still work
        protein_emb = torch.randn(4, 50)
        out = model(protein_emb)
        assert out["cell_painting"].shape == (4, 100)


# ============================================================================
# NTXentLoss
# ============================================================================


class TestNTXentLoss:
    """Tests for the contrastive loss."""

    def test_basic(self):
        loss_fn = NTXentLoss(temperature=0.1)
        latent = torch.randn(8, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(latent, labels)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_all_same_label(self):
        loss_fn = NTXentLoss(temperature=0.1)
        latent = torch.randn(4, 16)
        labels = torch.tensor([0, 0, 0, 0])
        loss = loss_fn(latent, labels)
        # Should still compute (all pairs are positive)
        assert loss.item() >= 0

    def test_all_unique_labels(self):
        loss_fn = NTXentLoss(temperature=0.1)
        latent = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(latent, labels)
        # No positive pairs → loss should be 0
        assert loss.item() == 0.0

    def test_single_sample(self):
        loss_fn = NTXentLoss(temperature=0.1)
        latent = torch.randn(1, 16)
        labels = torch.tensor([0])
        loss = loss_fn(latent, labels)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        loss_fn = NTXentLoss(temperature=0.1)
        latent = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(latent, labels)
        loss.backward()
        assert latent.grad is not None


class TestNTXentLossVectorised:
    """Tests for the vectorised NT-Xent loss."""

    def test_basic(self):
        loss_fn = NTXentLossVectorised(temperature=0.1)
        latent = torch.randn(8, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(latent, labels)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_matches_loop_version(self):
        """Both implementations should give similar results."""
        torch.manual_seed(42)
        latent = torch.randn(16, 32)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5])

        loop_loss = NTXentLoss(temperature=0.1)(latent, labels)
        vec_loss = NTXentLossVectorised(temperature=0.1)(latent, labels)

        torch.testing.assert_close(loop_loss, vec_loss, atol=1e-5, rtol=1e-5)

    def test_all_unique_labels(self):
        loss_fn = NTXentLossVectorised(temperature=0.1)
        latent = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(latent, labels)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        loss_fn = NTXentLossVectorised(temperature=0.1)
        latent = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(latent, labels)
        loss.backward()
        assert latent.grad is not None

# ============================================================================
# AutoencoderLoss
# ============================================================================


class TestAutoencoderLoss:
    """Tests for the combined Phase 1 loss."""

    def test_basic_ae(self, ae, batch_features, treatment_labels):
        loss_fn = AutoencoderLoss(variational=False)
        outputs = ae(batch_features)
        losses = loss_fn(outputs, batch_features, treatment_labels)

        assert "total" in losses
        assert "reconstruction" in losses
        assert "contrastive" in losses
        assert "kl" in losses
        assert losses["total"].ndim == 0
        assert losses["reconstruction"].item() >= 0
        assert losses["kl"].item() == 0.0  # Not variational

    def test_basic_vae(self, vae, batch_features, treatment_labels):
        loss_fn = AutoencoderLoss(
            config=Phase1LossConfig(kl_weight=0.01),
            variational=True,
        )
        vae.train()
        outputs = vae(batch_features)
        losses = loss_fn(outputs, batch_features, treatment_labels)

        assert losses["kl"].item() > 0  # KL should be positive for VAE
        assert losses["total"].item() > 0

    def test_no_contrastive_when_weight_zero(self, ae, batch_features):
        loss_fn = AutoencoderLoss(
            config=Phase1LossConfig(contrastive_weight=0.0),
        )
        outputs = ae(batch_features)
        losses = loss_fn(outputs, batch_features, treatment_labels=None)

        assert losses["contrastive"].item() == 0.0

    def test_no_contrastive_when_labels_none(self, ae, batch_features):
        loss_fn = AutoencoderLoss(
            config=Phase1LossConfig(contrastive_weight=0.5),
        )
        outputs = ae(batch_features)
        losses = loss_fn(outputs, batch_features, treatment_labels=None)

        assert losses["contrastive"].item() == 0.0

    def test_huber_reconstruction(self, ae, batch_features, treatment_labels):
        loss_fn = AutoencoderLoss(
            config=Phase1LossConfig(reconstruction_type="huber", huber_delta=0.5),
        )
        outputs = ae(batch_features)
        losses = loss_fn(outputs, batch_features, treatment_labels)

        assert losses["reconstruction"].item() >= 0
        assert losses["total"].item() > 0

    def test_feature_group_weights(self, ae, batch_features, treatment_labels):
        # Split 100 features into two groups
        group_indices = {
            "nucleus": torch.arange(0, 50),
            "cytoplasm": torch.arange(50, 100),
        }
        loss_fn = AutoencoderLoss(
            config=Phase1LossConfig(
                feature_group_weights={"nucleus": 2.0, "cytoplasm": 0.5},
            ),
            feature_group_indices=group_indices,
        )
        outputs = ae(batch_features)
        losses = loss_fn(outputs, batch_features, treatment_labels)

        assert losses["total"].item() > 0
        # Verify the buffer was created
        assert loss_fn.feature_weights is not None
        assert loss_fn.feature_weights.shape == (100,)
        assert loss_fn.feature_weights[:50].unique().item() == 2.0
        assert loss_fn.feature_weights[50:].unique().item() == 0.5

    def test_gradient_flows_through_loss(self, ae, batch_features, treatment_labels):
        loss_fn = AutoencoderLoss(variational=False)
        outputs = ae(batch_features)
        losses = loss_fn(outputs, batch_features, treatment_labels)
        losses["total"].backward()

        has_grad = False
        for p in ae.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_loss_weights_affect_total(self, ae, batch_features, treatment_labels):
        """Changing loss weights should change the total loss."""
        outputs = ae(batch_features)

        loss_fn_a = AutoencoderLoss(
            config=Phase1LossConfig(
                reconstruction_weight=1.0,
                contrastive_weight=0.0,
            ),
        )
        loss_fn_b = AutoencoderLoss(
            config=Phase1LossConfig(
                reconstruction_weight=1.0,
                contrastive_weight=1.0,
            ),
        )

        losses_a = loss_fn_a(outputs, batch_features, treatment_labels)
        losses_b = loss_fn_b(outputs, batch_features, treatment_labels)

        # Reconstruction should be the same
        torch.testing.assert_close(
            losses_a["reconstruction"], losses_b["reconstruction"],
        )
        # Total should differ if contrastive loss > 0
        # (it could be 0 if no positive pairs, but with our labels it won't be)
        if losses_b["contrastive"].item() > 0:
            assert losses_a["total"].item() != losses_b["total"].item()


# ============================================================================
# PretrainingDataset
# ============================================================================


class TestPretrainingDataset:
    """Tests for the Phase 1 dataset."""

    @pytest.fixture
    def dataset(self) -> PretrainingDataset:
        features = np.random.randn(100, 50).astype(np.float32)
        labels = [f"gene_{i % 10}" for i in range(100)]
        plates = [f"plate_{i % 5}" for i in range(100)]
        weights = np.ones(100, dtype=np.float32)
        return PretrainingDataset(
            phenotype_features=features,
            treatment_labels=labels,
            plate_ids=plates,
            sample_weights=weights,
            augmentation_noise_std=0.0,
        )

    def test_length(self, dataset):
        assert len(dataset) == 100

    def test_n_features(self, dataset):
        assert dataset.n_features == 50

    def test_n_treatments(self, dataset):
        assert dataset.n_treatments == 10

    def test_getitem_keys(self, dataset):
        sample = dataset[0]
        assert "phenotype_features" in sample
        assert "treatment_label" in sample
        assert "plate_id" in sample
        assert "sample_weight" in sample

    def test_getitem_types(self, dataset):
        sample = dataset[0]
        assert isinstance(sample["phenotype_features"], torch.Tensor)
        assert sample["phenotype_features"].dtype == torch.float32
        assert isinstance(sample["treatment_label"], torch.Tensor)
        assert sample["treatment_label"].dtype == torch.long
        assert isinstance(sample["plate_id"], str)
        assert isinstance(sample["sample_weight"], torch.Tensor)
        assert sample["sample_weight"].dtype == torch.float32

    def test_getitem_shapes(self, dataset):
        sample = dataset[0]
        assert sample["phenotype_features"].shape == (50,)
        assert sample["treatment_label"].shape == ()
        assert sample["sample_weight"].shape == ()

    def test_label_encoding(self, dataset):
        """Same string label should map to same integer."""
        s0 = dataset[0]
        s10 = dataset[10]
        # Both are gene_0
        assert s0["treatment_label"].item() == s10["treatment_label"].item()

    def test_different_labels_different_ints(self, dataset):
        s0 = dataset[0]  # gene_0
        s1 = dataset[1]  # gene_1
        assert s0["treatment_label"].item() != s1["treatment_label"].item()

    def test_augmentation_noise(self):
        features = np.zeros((10, 20), dtype=np.float32)
        labels = list(range(10))
        ds = PretrainingDataset(
            phenotype_features=features,
            treatment_labels=labels,
            augmentation_noise_std=1.0,
        )
        sample = ds[0]
        # With noise, zero features should now be non-zero (extremely likely)
        assert sample["phenotype_features"].abs().sum() > 0

    def test_no_augmentation_noise(self):
        features = np.zeros((10, 20), dtype=np.float32)
        labels = list(range(10))
        ds = PretrainingDataset(
            phenotype_features=features,
            treatment_labels=labels,
            augmentation_noise_std=0.0,
        )
        sample = ds[0]
        assert sample["phenotype_features"].abs().sum() == 0

    def test_default_plate_ids(self):
        features = np.random.randn(5, 10).astype(np.float32)
        labels = list(range(5))
        ds = PretrainingDataset(features, labels)
        sample = ds[0]
        assert sample["plate_id"] == "unknown"

    def test_default_weights(self):
        features = np.random.randn(5, 10).astype(np.float32)
        labels = list(range(5))
        ds = PretrainingDataset(features, labels)
        sample = ds[0]
        assert sample["sample_weight"].item() == 1.0

    def test_split(self, dataset):
        train_ds, val_ds = dataset.split(train_frac=0.8, val_frac=0.2, seed=42)
        assert len(train_ds) == 80
        assert len(val_ds) == 20
        # Val should have no augmentation
        assert val_ds.augmentation_noise_std == 0.0

    def test_split_preserves_label_semantics(self, dataset):
        """After split, same string label should still map to same int."""
        train_ds, val_ds = dataset.split(train_frac=0.8, val_frac=0.2)
        # Verify at least one label appears in both
        train_labels = set(train_ds.label_to_int.keys())
        val_labels = set(val_ds.label_to_int.keys())
        # With 100 samples and 10 labels, overlap is virtually certain
        overlap = train_labels & val_labels
        assert len(overlap) > 0

    def test_numpy_label_array(self):
        features = np.random.randn(10, 5).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        ds = PretrainingDataset(features, labels)
        assert ds.n_treatments == 5

    def test_from_parquet(self, tmp_path):
        """Test loading from Parquet file."""
        import pandas as pd

        n_samples = 50
        n_features = 30
        feature_cols = [f"feat_{i}" for i in range(n_features)]
        data = {col: np.random.randn(n_samples).astype(np.float32) for col in feature_cols}
        data["treatment_label"] = [f"trt_{i % 5}" for i in range(n_samples)]
        data["plate_id"] = [f"plate_{i % 3}" for i in range(n_samples)]
        data["Metadata_Source"] = ["source1"] * n_samples  # should be excluded

        df = pd.DataFrame(data)
        parquet_path = tmp_path / "test_data.parquet"
        df.to_parquet(parquet_path, index=False)

        ds = PretrainingDataset.from_parquet(
            parquet_path,
            treatment_column="treatment_label",
            plate_column="plate_id",
        )

        assert len(ds) == n_samples
        assert ds.n_features == n_features
        assert ds.n_treatments == 5

    def test_from_parquet_with_explicit_features(self, tmp_path):
        """Test loading from Parquet with explicit feature column list."""
        import pandas as pd

        n_samples = 20
        data = {
            "feat_a": np.random.randn(n_samples).astype(np.float32),
            "feat_b": np.random.randn(n_samples).astype(np.float32),
            "feat_c": np.random.randn(n_samples).astype(np.float32),
            "treatment_label": [f"trt_{i % 3}" for i in range(n_samples)],
            "plate_id": ["plate_0"] * n_samples,
        }
        df = pd.DataFrame(data)
        parquet_path = tmp_path / "test_explicit.parquet"
        df.to_parquet(parquet_path, index=False)

        ds = PretrainingDataset.from_parquet(
            parquet_path,
            feature_columns=["feat_a", "feat_b"],
        )

        assert ds.n_features == 2  # Only feat_a and feat_b

    def test_repr(self, dataset):
        r = repr(dataset)
        assert "PretrainingDataset" in r
        assert "100" in r

    def test_works_with_dataloader(self, dataset):
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            # plate_id is a string, need custom collate or drop it
            collate_fn=_collate_pretraining,
        )
        batch = next(iter(loader))
        assert batch["phenotype_features"].shape == (16, 50)
        assert batch["treatment_label"].shape == (16,)
        assert batch["sample_weight"].shape == (16,)
        assert len(batch["plate_id"]) == 16


def _collate_pretraining(samples):
    """Custom collate for PretrainingDataset batches with string fields."""
    batch = {}
    batch["phenotype_features"] = torch.stack(
        [s["phenotype_features"] for s in samples]
    )
    batch["treatment_label"] = torch.stack(
        [s["treatment_label"] for s in samples]
    )
    batch["sample_weight"] = torch.stack(
        [s["sample_weight"] for s in samples]
    )
    batch["plate_id"] = [s["plate_id"] for s in samples]
    return batch


# ============================================================================
# PretrainingConfig
# ============================================================================


class TestPretrainingConfig:
    """Tests for the YAML-loadable configuration."""

    def test_defaults(self, pretraining_config):
        assert pretraining_config.autoencoder.latent_dim == 256
        assert pretraining_config.phase1.training.epochs == 100
        assert pretraining_config.phase2.freeze.decoder is True
        assert pretraining_config.output.checkpoint_dir == "./data/checkpoints/pretraining"

    def test_yaml_roundtrip(self, tmp_path, pretraining_config):
        yaml_path = tmp_path / "test_config.yaml"
        pretraining_config.save(yaml_path)

        loaded = PretrainingConfig.from_yaml(yaml_path)
        assert loaded.autoencoder.latent_dim == pretraining_config.autoencoder.latent_dim
        assert loaded.phase1.training.epochs == pretraining_config.phase1.training.epochs
        assert loaded.phase2.freeze.decoder == pretraining_config.phase2.freeze.decoder
        assert loaded.output.log_dir == pretraining_config.output.log_dir

    def test_yaml_partial_override(self, tmp_path):
        """Loading a YAML with partial keys should use defaults for the rest."""
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text(
            "autoencoder:\n"
            "  latent_dim: 128\n"
            "  variational: true\n"
            "phase1:\n"
            "  training:\n"
            "    epochs: 50\n"
        )

        loaded = PretrainingConfig.from_yaml(yaml_path)
        assert loaded.autoencoder.latent_dim == 128
        assert loaded.autoencoder.variational is True
        assert loaded.phase1.training.epochs == 50
        # Defaults preserved
        assert loaded.autoencoder.input_dim == 1500
        assert loaded.phase2.freeze.decoder is True

    def test_from_dict(self):
        d = {
            "autoencoder": {"latent_dim": 64, "variational": True},
            "phase1": {"training": {"batch_size": 512}},
        }
        config = PretrainingConfig.from_dict(d)
        assert config.autoencoder.latent_dim == 64
        assert config.autoencoder.variational is True
        assert config.phase1.training.batch_size == 512
        # Defaults
        assert config.phase2.training.epochs == 50

    def test_phase2_output_dim_matches_latent(self, pretraining_config):
        """Phase 2 protein encoder output_dim should match autoencoder latent_dim."""
        assert (
            pretraining_config.phase2.protein_encoder.output_dim
            == pretraining_config.autoencoder.latent_dim
        )

    def test_effective_decoder_dims(self):
        cfg = PhenotypeAutoencoderConfig(
            encoder_hidden_dims=[1024, 512],
            decoder_hidden_dims=None,
        )
        assert cfg.effective_decoder_hidden_dims == [512, 1024]

    def test_explicit_decoder_dims(self):
        cfg = PhenotypeAutoencoderConfig(
            encoder_hidden_dims=[1024, 512],
            decoder_hidden_dims=[256, 768],
        )
        assert cfg.effective_decoder_hidden_dims == [256, 768]


# ============================================================================
# Checkpoint save/load
# ============================================================================


class TestCheckpoints:
    """Tests for Phase 1 and Phase 2 checkpoint helpers."""

    def test_save_and_load_phase1(self, ae, matching_pretraining_config, tmp_path):
        ckpt_path = tmp_path / "phase1.pt"

        save_phase1_checkpoint(
            path=ckpt_path,
            autoencoder=ae,
            config=matching_pretraining_config,
            epoch=10,
            global_step=500,
            best_val_loss=0.05,
            metrics={"reconstruction": 0.05, "silhouette": 0.3},
        )

        assert ckpt_path.exists()

        loaded_ae, ckpt = load_autoencoder_from_checkpoint(ckpt_path)

        assert ckpt["phase"] == 1
        assert ckpt["epoch"] == 10
        assert ckpt["global_step"] == 500
        assert ckpt["best_val_loss"] == 0.05
        assert "autoencoder_config" in ckpt
        assert "pretraining_config" in ckpt
        assert ckpt["metrics"]["silhouette"] == 0.3

        # Verify loaded model produces same output
        ae.eval()
        loaded_ae.eval()
        x = torch.randn(4, 100)
        with torch.no_grad():
            out_orig = ae(x)
            out_loaded = loaded_ae(x)
        torch.testing.assert_close(
            out_orig["reconstruction"], out_loaded["reconstruction"],
        )
        torch.testing.assert_close(
            out_orig["latent"], out_loaded["latent"],
        )

    def test_phase1_checkpoint_has_trainer_keys(self, ae, matching_pretraining_config, tmp_path):
        """Phase 1 checkpoint must have keys expected by Trainer._load_checkpoint()."""
        ckpt_path = tmp_path / "phase1_compat.pt"
        save_phase1_checkpoint(
            path=ckpt_path,
            autoencoder=ae,
            config=matching_pretraining_config,
            epoch=5,
        )

        ckpt = torch.load(ckpt_path, weights_only=False)
        # Session 6 Trainer expects these keys
        assert "epoch" in ckpt
        assert "global_step" in ckpt
        assert "model_state_dict" in ckpt
        assert "config" in ckpt
        assert "best_val_loss" in ckpt

    def test_save_phase2_checkpoint(self, ae, matching_pretraining_config, tmp_path):
        """Test Phase 2 checkpoint structure."""
        from protophen.models.protophen import ProToPhenModel, ProToPhenConfig

        model = ProToPhenModel(ProToPhenConfig(
            protein_embedding_dim=50,
            encoder_hidden_dims=[32],
            encoder_output_dim=16,
            cell_painting_dim=100,
            predict_viability=False,
        ))

        ckpt_path = tmp_path / "phase2.pt"
        phase1_path = tmp_path / "phase1_ref.pt"

        save_phase2_checkpoint(
            path=ckpt_path,
            model=model,
            config=matching_pretraining_config,
            phase1_checkpoint_path=phase1_path,
            epoch=25,
            global_step=1000,
            best_val_loss=0.03,
        )

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["phase"] == 2
        assert ckpt["phase1_checkpoint"] == str(phase1_path)
        assert ckpt["epoch"] == 25
        assert "autoencoder_config" in ckpt
        assert "model_state_dict" in ckpt

    def test_checkpoint_python_types(self, ae, matching_pretraining_config, tmp_path):
        """Checkpoint values should be Python-native types (no numpy)."""
        ckpt_path = tmp_path / "phase1_types.pt"
        save_phase1_checkpoint(
            path=ckpt_path,
            autoencoder=ae,
            config=matching_pretraining_config,
            epoch=1,
        )

        # Load with weights_only=True should work for non-state-dict fields
        # (state_dict requires weights_only=False due to tensor content)
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert isinstance(ckpt["epoch"], int)
        assert isinstance(ckpt["global_step"], int)
        assert isinstance(ckpt["best_val_loss"], float)
        assert isinstance(ckpt["phase"], int)


# ============================================================================
# Latent space quality utilities
# ============================================================================


class TestLatentQuality:
    """Tests for latent space quality metrics."""

    def test_replicate_correlation_perfect(self):
        """Identical replicates should give correlation ~1."""
        # 4 treatments, 3 replicates each, identical within treatment
        latent = torch.zeros(12, 16)
        for i in range(4):
            latent[i * 3:(i + 1) * 3] = torch.randn(1, 16)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

        score = compute_replicate_correlation(latent, labels)
        assert score > 0.99

    def test_replicate_correlation_random(self):
        """Random vectors should have low replicate correlation."""
        torch.manual_seed(42)
        latent = torch.randn(100, 32)
        labels = torch.arange(100) % 10  # 10 treatments, 10 reps each

        score = compute_replicate_correlation(latent, labels)
        # Random 32-dim vectors: expected cosine sim ≈ 0
        assert abs(score) < 0.3

    def test_replicate_correlation_no_replicates(self):
        """All unique labels should return 0."""
        latent = torch.randn(10, 16)
        labels = torch.arange(10)

        score = compute_replicate_correlation(latent, labels)
        assert score == 0.0

    def test_silhouette_separable(self):
        """Well-separated clusters should give high silhouette."""
        torch.manual_seed(42)
        # Create 3 clusters far apart
        latent = torch.cat([
            torch.randn(20, 8) + torch.tensor([10.0, 0, 0, 0, 0, 0, 0, 0]),
            torch.randn(20, 8) + torch.tensor([0, 10.0, 0, 0, 0, 0, 0, 0]),
            torch.randn(20, 8) + torch.tensor([0, 0, 10.0, 0, 0, 0, 0, 0]),
        ])
        labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)

        score = compute_latent_silhouette(latent, labels)
        assert score > 0.5

    def test_silhouette_single_label(self):
        """Single label should return 0."""
        latent = torch.randn(10, 8)
        labels = torch.zeros(10, dtype=torch.long)

        score = compute_latent_silhouette(latent, labels)
        assert score == 0.0

    def test_silhouette_subsampling(self):
        """Should work with sample_size smaller than data."""
        latent = torch.randn(100, 8)
        labels = torch.arange(100) % 5

        score = compute_latent_silhouette(latent, labels, sample_size=50)
        # Just check it runs and returns a float
        assert isinstance(score, float)


# ============================================================================
# End-to-end integration
# ============================================================================


class TestEndToEnd:
    """Integration tests for the full Phase 1 → Phase 2 workflow."""

    def test_phase1_training_step(self, ae, batch_features, treatment_labels):
        """Simulate a single Phase 1 training step."""
        loss_fn = AutoencoderLoss(variational=False)
        optimiser = torch.optim.Adam(ae.parameters(), lr=1e-3)

        ae.train()
        outputs = ae(batch_features)
        losses = loss_fn(outputs, batch_features, treatment_labels)

        optimiser.zero_grad()
        losses["total"].backward()
        optimiser.step()

        # Loss should be finite
        assert torch.isfinite(losses["total"])

    def test_phase1_to_phase2_transition(self, ae):
        """Verify the Phase 1 → Phase 2 model composition."""
        from protophen.models.protophen import ProToPhenModel, ProToPhenConfig

        # Phase 1 is trained — now create Phase 2 model
        model_config = ProToPhenConfig(
            protein_embedding_dim=50,
            encoder_hidden_dims=[32],
            encoder_output_dim=ae.latent_dim,  # Match autoencoder latent
            cell_painting_dim=ae.input_dim,
            predict_viability=False,
        )
        model = ProToPhenModel(model_config)

        # Replace cell painting head with autoencoder decoder
        model.decoders["cell_painting"] = ae.get_decoder_head(freeze=True)

        # Verify protein encoder is trainable
        trainable_encoder = sum(
            p.numel() for p in model.encoder.parameters() if p.requires_grad
        )
        assert trainable_encoder > 0

        # Verify decoder is frozen
        trainable_decoder = sum(
            p.numel()
            for p in model.decoders["cell_painting"].parameters()
            if p.requires_grad
        )
        assert trainable_decoder == 0

        # Forward pass
        protein_emb = torch.randn(4, 50)
        out = model(protein_emb)
        assert out["cell_painting"].shape == (4, ae.input_dim)

    def test_phase2_gradients_dont_flow_to_decoder(self, ae):
        """Ensure gradients are blocked from the frozen decoder."""
        from protophen.models.protophen import ProToPhenModel, ProToPhenConfig

        model_config = ProToPhenConfig(
            protein_embedding_dim=50,
            encoder_hidden_dims=[32],
            encoder_output_dim=ae.latent_dim,
            cell_painting_dim=ae.input_dim,
            predict_viability=False,
        )
        model = ProToPhenModel(model_config)
        model.decoders["cell_painting"] = ae.get_decoder_head(freeze=True)

        protein_emb = torch.randn(4, 50)
        target = torch.randn(4, ae.input_dim)

        out = model(protein_emb)
        loss = F.mse_loss(out["cell_painting"], target)
        loss.backward()

        # Decoder params should have no gradient
        for p in model.decoders["cell_painting"].parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

        # Encoder params should have gradient
        encoder_has_grad = False
        for p in model.encoder.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                encoder_has_grad = True
                break
        assert encoder_has_grad

    def test_phase2_latent_retrieval(self, ae):
        """The decoder head should cache the phenotype latent for retrieval."""
        from protophen.models.protophen import ProToPhenModel, ProToPhenConfig

        model_config = ProToPhenConfig(
            protein_embedding_dim=50,
            encoder_hidden_dims=[32],
            encoder_output_dim=ae.latent_dim,
            cell_painting_dim=ae.input_dim,
            predict_viability=False,
        )
        model = ProToPhenModel(model_config)
        head = ae.get_decoder_head(freeze=True)
        model.decoders["cell_painting"] = head

        protein_emb = torch.randn(4, 50)
        out = model(protein_emb, return_latent=True)

        # Protein latent
        assert "latent" in out

        # Phenotype latent (from decoder head cache)
        phenotype_latent = head.get_last_latent()
        assert phenotype_latent is not None
        assert phenotype_latent.shape == (4, ae.latent_dim)

        # The protein latent fed to the decoder should match cached latent
        torch.testing.assert_close(out["latent"], phenotype_latent)

    def test_full_save_load_cycle(self, ae, matching_pretraining_config, tmp_path):
        """Save Phase 1 → load → create Phase 2 model → save Phase 2."""
        from protophen.models.protophen import ProToPhenModel, ProToPhenConfig

        # Save Phase 1
        p1_path = tmp_path / "phase1.pt"
        save_phase1_checkpoint(
            path=p1_path,
            autoencoder=ae,
            config=matching_pretraining_config,
            epoch=10,
        )

        # Load Phase 1
        loaded_ae, p1_ckpt = load_autoencoder_from_checkpoint(p1_path)

        # Build Phase 2 model
        model_config = ProToPhenConfig(
            protein_embedding_dim=50,
            encoder_hidden_dims=[32],
            encoder_output_dim=loaded_ae.latent_dim,
            cell_painting_dim=loaded_ae.input_dim,
            predict_viability=False,
        )
        model = ProToPhenModel(model_config)
        model.decoders["cell_painting"] = loaded_ae.get_decoder_head(freeze=True)

        # Save Phase 2
        p2_path = tmp_path / "phase2.pt"
        save_phase2_checkpoint(
            path=p2_path,
            model=model,
            config=matching_pretraining_config,
            phase1_checkpoint_path=p1_path,
            epoch=25,
        )

        # Load Phase 2 checkpoint and verify structure
        p2_ckpt = torch.load(p2_path, weights_only=False)
        assert p2_ckpt["phase"] == 2
        assert p2_ckpt["phase1_checkpoint"] == str(p1_path)
        assert "model_state_dict" in p2_ckpt

        # Verify model can load state dict
        model2 = ProToPhenModel(model_config)
        model2.decoders["cell_painting"] = PhenotypeAutoencoder(
            loaded_ae.config
        ).get_decoder_head(freeze=True)
        model2.load_state_dict(p2_ckpt["model_state_dict"])

        # Verify outputs match
        model.eval()
        model2.eval()
        x = torch.randn(2, 50)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        torch.testing.assert_close(
            out1["cell_painting"], out2["cell_painting"],
        )


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_single_hidden_layer(self):
        cfg = PhenotypeAutoencoderConfig(
            input_dim=50,
            latent_dim=8,
            encoder_hidden_dims=[32],
            use_skip_connections=False,
        )
        ae = PhenotypeAutoencoder(cfg)
        out = ae(torch.randn(2, 50))
        assert out["reconstruction"].shape == (2, 50)

    def test_no_hidden_layers(self):
        cfg = PhenotypeAutoencoderConfig(
            input_dim=50,
            latent_dim=8,
            encoder_hidden_dims=[],
            use_skip_connections=False,
        )
        ae = PhenotypeAutoencoder(cfg)
        out = ae(torch.randn(2, 50))
        assert out["reconstruction"].shape == (2, 50)
        assert out["latent"].shape == (2, 8)

    def test_single_sample_batch(self, ae):
        out = ae(torch.randn(1, 100))
        assert out["reconstruction"].shape == (1, 100)

    def test_large_latent_dim(self):
        """Latent dim larger than hidden dims."""
        cfg = PhenotypeAutoencoderConfig(
            input_dim=50,
            latent_dim=128,
            encoder_hidden_dims=[32],
            use_skip_connections=False,
        )
        ae = PhenotypeAutoencoder(cfg)
        out = ae(torch.randn(4, 50))
        assert out["latent"].shape == (4, 128)

    def test_equal_hidden_dims_with_residual(self):
        """Residual connections should activate when consecutive dims match."""
        cfg = PhenotypeAutoencoderConfig(
            input_dim=64,
            latent_dim=16,
            encoder_hidden_dims=[32, 32, 32],
            use_residual=True,
            use_skip_connections=False,
        )
        ae = PhenotypeAutoencoder(cfg)
        out = ae(torch.randn(4, 64))
        assert out["reconstruction"].shape == (4, 64)

    def test_dataset_integer_labels(self):
        """Dataset should handle integer labels directly."""
        features = np.random.randn(10, 20).astype(np.float32)
        labels = np.arange(10)
        ds = PretrainingDataset(features, labels)
        assert ds.n_treatments == 10

    def test_dataset_mixed_labels(self):
        """Dataset should handle mixed-type labels."""
        features = np.random.randn(6, 10).astype(np.float32)
        labels = ["gene_A", "gene_A", 42, 42, "gene_B", "gene_B"]
        ds = PretrainingDataset(features, labels)
        assert ds.n_treatments == 3

    def test_config_to_dict(self, ae_config):
        d = ae_config.to_dict()
        assert d["input_dim"] == 100
        assert d["latent_dim"] == 16
        assert "effective_decoder_hidden_dims" in d