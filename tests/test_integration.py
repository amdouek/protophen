"""
Integration tests for ProToPhen.

These tests verify that different modules work correctly together
in realistic end-to-end workflows.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Data modules
from protophen.data.dataset import (
    DatasetConfig,
    ProtoPhenDataset,
    ProtoPhenSample,
    ProteinInferenceDataset,
)
from protophen.data.loaders import (
    create_dataloader,
    create_dataloaders,
    split_by_protein,
    protophen_collate_fn,
)

# Model modules
from protophen.models.protophen import (
    ProToPhenConfig,
    ProToPhenModel,
    create_protophen_model,
    create_lightweight_model,
)
from protophen.models.encoders import (
    ProteinEncoder,
    ProteinEncoderConfig,
    MLPBlock,
)
from protophen.models.decoders import (
    PhenotypeDecoder,
    CellPaintingHead,
    ViabilityHead,
    MultiTaskHead,
)

# Training modules
from protophen.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingState,
)
from protophen.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from protophen.training.metrics import (
    MetricCollection,
    MSEMetric,
    R2Metric,
    PearsonCorrelationMetric,
    create_default_metrics,
    compute_regression_metrics,
)

# Active learning modules
from protophen.active_learning.uncertainty import (
    MCDropoutEstimator,
    UncertaintyEstimate,
    UncertaintyType,
    estimate_uncertainty,
)
from protophen.active_learning.acquisition import (
    UncertaintySampling,
    HybridAcquisition,
    DiversitySampling,
)
from protophen.active_learning.selection import (
    ExperimentSelector,
    SelectionConfig,
    SelectionResult,
    select_next_experiments,
)

# Analysis modules
from protophen.analysis.clustering import (
    PhenotypeClustering,
    ClusteringResult,
    hierarchical_clustering,
    kmeans_clustering,
)
from protophen.analysis.interpretation import (
    GradientInterpreter,
    IntegratedGradientsInterpreter,
    FeatureAblationInterpreter,
    ModelInterpreter,
    InterpretationConfig,
    compute_feature_importance,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def embedding_dim():
    """Standard embedding dimension for tests."""
    return 256


@pytest.fixture
def phenotype_dim():
    """Standard phenotype dimension for tests."""
    return 100


@pytest.fixture
def n_samples():
    """Number of samples for test datasets."""
    return 64


@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 8


@pytest.fixture
def sample_proteins(n_samples, embedding_dim):
    """Create sample protein embeddings."""
    np.random.seed(42)
    return np.random.randn(n_samples, embedding_dim).astype(np.float32)


@pytest.fixture
def sample_phenotypes(n_samples, phenotype_dim):
    """Create sample phenotype features."""
    np.random.seed(42)
    return np.random.randn(n_samples, phenotype_dim).astype(np.float32)


@pytest.fixture
def protophen_samples(sample_proteins, sample_phenotypes):
    """Create ProtoPhenSample objects."""
    samples = []
    for i in range(len(sample_proteins)):
        sample = ProtoPhenSample(
            protein_id=f"protein_{i}",
            protein_embedding=sample_proteins[i],
            phenotypes={"cell_painting": sample_phenotypes[i]},
            metadata={
                "protein_name": f"TestProtein_{i}",
                "plate_id": f"plate_{i % 4}",
                "well_id": f"A{i % 12 + 1}",
            },
        )
        samples.append(sample)
    return samples


@pytest.fixture
def protophen_dataset(protophen_samples, phenotype_dim):
    """Create a ProtoPhenDataset."""
    config = DatasetConfig(
        phenotype_tasks=["cell_painting"],
        embedding_noise_std=0.0,
        feature_dropout=0.0,
    )
    return ProtoPhenDataset(samples=protophen_samples, config=config)


@pytest.fixture
def model_config(embedding_dim, phenotype_dim):
    """Create model configuration."""
    return ProToPhenConfig(
        protein_embedding_dim=embedding_dim,
        encoder_hidden_dims=[128, 64],
        encoder_output_dim=32,
        decoder_hidden_dims=[64, 128],
        cell_painting_dim=phenotype_dim,
        predict_viability=False,
        predict_uncertainty=False,
        mc_dropout=True,
    )


@pytest.fixture
def protophen_model(model_config):
    """Create a ProToPhenModel."""
    return ProToPhenModel(model_config)


@pytest.fixture
def trainer_config():
    """Create trainer configuration."""
    return TrainerConfig(
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer="adamw",
        scheduler="cosine",
        warmup_steps=0,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        use_amp=False,
        eval_every_n_epochs=1,
        tasks=["cell_painting"],
        device="cpu",
        seed=42,
    )


class MockLoss(nn.Module):
    """Simple mock loss function for testing."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, masks=None):
        total_loss = torch.tensor(0.0)
        losses = {}
        
        for task, pred in predictions.items():
            if task in targets:
                mask = masks.get(task) if masks else None
                if mask is not None:
                    pred = pred[mask]
                    tgt = targets[task][mask]
                else:
                    tgt = targets[task]
                
                if pred.numel() > 0:
                    task_loss = self.mse(pred, tgt)
                    losses[task] = task_loss
                    total_loss = total_loss + task_loss
        
        losses["total"] = total_loss
        return losses


# =============================================================================
# Data Pipeline Integration Tests
# =============================================================================

class TestDataPipelineIntegration:
    """Test data loading and preprocessing pipeline."""
    
    def test_dataset_to_dataloader(self, protophen_dataset, batch_size):
        """Test creating DataLoader from ProtoPhenDataset."""
        loader = create_dataloader(
            protophen_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        batch = next(iter(loader))
        
        assert "protein_embedding" in batch
        assert "cell_painting" in batch
        assert "protein_id" in batch
        assert batch["protein_embedding"].shape[0] == batch_size
    
    def test_split_and_create_loaders(self, protophen_dataset, batch_size):
        """Test splitting dataset and creating train/val/test loaders."""
        train_ds, val_ds, test_ds = split_by_protein(
            protophen_dataset,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
            seed=42,
        )
        
        loaders = create_dataloaders(
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            batch_size=batch_size,
            num_workers=0,
        )
        
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders
        
        # Verify no data leakage
        train_ids = {s.protein_id for s in train_ds.samples}
        val_ids = {s.protein_id for s in val_ds.samples}
        test_ids = {s.protein_id for s in test_ds.samples}
        
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
    
    def test_dataset_from_arrays(self, sample_proteins, sample_phenotypes):
        """Test creating dataset from numpy arrays."""
        dataset = ProtoPhenDataset.from_arrays(
            protein_embeddings=sample_proteins,
            phenotype_features=sample_phenotypes,
        )
        
        assert len(dataset) == len(sample_proteins)
        
        sample = dataset[0]
        assert "protein_embedding" in sample
        assert "cell_painting" in sample
    
    def test_inference_dataset_pipeline(self, sample_proteins):
        """Test inference dataset for prediction on new proteins."""
        protein_ids = [f"new_protein_{i}" for i in range(len(sample_proteins))]
        
        inference_ds = ProteinInferenceDataset(
            protein_embeddings=sample_proteins,
            protein_ids=protein_ids,
            protein_names=protein_ids,
        )
        
        loader = create_dataloader(inference_ds, batch_size=8, num_workers=0)
        
        batch = next(iter(loader))
        assert "protein_embedding" in batch
        assert "protein_id" in batch
        assert batch["protein_embedding"].shape[0] == 8


# =============================================================================
# Model Architecture Integration Tests
# =============================================================================

class TestModelArchitectureIntegration:
    """Test model components work together correctly."""
    
    def test_encoder_decoder_connection(self, embedding_dim, phenotype_dim):
        """Test encoder output feeds correctly to decoder."""
        encoder_config = ProteinEncoderConfig(
            input_dim=embedding_dim,
            hidden_dims=[128],
            output_dim=64,
        )
        encoder = ProteinEncoder(encoder_config)
        
        decoder = CellPaintingHead(
            input_dim=64,
            output_dim=phenotype_dim,
            hidden_dims=[128],
        )
        
        # Forward pass
        x = torch.randn(8, embedding_dim)
        latent = encoder(x)
        output = decoder(latent)
        
        assert latent.shape == (8, 64)
        assert output.shape == (8, phenotype_dim)
    
    def test_full_model_forward_pass(self, protophen_model, embedding_dim, phenotype_dim):
        """Test complete model forward pass."""
        x = torch.randn(8, embedding_dim)
        
        outputs = protophen_model(x)
        
        assert "cell_painting" in outputs
        assert outputs["cell_painting"].shape == (8, phenotype_dim)
    
    def test_model_with_multiple_tasks(self, embedding_dim):
        """Test model with multiple output tasks."""
        config = ProToPhenConfig(
            protein_embedding_dim=embedding_dim,
            encoder_hidden_dims=[128],
            encoder_output_dim=64,
            cell_painting_dim=100,
            predict_viability=True,
            predict_transcriptomics=True,
            transcriptomics_dim=50,
        )
        model = ProToPhenModel(config)
        
        x = torch.randn(8, embedding_dim)
        outputs = model(x)
        
        assert "cell_painting" in outputs
        assert "viability" in outputs
        assert "transcriptomics" in outputs
        
        assert outputs["cell_painting"].shape == (8, 100)
        assert outputs["viability"].shape == (8, 1)
        assert outputs["transcriptomics"].shape == (8, 50)
    
    def test_model_with_uncertainty(self, embedding_dim, phenotype_dim):
        """Test model with uncertainty prediction."""
        config = ProToPhenConfig(
            protein_embedding_dim=embedding_dim,
            encoder_hidden_dims=[64],
            encoder_output_dim=32,
            cell_painting_dim=phenotype_dim,
            predict_uncertainty=True,
        )
        model = ProToPhenModel(config)
        
        x = torch.randn(8, embedding_dim)
        outputs = model(x, return_uncertainty=True)
        
        assert "cell_painting" in outputs
        assert "cell_painting_log_var" in outputs
        assert "cell_painting_std" in outputs
    
    def test_model_latent_extraction(self, protophen_model, embedding_dim):
        """Test extracting latent representations."""
        x = torch.randn(8, embedding_dim)
        
        # Via forward pass
        outputs = protophen_model(x, return_latent=True)
        assert "latent" in outputs
        assert outputs["latent"].shape == (8, protophen_model.latent_dim)
        
        # Via direct method
        latent = protophen_model.get_latent(x)
        assert latent.shape == (8, protophen_model.latent_dim)
    
    def test_model_freeze_unfreeze(self, protophen_model):
        """Test freezing and unfreezing model components."""
        # Initially all trainable
        initial_trainable = protophen_model.n_trainable_parameters
        assert initial_trainable > 0
        
        # Freeze encoder
        protophen_model.freeze_encoder()
        assert protophen_model.n_trainable_parameters < initial_trainable
        
        # Unfreeze encoder
        protophen_model.unfreeze_encoder()
        assert protophen_model.n_trainable_parameters == initial_trainable


# =============================================================================
# Training Pipeline Integration Tests
# =============================================================================

class TestTrainingPipelineIntegration:
    """Test complete training workflow."""
    
    def test_training_loop(
        self,
        protophen_model,
        protophen_dataset,
        trainer_config,
        batch_size,
    ):
        """Test basic training loop execution."""
        train_ds, val_ds, _ = protophen_dataset.split(
            train_frac=0.7, val_frac=0.3, test_frac=0.0
        )
        
        train_loader = create_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = create_dataloader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        trainer = Trainer(
            model=protophen_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            loss_fn=MockLoss(),
        )
        
        history = trainer.train(epochs=2)
        
        assert "train_losses" in history
        assert "val_losses" in history
        assert len(history["train_losses"]) == 2
    
    def test_training_with_callbacks(
        self,
        protophen_model,
        protophen_dataset,
        trainer_config,
        batch_size,
    ):
        """Test training with callbacks."""
        train_ds, val_ds, _ = protophen_dataset.split(
            train_frac=0.7, val_frac=0.3, test_frac=0.0
        )
        
        train_loader = create_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = create_dataloader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = [
                CheckpointCallback(
                    checkpoint_dir=tmpdir,
                    save_best=True,
                    monitor="val_loss",
                ),
                EarlyStoppingCallback(
                    monitor="val_loss",
                    patience=5,
                ),
                LoggingCallback(log_every_n_steps=1),
            ]
            
            trainer = Trainer(
                model=protophen_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=trainer_config,
                loss_fn=MockLoss(),
                callbacks=callbacks,
            )
            
            history = trainer.train(epochs=2)
            
            # Check checkpoint was saved
            assert Path(tmpdir).exists()
    
    def test_training_with_metrics(
        self,
        protophen_model,
        protophen_dataset,
        trainer_config,
        batch_size,
    ):
        """Test training with metric collection."""
        train_ds, val_ds, _ = protophen_dataset.split(
            train_frac=0.7, val_frac=0.3, test_frac=0.0
        )
        
        train_loader = create_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = create_dataloader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        metrics = create_default_metrics()
        
        trainer = Trainer(
            model=protophen_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            loss_fn=MockLoss(),
            metrics=metrics,
        )
        
        history = trainer.train(epochs=1)
        
        assert "val_metrics" in history
        assert len(history["val_metrics"]) == 1
    
    def test_checkpoint_save_and_resume(
        self,
        protophen_model,
        protophen_dataset,
        trainer_config,
        batch_size,
        model_config,
    ):
        """Test saving checkpoint and resuming training."""
        train_loader = create_dataloader(
            protophen_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            # Train and save
            trainer1 = Trainer(
                model=protophen_model,
                train_loader=train_loader,
                config=trainer_config,
                loss_fn=MockLoss(),
            )
            trainer1.train(epochs=1)
            trainer1.save_checkpoint(checkpoint_path)
            
            # Create new model and trainer
            new_model = ProToPhenModel(model_config)
            trainer2 = Trainer(
                model=new_model,
                train_loader=train_loader,
                config=trainer_config,
                loss_fn=MockLoss(),
            )
            
            # Resume training
            history = trainer2.train(epochs=2, resume_from=checkpoint_path)
            
            assert trainer2.state.epoch == 2
    
    def test_prediction_after_training(
        self,
        protophen_model,
        protophen_dataset,
        trainer_config,
        batch_size,
    ):
        """Test making predictions after training."""
        train_loader = create_dataloader(
            protophen_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        
        trainer = Trainer(
            model=protophen_model,
            train_loader=train_loader,
            config=trainer_config,
            loss_fn=MockLoss(),
        )
        
        trainer.train(epochs=1)
        
        # Make predictions
        results = trainer.predict(train_loader, return_targets=True)
        
        assert "protein_ids" in results
        assert "cell_painting_predictions" in results
        assert "cell_painting_targets" in results


# =============================================================================
# Active Learning Integration Tests
# =============================================================================

class TestActiveLearningIntegration:
    """Test active learning pipeline integration."""
    
    def test_uncertainty_estimation_with_model(
        self,
        protophen_model,
        protophen_dataset,
        batch_size,
    ):
        """Test uncertainty estimation on trained model."""
        loader = create_dataloader(
            protophen_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        uncertainty = estimate_uncertainty(
            model=protophen_model,
            dataloader=loader,
            method="mc_dropout",
            n_samples=5,
            tasks=["cell_painting"],
            show_progress=False,
        )
        
        assert isinstance(uncertainty, UncertaintyEstimate)
        assert uncertainty.n_samples == len(protophen_dataset)
        assert uncertainty.epistemic is not None
    
    def test_acquisition_function_with_uncertainty(
        self,
        protophen_model,
        protophen_dataset,
        batch_size,
    ):
        """Test acquisition function scoring and selection."""
        loader = create_dataloader(
            protophen_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Get uncertainty
        uncertainty = estimate_uncertainty(
            model=protophen_model,
            dataloader=loader,
            method="mc_dropout",
            n_samples=3,
            show_progress=False,
        )
        
        # Test different acquisition functions
        for acq_fn in [UncertaintySampling(), HybridAcquisition()]:
            scores = acq_fn.score(uncertainty)
            selected = acq_fn.select(uncertainty, n_select=5)
            
            assert len(scores) == len(protophen_dataset)
            assert len(selected) == 5
    
    def test_experiment_selector_full_workflow(
        self,
        protophen_model,
        protophen_dataset,
        batch_size,
    ):
        """Test complete experiment selection workflow."""
        loader = create_dataloader(
            protophen_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        config = SelectionConfig(
            n_select=5,
            uncertainty_method="mc_dropout",
            n_mc_samples=3,
            acquisition_method="hybrid",
        )
        
        selector = ExperimentSelector(
            model=protophen_model,
            config=config,
            device="cpu",
        )
        
        result = selector.select(loader, show_progress=False)
        
        assert isinstance(result, SelectionResult)
        assert len(result.selected_ids) == 5
        assert len(result.selected_indices) == 5
    
    def test_iterative_selection_excludes_previous(
        self,
        protophen_model,
        protophen_dataset,
        batch_size,
    ):
        """Test that iterative selection properly excludes previous selections."""
        loader = create_dataloader(
            protophen_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        config = SelectionConfig(
            n_select=5,
            n_mc_samples=3,
        )
        
        selector = ExperimentSelector(
            model=protophen_model,
            config=config,
            device="cpu",
        )
        
        # First round
        result1 = selector.select(loader, show_progress=False)
        first_ids = set(result1.selected_ids)
        
        # Second round
        result2 = selector.select(loader, show_progress=False)
        second_ids = set(result2.selected_ids)
        
        # No overlap
        assert first_ids.isdisjoint(second_ids)
    
    def test_model_predict_with_uncertainty(
        self,
        protophen_model,
        embedding_dim,
    ):
        """Test model's built-in uncertainty prediction."""
        x = torch.randn(8, embedding_dim)
        
        results = protophen_model.predict_with_uncertainty(
            x,
            n_samples=5,
            tasks=["cell_painting"],
        )
        
        assert "cell_painting" in results
        assert "mean" in results["cell_painting"]
        assert "std" in results["cell_painting"]
        assert "samples" in results["cell_painting"]


# =============================================================================
# Analysis Pipeline Integration Tests
# =============================================================================

class TestAnalysisPipelineIntegration:
    """Test analysis module integration with model and data."""
    
    def test_clustering_model_predictions(
        self,
        protophen_model,
        protophen_dataset,
        batch_size,
    ):
        """Test clustering model predictions."""
        loader = create_dataloader(
            protophen_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Get predictions
        protophen_model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in loader:
                pred = protophen_model(batch["protein_embedding"])
                all_predictions.append(pred["cell_painting"])
        
        predictions = torch.cat(all_predictions, dim=0).numpy()
        
        # Cluster predictions
        result = kmeans_clustering(
            predictions,
            n_clusters=3,
            sample_ids=[s.protein_id for s in protophen_dataset.samples],
        )
        
        assert result.n_clusters == 3
        assert len(result.labels) == len(protophen_dataset)
    
    def test_clustering_latent_space(
        self,
        protophen_model,
        protophen_dataset,
        batch_size,
    ):
        """Test clustering in latent space."""
        loader = create_dataloader(
            protophen_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Get latent representations
        protophen_model.eval()
        all_latents = []
        
        with torch.no_grad():
            for batch in loader:
                latent = protophen_model.get_latent(batch["protein_embedding"])
                all_latents.append(latent)
        
        latents = torch.cat(all_latents, dim=0).numpy()
        
        # Cluster latent space
        result = hierarchical_clustering(latents, n_clusters=4)
        
        assert result.n_clusters == 4
    
    def test_interpretation_with_model(
        self,
        protophen_model,
        embedding_dim,
    ):
        """Test model interpretation."""
        config = InterpretationConfig(device="cpu", ig_n_steps=5)
        
        x = torch.randn(5, embedding_dim)
        
        # Gradient interpretation
        grad_interpreter = GradientInterpreter(protophen_model, config)
        grad_results = grad_interpreter.explain(x, task="cell_painting")
        
        assert "gradients" in grad_results
        assert "importance" in grad_results
        assert grad_results["gradients"].shape == x.shape
    
    def test_feature_importance_pipeline(
        self,
        protophen_model,
        embedding_dim,
    ):
        """Test feature importance computation."""
        x = torch.randn(10, embedding_dim)
        
        importance = compute_feature_importance(
            protophen_model,
            x,
            task="cell_painting",
            method="gradient",
        )
        
        assert importance.shape == (embedding_dim,)
        assert not np.isnan(importance).any()
    
    def test_model_interpreter_comprehensive(
        self,
        protophen_model,
        embedding_dim,
    ):
        """Test comprehensive model interpretation."""
        config = InterpretationConfig(device="cpu", ig_n_steps=3)
        interpreter = ModelInterpreter(protophen_model, config=config)
        
        x = torch.randn(3, embedding_dim)
        
        # Multiple interpretation methods
        grad_results = interpreter.gradient_importance(x)
        ig_results = interpreter.integrated_gradients(x)
        
        assert "feature_importance" in grad_results
        assert "attributions" in ig_results


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

class TestEndToEndPipeline:
    """Test complete end-to-end workflows."""
    
    def test_full_train_predict_analyse_pipeline(
        self,
        protophen_dataset,
        model_config,
        trainer_config,
        batch_size,
        embedding_dim,
    ):
        """Test complete pipeline from training to analysis."""
        # 1. Split data
        train_ds, val_ds, test_ds = protophen_dataset.split(
            train_frac=0.6, val_frac=0.2, test_frac=0.2
        )
        
        train_loader = create_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = create_dataloader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = create_dataloader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # 2. Create and train model
        model = ProToPhenModel(model_config)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            loss_fn=MockLoss(),
        )
        
        history = trainer.train(epochs=2)
        
        assert len(history["train_losses"]) == 2
        
        # 3. Make predictions on test set
        results = trainer.predict(test_loader, return_targets=True)
        
        assert "cell_painting_predictions" in results
        assert "cell_painting_targets" in results
        
        # 4. Compute metrics
        predictions = torch.from_numpy(results["cell_painting_predictions"])
        targets = torch.from_numpy(results["cell_painting_targets"])
        
        metrics = compute_regression_metrics(predictions, targets)
        
        assert "mse" in metrics
        assert "r2" in metrics
        
        # 5. Cluster predictions
        cluster_result = kmeans_clustering(
            results["cell_painting_predictions"],
            n_clusters=3,
        )
        
        assert cluster_result.n_clusters == 3
        
        # 6. Interpret model
        sample_input = torch.randn(5, embedding_dim)
        importance = compute_feature_importance(
            model, sample_input, method="gradient"
        )
        
        assert importance.shape == (embedding_dim,)
    
    def test_active_learning_simulation(
        self,
        protophen_dataset,
        model_config,
        trainer_config,
        batch_size,
    ):
        """Simulate active learning loop."""
        # Split into initial labeled and pool
        initial_size = 20
        pool_size = len(protophen_dataset) - initial_size
        
        initial_samples = protophen_dataset.samples[:initial_size]
        pool_samples = protophen_dataset.samples[initial_size:]
        
        labeled_ds = ProtoPhenDataset(
            samples=initial_samples,
            config=protophen_dataset.config,
        )
        pool_ds = ProtoPhenDataset(
            samples=pool_samples,
            config=protophen_dataset.config,
        )
        
        # Create model and train on initial data
        model = ProToPhenModel(model_config)
        
        train_loader = create_dataloader(
            labeled_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            config=trainer_config,
            loss_fn=MockLoss(),
        )
        trainer.train(epochs=1)
        
        # Select samples from pool
        pool_loader = create_dataloader(
            pool_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        result = select_next_experiments(
            model=model,
            dataloader=pool_loader,
            n_select=5,
            method="hybrid",
            n_mc_samples=3,
            show_progress=False,
        )
        
        assert len(result.selected_ids) == 5
        
        # Simulate adding selected samples to training set
        selected_indices = result.selected_indices
        new_samples = [pool_samples[i] for i in selected_indices]
        
        updated_labeled = ProtoPhenDataset(
            samples=initial_samples + new_samples,
            config=protophen_dataset.config,
        )
        
        assert len(updated_labeled) == initial_size + 5
    
    def test_transfer_learning_workflow(
        self,
        model_config,
        embedding_dim,
        phenotype_dim,
        batch_size,
    ):
        """Test transfer learning scenario."""
        # Create source model and "pretrain"
        source_model = ProToPhenModel(model_config)
        
        # Simulate pretraining (just a forward pass)
        x = torch.randn(batch_size, embedding_dim)
        _ = source_model(x)
        
        # Create target model with same encoder
        target_config = ProToPhenConfig(
            protein_embedding_dim=embedding_dim,
            encoder_hidden_dims=model_config.encoder_hidden_dims,
            encoder_output_dim=model_config.encoder_output_dim,
            cell_painting_dim=50,  # Different output dim
        )
        target_model = ProToPhenModel(target_config)
        
        # Transfer encoder weights
        target_model.encoder.load_state_dict(source_model.encoder.state_dict())
        
        # Freeze encoder for fine-tuning
        target_model.freeze_encoder()
        
        # Verify encoder is frozen
        encoder_trainable = sum(
            p.numel() for p in target_model.encoder.parameters() if p.requires_grad
        )
        assert encoder_trainable == 0
        
        # Decoder should still be trainable
        decoder_trainable = sum(
            p.numel() for p in target_model.decoders["cell_painting"].parameters()
            if p.requires_grad
        )
        assert decoder_trainable > 0
    
    def test_multi_task_prediction_workflow(
        self,
        embedding_dim,
        batch_size,
    ):
        """Test multi-task learning workflow."""
        config = ProToPhenConfig(
            protein_embedding_dim=embedding_dim,
            encoder_hidden_dims=[64],
            encoder_output_dim=32,
            cell_painting_dim=100,
            predict_viability=True,
            predict_transcriptomics=True,
            transcriptomics_dim=50,
        )
        model = ProToPhenModel(config)
        
        x = torch.randn(batch_size, embedding_dim)
        
        # Predict all tasks
        all_outputs = model(x)
        assert len(all_outputs) == 3
        
        # Predict subset of tasks
        subset_outputs = model(x, tasks=["cell_painting", "viability"])
        assert "cell_painting" in subset_outputs
        assert "viability" in subset_outputs
        assert "transcriptomics" not in subset_outputs
        
        # Add new task dynamically
        model.add_task("new_task", output_dim=25)
        
        assert "new_task" in model.task_names
        
        new_outputs = model(x, tasks=["new_task"])
        assert new_outputs["new_task"].shape == (batch_size, 25)


# =============================================================================
# Error Handling and Edge Cases
# =============================================================================

class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        empty_dataset = ProtoPhenDataset(samples=[])
        
        assert len(empty_dataset) == 0
        assert empty_dataset.embedding_dim == 0
    
    def test_mismatched_dimensions(self, embedding_dim, phenotype_dim):
        """Test handling of mismatched dimensions."""
        model = create_protophen_model(
            protein_embedding_dim=embedding_dim,
            cell_painting_dim=phenotype_dim,
        )
        
        # Wrong input dimension should raise
        wrong_dim_input = torch.randn(8, embedding_dim + 100)
        
        with pytest.raises(RuntimeError):
            model(wrong_dim_input)
    
    def test_missing_task_handling(self, protophen_model, embedding_dim):
        """Test handling of missing task in forward pass."""
        x = torch.randn(8, embedding_dim)
        
        # Request non-existent task - should be silently ignored
        outputs = protophen_model(x, tasks=["cell_painting", "nonexistent_task"])
        
        assert "cell_painting" in outputs
        assert "nonexistent_task" not in outputs
    
    def test_single_sample_batch(self, protophen_model, embedding_dim):
        """Test handling of single sample batches."""
        x = torch.randn(1, embedding_dim)
        
        outputs = protophen_model(x)
        
        assert outputs["cell_painting"].shape[0] == 1


# =============================================================================
# Performance and Scalability Tests
# =============================================================================

class TestPerformanceIntegration:
    """Test performance-related integration aspects."""
    
    def test_model_parameter_counts(self, model_config):
        """Test model parameter counting."""
        model = ProToPhenModel(model_config)
        
        summary = model.summary()
        
        assert summary["n_parameters"] > 0
        assert summary["n_trainable_parameters"] > 0
        assert summary["n_trainable_parameters"] <= summary["n_parameters"]
    
    def test_gradient_flow(self, protophen_model, embedding_dim, phenotype_dim):
        """Test that gradients flow through the entire model."""
        x = torch.randn(4, embedding_dim, requires_grad=True)
        target = torch.randn(4, phenotype_dim)
        
        outputs = protophen_model(x)
        loss = ((outputs["cell_painting"] - target) ** 2).mean()
        loss.backward()
        
        # Check gradients exist for encoder and decoder
        for name, param in protophen_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_batch_consistency(self, protophen_model, embedding_dim):
        """Test that predictions are consistent across batch sizes."""
        x = torch.randn(16, embedding_dim)
        
        protophen_model.eval()
        with torch.no_grad():
            # Full batch
            full_output = protophen_model(x)["cell_painting"]
            
            # Split into smaller batches
            split_outputs = []
            for i in range(0, 16, 4):
                out = protophen_model(x[i:i+4])["cell_painting"]
                split_outputs.append(out)
            
            combined_output = torch.cat(split_outputs, dim=0)
        
        # Should be identical
        torch.testing.assert_close(full_output, combined_output)