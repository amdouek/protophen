"""
Unit tests for ProToPhen dataset classes.
"""

import numpy as np
import pytest
import torch

from protophen.data.dataset import (
    DatasetConfig,
    ProteinInferenceDataset,
    ProtoPhenDataset,
    ProtoPhenSample,
)
from protophen.data.phenotype import Phenotype, PhenotypeDataset
from protophen.data.protein import Protein, ProteinLibrary


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_proteins():
    """Create sample proteins with embeddings."""
    proteins = []
    for i in range(10):
        protein = Protein(
            sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS"[: 20 + i],
            name=f"protein_{i}",
            source="de_novo",
        )
        # Add mock embeddings
        protein.set_embedding("esm2", np.random.randn(1280).astype(np.float32))
        protein.set_embedding("physicochemical", np.random.randn(439).astype(np.float32))
        protein.set_embedding("fused", np.random.randn(1719).astype(np.float32))
        proteins.append(protein)
    return proteins


@pytest.fixture
def sample_phenotypes(sample_proteins):
    """Create sample phenotypes matching proteins."""
    phenotypes = []
    for i, protein in enumerate(sample_proteins):
        phenotype = Phenotype(
            features=np.random.randn(1500).astype(np.float32),
            sample_id=f"sample_{i}",
            protein_id=protein.hash,
            well_id=f"A{i+1:02d}",
            plate_id="plate_001",
            cell_count=100 + i * 10,
            qc_passed=True,
        )
        phenotypes.append(phenotype)
    return phenotypes


@pytest.fixture
def protein_library(sample_proteins):
    """Create a protein library."""
    return ProteinLibrary(proteins=sample_proteins, name="test_library")


@pytest.fixture
def phenotype_dataset(sample_phenotypes):
    """Create a phenotype dataset."""
    return PhenotypeDataset(
        phenotypes=sample_phenotypes,
        feature_names=[f"feature_{i}" for i in range(1500)],
        name="test_phenotypes",
    )


# =============================================================================
# ProtoPhenSample Tests
# =============================================================================

class TestProtoPhenSample:
    """Tests for ProtoPhenSample dataclass."""

    def test_sample_creation(self):
        """Test basic sample creation."""
        sample = ProtoPhenSample(
            protein_id="test_protein",
            protein_embedding=np.random.randn(1280).astype(np.float32),
            phenotypes={"cell_painting": np.random.randn(1500).astype(np.float32)},
            metadata={"well": "A01"},
        )
        
        assert sample.protein_id == "test_protein"
        assert sample.protein_embedding.shape == (1280,)
        assert "cell_painting" in sample.phenotypes
        assert sample.metadata["well"] == "A01"

    def test_sample_to_dict(self):
        """Test conversion to dictionary."""
        sample = ProtoPhenSample(
            protein_id="test_protein",
            protein_embedding=np.random.randn(1280).astype(np.float32),
        )
        
        data = sample.to_dict()
        
        assert "protein_id" in data
        assert "protein_embedding" in data
        assert "phenotypes" in data
        assert "metadata" in data


# =============================================================================
# DatasetConfig Tests
# =============================================================================

class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig()
        
        assert config.protein_embedding_key == "fused"
        assert config.require_qc_passed is True
        assert config.embedding_noise_std == 0.0
        assert config.feature_dropout == 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DatasetConfig(
            protein_embedding_key="esm2",
            embedding_noise_std=0.1,
            min_cell_count=50,
        )
        
        assert config.protein_embedding_key == "esm2"
        assert config.embedding_noise_std == 0.1
        assert config.min_cell_count == 50


# =============================================================================
# ProtoPhenDataset Tests
# =============================================================================

class TestProtoPhenDataset:
    """Tests for ProtoPhenDataset."""

    def test_dataset_creation_empty(self):
        """Test creating empty dataset."""
        dataset = ProtoPhenDataset()
        
        assert len(dataset) == 0
        assert dataset.embedding_dim == 0

    def test_dataset_from_data(self, protein_library, phenotype_dataset):
        """Test creating dataset from protein and phenotype data."""
        dataset = ProtoPhenDataset.from_data(
            proteins=protein_library,
            phenotypes=phenotype_dataset,
            embedding_key="fused",
        )
        
        assert len(dataset) > 0
        assert dataset.embedding_dim == 1719

    def test_dataset_from_arrays(self):
        """Test creating dataset from numpy arrays."""
        n_samples = 50
        embed_dim = 1280
        n_features = 1500
        
        embeddings = np.random.randn(n_samples, embed_dim).astype(np.float32)
        features = np.random.randn(n_samples, n_features).astype(np.float32)
        
        dataset = ProtoPhenDataset.from_arrays(
            protein_embeddings=embeddings,
            phenotype_features=features,
        )
        
        assert len(dataset) == n_samples
        assert dataset.embedding_dim == embed_dim
        assert dataset.phenotype_dims["cell_painting"] == n_features

    def test_dataset_getitem(self):
        """Test indexing dataset."""
        embeddings = np.random.randn(10, 1280).astype(np.float32)
        features = np.random.randn(10, 1500).astype(np.float32)
        
        dataset = ProtoPhenDataset.from_arrays(embeddings, features)
        
        sample = dataset[0]
        
        assert "protein_embedding" in sample
        assert "cell_painting" in sample
        assert "protein_id" in sample
        assert isinstance(sample["protein_embedding"], torch.Tensor)
        assert sample["protein_embedding"].shape == (1280,)

    def test_dataset_with_augmentation(self):
        """Test dataset with data augmentation."""
        embeddings = np.random.randn(10, 1280).astype(np.float32)
        features = np.random.randn(10, 1500).astype(np.float32)
        
        config = DatasetConfig(
            embedding_noise_std=0.1,
            feature_dropout=0.1,
        )
        
        dataset = ProtoPhenDataset.from_arrays(embeddings, features, config=config)
        
        # Get same sample twice - should differ due to augmentation
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # With noise, samples should be different
        assert not torch.allclose(
            sample1["protein_embedding"],
            sample2["protein_embedding"],
        )

    def test_dataset_split(self):
        """Test train/val/test splitting."""
        embeddings = np.random.randn(100, 1280).astype(np.float32)
        features = np.random.randn(100, 1500).astype(np.float32)
        
        dataset = ProtoPhenDataset.from_arrays(embeddings, features)
        
        train, val, test = dataset.split(
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
        
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        
        # Check no overlap
        train_ids = {s.protein_id for s in train.samples}
        val_ids = {s.protein_id for s in val.samples}
        test_ids = {s.protein_id for s in test.samples}
        
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_dataset_statistics(self):
        """Test computing dataset statistics."""
        embeddings = np.random.randn(50, 1280).astype(np.float32)
        features = np.random.randn(50, 1500).astype(np.float32)
        
        dataset = ProtoPhenDataset.from_arrays(embeddings, features)
        
        stats = dataset.get_statistics()
        
        assert stats["n_samples"] == 50
        assert stats["embedding_dim"] == 1280
        assert "embedding_mean" in stats
        assert "embedding_std" in stats

    def test_dataset_normalisation(self):
        """Test on-the-fly normalisation."""
        embeddings = np.random.randn(50, 1280).astype(np.float32)
        features = np.random.randn(50, 1500).astype(np.float32) * 10 + 5
        
        dataset = ProtoPhenDataset.from_arrays(embeddings, features)
        
        # Compute and set normalisation stats
        mean, std = dataset.compute_normalisation_stats("cell_painting")
        dataset.set_normalisation_stats(mean, std)
        
        sample = dataset[0]
        
        # Normalised features should be roughly zero-mean, unit-variance
        assert sample["cell_painting"].mean().abs() < 1.0

    def test_add_phenotype_task(self):
        """Test adding a new phenotype task."""
        embeddings = np.random.randn(10, 1280).astype(np.float32)
        features = np.random.randn(10, 1500).astype(np.float32)
        
        dataset = ProtoPhenDataset.from_arrays(embeddings, features)
        
        # Add viability task
        viability = {
            f"protein_{i}": np.array([0.5 + np.random.rand() * 0.5])
            for i in range(10)
        }
        dataset.add_phenotype_task("viability", viability)
        
        assert "viability" in dataset.config.phenotype_tasks


# =============================================================================
# ProteinInferenceDataset Tests
# =============================================================================

class TestProteinInferenceDataset:
    """Tests for ProteinInferenceDataset."""

    def test_inference_dataset_creation(self):
        """Test creating inference dataset."""
        embeddings = np.random.randn(20, 1280).astype(np.float32)
        ids = [f"protein_{i}" for i in range(20)]
        
        dataset = ProteinInferenceDataset(
            protein_embeddings=embeddings,
            protein_ids=ids,
        )
        
        assert len(dataset) == 20
        assert dataset.embedding_dim == 1280

    def test_inference_dataset_from_library(self, protein_library):
        """Test creating from protein library."""
        dataset = ProteinInferenceDataset.from_library(
            proteins=protein_library,
            embedding_key="fused",
        )
        
        assert len(dataset) == len(protein_library)

    def test_inference_dataset_getitem(self):
        """Test indexing inference dataset."""
        embeddings = np.random.randn(10, 1280).astype(np.float32)
        ids = [f"protein_{i}" for i in range(10)]
        names = [f"Protein {i}" for i in range(10)]
        
        dataset = ProteinInferenceDataset(
            protein_embeddings=embeddings,
            protein_ids=ids,
            protein_names=names,
        )
        
        sample = dataset[0]
        
        assert "protein_embedding" in sample
        assert "protein_id" in sample
        assert "protein_name" in sample
        assert isinstance(sample["protein_embedding"], torch.Tensor)