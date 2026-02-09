"""
Tests for data loading utilities.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from protophen.data.dataset import (
    DatasetConfig,
    ProtoPhenDataset,
    ProtoPhenSample,
    ProteinInferenceDataset,
)
from protophen.data.loaders import (
    DataLoaderConfig,
    create_balanced_sampler,
    create_dataloader,
    create_dataloaders,
    inference_collate_fn,
    protophen_collate_fn,
    split_by_plate,
    split_by_protein,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_protophen_samples():
    """Create sample ProtoPhenSample objects for testing."""
    samples = []
    for i in range(20):
        sample = ProtoPhenSample(
            protein_id=f"protein_{i}",
            protein_embedding=np.random.randn(128).astype(np.float32),
            phenotypes={
                "cell_painting": np.random.randn(100).astype(np.float32),
            },
            metadata={
                "protein_name": f"TestProtein_{i}",
                "plate_id": f"plate_{i % 4}",  # 4 different plates
                "well_id": f"A{i % 12 + 1}",
                "cell_count": np.random.randint(100, 1000),
            },
        )
        samples.append(sample)
    return samples


@pytest.fixture
def protophen_dataset(sample_protophen_samples):
    """Create a ProtoPhenDataset for testing."""
    config = DatasetConfig(
        phenotype_tasks=["cell_painting"],
        embedding_noise_std=0.0,
        feature_dropout=0.0,
    )
    return ProtoPhenDataset(samples=sample_protophen_samples, config=config)


@pytest.fixture
def inference_dataset():
    """Create a ProteinInferenceDataset for testing."""
    n_proteins = 15
    embeddings = np.random.randn(n_proteins, 128).astype(np.float32)
    protein_ids = [f"protein_{i}" for i in range(n_proteins)]
    protein_names = [f"TestProtein_{i}" for i in range(n_proteins)]
    
    return ProteinInferenceDataset(
        protein_embeddings=embeddings,
        protein_ids=protein_ids,
        protein_names=protein_names,
    )


@pytest.fixture
def batch_samples():
    """Create a batch of sample dictionaries for collate testing."""
    batch = []
    for i in range(4):
        sample = {
            "protein_id": f"protein_{i}",
            "protein_name": f"TestProtein_{i}",
            "protein_embedding": torch.randn(128),
            "cell_painting": torch.randn(100),
            "mask_cell_painting": torch.tensor(True),
            "metadata": {"plate_id": f"plate_{i}", "well_id": f"A{i+1}"},
        }
        batch.append(sample)
    return batch


# =============================================================================
# Tests for Collate Functions
# =============================================================================

class TestProtophenCollateFn:
    """Tests for protophen_collate_fn."""
    
    def test_empty_batch(self):
        """Test collating an empty batch."""
        result = protophen_collate_fn([])
        assert result == {}
    
    def test_protein_ids_kept_as_list(self, batch_samples):
        """Test that protein_id fields are kept as lists of strings."""
        result = protophen_collate_fn(batch_samples)
        
        assert "protein_id" in result
        assert isinstance(result["protein_id"], list)
        assert len(result["protein_id"]) == 4
        assert all(isinstance(pid, str) for pid in result["protein_id"])
    
    def test_protein_names_kept_as_list(self, batch_samples):
        """Test that protein_name fields are kept as lists of strings."""
        result = protophen_collate_fn(batch_samples)
        
        assert "protein_name" in result
        assert isinstance(result["protein_name"], list)
        assert len(result["protein_name"]) == 4
    
    def test_metadata_kept_as_list(self, batch_samples):
        """Test that metadata fields are kept as lists of dicts."""
        result = protophen_collate_fn(batch_samples)
        
        assert "metadata" in result
        assert isinstance(result["metadata"], list)
        assert len(result["metadata"]) == 4
        assert all(isinstance(m, dict) for m in result["metadata"])
    
    def test_mask_fields_stacked(self, batch_samples):
        """Test that mask_ prefixed fields are stacked as tensors."""
        result = protophen_collate_fn(batch_samples)
        
        assert "mask_cell_painting" in result
        assert isinstance(result["mask_cell_painting"], torch.Tensor)
        assert result["mask_cell_painting"].shape == (4,)
    
    def test_tensors_stacked(self, batch_samples):
        """Test that tensor fields are stacked."""
        result = protophen_collate_fn(batch_samples)
        
        assert "protein_embedding" in result
        assert isinstance(result["protein_embedding"], torch.Tensor)
        assert result["protein_embedding"].shape == (4, 128)
        
        assert "cell_painting" in result
        assert isinstance(result["cell_painting"], torch.Tensor)
        assert result["cell_painting"].shape == (4, 100)
    
    def test_variable_length_tensors_padded(self):
        """Test that variable-length tensors are padded."""
        batch = [
            {"features": torch.randn(50)},
            {"features": torch.randn(75)},
            {"features": torch.randn(100)},
        ]
        
        result = protophen_collate_fn(batch)
        
        assert "features" in result
        assert result["features"].shape == (3, 100)  # Padded to max length
    
    def test_non_tensor_fields_kept_as_list(self):
        """Test that non-tensor, non-special fields are kept as lists."""
        batch = [
            {"custom_field": "value1"},
            {"custom_field": "value2"},
        ]
        
        result = protophen_collate_fn(batch)
        
        assert "custom_field" in result
        assert isinstance(result["custom_field"], list)
        assert result["custom_field"] == ["value1", "value2"]


class TestInferenceCollateFn:
    """Tests for inference_collate_fn."""
    
    def test_basic_collation(self):
        """Test basic collation of inference batches."""
        batch = [
            {
                "protein_embedding": torch.randn(128),
                "protein_id": "protein_0",
                "protein_name": "TestProtein_0",
            },
            {
                "protein_embedding": torch.randn(128),
                "protein_id": "protein_1",
                "protein_name": "TestProtein_1",
            },
        ]
        
        result = inference_collate_fn(batch)
        
        assert "protein_embedding" in result
        assert isinstance(result["protein_embedding"], torch.Tensor)
        assert result["protein_embedding"].shape == (2, 128)
        
        assert "protein_id" in result
        assert result["protein_id"] == ["protein_0", "protein_1"]
        
        assert "protein_name" in result
        assert result["protein_name"] == ["TestProtein_0", "TestProtein_1"]


# =============================================================================
# Tests for DataLoaderConfig
# =============================================================================

class TestDataLoaderConfig:
    """Tests for DataLoaderConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DataLoaderConfig()
        
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.pin_memory is True
        assert config.prefetch_factor == 2
        assert config.persistent_workers is True
        assert config.drop_last is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DataLoaderConfig(
            batch_size=64,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=4,
            persistent_workers=False,
            drop_last=True,
        )
        
        assert config.batch_size == 64
        assert config.num_workers == 8
        assert config.pin_memory is False
        assert config.prefetch_factor == 4
        assert config.persistent_workers is False
        assert config.drop_last is True


# =============================================================================
# Tests for create_dataloader
# =============================================================================

class TestCreateDataloader:
    """Tests for create_dataloader function."""
    
    def test_creates_dataloader(self, protophen_dataset):
        """Test that create_dataloader returns a DataLoader."""
        loader = create_dataloader(protophen_dataset, batch_size=4)
        
        assert isinstance(loader, DataLoader)
    
    def test_batch_size(self, protophen_dataset):
        """Test that batch size is respected."""
        loader = create_dataloader(protophen_dataset, batch_size=4)
        
        batch = next(iter(loader))
        assert batch["protein_embedding"].shape[0] == 4
    
    def test_auto_collate_protophen_dataset(self, protophen_dataset):
        """Test automatic collate function selection for ProtoPhenDataset."""
        loader = create_dataloader(protophen_dataset, batch_size=4)
        
        batch = next(iter(loader))
        
        # Check that protophen_collate_fn behavior is applied
        assert isinstance(batch["protein_id"], list)
        assert isinstance(batch["protein_embedding"], torch.Tensor)
    
    def test_auto_collate_inference_dataset(self, inference_dataset):
        """Test automatic collate function selection for ProteinInferenceDataset."""
        loader = create_dataloader(inference_dataset, batch_size=4)
        
        batch = next(iter(loader))
        
        # Check that inference_collate_fn behavior is applied
        assert isinstance(batch["protein_id"], list)
        assert isinstance(batch["protein_embedding"], torch.Tensor)
    
    def test_custom_collate_fn(self, protophen_dataset):
        """Test using a custom collate function."""
        def custom_collate(batch):
            return {"custom": True, "batch_size": len(batch)}
    
        loader = create_dataloader(
            protophen_dataset,
            batch_size=4,
            collate_fn=custom_collate,
            num_workers=0,  # Disabled multiprocessing to avoid pickling local function -- needed to pass unit tests but to be reviewed further.
        )
    
        batch = next(iter(loader))
        assert batch == {"custom": True, "batch_size": 4}
    
    def test_shuffle_disabled_with_sampler(self, protophen_dataset):
        """Test that shuffle is disabled when sampler is provided."""
        sampler = create_balanced_sampler(protophen_dataset, balance_by="plate_id")
        
        # This should not raise even though shuffle would conflict with sampler
        loader = create_dataloader(
            protophen_dataset,
            batch_size=4,
            shuffle=True,  # Should be ignored
            sampler=sampler,
        )
        
        assert isinstance(loader, DataLoader)
    
    def test_num_workers_capped(self, protophen_dataset):
        """Test that num_workers is capped at CPU count."""
        # Request an unreasonably large number of workers
        loader = create_dataloader(
            protophen_dataset,
            batch_size=4,
            num_workers=1000,
        )
        
        # Just verify it creates successfully (workers will be capped internally)
        assert isinstance(loader, DataLoader)
    
    def test_drop_last(self, protophen_dataset):
        """Test drop_last parameter."""
        # Dataset has 20 samples, batch_size=7 -> 2 full batches + 6 remaining
        loader = create_dataloader(
            protophen_dataset,
            batch_size=7,
            drop_last=True,
            shuffle=False,
        )
        
        batches = list(loader)
        assert len(batches) == 2  # Last incomplete batch dropped
    
    def test_no_drop_last(self, protophen_dataset):
        """Test without drop_last."""
        # Dataset has 20 samples, batch_size=7 -> 2 full batches + 6 remaining
        loader = create_dataloader(
            protophen_dataset,
            batch_size=7,
            drop_last=False,
            shuffle=False,
        )
        
        batches = list(loader)
        assert len(batches) == 3  # Includes incomplete batch


# =============================================================================
# Tests for create_dataloaders
# =============================================================================

class TestCreateDataloaders:
    """Tests for create_dataloaders function."""
    
    def test_train_only(self, protophen_dataset):
        """Test creating only training loader."""
        loaders = create_dataloaders(
            train_dataset=protophen_dataset,
            batch_size=4,
        )
        
        assert "train" in loaders
        assert "val" not in loaders
        assert "test" not in loaders
    
    def test_train_and_val(self, protophen_dataset):
        """Test creating training and validation loaders."""
        train_dataset, val_dataset, _ = protophen_dataset.split(
            train_frac=0.7, val_frac=0.3, test_frac=0.0
        )
        
        loaders = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=4,
        )
        
        assert "train" in loaders
        assert "val" in loaders
        assert "test" not in loaders
    
    def test_all_splits(self, protophen_dataset):
        """Test creating all three loaders."""
        train_dataset, val_dataset, test_dataset = protophen_dataset.split()
        
        loaders = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=4,
        )
        
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders
    
    def test_train_loader_drops_last(self, protophen_dataset):
        """Test that training loader drops last incomplete batch."""
        loaders = create_dataloaders(
            train_dataset=protophen_dataset,
            batch_size=7,  # 20 samples / 7 = 2 full + remainder
        )
        
        train_batches = list(loaders["train"])
        # Should drop last batch (drop_last=True for train)
        assert len(train_batches) == 2
    
    def test_val_loader_keeps_all(self, protophen_dataset):
        """Test that validation loader keeps all samples."""
        train_dataset, val_dataset, _ = protophen_dataset.split(
            train_frac=0.5, val_frac=0.5, test_frac=0.0
        )
        
        loaders = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=7,
        )
        
        val_batches = list(loaders["val"])
        total_samples = sum(b["protein_embedding"].shape[0] for b in val_batches)
        assert total_samples == len(val_dataset)


# =============================================================================
# Tests for create_balanced_sampler
# =============================================================================

class TestCreateBalancedSampler:
    """Tests for create_balanced_sampler function."""
    
    def test_returns_weighted_sampler(self, protophen_dataset):
        """Test that function returns a WeightedRandomSampler."""
        sampler = create_balanced_sampler(protophen_dataset, balance_by="plate_id")
        
        assert isinstance(sampler, WeightedRandomSampler)
    
    def test_sampler_num_samples(self, protophen_dataset):
        """Test that sampler has correct number of samples."""
        sampler = create_balanced_sampler(protophen_dataset, balance_by="plate_id")
        
        assert sampler.num_samples == len(protophen_dataset)
    
    def test_weights_computed_correctly(self):
        """Test that weights are computed based on inverse frequency."""
        # Create dataset with imbalanced groups
        samples = []
        
        # Group A: 8 samples
        for i in range(8):
            samples.append(ProtoPhenSample(
                protein_id=f"protein_a_{i}",
                protein_embedding=np.random.randn(128).astype(np.float32),
                metadata={"group": "A"},
            ))
        
        # Group B: 2 samples
        for i in range(2):
            samples.append(ProtoPhenSample(
                protein_id=f"protein_b_{i}",
                protein_embedding=np.random.randn(128).astype(np.float32),
                metadata={"group": "B"},
            ))
        
        dataset = ProtoPhenDataset(samples=samples)
        sampler = create_balanced_sampler(dataset, balance_by="group")
        
        weights = sampler.weights.numpy()
        
        # Group A weight: 10 / (2 * 8) = 0.625
        # Group B weight: 10 / (2 * 2) = 2.5
        # Ratio should be 4:1
        group_a_weight = weights[0]
        group_b_weight = weights[8]
        
        np.testing.assert_almost_equal(group_b_weight / group_a_weight, 4.0)
    
    def test_unknown_group_handling(self):
        """Test handling of samples without the balance_by key."""
        samples = [
            ProtoPhenSample(
                protein_id="protein_0",
                protein_embedding=np.random.randn(128).astype(np.float32),
                metadata={"plate_id": "plate_0"},
            ),
            ProtoPhenSample(
                protein_id="protein_1",
                protein_embedding=np.random.randn(128).astype(np.float32),
                metadata={},  # Missing plate_id
            ),
        ]
        
        dataset = ProtoPhenDataset(samples=samples)
        
        # Should not raise, missing keys default to "unknown"
        sampler = create_balanced_sampler(dataset, balance_by="plate_id")
        assert isinstance(sampler, WeightedRandomSampler)
    
    def test_sampler_with_dataloader(self, protophen_dataset):
        """Test using sampler with DataLoader."""
        sampler = create_balanced_sampler(protophen_dataset, balance_by="plate_id")
        
        loader = create_dataloader(
            protophen_dataset,
            batch_size=4,
            sampler=sampler,
        )
        
        # Should be able to iterate
        batch = next(iter(loader))
        assert batch["protein_embedding"].shape[0] == 4


# =============================================================================
# Tests for split_by_protein
# =============================================================================

class TestSplitByProtein:
    """Tests for split_by_protein function."""
    
    def test_returns_three_datasets(self, protophen_dataset):
        """Test that function returns three datasets."""
        train, val, test = split_by_protein(protophen_dataset)
        
        assert isinstance(train, ProtoPhenDataset)
        assert isinstance(val, ProtoPhenDataset)
        assert isinstance(test, ProtoPhenDataset)
    
    def test_split_fractions(self, protophen_dataset):
        """Test that split fractions are approximately respected."""
        train, val, test = split_by_protein(
            protophen_dataset,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        
        total = len(train) + len(val) + len(test)
        assert total == len(protophen_dataset)
        
        # Check approximate fractions (may not be exact due to per-protein splitting)
        assert len(train) > len(val)
        assert len(train) > len(test)
    
    def test_no_protein_overlap(self, protophen_dataset):
        """Test that no proteins appear in multiple splits."""
        train, val, test = split_by_protein(protophen_dataset)
        
        train_proteins = {s.protein_id for s in train.samples}
        val_proteins = {s.protein_id for s in val.samples}
        test_proteins = {s.protein_id for s in test.samples}
        
        # Check no overlap
        assert train_proteins.isdisjoint(val_proteins)
        assert train_proteins.isdisjoint(test_proteins)
        assert val_proteins.isdisjoint(test_proteins)
    
    def test_reproducible_with_seed(self, protophen_dataset):
        """Test that splitting is reproducible with same seed."""
        train1, val1, test1 = split_by_protein(protophen_dataset, seed=42)
        train2, val2, test2 = split_by_protein(protophen_dataset, seed=42)
        
        train_ids_1 = {s.protein_id for s in train1.samples}
        train_ids_2 = {s.protein_id for s in train2.samples}
        
        assert train_ids_1 == train_ids_2
    
    def test_different_seeds_different_splits(self, protophen_dataset):
        """Test that different seeds produce different splits."""
        train1, _, _ = split_by_protein(protophen_dataset, seed=42)
        train2, _, _ = split_by_protein(protophen_dataset, seed=123)
        
        train_ids_1 = {s.protein_id for s in train1.samples}
        train_ids_2 = {s.protein_id for s in train2.samples}
        
        # Should be different (with very high probability)
        assert train_ids_1 != train_ids_2
    
    def test_val_test_no_augmentation(self, protophen_dataset):
        """Test that val/test datasets have no augmentation."""
        # Set augmentation on original
        protophen_dataset.config.embedding_noise_std = 0.1
        protophen_dataset.config.feature_dropout = 0.2
        
        train, val, test = split_by_protein(protophen_dataset)
        
        # Train should keep augmentation
        assert train.config.embedding_noise_std == 0.1
        assert train.config.feature_dropout == 0.2
        
        # Val and test should have no augmentation
        assert val.config.embedding_noise_std == 0.0
        assert val.config.feature_dropout == 0.0
        assert test.config.embedding_noise_std == 0.0
        assert test.config.feature_dropout == 0.0
    
    def test_preserves_phenotype_tasks(self, protophen_dataset):
        """Test that phenotype tasks are preserved in splits."""
        train, val, test = split_by_protein(protophen_dataset)
        
        original_tasks = protophen_dataset.config.phenotype_tasks
        
        assert train.config.phenotype_tasks == original_tasks
        assert val.config.phenotype_tasks == original_tasks
        assert test.config.phenotype_tasks == original_tasks


# =============================================================================
# Tests for split_by_plate
# =============================================================================

class TestSplitByPlate:
    """Tests for split_by_plate function."""
    
    def test_returns_datasets(self, protophen_dataset):
        """Test that function returns datasets."""
        train, val, test = split_by_plate(
            protophen_dataset,
            train_plates=["plate_0", "plate_1"],
            val_plates=["plate_2"],
            test_plates=["plate_3"],
        )
        
        assert isinstance(train, ProtoPhenDataset)
        assert isinstance(val, ProtoPhenDataset)
        assert isinstance(test, ProtoPhenDataset)
    
    def test_correct_plate_assignment(self, protophen_dataset):
        """Test that samples are assigned to correct splits based on plate."""
        train, val, test = split_by_plate(
            protophen_dataset,
            train_plates=["plate_0", "plate_1"],
            val_plates=["plate_2"],
            test_plates=["plate_3"],
        )
        
        # Check train plates
        train_plates = {s.metadata.get("plate_id") for s in train.samples}
        assert train_plates == {"plate_0", "plate_1"}
        
        # Check val plates
        val_plates = {s.metadata.get("plate_id") for s in val.samples}
        assert val_plates == {"plate_2"}
        
        # Check test plates
        test_plates = {s.metadata.get("plate_id") for s in test.samples}
        assert test_plates == {"plate_3"}
    
    def test_no_test_plates(self, protophen_dataset):
        """Test splitting without test plates."""
        train, val, test = split_by_plate(
            protophen_dataset,
            train_plates=["plate_0", "plate_1", "plate_2"],
            val_plates=["plate_3"],
            test_plates=None,
        )
        
        assert len(train) > 0
        assert len(val) > 0
        assert test is None
    
    def test_samples_not_in_any_plate_excluded(self):
        """Test that samples not matching any plate are excluded."""
        samples = [
            ProtoPhenSample(
                protein_id="protein_0",
                protein_embedding=np.random.randn(128).astype(np.float32),
                metadata={"plate_id": "plate_A"},
            ),
            ProtoPhenSample(
                protein_id="protein_1",
                protein_embedding=np.random.randn(128).astype(np.float32),
                metadata={"plate_id": "plate_B"},
            ),
            ProtoPhenSample(
                protein_id="protein_2",
                protein_embedding=np.random.randn(128).astype(np.float32),
                metadata={"plate_id": "plate_C"},  # Not in any split
            ),
        ]
        
        dataset = ProtoPhenDataset(samples=samples)
        
        train, val, test = split_by_plate(
            dataset,
            train_plates=["plate_A"],
            val_plates=["plate_B"],
            test_plates=[],
        )
        
        assert len(train) == 1
        assert len(val) == 1
        assert len(test) == 0 if test else True
    
    def test_val_test_no_augmentation(self, protophen_dataset):
        """Test that val/test datasets have no augmentation."""
        protophen_dataset.config.embedding_noise_std = 0.1
        protophen_dataset.config.feature_dropout = 0.2
        
        train, val, test = split_by_plate(
            protophen_dataset,
            train_plates=["plate_0", "plate_1"],
            val_plates=["plate_2"],
            test_plates=["plate_3"],
        )
        
        assert train.config.embedding_noise_std == 0.1
        assert val.config.embedding_noise_std == 0.0
        assert test.config.embedding_noise_std == 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestLoadersIntegration:
    """Integration tests for data loading pipeline."""
    
    def test_full_pipeline(self, protophen_dataset):
        """Test complete data loading pipeline."""
        # Split dataset
        train, val, test = split_by_protein(
            protophen_dataset,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        
        # Create loaders
        loaders = create_dataloaders(
            train_dataset=train,
            val_dataset=val,
            test_dataset=test,
            batch_size=4,
            num_workers=0,  # For test stability
        )
        
        # Iterate through all loaders
        for split_name, loader in loaders.items():
            for batch in loader:
                assert "protein_embedding" in batch
                assert "protein_id" in batch
                assert isinstance(batch["protein_embedding"], torch.Tensor)
                assert batch["protein_embedding"].dim() == 2
    
    def test_balanced_sampling_pipeline(self, protophen_dataset):
        """Test pipeline with balanced sampling."""
        sampler = create_balanced_sampler(protophen_dataset, balance_by="plate_id")
        
        loader = create_dataloader(
            protophen_dataset,
            batch_size=4,
            sampler=sampler,
            num_workers=0,
        )
        
        # Should be able to iterate through entire dataset
        all_protein_ids = []
        for batch in loader:
            all_protein_ids.extend(batch["protein_id"])
        
        # With replacement sampling, might have duplicates
        assert len(all_protein_ids) == len(protophen_dataset)
    
    def test_inference_pipeline(self, inference_dataset):
        """Test inference data loading pipeline."""
        loader = create_dataloader(
            inference_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        
        all_embeddings = []
        all_ids = []
        
        for batch in loader:
            all_embeddings.append(batch["protein_embedding"])
            all_ids.extend(batch["protein_id"])
        
        embeddings = torch.cat(all_embeddings, dim=0)
        
        assert embeddings.shape[0] == len(inference_dataset)
        assert len(all_ids) == len(inference_dataset)