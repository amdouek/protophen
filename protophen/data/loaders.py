"""
Data loading utilities for ProToPhen.

This module provides utilities for creating PyTorch DataLoaders
with appropriate batching, sampling, and collation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from protophen.data.dataset import (
    DatasetConfig,
    ProtoPhenDataset, 
    ProteinInferenceDataset,)

from protophen.utils.logging import logger


# =============================================================================
# Collate Functions
# =============================================================================

def protophen_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for ProtoPhen batches.
    
    Handles variable phenotype tasks and metadata.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with tensors stacked appropriately
    """
    if not batch:
        return {}
    
    collated = {}
    
    # Get all keys from first sample
    sample_keys = batch[0].keys()
    
    for key in sample_keys:
        values = [sample[key] for sample in batch]
        
        if key == "protein_id" or key == "protein_name":
            # Keep as list of strings
            collated[key] = values
        
        elif key == "metadata":
            # Keep metadata as list of dicts
            collated[key] = values
        
        elif key.startswith("mask_"):
            # Stack boolean masks
            collated[key] = torch.stack(values)
        
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                # Tensors have different shapes - pad if needed
                max_len = max(v.shape[-1] for v in values)
                padded = []
                for v in values:
                    if v.shape[-1] < max_len:
                        pad_size = max_len - v.shape[-1]
                        v = torch.nn.functional.pad(v, (0, pad_size))
                    padded.append(v)
                collated[key] = torch.stack(padded)
        
        else:
            # Keep as list
            collated[key] = values
    
    return collated


def inference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for inference batches.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    collated = {
        "protein_embedding": torch.stack([s["protein_embedding"] for s in batch]),
        "protein_id": [s["protein_id"] for s in batch],
        "protein_name": [s["protein_name"] for s in batch],
    }
    return collated


# =============================================================================
# DataLoader Factory
# =============================================================================

@dataclass
class DataLoaderConfig:
    """Configuration for data loaders."""
    
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    sampler: Optional[Sampler] = None,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if sampler provided)
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete last batch
        sampler: Optional sampler
        collate_fn: Optional collate function
        
    Returns:
        DataLoader instance
    """
    # Determine collate function
    if collate_fn is None:
        if isinstance(dataset, ProtoPhenDataset):
            collate_fn = protophen_collate_fn
        elif isinstance(dataset, ProteinInferenceDataset):
            collate_fn = inference_collate_fn
    
    # Disable shuffle if sampler provided
    if sampler is not None:
        shuffle = False
    
    # Adjust num_workers if necessary
    if num_workers > 0:
        try:
            import multiprocessing
            max_workers = multiprocessing.cpu_count()
            num_workers = min(num_workers, max_workers)
        except Exception:
            pass
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )


def create_dataloaders(
    train_dataset: ProtoPhenDataset,
    val_dataset: Optional[ProtoPhenDataset] = None,
    test_dataset: Optional[ProtoPhenDataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for GPU
        
    Returns:
        Dictionary with "train", "val", "test" loaders
    """
    loaders = {}
    
    # Training loader (shuffled)
    loaders["train"] = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for training stability
    )
    
    # Validation loader
    if val_dataset is not None:
        loaders["val"] = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    # Test loader
    if test_dataset is not None:
        loaders["test"] = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    logger.info(
        f"Created DataLoaders: "
        f"train={len(loaders['train'])} batches, "
        f"val={len(loaders.get('val', [])) or 'N/A'} batches, "
        f"test={len(loaders.get('test', [])) or 'N/A'} batches"
    )
    
    return loaders


# =============================================================================
# Samplers
# =============================================================================

def create_balanced_sampler(
    dataset: ProtoPhenDataset,
    balance_by: str = "plate_id",
) -> WeightedRandomSampler:
    """
    Create a sampler that balances samples across groups.
    
    Useful for handling plate imbalance in Cell Painting data.
    
    Args:
        dataset: ProtoPhen dataset
        balance_by: Metadata key to balance by
        
    Returns:
        WeightedRandomSampler
    """
    # Count samples per group
    group_counts: Dict[str, int] = {}
    sample_groups: List[str] = []
    
    for sample in dataset.samples:
        group = sample.metadata.get(balance_by, "unknown")
        group_counts[group] = group_counts.get(group, 0) + 1
        sample_groups.append(group)
    
    # Compute weights (inverse frequency)
    n_samples = len(dataset)
    n_groups = len(group_counts)
    
    weights = []
    for group in sample_groups:
        weight = n_samples / (n_groups * group_counts[group])
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float64)
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=n_samples,
        replacement=True,
    )


# =============================================================================
# Data Splitting Utilities
# =============================================================================

def split_by_protein(
    dataset: ProtoPhenDataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[ProtoPhenDataset, ProtoPhenDataset, ProtoPhenDataset]:
    """
    Split dataset ensuring proteins don't appear in multiple splits.
    
    This is important for proper evaluation - we want to test
    generalisation to new proteins, not interpolation.
    
    Args:
        dataset: ProtoPhen dataset
        train_frac: Training fraction
        val_frac: Validation fraction
        test_frac: Test fraction
        seed: Random seed
        
    Returns:
        Tuple of (train, val, test) datasets
    """
    # Get unique proteins
    protein_ids = list(set(s.protein_id for s in dataset.samples))
    n_proteins = len(protein_ids)
    
    # Shuffle proteins
    rng = np.random.default_rng(seed)
    rng.shuffle(protein_ids)
    
    # Split proteins
    n_train = int(n_proteins * train_frac)
    n_val = int(n_proteins * val_frac)
    
    train_proteins = set(protein_ids[:n_train])
    val_proteins = set(protein_ids[n_train:n_train + n_val])
    test_proteins = set(protein_ids[n_train + n_val:])
    
    # Assign samples to splits
    train_samples = []
    val_samples = []
    test_samples = []
    
    for sample in dataset.samples:
        if sample.protein_id in train_proteins:
            train_samples.append(sample)
        elif sample.protein_id in val_proteins:
            val_samples.append(sample)
        else:
            test_samples.append(sample)
    
    # Create datasets
    train_config = dataset.config
    val_config = DatasetConfig(
        protein_embedding_key=train_config.protein_embedding_key,
        phenotype_tasks=train_config.phenotype_tasks,
        embedding_noise_std=0.0,
        feature_dropout=0.0,
    )
    
    train_dataset = ProtoPhenDataset(samples=train_samples, config=train_config)
    val_dataset = ProtoPhenDataset(samples=val_samples, config=val_config)
    test_dataset = ProtoPhenDataset(samples=test_samples, config=val_config)
    
    logger.info(
        f"Split by protein: train={len(train_dataset)} samples ({len(train_proteins)} proteins), "
        f"val={len(val_dataset)} samples ({len(val_proteins)} proteins), "
        f"test={len(test_dataset)} samples ({len(test_proteins)} proteins)"
    )
    
    return train_dataset, val_dataset, test_dataset


def split_by_plate(
    dataset: ProtoPhenDataset,
    train_plates: List[str],
    val_plates: List[str],
    test_plates: Optional[List[str]] = None,
) -> Tuple[ProtoPhenDataset, ProtoPhenDataset, Optional[ProtoPhenDataset]]:
    """
    Split dataset by plate ID.
    
    Useful for testing generalisation across experimental batches.
    
    Args:
        dataset: ProtoPhen dataset
        train_plates: List of training plate IDs
        val_plates: List of validation plate IDs
        test_plates: Optional list of test plate IDs
        
    Returns:
        Tuple of datasets
    """
    train_plates = set(train_plates)
    val_plates = set(val_plates)
    test_plates = set(test_plates) if test_plates else set()
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for sample in dataset.samples:
        plate_id = sample.metadata.get("plate_id")
        
        if plate_id in train_plates:
            train_samples.append(sample)
        elif plate_id in val_plates:
            val_samples.append(sample)
        elif plate_id in test_plates:
            test_samples.append(sample)
    
    train_config = dataset.config
    val_config = DatasetConfig(
        protein_embedding_key=train_config.protein_embedding_key,
        phenotype_tasks=train_config.phenotype_tasks,
        embedding_noise_std=0.0,
        feature_dropout=0.0,
    )
    
    train_dataset = ProtoPhenDataset(samples=train_samples, config=train_config)
    val_dataset = ProtoPhenDataset(samples=val_samples, config=val_config)
    test_dataset = ProtoPhenDataset(samples=test_samples, config=val_config) if test_samples else None
    
    return train_dataset, val_dataset, test_dataset