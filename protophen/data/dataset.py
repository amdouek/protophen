"""
PyTorch datasets for ProToPhen.

This module provides Dataset classes for training protein-to-phenotype
prediction models, handling the pairing of protein embeddings with
phenotypic measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from protophen.data.phenotype import Phenotype, PhenotypeDataset
from protophen.data.protein import Protein, ProteinLibrary
from protophen.utils.logging import logger


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for ProtoPhen datasets."""
    
    # Embedding settings
    protein_embedding_key: str = "fused"
    fallback_embedding_keys: List[str] = field(default_factory=lambda: ["esm2", "physicochemical"])
    
    # Phenotype settings
    phenotype_tasks: List[str] = field(default_factory=lambda: ["cell_painting"])
    
    # Data augmentation
    embedding_noise_std: float = 0.0  # Add Gaussian noise to embeddings
    feature_dropout: float = 0.0  # Randomly zero phenotype features
    
    # Filtering
    require_qc_passed: bool = True
    min_cell_count: Optional[int] = None
    
    # Normalisation (applied on-the-fly)
    normalise_phenotypes: bool = False
    phenotype_mean: Optional[np.ndarray] = None
    phenotype_std: Optional[np.ndarray] = None


# =============================================================================
# Sample Data Structure
# =============================================================================

@dataclass
class ProtoPhenSample:
    """
    A single sample for model training/inference.
    
    Attributes:
        protein_id: Unique protein identifier
        protein_embedding: Protein embedding vector
        phenotypes: Dictionary of phenotype vectors by task
        metadata: Additional metadata
    """
    protein_id: str
    protein_embedding: np.ndarray
    phenotypes: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "protein_id": self.protein_id,
            "protein_embedding": self.protein_embedding,
            "phenotypes": self.phenotypes,
            "metadata": self.metadata,
        }


# =============================================================================
# Main Dataset Class
# =============================================================================

class ProtoPhenDataset(Dataset):
    """
    PyTorch Dataset for protein-phenotype prediction.
    
    This dataset pairs protein embeddings with their corresponding
    phenotypic measurements for supervised learning.
    
    Attributes:
        samples: List of ProtoPhenSample objects
        config: Dataset configuration
        
    Example:
        >>> # Create from protein library and phenotype dataset
        >>> dataset = ProtoPhenDataset.from_data(
        ...     proteins=protein_library,
        ...     phenotypes=phenotype_dataset,
        ...     embedding_key="fused",
        ... )
        >>> 
        >>> # Access samples
        >>> sample = dataset[0]
        >>> print(sample["protein_embedding"].shape)
        >>> print(sample["cell_painting"].shape)
        >>> 
        >>> # Use with DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        samples: Optional[List[ProtoPhenSample]] = None,
        config: Optional[DatasetConfig] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialise dataset.
        
        Args:
            samples: List of ProtoPhenSample objects
            config: Dataset configuration
            transform: Optional transform to apply to samples
        """
        self.samples = samples or []
        self.config = config or DatasetConfig()
        self.transform = transform
        
        # Cache for fast lookup
        self._protein_id_to_idx: Dict[str, int] = {}
        self._rebuild_index()
        
        # Statistics (computed lazily)
        self._embedding_dim: Optional[int] = None
        self._phenotype_dims: Dict[str, int] = {}
    
    def _rebuild_index(self) -> None:
        """Rebuild protein ID index."""
        self._protein_id_to_idx = {
            sample.protein_id: idx for idx, sample in enumerate(self.samples)
        }
    
    @classmethod
    def from_data(
        cls,
        proteins: Union[ProteinLibrary, List[Protein]],
        phenotypes: Union[PhenotypeDataset, List[Phenotype]],
        embedding_key: str = "fused",
        fallback_keys: Optional[List[str]] = None,
        config: Optional[DatasetConfig] = None,
    ) -> "ProtoPhenDataset":
        """
        Create dataset from protein and phenotype data.
        
        Args:
            proteins: Protein library or list of proteins
            phenotypes: Phenotype dataset or list of phenotypes
            embedding_key: Key for protein embeddings
            fallback_keys: Fallback embedding keys if primary not found
            config: Dataset configuration
            
        Returns:
            ProtoPhenDataset instance
        """
        config = config or DatasetConfig()
        config.protein_embedding_key = embedding_key
        if fallback_keys:
            config.fallback_embedding_keys = fallback_keys
        
        # Convert to lists if necessary
        if isinstance(proteins, ProteinLibrary):
            protein_list = list(proteins)
        else:
            protein_list = proteins
        
        if isinstance(phenotypes, PhenotypeDataset):
            phenotype_list = list(phenotypes)
        else:
            phenotype_list = phenotypes
        
        # Build protein lookup
        protein_by_id: Dict[str, Protein] = {}
        protein_by_name: Dict[str, Protein] = {}
        protein_by_hash: Dict[str, Protein] = {}
        
        for protein in protein_list:
            protein_by_hash[protein.hash] = protein
            if protein.name:
                protein_by_name[protein.name] = protein
        
        # Match proteins to phenotypes
        samples = []
        matched = 0
        unmatched_proteins = []
        
        for phenotype in phenotype_list:
            # Skip QC-failed samples if configured
            if config.require_qc_passed and not phenotype.qc_passed:
                continue
            
            # Skip samples with low cell count
            if config.min_cell_count and phenotype.cell_count:
                if phenotype.cell_count < config.min_cell_count:
                    continue
            
            # Find matching protein
            protein = None
            protein_id = phenotype.protein_id
            
            if protein_id:
                # Try direct lookup
                protein = protein_by_hash.get(protein_id) or protein_by_name.get(protein_id)
            
            if protein is None:
                unmatched_proteins.append(phenotype.sample_id)
                continue
            
            # Get protein embedding
            embedding = cls._get_embedding(protein, config)
            if embedding is None:
                continue
            
            # Create sample
            sample = ProtoPhenSample(
                protein_id=protein.hash,
                protein_embedding=embedding,
                phenotypes={"cell_painting": phenotype.features},
                metadata={
                    "protein_name": protein.name,
                    "sample_id": phenotype.sample_id,
                    "well_id": phenotype.well_id,
                    "plate_id": phenotype.plate_id,
                    "cell_count": phenotype.cell_count,
                },
            )
            samples.append(sample)
            matched += 1
        
        if unmatched_proteins:
            logger.warning(
                f"Could not match {len(unmatched_proteins)} phenotypes to proteins. "
                f"First few: {unmatched_proteins[:5]}"
            )
        
        logger.info(f"Created dataset with {matched} matched protein-phenotype pairs")
        
        return cls(samples=samples, config=config)
    
    @staticmethod
    def _get_embedding(protein: Protein, config: DatasetConfig) -> Optional[np.ndarray]:
        """Get embedding from protein, trying fallback keys if necessary."""
        # Try primary key
        if config.protein_embedding_key in protein.embeddings:
            return protein.embeddings[config.protein_embedding_key]
        
        # Try fallback keys
        for key in config.fallback_embedding_keys:
            if key in protein.embeddings:
                return protein.embeddings[key]
        
        # Try concatenating fallback keys
        available = [
            protein.embeddings[k] 
            for k in config.fallback_embedding_keys 
            if k in protein.embeddings
        ]
        if available:
            return np.concatenate(available, axis=-1)
        
        logger.warning(f"No embedding found for protein {protein.name}")
        return None
    
    @classmethod
    def from_arrays(
        cls,
        protein_embeddings: np.ndarray,
        phenotype_features: np.ndarray,
        protein_ids: Optional[List[str]] = None,
        config: Optional[DatasetConfig] = None,
    ) -> "ProtoPhenDataset":
        """
        Create dataset from numpy arrays.
        
        Args:
            protein_embeddings: Protein embedding matrix (n_samples, embed_dim)
            phenotype_features: Phenotype feature matrix (n_samples, n_features)
            protein_ids: Optional list of protein IDs
            config: Dataset configuration
            
        Returns:
            ProtoPhenDataset instance
        """
        if protein_embeddings.shape[0] != phenotype_features.shape[0]:
            raise ValueError(
                f"Number of proteins ({protein_embeddings.shape[0]}) must match "
                f"number of phenotypes ({phenotype_features.shape[0]})"
            )
        
        n_samples = protein_embeddings.shape[0]
        if protein_ids is None:
            protein_ids = [f"protein_{i}" for i in range(n_samples)]
        
        samples = []
        for i in range(n_samples):
            sample = ProtoPhenSample(
                protein_id=protein_ids[i],
                protein_embedding=protein_embeddings[i],
                phenotypes={"cell_painting": phenotype_features[i]},
            )
            samples.append(sample)
        
        return cls(samples=samples, config=config)
    
    def add_phenotype_task(
        self,
        task_name: str,
        features: Dict[str, np.ndarray],
    ) -> None:
        """
        Add a new phenotype task to existing samples.
        
        Args:
            task_name: Name of the phenotype task (e.g., "viability")
            features: Dictionary mapping protein IDs to feature arrays
        """
        added = 0
        for sample in self.samples:
            if sample.protein_id in features:
                sample.phenotypes[task_name] = features[sample.protein_id]
                added += 1
        
        if task_name not in self.config.phenotype_tasks:
            self.config.phenotype_tasks.append(task_name)
        
        logger.info(f"Added '{task_name}' task to {added} samples")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - protein_embedding: Tensor of shape (embed_dim,)
                - {task_name}: Tensor of shape (n_features,) for each task
                - protein_id: String identifier
                - mask_{task_name}: Boolean indicating if task is available
        """
        sample = self.samples[idx]
        
        # Get protein embedding
        embedding = sample.protein_embedding.copy()
        
        # Apply embedding noise during training
        if self.config.embedding_noise_std > 0:
            noise = np.random.randn(*embedding.shape) * self.config.embedding_noise_std
            embedding = embedding + noise
        
        # Build output dictionary
        output = {
            "protein_embedding": torch.from_numpy(embedding).float(),
            "protein_id": sample.protein_id,
        }
        
        # Add phenotype tasks
        for task_name in self.config.phenotype_tasks:
            if task_name in sample.phenotypes:
                features = sample.phenotypes[task_name].copy()
                
                # Apply normalisation if configured
                if self.config.normalise_phenotypes:
                    if self.config.phenotype_mean is not None:
                        features = features - self.config.phenotype_mean
                    if self.config.phenotype_std is not None:
                        features = features / (self.config.phenotype_std + 1e-8)
                
                # Apply feature dropout during training
                if self.config.feature_dropout > 0:
                    mask = np.random.rand(*features.shape) > self.config.feature_dropout
                    features = features * mask
                
                output[task_name] = torch.from_numpy(features).float()
                output[f"mask_{task_name}"] = torch.tensor(True)
            else:
                # Task not available for this sample
                output[f"mask_{task_name}"] = torch.tensor(False)
        
        # Add metadata
        output["metadata"] = sample.metadata
        
        # Apply transform if provided
        if self.transform is not None:
            output = self.transform(output)
        
        return output
    
    def get_by_protein_id(self, protein_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get sample by protein ID."""
        idx = self._protein_id_to_idx.get(protein_id)
        if idx is not None:
            return self[idx]
        return None
    
    @property
    def embedding_dim(self) -> int:
        """Get protein embedding dimension."""
        if self._embedding_dim is None and len(self.samples) > 0:
            self._embedding_dim = self.samples[0].protein_embedding.shape[-1]
        return self._embedding_dim or 0
    
    @property
    def phenotype_dims(self) -> Dict[str, int]:
        """Get phenotype dimensions for each task."""
        if not self._phenotype_dims and len(self.samples) > 0:
            for task_name in self.config.phenotype_tasks:
                for sample in self.samples:
                    if task_name in sample.phenotypes:
                        self._phenotype_dims[task_name] = sample.phenotypes[task_name].shape[-1]
                        break
        return self._phenotype_dims
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary with statistics for embeddings and phenotypes
        """
        stats = {
            "n_samples": len(self.samples),
            "embedding_dim": self.embedding_dim,
            "phenotype_dims": self.phenotype_dims,
        }
        
        if len(self.samples) > 0:
            # Embedding statistics
            embeddings = np.stack([s.protein_embedding for s in self.samples])
            stats["embedding_mean"] = float(np.mean(embeddings))
            stats["embedding_std"] = float(np.std(embeddings))
            
            # Phenotype statistics per task
            for task_name in self.config.phenotype_tasks:
                task_features = [
                    s.phenotypes[task_name] 
                    for s in self.samples 
                    if task_name in s.phenotypes
                ]
                if task_features:
                    features = np.stack(task_features)
                    stats[f"{task_name}_mean"] = float(np.mean(features))
                    stats[f"{task_name}_std"] = float(np.std(features))
                    stats[f"{task_name}_n_samples"] = len(task_features)
        
        return stats
    
    def compute_normalisation_stats(
        self,
        task_name: str = "cell_painting",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std for phenotype normalisation.
        
        Args:
            task_name: Which phenotype task to compute stats for.
            
        Returns:
            Tuple of (mean, std) arrays.
        """
        features = [
            s.phenotypes[task_name] 
            for s in self.samples 
            if task_name in s.phenotypes
        ]
        
        if not features:
            raise ValueError(f"No samples have task '{task_name}'")
        
        features = np.stack(features)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        
        return mean, std
    
    def set_normalisation_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        """
        Set normalisation statistics for on-the-fly normalisation.
        
        Args:
            mean: Feature means
            std: Feature standard deviations
        """
        self.config.normalise_phenotypes = True
        self.config.phenotype_mean = mean
        self.config.phenotype_std = std
    
    def split(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        stratify_by: Optional[str] = None,
    ) -> Tuple["ProtoPhenDataset", "ProtoPhenDataset", "ProtoPhenDataset"]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            train_frac: Fraction for training
            val_frac: Fraction for validation
            test_frac: Fraction for testing
            seed: Random seed
            stratify_by: Optional metadata key for stratified splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
        
        n_samples = len(self.samples)
        indices = np.arange(n_samples)
        
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        
        n_train = int(n_samples * train_frac)
        n_val = int(n_samples * val_frac)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create new datasets
        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]
        test_samples = [self.samples[i] for i in test_indices]
        
        # Create configs without augmentation for val/test
        val_config = DatasetConfig(
            protein_embedding_key=self.config.protein_embedding_key,
            phenotype_tasks=self.config.phenotype_tasks,
            embedding_noise_std=0.0,  # No augmentation
            feature_dropout=0.0,
        )
        
        train_dataset = ProtoPhenDataset(samples=train_samples, config=self.config)
        val_dataset = ProtoPhenDataset(samples=val_samples, config=val_config)
        test_dataset = ProtoPhenDataset(samples=test_samples, config=val_config)
        
        logger.info(
            f"Split dataset: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def subset(self, indices: Sequence[int]) -> "ProtoPhenDataset":
        """
        Create a subset of the dataset.
        
        Args:
            indices: Indices to include
            
        Returns:
            New ProtoPhenDataset with selected samples
        """
        samples = [self.samples[i] for i in indices]
        return ProtoPhenDataset(samples=samples, config=self.config)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProtoPhenDataset(n_samples={len(self)}, "
            f"embedding_dim={self.embedding_dim}, "
            f"tasks={self.config.phenotype_tasks})"
        )


# =============================================================================
# Inference Dataset (Proteins Only)
# =============================================================================

class ProteinInferenceDataset(Dataset):
    """
    Dataset for inference on proteins without phenotype labels.
    
    Used for predicting cellular phenotypes for new/unseen proteins.
    
    Example:
        >>> dataset = ProteinInferenceDataset.from_library(
        ...     proteins=new_proteins,
        ...     embedding_key="fused",
        ... )
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> 
        >>> predictions = []
        >>> for batch in loader:
        ...     pred = model(batch["protein_embedding"])
        ...     predictions.append(pred)
    """
    
    def __init__(
        self,
        protein_embeddings: np.ndarray,
        protein_ids: List[str],
        protein_names: Optional[List[str]] = None,
    ):
        """
        Initialise inference dataset.
        
        Args:
            protein_embeddings: Embedding matrix (n_proteins, embed_dim)
            protein_ids: List of protein IDs (hashes)
            protein_names: Optional list of protein names
        """
        self.protein_embeddings = protein_embeddings
        self.protein_ids = protein_ids
        self.protein_names = protein_names or protein_ids
    
    @classmethod
    def from_library(
        cls,
        proteins: Union[ProteinLibrary, List[Protein]],
        embedding_key: str = "fused",
        fallback_keys: Optional[List[str]] = None,
    ) -> "ProteinInferenceDataset":
        """
        Create from protein library.
        
        Args:
            proteins: Protein library or list
            embedding_key: Key for embeddings
            fallback_keys: Fallback embedding keys
            
        Returns:
            ProteinInferenceDataset instance
        """
        if isinstance(proteins, ProteinLibrary):
            protein_list = list(proteins)
        else:
            protein_list = proteins
        
        fallback_keys = fallback_keys or ["esm2", "physicochemical"]
        
        embeddings = []
        ids = []
        names = []
        
        for protein in protein_list:
            # Get embedding
            emb = None
            if embedding_key in protein.embeddings:
                emb = protein.embeddings[embedding_key]
            else:
                for key in fallback_keys:
                    if key in protein.embeddings:
                        emb = protein.embeddings[key]
                        break
                
                # Try concatenating
                if emb is None:
                    available = [
                        protein.embeddings[k] 
                        for k in fallback_keys 
                        if k in protein.embeddings
                    ]
                    if available:
                        emb = np.concatenate(available, axis=-1)
            
            if emb is not None:
                embeddings.append(emb)
                ids.append(protein.hash)
                names.append(protein.name)
        
        embeddings = np.stack(embeddings)
        
        return cls(
            protein_embeddings=embeddings,
            protein_ids=ids,
            protein_names=names,
        )
    
    def __len__(self) -> int:
        return len(self.protein_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "protein_embedding": torch.from_numpy(self.protein_embeddings[idx]).float(),
            "protein_id": self.protein_ids[idx],
            "protein_name": self.protein_names[idx],
        }
    
    @property
    def embedding_dim(self) -> int:
        return self.protein_embeddings.shape[-1]
    
    def __repr__(self) -> str:
        return f"ProteinInferenceDataset(n_proteins={len(self)}, embedding_dim={self.embedding_dim})"