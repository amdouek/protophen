"""
Configuration management for ProToPhen.

This module provides a hierarchical configuration system using dataclasses and YAML files for easy experiment tracking and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import OmegaConf

# =========================
# Configuration Dataclasses
# =========================

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Paths
    protein_library_path: Optional[str] = None
    phenotype_data_path: Optional[str] = None
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    
    # Protein filtering
    min_sequence_length: int = 10
    max_sequence_length: int = 2000
    
    # Train/val/test splits
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 42
    
@dataclass
class EmbeddingConfig:
    """Configuration for protein embeddings."""
    
    # ESM-2 settings
    esm_model_name: str = "esm2_t33_650M_UR50D" # Options: esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D
    esm_layer: int = -1 # Which layer to extract; -1 indicates the last layer
    esm_pooling: str = "mean" # Options: mean, cls, max
    
    # ESMFold settings
    use_structure: bool = False
    esmfold_chunk_size: Optional[int] = None # If None, process full sequences
    
    # Physicochemical features
    include_physicochemical: bool = True
    
    # Fusion
    fusion_method: str = "concatenate" # Options: concatenate, attention, gated
    
    # Computation
    batch_size: int = 8
    device: str = "cuda" # Options: cuda, cpu, mps
    use_fp16: bool = True
    
@dataclass
class PhenotypeConfig:
    """Configuration for phenotype data processing."""
    
    # Cell Painting settings
    feature_selection_method: str = "variance" # Options: variance, correlation, none
    n_features_to_keep: Optional[int] = None # If None, keep all after selection
    
    # Normalisation
    plate_normalisation: str = "robust_mad" # Options: robust_mad, zscore, none
    feature_normalisation: str = "standardise" # Options: standardise, minmax, none
    
    # Batch correction
    batch_correction_method: Optional[str] = None # Options: combat, harmony, none
    
    # Outlier handling
    clip_outliers: bool = True
    outlier_threshold: float = 5.0 # Standard deviations
    
@dataclass
class ModelConfig:
    """Configuration for the ProToPhen model."""
    
    # Architecture
    protein_embedding_dim: int = 1280 # ESM-2 650M output dim
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    dropout_rate: float = 0.1
    activation: str = "gelu" # Options: relu, gelu, silu
    
    # Output heads
    predict_cell_painting: bool = True
    predict_viability: bool = True
    predict_transcriptomics: bool = False
    
    # Cell Painting head
    cell_painting_dim: int = 1500 # Approx number of Cell Painting features
    
    # Multitask learning
    task_weights: dict[str, float] = field(default_factory=lambda: {
        "cell_painting": 1.0,
        "viability": 0.5,
        "transcriptomics": 0.5,
    })
    
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Optimisation
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimiser: str = "adamw" # Options: adam, adamw, sgd
    scheduler: str = "cosine" # Options: cosine, linear, constant, plateau
    warmup_steps: int = 100
    
    # Training loop
    epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 10
    evaluate_every_n_epochs: int = 1
    
    # Reproducibility
    seed: int = 42
    
@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    
    # Uncertainty estimation
    uncertainty_method: str = "mc_dropout" # Options: mc_dropout, ensemble, evidential
    n_mc_samples: int = 20
    
    # Acquisition function
    acquisition_function: str = "expected_improvement" # Options: expected_improvement, uncertainty, diversity
    
    # Selection
    batch_size: int = 10 # Number of proteins to select per iteration
    diversity_weight: float = 0.3 # Weight for diversity vs uncertainty in acquisition
    
@dataclass
class ProtoPhenConfig:
    """Master config for ProToPhen.
    
    This dataclass aggregates all sub-configurations and provides methods for loading/saving from YAML files.
    
    Example:
        >>> config = ProtoPhenConfig()
        >>> config.save("configs/experiment_1.yaml")
        >>> 
        >>> # Load and modify
        >>> config = ProtoPhenConfig.from_yaml("configs/experiment_1.yaml")
        >>> config.training.learning_rate = 5e-5
    """
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    phenotype: PhenotypeConfig = field(default_factory=PhenotypeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    
    # Experiment metadata
    experiment_name: str = "default"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    
    def save(self, path: str | Path) -> None:
        """Save the configuration to a YAML file.
        
        Args:
            path: Output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to Omegaconf for nice YAML output
        conf = OmegaConf.structured(self)
        
        with open(path, "w") as f:
            OmegaConf.save(conf, f)
            
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProtoPhenConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            ProtoPhenConfig instance.
        """
        path = Path(path)
        
        with open(path) as f:
            raw_config = yaml.safe_load(f)
        
        # Use OmegaConf for merging with defaults
        default_conf = OmegaConf.structured(cls())
        loaded_conf = OmegaConf.create(raw_config)
        merged_conf = OmegaConf.merge(default_conf, loaded_conf)
        
        return OmegaConf.to_object(merged_conf)
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ProtoPhenConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary.
            
        Returns:
            ProtoPhenConfig instance.
        """
        default_conf = OmegaConf.structured(cls())
        loaded_conf = OmegaConf.create(config_dict)
        merged_conf = OmegaConf.merge(default_conf, loaded_conf)
        
        return OmegaConf.to_object(merged_conf)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as a dictionary.
        """
        return OmegaConf.to_container(OmegaConf.structured(self))
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ProtoPhenConfig(experiment_name='{self.experiment_name}')"
    
# =================
# Utility Functions
# =================

def load_config(path: str | Path) -> ProtoPhenConfig:
    """
    Load configuration from YAML file.
    
    Convenience function that wraps ProtoPhenConfig.from_yaml().
    
    Args:
        path: Path to YAML configuration file.
        
    Returns:
        ProtoPhenConfig instance.
    """
    return ProtoPhenConfig.from_yaml(path)

def create_default_config(output_path: Optional[str | Path] = None) -> ProtoPhenConfig:
    """
    Create a default configuration, optionally saving to a file.
    
    Args:
        output_path: If provided, save config to this path.
        
    Returns:
        Default ProtoPhenConfig instance.
    """
    config = ProtoPhenConfig()
    
    if output_path:
        config.save(output_path)
        
    return config