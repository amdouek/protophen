"""
Main ProToPhen model architecture.

This module provides the complete model for predicting cellular
phenotypes from protein embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from protophen.models.encoders import ProteinEncoder, ProteinEncoderConfig
from protophen.models.decoders import (
    CellPaintingHead,
    MultiTaskHead,
    PhenotypeDecoder,
    ViabilityHead,
)
from protophen.utils.logging import logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProToPhenConfig:
    """Configuration for the ProToPhen model."""
    
    # Input dimensions
    protein_embedding_dim: int = 1280  # ESM-2 or fused embedding dim
    
    # Encoder settings
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])
    encoder_output_dim: int = 256
    encoder_dropout: float = 0.1
    encoder_activation: str = "gelu"
    
    # Decoder settings
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 1024])
    decoder_dropout: float = 0.1
    
    # Task-specific settings
    cell_painting_dim: int = 1500
    predict_viability: bool = True
    predict_transcriptomics: bool = False
    transcriptomics_dim: int = 978  # Placeholder for L1000 assay - consider to change if the transcriptomics data is from RNA-seq.
    
    # Uncertainty estimation
    predict_uncertainty: bool = False
    
    # Regularisation
    use_spectral_norm: bool = False
    
    # MC Dropout for uncertainty
    mc_dropout: bool = True


# =============================================================================
# Main Model
# =============================================================================

class ProToPhenModel(nn.Module):
    """
    ProToPhen: Protein to Phenotype prediction model.
    
    This model takes pre-computed protein embeddings and predicts
    cellular phenotypes including Cell Painting morphological features,
    viability, and optionally transcriptomics.
    
    Architecture:
        Protein Embedding -> Encoder -> Latent -> Decoder(s) -> Phenotype(s)
    
    Attributes:
        config: Model configuration
        encoder: Protein embedding encoder
        decoders: Task-specific decoder heads
        
    Example:
        >>> config = ProToPhenConfig(
        ...     protein_embedding_dim=1280 + 439,  # ESM-2 + physicochemical
        ...     cell_painting_dim=1500,
        ... )
        >>> model = ProToPhenModel(config)
        >>> 
        >>> # Forward pass
        >>> embeddings = torch.randn(32, 1719)
        >>> outputs = model(embeddings)
        >>> print(outputs["cell_painting"].shape)  # (32, 1500)
        >>> 
        >>> # With uncertainty
        >>> model.config.predict_uncertainty = True
        >>> outputs = model(embeddings, return_uncertainty=True)
    """
    
    def __init__(
        self,
        config: Optional[ProToPhenConfig] = None,
        **kwargs,
    ):
        """
        Initialise ProToPhen model.
        
        Args:
            config: Model configuration
            **kwargs: Override config parameters
        """
        super().__init__()
        
        # Build config
        if config is None:
            config = ProToPhenConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        
        # Build encoder
        encoder_config = ProteinEncoderConfig(
            input_dim=config.protein_embedding_dim,
            hidden_dims=config.encoder_hidden_dims,
            output_dim=config.encoder_output_dim,
            activation=config.encoder_activation,
            dropout=config.encoder_dropout,
        )
        self.encoder = ProteinEncoder(encoder_config)
        
        # Build decoders
        self.decoders = nn.ModuleDict()
        
        # Cell Painting head (always present)
        self.decoders["cell_painting"] = CellPaintingHead(
            input_dim=config.encoder_output_dim,
            output_dim=config.cell_painting_dim,
            hidden_dims=config.decoder_hidden_dims,
            dropout=config.decoder_dropout,
            predict_uncertainty=config.predict_uncertainty,
        )
        
        # Viability head (optional)
        if config.predict_viability:
            self.decoders["viability"] = ViabilityHead(
                input_dim=config.encoder_output_dim,
                hidden_dims=[128, 64],
                dropout=config.decoder_dropout,
            )
        
        # Transcriptomics head (optional)
        if config.predict_transcriptomics:
            self.decoders["transcriptomics"] = PhenotypeDecoder(
                input_dim=config.encoder_output_dim,
                output_dim=config.transcriptomics_dim,
                hidden_dims=config.decoder_hidden_dims,
                dropout=config.decoder_dropout,
            )
        
        # Apply spectral normalisation if requested
        if config.use_spectral_norm:
            self._apply_spectral_norm()
        
        # Count parameters
        self._n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Initialised ProToPhenModel: "
            f"{self._n_params:,} parameters, "
            f"tasks={list(self.decoders.keys())}"
        )
    
    def _apply_spectral_norm(self) -> None:
        """Apply spectral normalisation to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.utils.spectral_norm(module)
    
    def forward(
        self,
        protein_embedding: torch.Tensor,
        tasks: Optional[List[str]] = None,
        return_latent: bool = False,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            protein_embedding: Protein embeddings of shape (batch_size, embed_dim)
            tasks: List of tasks to predict (None = all available)
            return_latent: Whether to include latent representation in output
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing predictions for each task
        """
        # Encode protein embeddings
        latent = self.encoder(protein_embedding)
        
        # Determine tasks to compute
        if tasks is None:
            tasks = list(self.decoders.keys())
        
        # Compute predictions
        outputs = {}
        
        for task_name in tasks:
            if task_name not in self.decoders:
                continue
            
            decoder = self.decoders[task_name]
            
            # Handle uncertainty prediction for Cell Painting
            if task_name == "cell_painting" and self.config.predict_uncertainty:
                mean, log_var = decoder(latent)
                outputs[task_name] = mean
                if return_uncertainty:
                    outputs[f"{task_name}_log_var"] = log_var
                    outputs[f"{task_name}_std"] = torch.exp(0.5 * log_var)
            else:
                outputs[task_name] = decoder(latent)
        
        # Optionally include latent representation
        if return_latent:
            outputs["latent"] = latent
        
        return outputs
    
    def predict(
        self,
        protein_embedding: torch.Tensor,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions (evaluation mode).
        
        Args:
            protein_embedding: Protein embeddings
            tasks: Tasks to predict
            
        Returns:
            Dictionary of predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(protein_embedding, tasks=tasks)
    
    def predict_with_uncertainty(
        self,
        protein_embedding: torch.Tensor,
        n_samples: int = 20,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Make predictions with MC Dropout uncertainty estimation.
        
        Args:
            protein_embedding: Protein embeddings of shape (batch_size, embed_dim)
            n_samples: Number of MC samples
            tasks: Tasks to predict
            
        Returns:
            Dictionary containing for each task:
                - mean: Mean prediction
                - std: Standard deviation (epistemic uncertainty)
                - samples: All MC samples
        """
        if not self.config.mc_dropout:
            logger.warning("MC Dropout not enabled in config")
        
        # Enable dropout during inference
        self.train()
        
        # Determine tasks
        if tasks is None:
            tasks = list(self.decoders.keys())
        
        # Collect samples
        all_samples = {task: [] for task in tasks}
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(protein_embedding, tasks=tasks)
                for task in tasks:
                    if task in outputs:
                        all_samples[task].append(outputs[task])
        
        # Compute statistics
        results = {}
        for task in tasks:
            if all_samples[task]:
                samples = torch.stack(all_samples[task], dim=0)  # (n_samples, batch, features)
                results[task] = {
                    "mean": samples.mean(dim=0),
                    "std": samples.std(dim=0),
                    "samples": samples,
                }
        
        # Return to eval mode
        self.eval()
        
        return results
    
    def get_latent(self, protein_embedding: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation without decoding.
        
        Args:
            protein_embedding: Protein embeddings
            
        Returns:
            Latent representation of shape (batch_size, encoder_output_dim)
        """
        return self.encoder(protein_embedding)
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters (for transfer learning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def freeze_decoder(self, task_name: str) -> None:
        """Freeze a specific decoder."""
        if task_name in self.decoders:
            for param in self.decoders[task_name].parameters():
                param.requires_grad = False
            logger.info(f"Decoder '{task_name}' frozen")
    
    def add_task(
        self,
        task_name: str,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """
        Add a new prediction task.
        
        Args:
            task_name: Name for the new task
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
        """
        if task_name in self.decoders:
            logger.warning(f"Task '{task_name}' already exists, replacing")
        
        self.decoders[task_name] = PhenotypeDecoder(
            input_dim=self.config.encoder_output_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims or self.config.decoder_hidden_dims,
            dropout=self.config.decoder_dropout,
        )
        
        logger.info(f"Added task '{task_name}' with output_dim={output_dim}")
    
    @property
    def n_parameters(self) -> int:
        """Total number of parameters."""
        return self._n_params
    
    @property
    def n_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def task_names(self) -> List[str]:
        """List of available tasks."""
        return list(self.decoders.keys())
    
    @property
    def latent_dim(self) -> int:
        """Latent representation dimension."""
        return self.config.encoder_output_dim
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary."""
        return {
            "n_parameters": self.n_parameters,
            "n_trainable_parameters": self.n_trainable_parameters,
            "protein_embedding_dim": self.config.protein_embedding_dim,
            "latent_dim": self.latent_dim,
            "tasks": {
                name: decoder.output_dim 
                for name, decoder in self.decoders.items()
            },
            "encoder_hidden_dims": self.config.encoder_hidden_dims,
            "decoder_hidden_dims": self.config.decoder_hidden_dims,
        }
    
    def __repr__(self) -> str:
        tasks_str = ", ".join(
            f"{k}:{v.output_dim}" for k, v in self.decoders.items()
        )
        return (
            f"ProToPhenModel("
            f"input_dim={self.config.protein_embedding_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"tasks=[{tasks_str}], "
            f"params={self.n_parameters:,})"
        )


# =============================================================================
# Model Factory Functions
# =============================================================================

def create_protophen_model(
    protein_embedding_dim: int,
    cell_painting_dim: int = 1500,
    predict_viability: bool = True,
    predict_uncertainty: bool = False,
    **kwargs,
) -> ProToPhenModel:
    """
    Create a ProToPhen model with common settings.
    
    Args:
        protein_embedding_dim: Dimension of protein embeddings
        cell_painting_dim: Number of Cell Painting features
        predict_viability: Whether to predict viability
        predict_uncertainty: Whether to predict uncertainty
        **kwargs: Additional config parameters
        
    Returns:
        ProToPhenModel instance
    """
    config = ProToPhenConfig(
        protein_embedding_dim=protein_embedding_dim,
        cell_painting_dim=cell_painting_dim,
        predict_viability=predict_viability,
        predict_uncertainty=predict_uncertainty,
        **kwargs,
    )
    return ProToPhenModel(config)


def create_lightweight_model(
    protein_embedding_dim: int,
    cell_painting_dim: int = 1500,
) -> ProToPhenModel:
    """
    Create a lightweight model for quick experiments.
    
    Args:
        protein_embedding_dim: Dimension of protein embeddings
        cell_painting_dim: Number of Cell Painting features
        
    Returns:
        ProToPhenModel instance
    """
    config = ProToPhenConfig(
        protein_embedding_dim=protein_embedding_dim,
        encoder_hidden_dims=[512, 256],
        encoder_output_dim=128,
        decoder_hidden_dims=[256, 512],
        cell_painting_dim=cell_painting_dim,
        predict_viability=False,
        predict_transcriptomics=False,
    )
    return ProToPhenModel(config)