"""
Protein encoder modules for ProToPhen.

This module provides neural network components for encoding
protein embeddings into representations suitable for phenotype prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProteinEncoderConfig:
    """Configuration for protein encoder."""
    
    input_dim: int = 1280  # ESM-2 650M dimension
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    output_dim: int = 256
    
    # Architecture options
    activation: Literal["relu", "gelu", "silu", "tanh"] = "gelu"
    dropout: float = 0.1
    use_batch_norm: bool = False
    use_layer_norm: bool = True
    use_residual: bool = True
    
    # Regularisation
    input_dropout: float = 0.0
    weight_init: Literal["xavier", "kaiming", "normal"] = "kaiming"


# =============================================================================
# Building Blocks
# =============================================================================

class MLPBlock(nn.Module):
    """
    MLP block with optional normalisation, dropout, and residual connection.
    
    Architecture:
        Linear -> Norm -> Activation -> Dropout [-> Residual]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        use_residual: bool = False,
    ):
        """
        Initialise MLP block.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            activation: Activation function
            dropout: Dropout rate
            use_batch_norm: Use batch normalisation
            use_layer_norm: Use layer normalisation
            use_residual: Use residual connection (requires in_features == out_features)
        """
        super().__init__()
        
        self.use_residual = use_residual and (in_features == out_features)
        
        # Main linear layer
        self.linear = nn.Linear(in_features, out_features)
        
        # Normalisation
        if use_batch_norm:
            self.norm = nn.BatchNorm1d(out_features)
        elif use_layer_norm:
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = nn.Identity()
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual projection if dimensions don't match
        if use_residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = None
    
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
        
        return out


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block with pre-normalisation.
    
    Architecture:
        x -> LayerNorm -> Linear -> Activation -> Dropout -> Linear -> + x
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        """
        Initialise residual block.
        
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (default: 4 * dim)
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()
        
        hidden_dim = hidden_dim or 4 * dim
        
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = MLPBlock._get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x + residual


# =============================================================================
# Main Encoder
# =============================================================================

class ProteinEncoder(nn.Module):
    """
    Encoder for protein embeddings.
    
    Takes pre-computed protein embeddings (e.g., from ESM-2 + physicochemical
    features) and transforms them into a latent representation suitable for
    phenotype prediction.
    
    Attributes:
        config: Encoder configuration
        encoder: Sequential encoder network
        
    Example:
        >>> config = ProteinEncoderConfig(
        ...     input_dim=1280 + 439,  # ESM-2 + physicochemical
        ...     hidden_dims=[1024, 512],
        ...     output_dim=256,
        ... )
        >>> encoder = ProteinEncoder(config)
        >>> 
        >>> embeddings = torch.randn(32, 1719)  # Batch of fused embeddings
        >>> latent = encoder(embeddings)
        >>> print(latent.shape)  # (32, 256)
    """
    
    def __init__(self, config: Optional[ProteinEncoderConfig] = None, **kwargs):
        """
        Initialise protein encoder.
        
        Args:
            config: Encoder configuration
            **kwargs: Override config parameters
        """
        super().__init__()
        
        # Build config
        if config is None:
            config = ProteinEncoderConfig(**kwargs)
        else:
            # Update config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        
        # Input dropout
        self.input_dropout = nn.Dropout(config.input_dropout) if config.input_dropout > 0 else nn.Identity()
        
        # Build encoder layers
        layers = []
        dims = [config.input_dim] + config.hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(
                MLPBlock(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    activation=config.activation,
                    dropout=config.dropout,
                    use_batch_norm=config.use_batch_norm,
                    use_layer_norm=config.use_layer_norm,
                    use_residual=config.use_residual and dims[i] == dims[i + 1],
                )
            )
        
        self.encoder = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(dims[-1], config.output_dim)
        self.output_norm = nn.LayerNorm(config.output_dim)
        
        # Initialise weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialise network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "kaiming":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.normal_(module.weight, std=0.02)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode protein embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, input_dim)
            return_hidden: Whether to return intermediate hidden states
            
        Returns:
            Encoded representation of shape (batch_size, output_dim)
            Optionally, list of hidden states from each layer
        """
        x = self.input_dropout(x)
        
        if return_hidden:
            hidden_states = []
            for layer in self.encoder:
                x = layer(x)
                hidden_states.append(x)
            
            x = self.output_proj(x)
            x = self.output_norm(x)
            
            return x, hidden_states
        else:
            x = self.encoder(x)
            x = self.output_proj(x)
            x = self.output_norm(x)
            
            return x
    
    @property
    def output_dim(self) -> int:
        """Output dimension of encoder."""
        return self.config.output_dim
    
    def __repr__(self) -> str:
        return (
            f"ProteinEncoder("
            f"input_dim={self.config.input_dim}, "
            f"hidden_dims={self.config.hidden_dims}, "
            f"output_dim={self.config.output_dim})"
        )


# =============================================================================
# Specialised Encoders
# =============================================================================

class AttentionEncoder(nn.Module):
    """
    Encoder with self-attention over embedding components.
    
    Useful when protein embeddings have multiple components
    (e.g., per-residue embeddings or multiple embedding types).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialise attention encoder.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        self._output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, seq_len, input_dim) or (batch, input_dim)
            
        Returns:
            Output of shape (batch, output_dim)
        """
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_proj(x)
        x = self.transformer(x)
        
        # Pool over sequence dimension
        x = x.mean(dim=1)
        
        x = self.output_proj(x)
        x = self.output_norm(x)
        
        return x
    
    @property
    def output_dim(self) -> int:
        return self._output_dim