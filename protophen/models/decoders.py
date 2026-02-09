"""
Phenotype decoder modules for ProToPhen.

This module provides neural network components for decoding
latent representations into phenotypic predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from protophen.models.encoders import MLPBlock


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DecoderConfig:
    """Configuration for phenotype decoders."""
    
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 1024])
    
    # Architecture
    activation: str = "gelu"
    dropout: float = 0.1
    use_layer_norm: bool = True
    
    # Output settings
    output_activation: Optional[str] = None  # None for linear output


# =============================================================================
# Base Decoder
# =============================================================================

class PhenotypeDecoder(nn.Module):
    """
    Base decoder for phenotype prediction.
    
    Transforms latent protein representations into phenotypic predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        output_activation: Optional[str] = None,
    ):
        """
        Initialise phenotype decoder.
        
        Args:
            input_dim: Input dimension (encoder output)
            output_dim: Output dimension (number of phenotype features)
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
            output_activation: Optional activation for output
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [512]
        
        # Build decoder layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(
                MLPBlock(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    activation=activation,
                    dropout=dropout,
                    use_layer_norm=True,
                )
            )
        
        self.decoder = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
        # Output activation
        if output_activation is not None:
            self.output_activation = MLPBlock._get_activation(output_activation)
        else:
            self.output_activation = nn.Identity()
        
        self._input_dim = input_dim
        self._output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to phenotype predictions.
        
        Args:
            x: Latent representation of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        x = self.decoder(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x
    
    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


# =============================================================================
# Task-Specific Heads
# =============================================================================

class CellPaintingHead(nn.Module):
    """
    Prediction head for Cell Painting features.
    
    Predicts ~1500 morphological features with optional feature grouping.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1500,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        predict_uncertainty: bool = False,
    ):
        """
        Initialise Cell Painting head.
        
        Args:
            input_dim: Input dimension
            output_dim: Number of Cell Painting features
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            predict_uncertainty: Whether to predict uncertainty (aleatoric)
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [1024, 2048]
        self.predict_uncertainty = predict_uncertainty
        
        # Shared layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        self.shared = nn.Sequential(*layers)
        
        # Mean prediction head
        self.mean_head = nn.Linear(dims[-1], output_dim)
        
        # Uncertainty prediction head (log variance)
        if predict_uncertainty:
            self.log_var_head = nn.Linear(dims[-1], output_dim)
        
        self._output_dim = output_dim
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch_size, input_dim)
            
        Returns:
            Mean predictions of shape (batch_size, output_dim)
            If predict_uncertainty: tuple of (mean, log_variance)
        """
        h = self.shared(x)
        mean = self.mean_head(h)
        
        if self.predict_uncertainty:
            log_var = self.log_var_head(h)
            # Clamp log variance for numerical stability
            log_var = torch.clamp(log_var, min=-10, max=10)
            return mean, log_var
        
        return mean
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class ViabilityHead(nn.Module):
    """
    Prediction head for cell viability.
    
    Predicts a single viability score (0-1).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        output_type: Literal["sigmoid", "linear", "beta"] = "sigmoid",
    ):
        """
        Initialise viability head.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            output_type: Output activation type
                - "sigmoid": Standard sigmoid for 0-1 prediction
                - "linear": Linear output (for pre-normalised targets)
                - "beta": Beta distribution parameters (for uncertainty)
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [128, 64]
        self.output_type = output_type
        
        # Build MLP
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer(s)
        if output_type == "beta":
            # Predict alpha and beta parameters of Beta distribution
            self.alpha_head = nn.Linear(dims[-1], 1)
            self.beta_head = nn.Linear(dims[-1], 1)
        else:
            self.output_layer = nn.Linear(dims[-1], 1)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch_size, input_dim)
            
        Returns:
            Viability prediction of shape (batch_size, 1)
            If output_type=="beta": tuple of (alpha, beta) parameters
        """
        h = self.mlp(x)
        
        if self.output_type == "beta":
            # Use softplus to ensure positive parameters
            alpha = F.softplus(self.alpha_head(h)) + 1.0
            beta = F.softplus(self.beta_head(h)) + 1.0
            return alpha, beta
        
        out = self.output_layer(h)
        
        if self.output_type == "sigmoid":
            out = torch.sigmoid(out)
        
        return out
    
    @property
    def output_dim(self) -> int:
        return 1


class TranscriptomicsHead(nn.Module):
    """
    Prediction head for transcriptomics data.
    
    Predicts gene expression values, optionally in a compressed space.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_bottleneck: bool = True,
        bottleneck_dim: int = 128,
    ):
        """
        Initialise transcriptomics head.
        
        Args:
            input_dim: Input dimension
            output_dim: Number of genes/features to predict
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_bottleneck: Use a bottleneck layer before expansion
            bottleneck_dim: Dimension of bottleneck
        """
        super().__init__()
        
        self.use_bottleneck = use_bottleneck
        hidden_dims = hidden_dims or [512]
        
        # Encoder to bottleneck
        if use_bottleneck:
            enc_layers = []
            dims = [input_dim] + hidden_dims + [bottleneck_dim]
            
            for i in range(len(dims) - 1):
                enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
                enc_layers.append(nn.LayerNorm(dims[i + 1]))
                enc_layers.append(nn.GELU())
                enc_layers.append(nn.Dropout(dropout))
            
            self.encoder = nn.Sequential(*enc_layers)
            
            # Decoder from bottleneck
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1], output_dim),
            )
        else:
            # Direct prediction
            layers = []
            dims = [input_dim] + hidden_dims
            
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(dims[-1], output_dim))
            self.predictor = nn.Sequential(*layers)
        
        self._output_dim = output_dim
        self._bottleneck_dim = bottleneck_dim if use_bottleneck else None
    
    def forward(
        self,
        x: torch.Tensor,
        return_bottleneck: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch_size, input_dim)
            return_bottleneck: Whether to return bottleneck representation
            
        Returns:
            Predictions of shape (batch_size, output_dim)
            If return_bottleneck: tuple of (predictions, bottleneck)
        """
        if self.use_bottleneck:
            bottleneck = self.encoder(x)
            out = self.decoder(bottleneck)
            
            if return_bottleneck:
                return out, bottleneck
            return out
        else:
            return self.predictor(x)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def bottleneck_dim(self) -> Optional[int]:
        return self._bottleneck_dim


# =============================================================================
# Multi-Task Head
# =============================================================================

class MultiTaskHead(nn.Module):
    """
    Combined multi-task prediction head.
    
    Manages multiple task-specific heads and handles missing tasks gracefully.
    
    Example:
        >>> head = MultiTaskHead(
        ...     input_dim=256,
        ...     task_configs={
        ...         "cell_painting": {"output_dim": 1500},
        ...         "viability": {"output_type": "sigmoid"},
        ...     }
        ... )
        >>> predictions = head(latent, tasks=["cell_painting", "viability"])
    """
    
    def __init__(
        self,
        input_dim: int,
        task_configs: Optional[Dict[str, Dict]] = None,
        shared_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        """
        Initialise multi-task head.
        
        Args:
            input_dim: Input dimension
            task_configs: Configuration for each task head
            shared_hidden_dims: Optional shared layers before task heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Default task configurations
        default_configs = {
            "cell_painting": {
                "type": "cell_painting",
                "output_dim": 1500,
                "hidden_dims": [1024],
                "predict_uncertainty": False,
            },
            "viability": {
                "type": "viability",
                "hidden_dims": [128, 64],
                "output_type": "sigmoid",
            },
            "transcriptomics": {
                "type": "transcriptomics",
                "output_dim": 978,  # L1000 genes
                "hidden_dims": [512],
                "use_bottleneck": True,
            },
        }
        
        task_configs = task_configs or default_configs
        
        # Optional shared layers
        if shared_hidden_dims:
            layers = []
            dims = [input_dim] + shared_hidden_dims
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            self.shared = nn.Sequential(*layers)
            head_input_dim = dims[-1]
        else:
            self.shared = nn.Identity()
            head_input_dim = input_dim
        
        # Create task-specific heads
        self.heads = nn.ModuleDict()
        self.task_configs = task_configs
        
        for task_name, config in task_configs.items():
            task_type = config.pop("type", task_name)
            
            if task_type == "cell_painting":
                self.heads[task_name] = CellPaintingHead(
                    input_dim=head_input_dim,
                    **config,
                )
            elif task_type == "viability":
                self.heads[task_name] = ViabilityHead(
                    input_dim=head_input_dim,
                    **config,
                )
            elif task_type == "transcriptomics":
                self.heads[task_name] = TranscriptomicsHead(
                    input_dim=head_input_dim,
                    **config,
                )
            else:
                # Generic decoder
                self.heads[task_name] = PhenotypeDecoder(
                    input_dim=head_input_dim,
                    **config,
                )
    
    def forward(
        self,
        x: torch.Tensor,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multiple tasks.
        
        Args:
            x: Input of shape (batch_size, input_dim)
            tasks: List of tasks to predict (None = all)
            
        Returns:
            Dictionary mapping task names to predictions
        """
        # Shared processing
        h = self.shared(x)
        
        # Determine which tasks to compute
        if tasks is None:
            tasks = list(self.heads.keys())
        
        # Compute predictions for each task
        outputs = {}
        for task_name in tasks:
            if task_name in self.heads:
                outputs[task_name] = self.heads[task_name](h)
        
        return outputs
    
    def get_task_output_dim(self, task_name: str) -> int:
        """Get output dimension for a specific task."""
        if task_name in self.heads:
            return self.heads[task_name].output_dim
        raise ValueError(f"Unknown task: {task_name}")
    
    @property
    def task_names(self) -> List[str]:
        """List of available tasks."""
        return list(self.heads.keys())
    
    def __repr__(self) -> str:
        tasks_str = ", ".join(f"{k}:{v.output_dim}" for k, v in self.heads.items())
        return f"MultiTaskHead(tasks=[{tasks_str}])"