"""
Embedding fusion module for combining multiple protein representations.

This module provides methods for combining different embedding types
(ESM-2, physicochemical features, structural features) into unified
protein representations.

Fusion strategies:
- Concatenation: Simple concatenation with optional projection
- Weighted: Learned or fixed weights for each embedding type
- Attention: Cross-attention between embedding types
- Gated: Gated fusion with learned gates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from protophen.data.protein import Protein, ProteinLibrary
from protophen.utils.logging import logger


# =============================================================================
# Configuration
# =============================================================================

class FusionMethod(str, Enum):
    """Available fusion methods."""
    CONCATENATE = "concatenate"
    WEIGHTED = "weighted"
    ATTENTION = "attention"
    GATED = "gated"


@dataclass
class FusionConfig:
    """Configuration for embedding fusion."""
    
    method: FusionMethod = FusionMethod.CONCATENATE
    
    # Input embedding dimensions (set automatically or manually)
    embedding_dims: Dict[str, int] = field(default_factory=dict)
    
    # Output dimension (None = sum of inputs for concat, or specified for projection)
    output_dim: Optional[int] = None
    
    # Projection settings
    use_projection: bool = False
    projection_hidden_dim: Optional[int] = None
    
    # Normalisation
    normalise_inputs: bool = True
    normalise_output: bool = False
    
    # For weighted fusion
    learnable_weights: bool = False
    initial_weights: Optional[Dict[str, float]] = None
    
    # For attention fusion
    attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # For gated fusion
    gate_activation: str = "sigmoid"


# =============================================================================
# Fusion Modules (PyTorch)
# =============================================================================

class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion with optional projection.
    
    Performs simple concatenation of embeddings along the feature dimension,
    optionally followed by a learned projection layer.
    """
    
    def __init__(
        self,
        embedding_dims: Dict[str, int],
        output_dim: Optional[int] = None,
        use_projection: bool = False,
        projection_hidden_dim: Optional[int] = None,
        normalise_inputs: bool = True,
    ):
        super().__init__()
        
        self.embedding_names = sorted(embedding_dims.keys())
        self.embedding_dims = embedding_dims
        self.normalise_inputs = normalise_inputs
        
        total_dim = sum(embedding_dims.values())
        
        if use_projection and output_dim is not None:
            hidden_dim = projection_hidden_dim or (total_dim + output_dim) // 2
            self.projection = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )
            self.output_dim = output_dim
        else:
            self.projection = None
            self.output_dim = total_dim
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse embeddings via concatenation.
        
        Args:
            embeddings: Dictionary mapping embedding names to tensors
            
        Returns:
            Fused embedding tensor
        """
        # Collect embeddings in consistent order
        tensors = []
        for name in self.embedding_names:
            if name not in embeddings:
                raise ValueError(f"Missing embedding: {name}")
            
            x = embeddings[name]
            
            if self.normalise_inputs:
                x = torch.nn.functional.normalize(x, p=2, dim=-1)
            
            tensors.append(x)
        
        # Concatenate
        fused = torch.cat(tensors, dim=-1)
        
        # Optional projection
        if self.projection is not None:
            fused = self.projection(fused)
        
        return fused


class WeightedFusion(nn.Module):
    """
    Weighted sum fusion.
    
    Projects each embedding to common dimension, then combines
    with learned or fixed weights.
    """
    
    def __init__(
        self,
        embedding_dims: Dict[str, int],
        output_dim: int,
        learnable_weights: bool = True,
        initial_weights: Optional[Dict[str, float]] = None,
        normalise_inputs: bool = True,
    ):
        super().__init__()
        
        self.embedding_names = sorted(embedding_dims.keys())
        self.output_dim = output_dim
        self.normalise_inputs = normalise_inputs
        
        # Projection layers for each embedding type
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in embedding_dims.items()
        })
        
        # Fusion weights
        n_embeddings = len(embedding_dims)
        if initial_weights is not None:
            weight_values = [initial_weights.get(name, 1.0 / n_embeddings) 
                          for name in self.embedding_names]
        else:
            weight_values = [1.0 / n_embeddings] * n_embeddings
        
        weights = torch.tensor(weight_values, dtype=torch.float32)
        
        if learnable_weights:
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer('weights', weights)
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings via weighted sum."""
        projected = []
        
        for name in self.embedding_names:
            if name not in embeddings:
                raise ValueError(f"Missing embedding: {name}")
            
            x = embeddings[name]
            
            if self.normalise_inputs:
                x = torch.nn.functional.normalize(x, p=2, dim=-1)
            
            proj = self.projections[name](x)
            projected.append(proj)
        
        # Stack and weight
        stacked = torch.stack(projected, dim=-1)  # (batch, output_dim, n_embeddings)
        
        # Softmax weights for proper weighting
        weights = torch.softmax(self.weights, dim=0)
        
        # Weighted sum
        fused = (stacked * weights).sum(dim=-1)
        
        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based fusion using cross-attention.
    
    Uses multi-head attention to learn relationships between
    different embedding types.
    """
    
    def __init__(
        self,
        embedding_dims: Dict[str, int],
        output_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        normalise_inputs: bool = True,
    ):
        super().__init__()
        
        self.embedding_names = sorted(embedding_dims.keys())
        self.output_dim = output_dim
        self.normalise_inputs = normalise_inputs
        
        # Project each embedding to common dimension
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in embedding_dims.items()
        })
        
        # Self-attention over embedding types
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings via attention."""
        projected = []
        
        for name in self.embedding_names:
            if name not in embeddings:
                raise ValueError(f"Missing embedding: {name}")
            
            x = embeddings[name]
            
            if self.normalise_inputs:
                x = torch.nn.functional.normalize(x, p=2, dim=-1)
            
            proj = self.projections[name](x)
            projected.append(proj)
        
        # Stack as sequence: (batch, n_embedding_types, output_dim)
        stacked = torch.stack(projected, dim=1)
        
        # Self-attention
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Pool over embedding types (mean)
        pooled = attended.mean(dim=1)
        
        # Output projection
        fused = self.output_proj(pooled)
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion with learned gates for each embedding type.
    
    Each embedding type has a learned gate that controls its
    contribution to the final representation.
    """
    
    def __init__(
        self,
        embedding_dims: Dict[str, int],
        output_dim: int,
        gate_activation: str = "sigmoid",
        normalise_inputs: bool = True,
    ):
        super().__init__()
        
        self.embedding_names = sorted(embedding_dims.keys())
        self.output_dim = output_dim
        self.normalise_inputs = normalise_inputs
        
        # Projection and gate for each embedding type
        self.projections = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        
        total_dim = sum(embedding_dims.values())
        
        for name, dim in embedding_dims.items():
            self.projections[name] = nn.Linear(dim, output_dim)
            # Gate takes concatenated input to make decision
            self.gates[name] = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.Sigmoid() if gate_activation == "sigmoid" else nn.Tanh(),
            )
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings via gating."""
        # Concatenate all inputs for gate computation
        all_inputs = []
        for name in self.embedding_names:
            if name not in embeddings:
                raise ValueError(f"Missing embedding: {name}")
            all_inputs.append(embeddings[name])
        
        concat_input = torch.cat(all_inputs, dim=-1)
        
        # Compute gated projections
        gated_outputs = []
        for name in self.embedding_names:
            x = embeddings[name]
            
            if self.normalise_inputs:
                x = torch.nn.functional.normalize(x, p=2, dim=-1)
            
            proj = self.projections[name](x)
            gate = self.gates[name](concat_input)
            
            gated = proj * gate
            gated_outputs.append(gated)
        
        # Sum gated outputs
        fused = sum(gated_outputs)
        fused = self.layer_norm(fused)
        
        return fused


# =============================================================================
# Main Fusion Class (NumPy-based for inference)
# =============================================================================

class EmbeddingFusion:
    """
    Combine multiple protein embeddings into unified representations.
    
    This class provides both simple NumPy-based fusion for inference
    and PyTorch modules for end-to-end training.
    
    Attributes:
        config: Fusion configuration
        embedding_names: List of expected embedding types
        
    Example:
        >>> fusion = EmbeddingFusion(method="concatenate")
        >>> 
        >>> # Fuse embeddings for a protein
        >>> protein.embeddings["esm2"] = esm_embedding
        >>> protein.embeddings["physicochemical"] = physchem_embedding
        >>> fused = fusion.fuse_protein(protein)
        >>> 
        >>> # Or fuse a library
        >>> fused_matrix = fusion.fuse_library(library)
    """
    
    def __init__(
        self,
        method: Union[str, FusionMethod] = "concatenate",
        embedding_names: Optional[List[str]] = None,
        output_dim: Optional[int] = None,
        normalise_inputs: bool = True,
        normalise_output: bool = False,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialise embedding fusion.
        
        Args:
            method: Fusion method (concatenate, weighted, attention, gated)
            embedding_names: Expected embedding types (auto-detected if None)
            output_dim: Output dimension (required for weighted fusion)
            normalise_inputs: Whether to L2-normalise input embeddings
            normalise_output: Whether to L2-normalise output embedding
            weights: Weights for weighted fusion
        """
        if isinstance(method, str):
            method = FusionMethod(method)
        
        self.config = FusionConfig(
            method=method,
            output_dim=output_dim,
            normalise_inputs=normalise_inputs,
            normalise_output=normalise_output,
            initial_weights=weights,
        )
        
        self.embedding_names = embedding_names
        self._torch_module: Optional[nn.Module] = None
        
        logger.info(f"Initialised EmbeddingFusion: method={method.value}")
    
    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """L2-normalise along last axis."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + 1e-8)
    
    def fuse(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse multiple embeddings into single representation.
        
        Args:
            embeddings: Dictionary mapping embedding names to arrays
            
        Returns:
            Fused embedding array
        """
        if self.embedding_names is None:
            self.embedding_names = sorted(embeddings.keys())
        
        # Validate embeddings
        for name in self.embedding_names:
            if name not in embeddings:
                raise ValueError(f"Missing embedding: {name}")
        
        # Collect and optionally normalise
        arrays = []
        for name in self.embedding_names:
            arr = embeddings[name]
            if self.config.normalise_inputs:
                arr = self._normalise(arr)
            arrays.append(arr)
        
        # Apply fusion method
        if self.config.method == FusionMethod.CONCATENATE:
            fused = np.concatenate(arrays, axis=-1)
        
        elif self.config.method == FusionMethod.WEIGHTED:
            if self.config.output_dim is None:
                raise ValueError("output_dim required for weighted fusion")
            
            # Simple weighted average (for inference without learned projections)
            # In practice, we'd use the PyTorch module for learned weighted fusion -- consider deprecating/removing during post-dev refactoring
            weights = self.config.initial_weights or {}
            n_emb = len(arrays)
            
            # Pad/truncate to common dimension
            max_dim = max(arr.shape[-1] for arr in arrays)
            padded = []
            for i, (name, arr) in enumerate(zip(self.embedding_names, arrays)):
                if arr.shape[-1] < max_dim:
                    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, max_dim - arr.shape[-1])]
                    arr = np.pad(arr, pad_width, mode='constant')
                elif arr.shape[-1] > max_dim:
                    arr = arr[..., :max_dim]
                
                w = weights.get(name, 1.0 / n_emb)
                padded.append(arr * w)
            
            fused = sum(padded)
        
        else:
            # For attention and gated, fall back to concatenation for NumPy
            # Use get_torch_module() for learned fusion
            logger.warning(
                f"Method '{self.config.method.value}' requires PyTorch for full functionality. "
                f"Falling back to concatenation."
            )
            fused = np.concatenate(arrays, axis=-1)
        
        # Optional output normalisation
        if self.config.normalise_output:
            fused = self._normalise(fused)
        
        return fused
    
    def fuse_protein(
        self,
        protein: Protein,
        embedding_key: str = "fused",
    ) -> np.ndarray:
        """
        Fuse embeddings for a single protein.
        
        Args:
            protein: Protein with computed embeddings
            embedding_key: Key to store fused embedding under
            
        Returns:
            Fused embedding array
        """
        # Auto-detect embedding names from protein if not set
        if self.embedding_names is None:
            self.embedding_names = sorted(protein.embedding_types)
        
        embeddings = {name: protein.get_embedding(name) for name in self.embedding_names}
        fused = self.fuse(embeddings)
        
        protein.set_embedding(embedding_key, fused)
        
        return fused
    
    def fuse_library(
        self,
        library: ProteinLibrary,
        embedding_key: str = "fused",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Fuse embeddings for all proteins in a library.
        
        Args:
            library: ProteinLibrary with computed embeddings
            embedding_key: Key under which fused embeddings are to be stored
            show_progress: Whether to show progress bar
            
        Returns:
            Stacked fused embeddings array
        """
        from tqdm.auto import tqdm
        
        # Auto-detect embedding names from first protein
        if self.embedding_names is None and len(library) > 0:
            self.embedding_names = sorted(library[0].embedding_types)
        
        fused_list = []
        iterator = tqdm(library, desc="Fusing embeddings", disable=not show_progress)
        
        for protein in iterator:
            fused = self.fuse_protein(protein, embedding_key=embedding_key)
            fused_list.append(fused)
        
        return np.stack(fused_list)
    
    def get_torch_module(
        self,
        embedding_dims: Dict[str, int],
    ) -> nn.Module:
        """
        Get PyTorch module for end-to-end training.
        
        Args:
            embedding_dims: Dictionary mapping embedding names to dimensions
            
        Returns:
            PyTorch fusion module
        """
        if self.embedding_names is None:
            self.embedding_names = sorted(embedding_dims.keys())
        
        self.config.embedding_dims = embedding_dims
        
        if self.config.method == FusionMethod.CONCATENATE:
            module = ConcatFusion(
                embedding_dims=embedding_dims,
                output_dim=self.config.output_dim,
                use_projection=self.config.use_projection,
                normalise_inputs=self.config.normalise_inputs,
            )
        
        elif self.config.method == FusionMethod.WEIGHTED:
            if self.config.output_dim is None:
                raise ValueError("output_dim required for weighted fusion")
            module = WeightedFusion(
                embedding_dims=embedding_dims,
                output_dim=self.config.output_dim,
                learnable_weights=self.config.learnable_weights,
                initial_weights=self.config.initial_weights,
                normalise_inputs=self.config.normalise_inputs,
            )
        
        elif self.config.method == FusionMethod.ATTENTION:
            if self.config.output_dim is None:
                raise ValueError("output_dim required for attention fusion")
            module = AttentionFusion(
                embedding_dims=embedding_dims,
                output_dim=self.config.output_dim,
                n_heads=self.config.attention_heads,
                dropout=self.config.attention_dropout,
                normalise_inputs=self.config.normalise_inputs,
            )
        
        elif self.config.method == FusionMethod.GATED:
            if self.config.output_dim is None:
                raise ValueError("output_dim required for gated fusion")
            module = GatedFusion(
                embedding_dims=embedding_dims,
                output_dim=self.config.output_dim,
                gate_activation=self.config.gate_activation,
                normalise_inputs=self.config.normalise_inputs,
            )
        
        else:
            raise ValueError(f"Unknown fusion method: {self.config.method}")
        
        self._torch_module = module
        return module
    
    @property
    def output_dim(self) -> Optional[int]:
        """Get output dimension (if known)."""
        if self._torch_module is not None:
            return self._torch_module.output_dim
        return self.config.output_dim
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EmbeddingFusion(method={self.config.method.value}, embeddings={self.embedding_names})"


# =============================================================================
# Convenience Functions
# =============================================================================

def fuse_embeddings(
    embeddings: Dict[str, np.ndarray],
    method: str = "concatenate",
    normalise: bool = True,
) -> np.ndarray:
    """
    Convenience function to fuse embeddings.
    
    Args:
        embeddings: Dictionary of embeddings
        method: Fusion method
        normalise: Whether to normalise inputs
        
    Returns:
        Fused embedding array
    """
    fusion = EmbeddingFusion(method=method, normalise_inputs=normalise)
    return fusion.fuse(embeddings)


def get_fusion_module(
    embedding_dims: Dict[str, int],
    method: str = "concatenate",
    output_dim: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Convenience function to get a PyTorch fusion module.
    
    Args:
        embedding_dims: Dictionary mapping embedding names to dimensions
        method: Fusion method
        output_dim: Output dimension
        **kwargs: Additional config options
        
    Returns:
        PyTorch fusion module
    """
    fusion = EmbeddingFusion(method=method, output_dim=output_dim, **kwargs)
    return fusion.get_torch_module(embedding_dims)