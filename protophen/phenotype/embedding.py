"""
Phenotype dimensionality reduction and embedding.

This module provides tools for reducing the dimensionality of
phenotypic feature spaces and creating embeddings for visualisation
and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from protophen.data.phenotype import PhenotypeDataset
from protophen.utils.logging import logger

# Try to import UMAP (optional dependency)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for phenotype embedding."""
    
    method: Literal["pca", "umap", "tsne"] = "pca"
    n_components: int = 50
    
    # PCA settings
    pca_whiten: bool = False
    
    # UMAP settings
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    
    # t-SNE settings
    tsne_perplexity: float = 30.0
    tsne_learning_rate: Union[float, str] = "auto"
    tsne_n_iter: int = 1000
    
    # Pre-processing
    standardise: bool = True
    
    # Reproducibility
    random_state: int = 42


# =============================================================================
# Phenotype Embedder
# =============================================================================

class PhenotypeEmbedder:
    """
    Reduce dimensionality of phenotypic features.
    
    This class provides methods for embedding high-dimensional Cell Painting
    features into lower-dimensional spaces for visualisation and analysis.
    
    Supports:
    - PCA: Linear dimensionality reduction
    - UMAP: Non-linear embedding (requires umap-learn)
    - t-SNE: Non-linear embedding for visualisation
    
    Attributes:
        config: Embedding configuration
        model: Fitted embedding model
        
    Example:
        >>> embedder = PhenotypeEmbedder(method="pca", n_components=50)
        >>> 
        >>> # Fit and transform
        >>> embedded = embedder.fit_transform(features)
        >>> 
        >>> # For visualisation, reduce further
        >>> vis_embedder = PhenotypeEmbedder(method="umap", n_components=2)
        >>> coords = vis_embedder.fit_transform(embedded)
    """
    
    def __init__(
        self,
        method: str = "pca",
        n_components: int = 50,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialise phenotype embedder.
        
        Args:
            method: Embedding method ("pca", "umap", "tsne")
            n_components: Number of output dimensions
            random_state: Random seed for reproducibility
            **kwargs: Additional method-specific parameters
        """
        self.config = EmbeddingConfig(
            method=method,
            n_components=n_components,
            random_state=random_state,
        )
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self._model = None
        self._scaler = None
        self.is_fitted = False
        
        # Validate method
        if method == "umap" and not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP not available. Install with: pip install umap-learn"
            )
        
        logger.info(f"Initialised PhenotypeEmbedder: method={method}, n_components={n_components}")
    
    def _create_model(self):
        """Create the embedding model based on configuration."""
        config = self.config
        
        if config.method == "pca":
            return PCA(
                n_components=config.n_components,
                whiten=config.pca_whiten,
                random_state=config.random_state,
            )
        
        elif config.method == "umap":
            return UMAP(
                n_components=config.n_components,
                n_neighbors=config.umap_n_neighbors,
                min_dist=config.umap_min_dist,
                metric=config.umap_metric,
                random_state=config.random_state,
            )
        
        elif config.method == "tsne":
            return TSNE(
                n_components=config.n_components,
                perplexity=config.tsne_perplexity,
                learning_rate=config.tsne_learning_rate,
                n_iter=config.tsne_n_iter,
                random_state=config.random_state,
            )
        
        else:
            raise ValueError(f"Unknown embedding method: {config.method}")
    
    def fit(self, X: np.ndarray) -> "PhenotypeEmbedder":
        """
        Fit the embedding model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            self
        """
        # Handle NaNs
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardise if requested
        if self.config.standardise:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        
        # Create and fit model
        self._model = self._create_model()
        
        if self.config.method == "tsne":
            # t-SNE doesn't have separate fit/transform
            logger.warning("t-SNE doesn't support separate fit/transform. Use fit_transform instead.")
        else:
            self._model.fit(X)
        
        self.is_fitted = True
        logger.info(f"Fitted {self.config.method.upper()} embedder")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Embedded features (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("Embedder must be fitted before transform")
        
        if self.config.method == "tsne":
            raise RuntimeError("t-SNE doesn't support transform on new data. Use fit_transform.")
        
        # Handle NaNs
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardise if fitted with standardisation
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        return self._model.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Embedded features (n_samples, n_components)
        """
        # Handle NaNs
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardise if requested
        if self.config.standardise:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        
        # Create and fit model
        self._model = self._create_model()
        
        embedded = self._model.fit_transform(X)
        
        self.is_fitted = True
        logger.info(f"Fitted and transformed with {self.config.method.upper()}")
        
        return embedded
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform (PCA only).
        
        Args:
            X: Embedded features (n_samples, n_components)
            
        Returns:
            Reconstructed features (n_samples, n_features)
        """
        if self.config.method != "pca":
            raise RuntimeError(f"Inverse transform not supported for {self.config.method}")
        
        if not self.is_fitted:
            raise RuntimeError("Embedder must be fitted before inverse_transform")
        
        reconstructed = self._model.inverse_transform(X)
        
        # Inverse standardisation
        if self._scaler is not None:
            reconstructed = self._scaler.inverse_transform(reconstructed)
        
        return reconstructed
    
    @property
    def explained_variance_ratio_(self) -> Optional[np.ndarray]:
        """Get explained variance ratio (PCA only)."""
        if self.config.method == "pca" and self.is_fitted:
            return self._model.explained_variance_ratio_
        return None
    
    @property
    def total_explained_variance_(self) -> Optional[float]:
        """Get total explained variance (PCA only)."""
        if self.explained_variance_ratio_ is not None:
            return float(np.sum(self.explained_variance_ratio_))
        return None
    
    def get_loadings(self) -> Optional[np.ndarray]:
        """
        Get PCA loadings (components).
        
        Returns:
            Loadings matrix (n_components, n_features) or None
        """
        if self.config.method == "pca" and self.is_fitted:
            return self._model.components_
        return None
    
    def __repr__(self) -> str:
        return f"PhenotypeEmbedder(method={self.config.method}, n_components={self.config.n_components})"


# =============================================================================
# Convenience Functions
# =============================================================================

def reduce_dimensions(
    features: np.ndarray,
    n_components: int = 50,
    method: str = "pca",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function to reduce feature dimensionality.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_components: Number of output dimensions
        method: Embedding method
        **kwargs: Additional parameters
        
    Returns:
        Reduced features (n_samples, n_components)
    """
    embedder = PhenotypeEmbedder(method=method, n_components=n_components, **kwargs)
    return embedder.fit_transform(features)


def get_pca_embedding(
    features: np.ndarray,
    n_components: int = 50,
    return_model: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, PhenotypeEmbedder]]:
    """
    Get PCA embedding of features.
    
    Args:
        features: Feature matrix
        n_components: Number of components
        return_model: Whether to return the fitted embedder
        
    Returns:
        Embedded features, and optionally the fitted embedder
    """
    embedder = PhenotypeEmbedder(method="pca", n_components=n_components)
    embedded = embedder.fit_transform(features)
    
    if return_model:
        return embedded, embedder
    return embedded


def get_umap_embedding(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """
    Get UMAP embedding of features (for visualisation).
    
    Args:
        features: Feature matrix
        n_components: Number of components (typically 2 for visualisation)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter
        **kwargs: Additional UMAP parameters
        
    Returns:
        Embedded features
    """
    embedder = PhenotypeEmbedder(
        method="umap",
        n_components=n_components,
        umap_n_neighbors=n_neighbors,
        umap_min_dist=min_dist,
        **kwargs,
    )
    return embedder.fit_transform(features)