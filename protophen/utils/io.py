"""
File I/O utilities for ProToPhen.

This module provides utilities for saving and loading various data formats,
including embeddings, configurations, and cached computations.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from protophen.utils.logging import logger


# =============================================================================
# Path Utilities
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_path(
    cache_dir: Union[str, Path],
    key: str,
    suffix: str = ".pkl",
) -> Path:
    """
    Generate a cache file path from a key.
    
    Args:
        cache_dir: Cache directory
        key: Unique key for the cached item
        suffix: File suffix
        
    Returns:
        Path to cache file
    """
    cache_dir = ensure_dir(cache_dir)
    # Hash the key if it's too long for a filename
    if len(key) > 200:
        key = hashlib.md5(key.encode()).hexdigest()
    # Sanitise key for filesystem
    safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
    return cache_dir / f"{safe_key}{suffix}"


# =============================================================================
# Embedding I/O
# =============================================================================

def save_embeddings(
    embeddings: dict[str, np.ndarray],
    path: Union[str, Path],
    compress: bool = True,
) -> None:
    """
    Save embeddings dictionary to file.
    
    Args:
        embeddings: Dictionary mapping IDs to embedding arrays
        path: Output path
        compress: Whether to use gzip compression
        
    Example:
        >>> embeddings = {"protein_1": np.random.randn(1280)}
        >>> save_embeddings(embeddings, "embeddings.pkl.gz")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress or path.suffix == ".gz":
        if not path.suffix == ".gz":
            path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(path, "wb") as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, "wb") as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.debug(f"Saved {len(embeddings)} embeddings to {path}")


def load_embeddings(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """
    Load embeddings dictionary from file.
    
    Args:
        path: Input path
        
    Returns:
        Dictionary mapping IDs to embedding arrays
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    
    if path.suffix == ".gz" or str(path).endswith(".pkl.gz"):
        with gzip.open(path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        with open(path, "rb") as f:
            embeddings = pickle.load(f)
    
    logger.debug(f"Loaded {len(embeddings)} embeddings from {path}")
    return embeddings


def save_numpy(
    array: np.ndarray,
    path: Union[str, Path],
    compress: bool = True,
) -> None:
    """
    Save numpy array to file.
    
    Args:
        array: Numpy array
        path: Output path
        compress: Whether to use compression
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        np.savez_compressed(path, data=array)
    else:
        np.save(path, array)
    
    logger.debug(f"Saved array with shape {array.shape} to {path}")


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """
    Load numpy array from file.
    
    Args:
        path: Input path
        
    Returns:
        Numpy array
    """
    path = Path(path)
    
    if path.suffix == ".npz":
        with np.load(path) as data:
            return data["data"]
    else:
        return np.load(path)


# =============================================================================
# Caching Decorator
# =============================================================================

class EmbeddingCache:
    """
    Cache for protein embeddings.
    
    This class manages a disk-based cache for protein embeddings,
    allowing efficient reuse of computed embeddings across sessions.
    
    Attributes:
        cache_dir: Directory for cache files
        
    Example:
        >>> cache = EmbeddingCache("./cache/esm2")
        >>> 
        >>> # Check if embedding is cached
        >>> if cache.has("protein_abc123"):
        ...     embedding = cache.get("protein_abc123")
        ... else:
        ...     embedding = compute_embedding(sequence)
        ...     cache.set("protein_abc123", embedding)
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialise embedding cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = ensure_dir(cache_dir)
        self._memory_cache: dict[str, np.ndarray] = {}
        self._index_path = self.cache_dir / "index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> dict[str, str]:
        """Load the cache index."""
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save the cache index."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)
    
    def _get_cache_file(self, key: str) -> Path:
        """Get the cache file path for a key."""
        # Use hash for filename to handle long keys
        file_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{file_hash}.npy"
    
    def has(self, key: str) -> bool:
        """
        Check if a key is in the cache.
        
        Args:
            key: Cache key (typically protein hash)
            
        Returns:
            True if key is cached
        """
        return key in self._index or key in self._memory_cache
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Embedding array or None if not found
        """
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        if key in self._index:
            cache_file = Path(self._index[key])
            if cache_file.exists():
                embedding = np.load(cache_file)
                # Populate memory cache
                self._memory_cache[key] = embedding
                return embedding
            else:
                # File missing, remove from index
                del self._index[key]
                self._save_index()
        
        return None
    
    def set(self, key: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            key: Cache key
            embedding: Embedding array
        """
        cache_file = self._get_cache_file(key)
        np.save(cache_file, embedding)
        
        self._index[key] = str(cache_file)
        self._memory_cache[key] = embedding
        self._save_index()
    
    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        """
        Get multiple embeddings from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of found embeddings
        """
        return {key: emb for key in keys if (emb := self.get(key)) is not None}
    
    def set_many(self, embeddings: dict[str, np.ndarray]) -> None:
        """
        Store multiple embeddings in cache.
        
        Args:
            embeddings: Dictionary of embeddings
        """
        for key, embedding in embeddings.items():
            self.set(key, embedding)
    
    def clear(self) -> None:
        """Clear the entire cache."""
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
        self._index = {}
        self._memory_cache = {}
        self._save_index()
        logger.info(f"Cleared cache at {self.cache_dir}")
    
    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._index)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EmbeddingCache(dir='{self.cache_dir}', size={len(self)})"


# =============================================================================
# Update utils __init__.py exports
# =============================================================================

__all__ = [
    "ensure_dir",
    "get_cache_path",
    "save_embeddings",
    "load_embeddings",
    "save_numpy",
    "load_numpy",
    "EmbeddingCache",
]