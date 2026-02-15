"""
Local caching system for JUMP-CP data.

This module provides a Parquet-based cache for JUMP-CP metadata and
morphological profiles, enabling efficient local storage, lazy loading,
and incremental updates without re-downloading from S3.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from protophen.utils.io import ensure_dir
from protophen.utils.logging import logger


# =============================================================================
# Cache Entry Metadata
# =============================================================================

@dataclass
class CacheEntry:
    """Metadata for a cached item."""

    key: str
    path: str
    size_bytes: int
    n_rows: int
    n_cols: int
    created_at: float
    accessed_at: float
    source: str = ""  # e.g., "s3://..." or "https://..."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(**data)


# =============================================================================
# JUMP-CP Local Cache
# =============================================================================

class JUMPCPCache:
    """
    Parquet-based local cache for JUMP-CP data.

    Manages two namespaces:
    - **metadata**: Small DataFrames (well maps, gene annotations).
    - **profiles**: Larger per-plate morphological profiles.

    Attributes:
        cache_dir: Root directory for all cached files.
        max_size_gb: Soft size limit; exceeded entries are evicted LRU.

    Example:
        >>> cache = JUMPCPCache("./data/raw/jumpcp")
        >>> cache.store_profiles("plate_001", df)
        >>> df = cache.get_profiles("plate_001")
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: float = 50.0,
    ):
        self.cache_dir = ensure_dir(Path(cache_dir))
        self.max_size_gb = max_size_gb

        # Sub-directories
        self._metadata_dir = ensure_dir(self.cache_dir / "metadata")
        self._profiles_dir = ensure_dir(self.cache_dir / "profiles")

        # Index
        self._index_path = self.cache_dir / "cache_index.json"
        self._index: Dict[str, CacheEntry] = self._load_index()

        logger.debug(
            f"JUMPCPCache initialised at {self.cache_dir} "
            f"({len(self._index)} entries cached)"
        )

    # =========================================================================
    # Index management
    # =========================================================================

    def _load_index(self) -> Dict[str, CacheEntry]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                raw = json.load(f)
            return {k: CacheEntry.from_dict(v) for k, v in raw.items()}
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._index.items()},
                f,
                indent=2,
            )

    def _make_key(self, namespace: str, name: str) -> str:
        return f"{namespace}/{name}"

    # =========================================================================
    # Metadata (small DataFrames)
    # =========================================================================

    def has_metadata(self, name: str) -> bool:
        """Check whether a metadata table is cached."""
        return self._make_key("metadata", name) in self._index

    def get_metadata(self, name: str) -> Optional[pd.DataFrame]:
        """Load a cached metadata DataFrame, or ``None``."""
        key = self._make_key("metadata", name)
        entry = self._index.get(key)
        if entry is None:
            return None

        path = Path(entry.path)
        if not path.exists():
            del self._index[key]
            self._save_index()
            return None

        entry.accessed_at = time.time()
        self._save_index()
        return pd.read_parquet(path)

    def store_metadata(
        self,
        name: str,
        df: pd.DataFrame,
        source: str = "",
    ) -> Path:
        """Cache a metadata DataFrame as Parquet."""
        path = self._metadata_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)

        key = self._make_key("metadata", name)
        self._index[key] = CacheEntry(
            key=key,
            path=str(path),
            size_bytes=path.stat().st_size,
            n_rows=len(df),
            n_cols=len(df.columns),
            created_at=time.time(),
            accessed_at=time.time(),
            source=source,
        )
        self._save_index()
        logger.debug(f"Cached metadata '{name}' ({len(df)} rows)")
        return path

    # =========================================================================
    # Profiles (per-plate DataFrames)
    # =========================================================================

    def has_profiles(self, plate_id: str) -> bool:
        """Check whether profiles for a plate are cached."""
        return self._make_key("profiles", plate_id) in self._index

    def get_profiles(self, plate_id: str) -> Optional[pd.DataFrame]:
        """Load cached plate profiles, or ``None``."""
        key = self._make_key("profiles", plate_id)
        entry = self._index.get(key)
        if entry is None:
            return None

        path = Path(entry.path)
        if not path.exists():
            del self._index[key]
            self._save_index()
            return None

        entry.accessed_at = time.time()
        self._save_index()
        return pd.read_parquet(path)

    def store_profiles(
        self,
        plate_id: str,
        df: pd.DataFrame,
        source: str = "",
    ) -> Path:
        """Cache plate profiles as Parquet."""
        path = self._profiles_dir / f"{plate_id}.parquet"
        df.to_parquet(path, index=False)

        key = self._make_key("profiles", plate_id)
        self._index[key] = CacheEntry(
            key=key,
            path=str(path),
            size_bytes=path.stat().st_size,
            n_rows=len(df),
            n_cols=len(df.columns),
            created_at=time.time(),
            accessed_at=time.time(),
            source=source,
        )
        self._save_index()
        logger.debug(f"Cached profiles for plate '{plate_id}' ({len(df)} wells)")
        return path

    # =========================================================================
    # Curated datasets
    # =========================================================================

    def store_curated_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        source: str = "curated",
    ) -> Path:
        """Store a curated pre-training dataset."""
        curated_dir = ensure_dir(self.cache_dir / "curated")
        path = curated_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)

        key = self._make_key("curated", name)
        self._index[key] = CacheEntry(
            key=key,
            path=str(path),
            size_bytes=path.stat().st_size,
            n_rows=len(df),
            n_cols=len(df.columns),
            created_at=time.time(),
            accessed_at=time.time(),
            source=source,
        )
        self._save_index()
        logger.info(f"Stored curated dataset '{name}' ({len(df)} rows)")
        return path

    def get_curated_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Load a curated dataset."""
        key = self._make_key("curated", name)
        entry = self._index.get(key)
        if entry is None:
            return None
        path = Path(entry.path)
        if not path.exists():
            del self._index[key]
            self._save_index()
            return None
        entry.accessed_at = time.time()
        self._save_index()
        return pd.read_parquet(path)

    # =========================================================================
    # Cache management
    # =========================================================================

    def get_cache_info(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_bytes = sum(e.size_bytes for e in self._index.values())
        profile_entries = {
            k: v for k, v in self._index.items() if k.startswith("profiles/")
        }
        metadata_entries = {
            k: v for k, v in self._index.items() if k.startswith("metadata/")
        }
        return {
            "cache_dir": str(self.cache_dir),
            "total_entries": len(self._index),
            "total_size_gb": total_bytes / (1024**3),
            "max_size_gb": self.max_size_gb,
            "n_profiles": len(profile_entries),
            "n_metadata": len(metadata_entries),
            "profile_size_gb": sum(
                e.size_bytes for e in profile_entries.values()
            )
            / (1024**3),
        }

    def evict_lru(self, target_size_gb: Optional[float] = None) -> int:
        """
        Evict least-recently-used entries until below *target_size_gb*.

        Returns:
            Number of entries evicted.
        """
        target = target_size_gb if target_size_gb is not None else self.max_size_gb
        target_bytes = target * (1024**3)

        total = sum(e.size_bytes for e in self._index.values())
        if total <= target_bytes:
            return 0

        # Sort by last access time (oldest first)
        sorted_entries = sorted(self._index.values(), key=lambda e: e.accessed_at)
        evicted = 0

        for entry in sorted_entries:
            if total <= target_bytes:
                break
            path = Path(entry.path)
            if path.exists():
                path.unlink()
            total -= entry.size_bytes
            del self._index[entry.key]
            evicted += 1

        self._save_index()
        logger.info(f"Evicted {evicted} cache entries")
        return evicted

    def clear(self) -> None:
        """Remove all cached data."""
        for entry in self._index.values():
            path = Path(entry.path)
            if path.exists():
                path.unlink()
        self._index = {}
        self._save_index()
        logger.info(f"Cleared JUMP-CP cache at {self.cache_dir}")

    # =========================================================================
    # Dunder
    # =========================================================================

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        info = self.get_cache_info()
        return (
            f"JUMPCPCache(entries={info['total_entries']}, "
            f"size={info['total_size_gb']:.2f} GB)"
        )