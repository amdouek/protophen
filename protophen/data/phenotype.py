"""
Phenotype data structures for ProToPhen.

This module defines data classes for representing Cell Painting
and other phenotypic readouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from protophen.utils.logging import logger


# =============================================================================
# Phenotype Data Class
# =============================================================================

class Phenotype(BaseModel):
    """
    Represents phenotypic measurements for a single sample.
    
    Attributes:
        features: Feature vector (e.g., Cell Painting features)
        sample_id: Unique sample identifier
        protein_id: Associated protein identifier (if applicable)
        well_id: Plate well identifier (e.g., "A01")
        plate_id: Plate identifier
        metadata: Additional metadata
        
    Example:
        >>> phenotype = Phenotype(
        ...     features=np.random.randn(1500),
        ...     sample_id="sample_001",
        ...     protein_id="protein_abc123",
        ...     well_id="A01",
        ...     plate_id="plate_001",
        ... )
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )
    
    # Core data
    features: np.ndarray
    
    # Identifiers
    sample_id: str
    protein_id: Optional[str] = None
    well_id: Optional[str] = None
    plate_id: Optional[str] = None
    
    # Treatment info
    treatment: Optional[str] = None
    concentration: Optional[float] = None
    timepoint: Optional[float] = None
    
    # Quality metrics
    cell_count: Optional[int] = None
    qc_passed: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @field_validator("features", mode="before")
    @classmethod
    def convert_features(cls, v):
        """Convert features to numpy array."""
        if isinstance(v, list):
            return np.array(v, dtype=np.float32)
        return v
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.features)
    
    def to_dict(self, include_features: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "sample_id": self.sample_id,
            "protein_id": self.protein_id,
            "well_id": self.well_id,
            "plate_id": self.plate_id,
            "treatment": self.treatment,
            "concentration": self.concentration,
            "timepoint": self.timepoint,
            "cell_count": self.cell_count,
            "qc_passed": self.qc_passed,
            "n_features": self.n_features,
            "metadata": self.metadata,
        }
        if include_features:
            data["features"] = self.features.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phenotype":
        """Create from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"Phenotype(sample_id='{self.sample_id}', n_features={self.n_features})"


# =============================================================================
# Phenotype Dataset
# =============================================================================

class PhenotypeDataset:
    """
    Collection of phenotype measurements with batch operations.
    
    This class manages a collection of Phenotype objects and provides
    methods for loading, processing, and accessing phenotypic data.
    
    Attributes:
        phenotypes: List of Phenotype objects
        feature_names: Names of features (if known)
        name: Dataset name
        
    Example:
        >>> dataset = PhenotypeDataset(name="experiment_001")
        >>> dataset.load_from_csv("cell_painting_features.csv")
        >>> 
        >>> # Get feature matrix
        >>> X = dataset.get_feature_matrix()
        >>> 
        >>> # Filter by plate
        >>> plate_data = dataset.filter(lambda p: p.plate_id == "plate_001")
    """
    
    def __init__(
        self,
        phenotypes: Optional[Sequence[Phenotype]] = None,
        feature_names: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialise phenotype dataset.
        
        Args:
            phenotypes: Initial list of phenotypes
            feature_names: Names of features
            name: Dataset name
        """
        self._phenotypes: List[Phenotype] = list(phenotypes) if phenotypes else []
        self.feature_names = feature_names
        self.name = name
        
        self._sample_index: Dict[str, int] = {}
        self._rebuild_index()
    
    def _rebuild_index(self) -> None:
        """Rebuild sample ID index."""
        self._sample_index = {p.sample_id: i for i, p in enumerate(self._phenotypes)}
    
    # =========================================================================
    # Data Access
    # =========================================================================
    
    def add(self, phenotype: Phenotype) -> None:
        """Add a phenotype to the dataset."""
        if phenotype.sample_id in self._sample_index:
            logger.warning(f"Duplicate sample_id: {phenotype.sample_id}")
            return
        
        self._sample_index[phenotype.sample_id] = len(self._phenotypes)
        self._phenotypes.append(phenotype)
    
    def get_by_sample_id(self, sample_id: str) -> Optional[Phenotype]:
        """Get phenotype by sample ID."""
        idx = self._sample_index.get(sample_id)
        return self._phenotypes[idx] if idx is not None else None
    
    def get_by_protein_id(self, protein_id: str) -> List[Phenotype]:
        """Get all phenotypes for a protein (may be multiple replicates)."""
        return [p for p in self._phenotypes if p.protein_id == protein_id]
    
    def filter(self, predicate: callable) -> "PhenotypeDataset":
        """Filter phenotypes by predicate."""
        filtered = [p for p in self._phenotypes if predicate(p)]
        return PhenotypeDataset(
            phenotypes=filtered,
            feature_names=self.feature_names,
            name=f"{self.name}_filtered" if self.name else None,
        )
    
    # =========================================================================
    # Feature Matrix Operations
    # =========================================================================
    
    def get_feature_matrix(self, qc_only: bool = True) -> np.ndarray:
        """
        Get stacked feature matrix.
        
        Args:
            qc_only: Only include QC-passed samples
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        phenotypes = self._phenotypes
        if qc_only:
            phenotypes = [p for p in phenotypes if p.qc_passed]
        
        if not phenotypes:
            return np.array([])
        
        return np.stack([p.features for p in phenotypes])
    
    def get_metadata_df(self) -> pd.DataFrame:
        """Get metadata as DataFrame."""
        records = [p.to_dict(include_features=False) for p in self._phenotypes]
        return pd.DataFrame(records)
    
    def to_dataframe(self, include_features: bool = True) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.
        
        Args:
            include_features: Whether to include feature columns
            
        Returns:
            DataFrame with metadata and optionally features
        """
        meta_df = self.get_metadata_df()
        
        if include_features and self._phenotypes:
            feature_matrix = self.get_feature_matrix(qc_only=False)
            
            if self.feature_names is not None:
                feature_cols = self.feature_names
            else:
                feature_cols = [f"feature_{i}" for i in range(feature_matrix.shape[1])]
            
            feature_df = pd.DataFrame(feature_matrix, columns=feature_cols)
            return pd.concat([meta_df, feature_df], axis=1)
        
        return meta_df
    
    # =========================================================================
    # I/O Operations
    # =========================================================================
    
    def load_from_csv(
        self,
        filepath: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
        feature_prefix: str = "Cells_",
        sample_id_column: str = "Metadata_Sample",
        protein_id_column: Optional[str] = "Metadata_Protein",
        well_column: Optional[str] = "Metadata_Well",
        plate_column: Optional[str] = "Metadata_Plate",
    ) -> int:
        """
        Load phenotypes from CSV file (CellProfiler format).
        
        Args:
            filepath: Path to CSV file
            feature_columns: Specific feature columns (auto-detect if None)
            feature_prefix: Prefix for feature columns (for auto-detection)
            sample_id_column: Column containing sample IDs
            protein_id_column: Column containing protein IDs
            well_column: Column containing well IDs
            plate_column: Column containing plate IDs
            
        Returns:
            Number of phenotypes loaded
        """
        filepath = Path(filepath)
        logger.info(f"Loading phenotypes from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Auto-detect feature columns
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c.startswith(feature_prefix)]
            if not feature_columns:
                # Try common Cell Painting prefixes -- placeholder, need to crossref this with actual Cell Painting nomenclature
                for prefix in ["Cells_", "Cytoplasm_", "Nuclei_", "Image_"]:
                    feature_columns.extend([c for c in df.columns if c.startswith(prefix)])
        
        if not feature_columns:
            raise ValueError(f"No feature columns found with prefix '{feature_prefix}'")
        
        self.feature_names = feature_columns
        logger.info(f"Found {len(feature_columns)} feature columns")
        
        # Load each row as a Phenotype
        count = 0
        for _, row in df.iterrows():
            features = row[feature_columns].values.astype(np.float32)
            
            phenotype = Phenotype(
                features=features,
                sample_id=str(row.get(sample_id_column, f"sample_{count}")),
                protein_id=str(row[protein_id_column]) if protein_id_column and protein_id_column in row else None,
                well_id=str(row[well_column]) if well_column and well_column in row else None,
                plate_id=str(row[plate_column]) if plate_column and plate_column in row else None,
            )
            
            self.add(phenotype)
            count += 1
        
        logger.info(f"Loaded {count} phenotypes")
        return count
    
    def save_to_csv(self, filepath: Union[str, Path]) -> None:
        """Save dataset to CSV."""
        df = self.to_dataframe(include_features=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self)} phenotypes to {filepath}")
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        if not self._phenotypes:
            return {"count": 0}
        
        feature_matrix = self.get_feature_matrix(qc_only=False)
        
        plates = set(p.plate_id for p in self._phenotypes if p.plate_id)
        proteins = set(p.protein_id for p in self._phenotypes if p.protein_id)
        
        return {
            "name": self.name,
            "n_samples": len(self._phenotypes),
            "n_features": feature_matrix.shape[1] if len(feature_matrix) > 0 else 0,
            "n_plates": len(plates),
            "n_proteins": len(proteins),
            "n_qc_passed": sum(1 for p in self._phenotypes if p.qc_passed),
            "feature_stats": {
                "mean": float(np.nanmean(feature_matrix)) if len(feature_matrix) > 0 else None,
                "std": float(np.nanstd(feature_matrix)) if len(feature_matrix) > 0 else None,
                "n_nan": int(np.isnan(feature_matrix).sum()) if len(feature_matrix) > 0 else 0,
            },
        }
    
    # =========================================================================
    # Dunder Methods
    # =========================================================================
    
    def __len__(self) -> int:
        return len(self._phenotypes)
    
    def __iter__(self):
        return iter(self._phenotypes)
    
    def __getitem__(self, idx: int) -> Phenotype:
        return self._phenotypes[idx]
    
    def __repr__(self) -> str:
        return f"PhenotypeDataset(name='{self.name}', n_samples={len(self)})"