"""
Cell Painting data processing.

This module provides tools for loading and processing Cell Painting
morphological profiling data from CellProfiler output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from protophen.data.phenotype import Phenotype, PhenotypeDataset
from protophen.utils.logging import logger


# =============================================================================
# Configuration
# =============================================================================

# Standard Cell Painting channel prefixes
CELLPAINTING_PREFIXES = [
    "Cells_",
    "Cytoplasm_",
    "Nuclei_",
]

# Feature categories
FEATURE_CATEGORIES = [
    "AreaShape",
    "Correlation",
    "Granularity",
    "Intensity",
    "Location",
    "Neighbors",
    "RadialDistribution",
    "Texture",
]


@dataclass
class CellPaintingConfig:
    """Configuration for Cell Painting data processing."""
    
    # Feature selection
    compartments: List[str] = field(default_factory=lambda: ["Cells", "Cytoplasm", "Nuclei"])
    feature_categories: Optional[List[str]] = None  # None = all
    
    # Filtering
    remove_blocklist: bool = True
    min_variance: float = 1e-6
    max_correlation: float = 0.95
    max_nan_fraction: float = 0.05
    
    # Aggregation
    aggregation_method: Literal["mean", "median"] = "median"
    
    # Quality control
    min_cell_count: int = 50
    
    # Metadata columns
    sample_id_col: str = "Metadata_Sample"
    protein_id_col: str = "Metadata_Protein"
    well_col: str = "Metadata_Well"
    plate_col: str = "Metadata_Plate"
    cell_count_col: str = "Metadata_CellCount"


# =============================================================================
# Blocklist Features (Known Problematic)
# =============================================================================

BLOCKLIST_FEATURES = [
    # Location features (batch-dependent)
    "Location_Center_X",
    "Location_Center_Y",
    "Location_Center_Z",
    # Object numbers
    "ObjectNumber",
    "Number_Object_Number",
    # Edge features (boundary artifacts)
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMaximum_Y",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    # Parent/children tracking
    "Parent_",
    "Children_",
]


def is_blocklisted(feature_name: str) -> bool:
    """Check if feature is blocklisted."""
    for blocked in BLOCKLIST_FEATURES:
        if blocked in feature_name:
            return True
    return False


# =============================================================================
# Cell Painting Processor
# =============================================================================

class CellPaintingProcessor:
    """
    Process Cell Painting morphological profiling data.
    
    This class handles loading, cleaning, and processing CellProfiler
    output from Cell Painting experiments.
    
    Attributes:
        config: Processing configuration
        feature_names: Selected feature names
        
    Example:
        >>> processor = CellPaintingProcessor()
        >>> 
        >>> # Load and process data
        >>> dataset = processor.load_and_process("cellprofiler_output.csv")
        >>> 
        >>> # Or load from multiple plates
        >>> dataset = processor.load_from_plates([
        ...     "plate1/features.csv",
        ...     "plate2/features.csv",
        ... ])
    """
    
    def __init__(self, config: Optional[CellPaintingConfig] = None):
        """
        Initialise processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or CellPaintingConfig()
        self.feature_names: Optional[List[str]] = None
        self._feature_stats: Optional[Dict[str, Dict]] = None
        
        logger.info("Initialised CellPaintingProcessor")
    
    def _identify_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify feature columns from DataFrame."""
        feature_cols = []
        
        for col in df.columns:
            # Check if column matches any compartment prefix
            is_feature = False
            for compartment in self.config.compartments:
                if col.startswith(f"{compartment}_"):
                    is_feature = True
                    break
            
            if not is_feature:
                continue
            
            # Check feature category if specified
            if self.config.feature_categories is not None:
                has_category = any(cat in col for cat in self.config.feature_categories)
                if not has_category:
                    continue
            
            # Check blocklist
            if self.config.remove_blocklist and is_blocklisted(col):
                continue
            
            feature_cols.append(col)
        
        return feature_cols
    
    def _filter_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> List[str]:
        """Filter features based on variance and correlation."""
        features_df = df[feature_cols]
        
        # Remove features with too many NaNs
        nan_fractions = features_df.isna().mean()
        valid_cols = nan_fractions[nan_fractions <= self.config.max_nan_fraction].index.tolist()
        
        if len(valid_cols) < len(feature_cols):
            removed = len(feature_cols) - len(valid_cols)
            logger.debug(f"Removed {removed} features due to NaN fraction")
        
        features_df = features_df[valid_cols]
        
        # Remove low-variance features
        variances = features_df.var()
        valid_cols = variances[variances > self.config.min_variance].index.tolist()
        
        if len(valid_cols) < features_df.shape[1]:
            removed = features_df.shape[1] - len(valid_cols)
            logger.debug(f"Removed {removed} low-variance features")
        
        # Remove highly correlated features
        if self.config.max_correlation < 1.0 and len(valid_cols) > 1:
            corr_matrix = features_df[valid_cols].corr().abs()
            upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            upper_corr = corr_matrix.where(upper_tri)
            
            to_drop = set()
            for col in upper_corr.columns:
                if col in to_drop:
                    continue
                highly_correlated = upper_corr.index[upper_corr[col] > self.config.max_correlation]
                to_drop.update(highly_correlated)
            
            valid_cols = [c for c in valid_cols if c not in to_drop]
            
            if len(to_drop) > 0:
                logger.debug(f"Removed {len(to_drop)} highly correlated features")
        
        return valid_cols
    
    def load_and_process(
        self,
        filepath: Union[str, Path],
        aggregate_wells: bool = True,
    ) -> PhenotypeDataset:
        """
        Load and process Cell Painting data from CSV.
        
        Args:
            filepath: Path to CellProfiler output CSV
            aggregate_wells: Whether to aggregate cell-level to well-level
            
        Returns:
            Processed PhenotypeDataset
        """
        filepath = Path(filepath)
        logger.info(f"Loading Cell Painting data from {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows")
        
        return self.process_dataframe(df, aggregate_wells=aggregate_wells)
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        aggregate_wells: bool = True,
    ) -> PhenotypeDataset:
        """
        Process a Cell Painting DataFrame.
        
        Args:
            df: DataFrame with CellProfiler output
            aggregate_wells: Whether to aggregate cell-level to well-level
            
        Returns:
            Processed PhenotypeDataset
        """
        # Identify feature columns
        feature_cols = self._identify_feature_columns(df)
        logger.info(f"Identified {len(feature_cols)} feature columns")
        
        # Filter features
        feature_cols = self._filter_features(df, feature_cols)
        logger.info(f"Retained {len(feature_cols)} features after filtering")
        
        self.feature_names = feature_cols
        
        # Aggregate to well level if requested
        if aggregate_wells:
            df = self._aggregate_to_wells(df, feature_cols)
        
        # Create dataset
        dataset = PhenotypeDataset(
            feature_names=feature_cols,
            name="cell_painting",
        )
        
        # Create Phenotype objects
        config = self.config
        for _, row in df.iterrows():
            features = row[feature_cols].values.astype(np.float32)
            
            # Handle NaNs
            features = np.nan_to_num(features, nan=0.0)
            
            phenotype = Phenotype(
                features=features,
                sample_id=str(row.get(config.sample_id_col, f"sample_{len(dataset)}")),
                protein_id=str(row[config.protein_id_col]) if config.protein_id_col in row else None,
                well_id=str(row[config.well_col]) if config.well_col in row else None,
                plate_id=str(row[config.plate_col]) if config.plate_col in row else None,
                cell_count=int(row[config.cell_count_col]) if config.cell_count_col in row else None,
            )
            
            # QC check
            if phenotype.cell_count is not None:
                phenotype.qc_passed = phenotype.cell_count >= config.min_cell_count
            
            dataset.add(phenotype)
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        return dataset
    
    def _aggregate_to_wells(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Aggregate cell-level data to well level."""
        config = self.config
        
        # Determine grouping columns
        group_cols = []
        for col in [config.plate_col, config.well_col, config.protein_id_col]:
            if col in df.columns:
                group_cols.append(col)
        
        if not group_cols:
            logger.warning("No grouping columns found, returning unaggregated data")
            return df
        
        # Aggregate
        agg_func = config.aggregation_method
        agg_df = df.groupby(group_cols)[feature_cols].agg(agg_func).reset_index()
        
        # Add cell count
        cell_counts = df.groupby(group_cols).size().reset_index(name=config.cell_count_col)
        agg_df = agg_df.merge(cell_counts, on=group_cols)
        
        # Create sample ID
        agg_df[config.sample_id_col] = agg_df[group_cols].astype(str).agg('_'.join, axis=1)
        
        logger.info(f"Aggregated to {len(agg_df)} wells from {len(df)} cells")
        
        return agg_df
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by category."""
        if self.feature_names is None:
            return {}
        
        categories = {}
        for feature in self.feature_names:
            for category in FEATURE_CATEGORIES:
                if category in feature:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(feature)
                    break
        
        return categories
    
    def __repr__(self) -> str:
        n_features = len(self.feature_names) if self.feature_names else 0
        return f"CellPaintingProcessor(n_features={n_features})"


# =============================================================================
# Convenience Functions
# =============================================================================

def load_cell_painting_data(
    filepath: Union[str, Path],
    aggregate_wells: bool = True,
    **config_kwargs,
) -> PhenotypeDataset:
    """
    Convenience function to load Cell Painting data.
    
    Args:
        filepath: Path to CellProfiler output
        aggregate_wells: Whether to aggregate to well level
        **config_kwargs: Additional configuration options
        
    Returns:
        PhenotypeDataset
    """
    config = CellPaintingConfig(**config_kwargs)
    processor = CellPaintingProcessor(config)
    return processor.load_and_process(filepath, aggregate_wells=aggregate_wells)