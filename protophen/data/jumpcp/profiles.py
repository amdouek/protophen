"""
Morphological profile handling for JUMP-CP.

This module loads well-level aggregated profiles, identifies feature columns,
performs treatment-level aggregation, and bridges JUMP-CP data into
ProToPhen's :class:`PhenotypeDataset` representation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from protophen.data.jumpcp.access import JUMPCPAccess, JUMPCPConfig
from protophen.data.jumpcp.metadata import JUMPCPMetadata
from protophen.data.phenotype import Phenotype, PhenotypeDataset
from protophen.phenotype.cellpainting import (
    CELLPAINTING_PREFIXES,
    is_blocklisted,
)
from protophen.utils.logging import logger


# =============================================================================
# Feature column detection
# =============================================================================

def identify_feature_columns(
    df: pd.DataFrame,
    compartments: Optional[List[str]] = None,
    remove_blocklist: bool = True,
) -> List[str]:
    """
    Identify morphological feature columns in a JUMP-CP profile DataFrame.

    JUMP-CP profiles use the CellProfiler naming convention:
    ``{Compartment}_{Category}_{Feature}_{Channel}``

    Args:
        df: DataFrame with JUMP-CP profiles.
        compartments: Compartments to include.  Defaults to
            ``["Cells", "Cytoplasm", "Nuclei"]``.
        remove_blocklist: Drop known-problematic features.

    Returns:
        Sorted list of feature column names.
    """
    compartments = compartments or ["Cells", "Cytoplasm", "Nuclei"]
    prefixes = tuple(f"{c}_" for c in compartments)

    features = []
    for col in df.columns:
        if not col.startswith(prefixes):
            continue
        if remove_blocklist and is_blocklisted(col):
            continue
        features.append(col)

    return sorted(features)


def identify_metadata_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that look like metadata (``Metadata_*``)."""
    return [c for c in df.columns if c.startswith("Metadata_")]


# =============================================================================
# Profile Loader
# =============================================================================

class ProfileLoader:
    """
    Load and process JUMP-CP morphological profiles.

    Wraps :class:`JUMPCPAccess` and :class:`JUMPCPMetadata` to provide
    high-level methods for fetching profiles, filtering features, and
    aggregating to treatment-level summaries.

    Attributes:
        access: Data access layer.
        metadata: Metadata manager.

    Example:
        >>> loader = ProfileLoader()
        >>> df = loader.load_plate_profiles("source_4", "2021_08_17_U2OS_48_X6", "BR00116991")
        >>> features = loader.feature_columns
        >>> print(f"{len(features)} features")
    """

    def __init__(
        self,
        access: Optional[JUMPCPAccess] = None,
        metadata: Optional[JUMPCPMetadata] = None,
        config: Optional[JUMPCPConfig] = None,
    ):
        self.access = access or JUMPCPAccess(config=config)
        self.metadata = metadata or JUMPCPMetadata(access=self.access)

        self._feature_columns: Optional[List[str]] = None
        self._metadata_columns: Optional[List[str]] = None

    @property
    def feature_columns(self) -> Optional[List[str]]:
        """Feature columns identified from the most recently loaded plate."""
        return self._feature_columns

    # =========================================================================
    # Single-plate loading
    # =========================================================================

    def load_plate_profiles(
        self,
        source: str,
        batch: str,
        plate: str,
        compartments: Optional[List[str]] = None,
        remove_blocklist: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load well-level profiles for a single plate.

        Args:
            source: Source identifier.
            batch: Batch identifier.
            plate: Plate identifier.
            compartments: Feature compartments to include.
            remove_blocklist: Remove known-bad features.
            force_refresh: Bypass cache.

        Returns:
            DataFrame with metadata + feature columns.
        """
        df = self.access.fetch_plate_profiles(
            source=source,
            batch=batch,
            plate=plate,
            force_refresh=force_refresh,
        )

        self._feature_columns = identify_feature_columns(
            df,
            compartments=compartments,
            remove_blocklist=remove_blocklist,
        )
        self._metadata_columns = identify_metadata_columns(df)

        logger.info(
            f"Plate '{plate}': {len(df)} wells, "
            f"{len(self._feature_columns)} features"
        )
        return df

    # =========================================================================
    # Multi-plate loading
    # =========================================================================

    def load_plates(
        self,
        plate_info: pd.DataFrame,
        compartments: Optional[List[str]] = None,
        remove_blocklist: bool = True,
        max_plates: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load profiles from multiple plates and concatenate.

        Args:
            plate_info: DataFrame with columns ``["plate", "source", "batch"]``
                (as returned by :meth:`JUMPCPMetadata.get_plate_batch_source_map`).
            compartments: Feature compartments to include.
            remove_blocklist: Remove known-bad features.
            max_plates: Limit the number of plates loaded (for testing).

        Returns:
            Concatenated DataFrame of all plate profiles.
        """
        frames: List[pd.DataFrame] = []
        total = min(len(plate_info), max_plates) if max_plates else len(plate_info)

        for idx, (_, row) in enumerate(plate_info.iterrows()):
            if max_plates is not None and idx >= max_plates:
                break

            try:
                df = self.load_plate_profiles(
                    source=row["source"],
                    batch=row["batch"],
                    plate=row["plate"],
                    compartments=compartments,
                    remove_blocklist=remove_blocklist,
                )
                frames.append(df)
                logger.debug(f"Loaded plate {idx + 1}/{total}: {row['plate']}")
            except Exception as exc:
                logger.warning(f"Failed to load plate '{row['plate']}': {exc}")
                continue

        if not frames:
            raise RuntimeError("No plates could be loaded")

        # Use intersection of feature columns across plates
        common_features = set(identify_feature_columns(frames[0]))
        for f in frames[1:]:
            common_features &= set(identify_feature_columns(f))
        self._feature_columns = sorted(common_features)

        # Subset to common columns
        meta_cols = set()
        for f in frames:
            meta_cols.update(identify_metadata_columns(f))
        keep_cols = sorted(meta_cols) + self._feature_columns

        combined = pd.concat(
            [f[[c for c in keep_cols if c in f.columns]] for f in frames],
            ignore_index=True,
        )

        logger.info(
            f"Loaded {len(combined)} wells from {len(frames)} plates, "
            f"{len(self._feature_columns)} common features"
        )
        return combined

    # =========================================================================
    # Treatment-level aggregation
    # =========================================================================

    def aggregate_to_treatments(
        self,
        df: pd.DataFrame,
        group_by: Optional[List[str]] = None,
        method: Literal["mean", "median"] = "median",
        min_replicates: int = 2,
    ) -> pd.DataFrame:
        """
        Aggregate well-level profiles to treatment-level summaries.

        Args:
            df: Well-level DataFrame.
            group_by: Columns to group by.  Defaults to heuristic detection
                of gene/treatment column.
            method: Aggregation function.
            min_replicates: Minimum number of wells for a treatment to be kept.

        Returns:
            Treatment-level DataFrame.
        """
        if self._feature_columns is None:
            self._feature_columns = identify_feature_columns(df)

        if group_by is None:
            # Heuristic: look for gene/perturbation column
            group_by = []
            for candidate in [
                "Metadata_JCP2022",
                "Metadata_Symbol",
                "Metadata_Gene",
                "Metadata_broad_sample",
            ]:
                if candidate in df.columns:
                    group_by.append(candidate)
                    break
            if not group_by:
                raise ValueError(
                    "Cannot auto-detect grouping column. "
                    "Please provide `group_by` explicitly."
                )

        # Count replicates
        rep_counts = df.groupby(group_by).size().reset_index(name="_n_replicates")
        valid = rep_counts[rep_counts["_n_replicates"] >= min_replicates]

        # Merge to filter
        df_filtered = df.merge(
            valid[group_by], on=group_by, how="inner"
        )

        agg_df = (
            df_filtered.groupby(group_by)[self._feature_columns]
            .agg(method)
            .reset_index()
        )

        # Re-attach replicate counts
        agg_df = agg_df.merge(rep_counts, on=group_by)

        logger.info(
            f"Aggregated to {len(agg_df)} treatments "
            f"({method}, min_replicates={min_replicates})"
        )
        return agg_df

    # =========================================================================
    # Conversion to PhenotypeDataset
    # =========================================================================

    def to_phenotype_dataset(
        self,
        df: pd.DataFrame,
        gene_col: Optional[str] = None,
        plate_col: Optional[str] = None,
        well_col: Optional[str] = None,
        dataset_name: str = "jumpcp",
    ) -> PhenotypeDataset:
        """
        Convert a profiles DataFrame to a :class:`PhenotypeDataset`.

        Args:
            df: DataFrame with metadata + feature columns.
            gene_col: Column containing gene / protein identifiers.
            plate_col: Column containing plate identifiers.
            well_col: Column containing well identifiers.
            dataset_name: Name for the resulting dataset.

        Returns:
            :class:`PhenotypeDataset` instance.
        """
        if self._feature_columns is None:
            self._feature_columns = identify_feature_columns(df)

        # Auto-detect columns
        if gene_col is None:
            for c in [
                "Metadata_Symbol",
                "Metadata_Gene",
                "Metadata_JCP2022",
            ]:
                if c in df.columns:
                    gene_col = c
                    break

        if plate_col is None:
            for c in ["Metadata_Plate", "Metadata_plate"]:
                if c in df.columns:
                    plate_col = c
                    break

        if well_col is None:
            for c in ["Metadata_Well", "Metadata_well"]:
                if c in df.columns:
                    well_col = c
                    break

        dataset = PhenotypeDataset(
            feature_names=self._feature_columns,
            name=dataset_name,
        )

        for idx, row in df.iterrows():
            features = row[self._feature_columns].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0)

            sample_id = f"jumpcp_{idx}"
            protein_id = (
                str(row[gene_col]) 
                if gene_col and gene_col in row.index 
                    and pd.notna(row[gene_col])
                else None
            )
            plate_id_val = (
                str(row[plate_col])
                if plate_col and plate_col in row.index
                    and pd.notna(row[plate_col])
                else None
            )
            well_id_val = (
                str(row[well_col])
                if well_col and well_col in row.index
                    and pd.notna(row[well_col])
                else None
            )

            # For aggregated data, plate_id may be present. Try to recover from other metadata columns.
            if plate_id_val is None or plate_id_val == "nan":
                plate_id_val = None
                for c in ["Metadata_Source", "Metadata_Batch"]:
                    if c in row.index and pd.notna(row[c]):
                        plate_id_val = str(row[c])
                        break
            
            phenotype = Phenotype(
                features=features,
                sample_id=sample_id,
                protein_id=protein_id,
                plate_id=plate_id_val,
                well_id=well_id_val,
                treatment=protein_id,
            )
            dataset.add(phenotype)

        logger.info(
            f"Created PhenotypeDataset '{dataset_name}' "
            f"with {len(dataset)} samples"
        )
        return dataset

    def __repr__(self) -> str:
        n_feat = (
            len(self._feature_columns) if self._feature_columns else 0
        )
        return f"ProfileLoader(n_features={n_feat})"