"""
Metadata parsing and filtering for JUMP-CP.

This module wraps the raw JUMP-CP metadata tables (well, plate, ORF, CRISPR)
and provides structured access: filtering by perturbation type, gene family,
replicate count, and plate layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from protophen.data.jumpcp.access import JUMPCPAccess, JUMPCPConfig
from protophen.utils.logging import logger


# =============================================================================
# Perturbation types
# =============================================================================

PERTURBATION_TYPES = Literal["orf", "crispr", "compound"]


# =============================================================================
# Metadata Manager
# =============================================================================

class JUMPCPMetadata:
    """
    Parse, merge, and filter JUMP-CP metadata tables.

    This class lazily fetches metadata tables via :class:`JUMPCPAccess` and
    provides convenient views for subsetting perturbations.

    Attributes:
        access: Data-access layer.

    Example:
        >>> meta = JUMPCPMetadata()
        >>> orf_genes = meta.get_orf_genes()
        >>> plates = meta.get_plates_for_perturbation("orf")
        >>> well_map = meta.get_well_map("BR00116991")
    """

    def __init__(
        self,
        access: Optional[JUMPCPAccess] = None,
        config: Optional[JUMPCPConfig] = None,
    ):
        self.access = access or JUMPCPAccess(config=config)
        self._tables: Dict[str, pd.DataFrame] = {}

    # =========================================================================
    # Lazy table loading
    # =========================================================================

    def _get_table(self, name: str, force_refresh: bool = False) -> pd.DataFrame:
        """Load a metadata table, caching in memory."""
        if name not in self._tables or force_refresh:
            self._tables[name] = self.access.fetch_metadata_table(
                name, force_refresh=force_refresh
            )
        return self._tables[name]

    @property
    def well_table(self) -> pd.DataFrame:
        return self._get_table("well")

    @property
    def plate_table(self) -> pd.DataFrame:
        return self._get_table("plate")

    @property
    def orf_table(self) -> pd.DataFrame:
        return self._get_table("orf")

    @property
    def crispr_table(self) -> pd.DataFrame:
        return self._get_table("crispr")

    @property
    def compound_table(self) -> pd.DataFrame:
        return self._get_table("compound")

    # =========================================================================
    # Gene / perturbation queries
    # =========================================================================

    def get_orf_genes(self) -> List[str]:
        """Return unique gene symbols in the ORF overexpression set."""
        df = self.orf_table
        col = self._find_gene_column(df)
        genes = df[col].dropna().unique().tolist()
        logger.info(f"ORF set contains {len(genes)} unique genes")
        return sorted(genes)

    def get_crispr_genes(self) -> List[str]:
        """Return unique gene symbols in the CRISPR set."""
        df = self.crispr_table
        col = self._find_gene_column(df)
        genes = df[col].dropna().unique().tolist()
        logger.info(f"CRISPR set contains {len(genes)} unique genes")
        return sorted(genes)

    @staticmethod
    def _find_gene_column(df: pd.DataFrame) -> str:
        """Heuristic to locate the gene-symbol column."""
        candidates = [
            "Metadata_Symbol",
            "Metadata_Gene",
            "gene",
            "Gene",
            "gene_symbol",
            "Metadata_JCP2022", # Perturbation ID - fallback
        ]
        for c in candidates:
            if c in df.columns:
                return c
        # Fall back to first column containing "gene" (case-insensitive)
        for c in df.columns:
            if "gene" in c.lower() or "symbol" in c.lower():
                return c
        raise KeyError(
            f"Cannot identify gene column in DataFrame with columns: {list(df.columns)}"
        )

    @staticmethod
    def _find_plate_column(df: pd.DataFrame) -> str:
        """Heuristic to locate the plate-ID column."""
        candidates = [
            "Metadata_Plate",
            "Metadata_plate",
            "plate",
            "Plate",
            "plate_id",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            if "plate" in c.lower():
                return c
        raise KeyError(
            f"Cannot identify plate column in DataFrame with columns: {list(df.columns)}"
        )

    @staticmethod
    def _find_well_column(df: pd.DataFrame) -> str:
        """Heuristic to locate the well-ID column."""
        candidates = [
            "Metadata_Well",
            "Metadata_well",
            "well",
            "Well",
            "well_position",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            if "well" in c.lower():
                return c
        raise KeyError(
            f"Cannot identify well column in DataFrame with columns: {list(df.columns)}"
        )

    @staticmethod
    def _find_source_column(df: pd.DataFrame) -> str:
        candidates = [
            "Metadata_Source",
            "Metadata_source",
            "source",
            "Source",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            if "source" in c.lower():
                return c
        raise KeyError(
            f"Cannot identify source column in DataFrame with columns: {list(df.columns)}"
        )

    @staticmethod
    def _find_batch_column(df: pd.DataFrame) -> str:
        candidates = [
            "Metadata_Batch",
            "Metadata_batch",
            "batch",
            "Batch",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            if "batch" in c.lower():
                return c
        raise KeyError(
            f"Cannot identify batch column in DataFrame with columns: {list(df.columns)}"
        )

    # =========================================================================
    # Plate queries
    # =========================================================================

    def get_plates_for_perturbation(
        self,
        perturbation_type: str,
    ) -> pd.DataFrame:
        """
        Return plate metadata filtered to a specific perturbation type.

        Args:
            perturbation_type: ``"orf"``, ``"crispr"``, or ``"compound"``.

        Returns:
            Subset of the plate table.
        """
        plate_df = self.plate_table

        # Candidate column names (in order of priority)
        pert_col = None
        for c in plate_df.columns:
            cl = c.lower()
            # Match "MetaData_PlateType", "Metadata_Perturbation_Type", etc.
            if "platetype" in cl or ("pert" in cl and "type" in cl):
                pert_col = c
                break

        if pert_col is not None:
            # Map perturbation type to possible PlateType values
            type_map = {
                "orf": ["orf"],
                "crispr": ["crispr"],
                "compound": ["compound", "compound_empty", "compound_full"],
                "target2": ["target2"],
            }
            accepted_values = type_map.get(
                perturbation_type.lower(), 
                [perturbation_type.lower()]
            )
            
            # Case-insensitive matching
            plate_types = plate_df[pert_col].str.lower().str.strip()
            mask = plate_types.isin(accepted_values)
            
            # Also try substring matches for compound variants
            if mask.sum() == 0:
                mask = plate_types.str.contains(
                    perturbation_type.lower(), na=False
                )
                
            filtered = plate_df[mask].copy()
            
            if len(filtered) == 0:
                unique_types = plate_df[pert_col].unique().tolist()
                logger.warning(
                    f"No plates matched perturbation type '{perturbation_type}'. "
                    f"Available PlateType values: {unique_types}"
                )
        else:
            logger.warning(
                "No perturbation-type column found in plate table; "
                "returning all plates."
            )
            filtered = plate_df.copy()

        logger.info(
            f"Found {len(filtered)} plates for perturbation type "
            f"'{perturbation_type}'"
        )
        return filtered

    def get_well_map(
        self,
        plate_id: str,
    ) -> pd.DataFrame:
        """
        Return the well-level metadata for a plate.

        Merges the ``well`` table with the appropriate perturbation table
        to provide gene symbols alongside well positions.
        """
        well_df = self.well_table
        plate_col = self._find_plate_column(well_df)

        plate_wells = well_df[well_df[plate_col] == plate_id].copy()
        logger.debug(f"Plate '{plate_id}': {len(plate_wells)} wells")
        return plate_wells

    # =========================================================================
    # Replicate and coverage queries
    # =========================================================================

    def count_replicates(
        self,
        perturbation_type: str = "orf",
    ) -> pd.DataFrame:
        """
        Count the number of replicate wells per gene.

        Args:
            perturbation_type: ``"orf"`` or ``"crispr"``.

        Returns:
            DataFrame with columns ``['gene', 'n_replicates']``.
        """
        if perturbation_type == "orf":
            df = self.orf_table
        elif perturbation_type == "crispr":
            df = self.crispr_table
        else:
            raise ValueError(f"Unsupported perturbation type: {perturbation_type}")

        gene_col = self._find_gene_column(df)
        counts = (
            df.groupby(gene_col)
            .size()
            .reset_index(name="n_replicates")
            .rename(columns={gene_col: "gene"})
            .sort_values("n_replicates", ascending=False)
        )
        return counts

    def filter_genes_by_replicates(
        self,
        perturbation_type: str = "orf",
        min_replicates: int = 2,
    ) -> List[str]:
        """
        Return genes that have at least *min_replicates* wells.

        Args:
            perturbation_type: ``"orf"`` or ``"crispr"``.
            min_replicates: Minimum number of replicate wells.

        Returns:
            Sorted list of gene symbols.
        """
        counts = self.count_replicates(perturbation_type)
        filtered = counts[counts["n_replicates"] >= min_replicates]["gene"].tolist()
        logger.info(
            f"{len(filtered)} genes with ≥{min_replicates} replicates "
            f"({perturbation_type})"
        )
        return sorted(filtered)

    def get_plate_batch_source_map(
        self,
        perturbation_type: str = "orf",
    ) -> pd.DataFrame:
        """
        Return a mapping of plate → (source, batch) for profile downloads.

        This is needed to construct S3 keys for each plate.
        """
        plate_df = self.get_plates_for_perturbation(perturbation_type)

        source_col = self._find_source_column(plate_df)
        batch_col = self._find_batch_column(plate_df)
        plate_col = self._find_plate_column(plate_df)

        mapping = plate_df[[plate_col, source_col, batch_col]].copy()
        mapping.columns = ["plate", "source", "batch"]
        mapping = mapping.drop_duplicates()

        logger.info(
            f"Plate→source/batch mapping: {len(mapping)} entries "
            f"({perturbation_type})"
        )
        return mapping

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Return a high-level summary of available metadata."""
        info: Dict[str, Any] = {}

        try:
            info["n_orf_genes"] = len(self.get_orf_genes())
        except Exception:
            info["n_orf_genes"] = None

        try:
            info["n_crispr_genes"] = len(self.get_crispr_genes())
        except Exception:
            info["n_crispr_genes"] = None

        try:
            info["n_plates"] = len(self.plate_table)
        except Exception:
            info["n_plates"] = None

        try:
            info["n_wells"] = len(self.well_table)
        except Exception:
            info["n_wells"] = None

        return info

    def __repr__(self) -> str:
        n_tables = len(self._tables)
        return f"JUMPCPMetadata(loaded_tables={n_tables})"