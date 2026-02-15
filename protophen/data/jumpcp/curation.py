"""
Subset selection and quality control for JUMP-CP pre-training data.

This module implements the :class:`DataCurator` which selects high-quality,
balanced subsets of the JUMP-CP dataset suitable for ProToPhen pre-training.
The full dataset is >100 TB; intelligent curation is essential.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from protophen.data.jumpcp.access import JUMPCPAccess, JUMPCPConfig
from protophen.data.jumpcp.cache import JUMPCPCache
from protophen.data.jumpcp.metadata import JUMPCPMetadata
from protophen.data.jumpcp.profiles import ProfileLoader, identify_feature_columns
from protophen.data.phenotype import PhenotypeDataset
from protophen.phenotype.normalisation import (
    BatchCorrector,
    FeatureSelector,
    Normaliser,
)
from protophen.utils.logging import logger


# =============================================================================
# Curation Configuration
# =============================================================================

@dataclass
class CurationConfig:
    """Configuration for pre-training data curation."""

    # Perturbation types to include
    perturbation_types: List[str] = field(
        default_factory=lambda: ["orf"]
    )

    # Gene filtering
    min_replicates: int = 2
    max_genes: Optional[int] = None  # None = keep all passing QC

    # Quality control
    min_cell_count: int = 50
    max_nan_fraction: float = 0.05
    min_feature_variance: float = 1e-6
    max_feature_correlation: float = 0.95

    # Normalisation
    normalisation_method: str = "robust_mad"
    batch_correction: bool = True
    batch_correction_method: str = "robust"
    clip_outliers: bool = True
    outlier_threshold: float = 5.0

    # Aggregation
    aggregation_method: Literal["mean", "median"] = "median"

    # Sampling strategy
    sampling_strategy: Literal[
        "all", "diversity", "coverage"
    ] = "all"
    target_n_genes: Optional[int] = None  # For diversity / coverage sampling

    # Size constraints
    max_plates: Optional[int] = None  # For quick tests

    # Output
    output_name: str = "pretraining_v1"


# =============================================================================
# Quality Control
# =============================================================================

class QualityController:
    """
    Quality-control filters for JUMP-CP well-level profiles.

    Checks:
    - NaN fraction per well
    - Cell count (if available)
    - Profile norm outliers (wells with abnormally low/high feature magnitudes)
    """

    def __init__(
        self,
        max_nan_fraction: float = 0.05,
        min_cell_count: int = 50,
        norm_outlier_mad: float = 3.0,
    ):
        self.max_nan_fraction = max_nan_fraction
        self.min_cell_count = min_cell_count
        self.norm_outlier_mad = norm_outlier_mad

    def flag_wells(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> pd.Series:
        """
        Return a boolean Series indicating which rows pass QC.

        Args:
            df: Well-level DataFrame.
            feature_columns: Feature column names.

        Returns:
            Boolean Series aligned with *df*.
        """
        n = len(df)
        passed = pd.Series(True, index=df.index)

        feature_matrix = df[feature_columns].values.astype(np.float32)

        # --- Check 1: NaN fraction ---
        nan_frac = np.isnan(feature_matrix).mean(axis=1)
        nan_fail = nan_frac > self.max_nan_fraction
        passed &= ~nan_fail
        logger.debug(f"QC: {nan_fail.sum()}/{n} wells failed NaN check")

        # --- Check 2: Cell count (if column exists) ---
        cell_col = None
        for c in [
            "Metadata_Count_Cells",
            "Metadata_Cells_Number_Object_Number",
            "Metadata_Nuclei_Number_Object_Number",
            "Metadata_Count_Nuclei",
            "Metadata_CellCount",
            "Metadata_Number_of_Cells",
            "Metadata_Object_Count",
        ]:
            if c in df.columns:
                cell_col = c
                break

        # Heuristic fallback
        if cell_col is None:
            for c in df.columns:
                cl = c.lower()
                if cl.startswith("metadata_"):
                    has_count = ("count" in cl or "number" in cl)
                    has_object = ("cell" in cl or "nuclei" in cl or "object" in cl)
                    if has_count and has_object:
                        try:
                            df[c].astype(float)
                            cell_col = c
                            break
                        except (ValueError, TypeError):
                            continue

        if cell_col is not None:
            low_cells = df[cell_col].fillna(0).astype(float).astype(int) < self.min_cell_count
            passed &= ~low_cells
            logger.debug(f"QC: {low_cells.sum()}/{n} wells failed cell-count check ({cell_col})")
        else:
            logger.debug(
                "QC: No cell-count column found in aggregated profiles; "
                "relying on NaN and profile-norm checks"
            )

        # --- Check 3: Profile norm outliers ---
        clean_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        norms = np.linalg.norm(clean_matrix, axis=1)

        if len(norms) > 2:
            norm_median = np.median(norms)
            norm_mad = np.median(np.abs(norms - norm_median)) * 1.4826

            if norm_mad > 1e-8:
                norm_lo = norm_median - self.norm_outlier_mad * norm_mad
                norm_hi = norm_median + self.norm_outlier_mad * norm_mad
                norm_fail = (norms < norm_lo) | (norms > norm_hi)
                passed &= ~norm_fail
                logger.debug(
                    f"QC: {norm_fail.sum()}/{n} wells failed profile-norm check "
                    f"(bounds=[{norm_lo:.1f}, {norm_hi:.1f}])"
                )

        logger.info(f"QC: {passed.sum()}/{n} wells passed all checks")
        return passed


# =============================================================================
# Data Curator
# =============================================================================

class DataCurator:
    """
    Construct curated pre-training datasets from JUMP-CP.

    Orchestrates metadata filtering, profile loading, QC, normalisation,
    feature selection, batch correction, and optional diversity-based
    gene sampling.

    Attributes:
        config: Curation configuration.
        access: Data access layer.
        metadata: Metadata manager.
        loader: Profile loader.

    Example:
        >>> curator = DataCurator()
        >>> dataset = curator.build_pretraining_set()
        >>> curator.save(dataset, "data/processed/pretraining")
    """

    def __init__(
        self,
        config: Optional[CurationConfig] = None,
        access: Optional[JUMPCPAccess] = None,
        jumpcp_config: Optional[JUMPCPConfig] = None,
    ):
        self.config = config or CurationConfig()
        self.access = access or JUMPCPAccess(config=jumpcp_config)
        self.metadata = JUMPCPMetadata(access=self.access)
        self.loader = ProfileLoader(access=self.access, metadata=self.metadata)
        self.qc = QualityController(
            max_nan_fraction=self.config.max_nan_fraction,
            min_cell_count=self.config.min_cell_count,
        )

        # Fitted transformers (stored for reproducibility)
        self._feature_selector: Optional[FeatureSelector] = None
        self._normaliser: Optional[Normaliser] = None
        self._batch_corrector: Optional[BatchCorrector] = None
        self._curation_report: Dict[str, Any] = {}

    # =========================================================================
    # Main pipeline
    # =========================================================================

    def build_pretraining_set(self) -> pd.DataFrame:
        """
        Run the full curation pipeline and return a clean DataFrame.

        Steps:
        1.  Determine plates to load from metadata.
        2.  Load well-level profiles.
        2b. Merge perturbation metadata (gene symbols).
        3.  QC filtering.
        4.  Feature selection.
        5.  Normalisation (per-plate).
        6.  Batch correction.
        7.  Treatment-level aggregation.
        8.  (Optional) Diversity-based gene sampling.

        Returns:
            Curated DataFrame ready for pre-training.
        """
        report: Dict[str, Any] = {"config": self.config.__dict__.copy()}

        # ------------------------------------------------------------------
        # Step 1: Determine plates
        # ------------------------------------------------------------------
        logger.info("Step 1: Resolving plates from metadata")
        all_plate_info = []

        for pert_type in self.config.perturbation_types:
            plate_map = self.metadata.get_plate_batch_source_map(pert_type)
            plate_map["perturbation_type"] = pert_type
            all_plate_info.append(plate_map)

        plate_info = pd.concat(all_plate_info, ignore_index=True)
        report["n_plates_available"] = len(plate_info)
        logger.info(f"Total plates available: {len(plate_info)}")

        # ------------------------------------------------------------------
        # Step 2: Load profiles
        # ------------------------------------------------------------------
        logger.info("Step 2: Loading well-level profiles")
        raw_df = self.loader.load_plates(
            plate_info,
            max_plates=self.config.max_plates,
        )
        feature_cols = self.loader.feature_columns
        report["n_wells_raw"] = len(raw_df)
        report["n_features_raw"] = len(feature_cols)

        # ------------------------------------------------------------------
        # Step 2b: Merge perturbation metadata
        # ------------------------------------------------------------------
        logger.info("Step 2b: Merging perturbation metadata with profiles")
        raw_df = self._merge_perturbation_metadata(raw_df)
        report["n_wells_after_merge"] = len(raw_df)

        # Verify a gene column now exists
        gene_col = self._detect_gene_column(raw_df)
        if gene_col is not None:
            n_genes = raw_df[gene_col].nunique()
            logger.info(f"Merged metadata: {n_genes} unique genes ({gene_col})")
            report["gene_column"] = gene_col
            report["n_genes_merged"] = n_genes
        else:
            logger.warning(
                "No gene column found after metadata merge. "
                "Treatment-level aggregation may fail."
            )

        # ------------------------------------------------------------------
        # Step 3: QC filtering
        # ------------------------------------------------------------------
        logger.info("Step 3: Quality control")
        qc_mask = self.qc.flag_wells(raw_df, feature_cols)
        df = raw_df[qc_mask].copy().reset_index(drop=True)
        report["n_wells_post_qc"] = len(df)

        # ------------------------------------------------------------------
        # Step 4: Feature selection
        # ------------------------------------------------------------------
        logger.info("Step 4: Feature selection")
        feature_matrix = df[feature_cols].values.astype(np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

        self._feature_selector = FeatureSelector(
            min_variance=self.config.min_feature_variance,
            max_correlation=self.config.max_feature_correlation,
            max_nan_fraction=self.config.max_nan_fraction,
        )
        feature_matrix = self._feature_selector.fit_transform(feature_matrix)
        selected_features = [
            feature_cols[i] for i in self._feature_selector.selected_indices_
        ]

        # Replace feature columns in df
        df = df.drop(columns=feature_cols)
        feat_df = pd.DataFrame(feature_matrix, columns=selected_features)
        df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
        feature_cols = selected_features
        report["n_features_selected"] = len(feature_cols)

        # ------------------------------------------------------------------
        # Step 5: Plate-level normalisation
        # ------------------------------------------------------------------
        logger.info("Step 5: Normalisation")
        self._normaliser = Normaliser(
            method=self.config.normalisation_method,
            clip_outliers=self.config.clip_outliers,
            outlier_threshold=self.config.outlier_threshold,
        )
        feature_matrix = df[feature_cols].values.astype(np.float32)
        feature_matrix = self._normaliser.fit_transform(feature_matrix)
        df[feature_cols] = feature_matrix

        # ------------------------------------------------------------------
        # Step 6: Batch correction
        # ------------------------------------------------------------------
        if self.config.batch_correction:
            logger.info("Step 6: Batch correction")
            plate_col = None
            for c in ["Metadata_Plate", "Metadata_plate"]:
                if c in df.columns:
                    plate_col = c
                    break

            if plate_col is not None and df[plate_col].nunique() > 1:
                self._batch_corrector = BatchCorrector(
                    method=self.config.batch_correction_method,
                )
                feature_matrix = df[feature_cols].values.astype(np.float32)
                batch_labels = df[plate_col].values
                feature_matrix = self._batch_corrector.fit_transform(
                    feature_matrix, batch_labels
                )
                df[feature_cols] = feature_matrix
                report["batch_correction_applied"] = True
                report["n_batches"] = int(df[plate_col].nunique())
            else:
                logger.info(
                    "Skipping batch correction (single plate or column missing)"
                )
                report["batch_correction_applied"] = False
        else:
            logger.info("Step 6: Batch correction disabled")
            report["batch_correction_applied"] = False

        # ------------------------------------------------------------------
        # Step 7: Treatment-level aggregation
        # ------------------------------------------------------------------
        logger.info("Step 7: Treatment-level aggregation")
        self.loader._feature_columns = feature_cols
        agg_df = self.loader.aggregate_to_treatments(
            df,
            method=self.config.aggregation_method,
            min_replicates=self.config.min_replicates,
        )
        report["n_treatments_aggregated"] = len(agg_df)

        # ------------------------------------------------------------------
        # Step 8: Gene sampling (optional)
        # ------------------------------------------------------------------
        if self.config.sampling_strategy != "all" and self.config.target_n_genes:
            logger.info(
                f"Step 8: Gene sampling (strategy={self.config.sampling_strategy}, "
                f"target={self.config.target_n_genes})"
            )
            agg_df = self._sample_genes(agg_df, feature_cols)
            report["n_treatments_sampled"] = len(agg_df)
        elif self.config.max_genes is not None:
            logger.info(f"Step 8: Limiting to {self.config.max_genes} genes")
            agg_df = self._limit_genes(agg_df)
            report["n_treatments_limited"] = len(agg_df)
        else:
            logger.info("Step 8: No gene sampling applied")

        report["n_treatments_final"] = len(agg_df)
        report["n_features_final"] = len(feature_cols)
        self._curation_report = report

        logger.info(
            f"Curation complete: {len(agg_df)} treatments, "
            f"{len(feature_cols)} features"
        )
        return agg_df

    # =========================================================================
    # Metadata merging
    # =========================================================================

    def _merge_perturbation_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge perturbation metadata (gene symbols, etc.) into profile data.

        JUMP-CP aggregated profiles only contain Plate/Well/Source columns.
        Gene identity requires a two-hop join:

        1. Profiles → well table (on Source, Plate, Well) → adds Metadata_JCP2022
        2. Result → orf/crispr table (on Metadata_JCP2022) → adds Metadata_Symbol

        Args:
            df: Well-level profiles DataFrame.

        Returns:
            DataFrame with perturbation metadata columns added.
        """
        initial_cols = set(df.columns)

        # --- Hop 1: Merge well metadata (adds Metadata_JCP2022) ---
        try:
            well_df = self.metadata._get_table("well")
            logger.info(
                f"Well metadata: {len(well_df)} rows, "
                f"columns={list(well_df.columns)}"
            )
            df = self._safe_merge(df, well_df, label="well")

            # Check if JCP2022 was added
            if "Metadata_JCP2022" in df.columns:
                n_matched = df["Metadata_JCP2022"].notna().sum()
                logger.info(
                    f"Hop 1 (well): Added Metadata_JCP2022, "
                    f"{n_matched}/{len(df)} wells matched"
                )
            else:
                logger.warning("Hop 1 (well): Metadata_JCP2022 not found after merge")
        except Exception as exc:
            logger.warning(f"Could not load/merge well metadata: {exc}")

        # --- Hop 2: Merge perturbation tables (adds Metadata_Symbol etc.) ---
        for pert_type in self.config.perturbation_types:
            try:
                pert_df = self.metadata._get_table(pert_type)
                logger.info(
                    f"{pert_type} metadata: {len(pert_df)} rows, "
                    f"columns={list(pert_df.columns)}"
                )
                df = self._safe_merge(df, pert_df, label=pert_type)
            except Exception as exc:
                logger.warning(f"Could not load/merge {pert_type} metadata: {exc}")

        new_cols = set(df.columns) - initial_cols
        if new_cols:
            logger.info(
                f"Metadata merge added {len(new_cols)} columns: "
                f"{sorted(new_cols)[:10]}{'...' if len(new_cols) > 10 else ''}"
            )
        else:
            logger.warning("Metadata merge did not add any new columns")

        return df

    @staticmethod
    def _safe_merge(
        profiles: pd.DataFrame,
        meta: pd.DataFrame,
        label: str = "",
    ) -> pd.DataFrame:
        """
        Merge metadata into profiles using the best available key columns.

        Tries merge keys in order of specificity:
        1. (Source, Plate, Well) — most specific, avoids cross-source collisions
        2. (Plate, Well) — sufficient if plates have unique IDs
        3. (JCP2022,) — perturbation-level join (for orf/crispr tables)

        Args:
            profiles: Profile DataFrame.
            meta: Metadata DataFrame to merge.
            label: Label for logging.

        Returns:
            Merged DataFrame (left join, preserving all profile rows).
        """

        def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        source_candidates = ["Metadata_Source", "Metadata_source", "source"]
        plate_candidates = ["Metadata_Plate", "Metadata_plate", "plate"]
        well_candidates = ["Metadata_Well", "Metadata_well", "well"]
        jcp_candidates = ["Metadata_JCP2022", "Metadata_jcp2022"]

        # Find matching columns in both DataFrames
        p_source = _find_col(profiles, source_candidates)
        p_plate = _find_col(profiles, plate_candidates)
        p_well = _find_col(profiles, well_candidates)
        p_jcp = _find_col(profiles, jcp_candidates)

        m_source = _find_col(meta, source_candidates)
        m_plate = _find_col(meta, plate_candidates)
        m_well = _find_col(meta, well_candidates)
        m_jcp = _find_col(meta, jcp_candidates)

        # Determine the best merge strategy
        merge_on = []

        # Strategy 1: Spatial keys (Source + Plate + Well)
        if all([p_source, p_plate, p_well, m_source, m_plate, m_well]):
            # All spatial keys available in both — best case
            merge_on = [
                (p_source, m_source),
                (p_plate, m_plate),
                (p_well, m_well),
            ]
        elif all([p_plate, p_well, m_plate, m_well]):
            # Plate + Well available
            merge_on = [
                (p_plate, m_plate),
                (p_well, m_well),
            ]
        # Strategy 2: JCP2022 identifier (for orf/crispr table joins)
        elif p_jcp and m_jcp:
            merge_on = [(p_jcp, m_jcp)]
        else:
            logger.warning(
                f"No viable merge keys for '{label}'. "
                f"Profile cols: {[c for c in profiles.columns if c.startswith('Metadata_')]} "
                f"Meta cols: {list(meta.columns)}"
            )
            return profiles

        left_keys = [k[0] for k in merge_on]
        right_keys = [k[1] for k in merge_on]

        logger.debug(
            f"Merge '{label}': left_keys={left_keys}, right_keys={right_keys}"
        )

        # Only bring in columns not already in profiles
        existing_cols = set(profiles.columns)
        new_meta_cols = [c for c in meta.columns if c not in existing_cols]
        keep_cols = right_keys + new_meta_cols
        # Deduplicate while preserving order
        seen = set()
        keep_cols_dedup = []
        for c in keep_cols:
            if c not in seen:
                keep_cols_dedup.append(c)
                seen.add(c)

        meta_subset = meta[keep_cols_dedup].drop_duplicates(subset=right_keys)

        n_before = len(profiles)

        if left_keys == right_keys:
            merged = profiles.merge(meta_subset, on=left_keys, how="left")
        else:
            merged = profiles.merge(
                meta_subset,
                left_on=left_keys,
                right_on=right_keys,
                how="left",
            )
            # Drop duplicate key columns from the right side
            for lk, rk in zip(left_keys, right_keys):
                if lk != rk and rk in merged.columns:
                    merged = merged.drop(columns=[rk])

        n_new_cols = len(merged.columns) - len(profiles.columns)
        if n_new_cols > 0:
            new_col_names = [c for c in merged.columns if c not in existing_cols]
            n_matched = merged[new_col_names[0]].notna().sum() if new_col_names else 0
        else:
            n_matched = 0

        logger.info(
            f"Merged '{label}': keys={left_keys}, "
            f"{n_matched}/{n_before} wells matched, "
            f"{n_new_cols} new columns"
        )

        return merged

    # =========================================================================
    # Gene detection
    # =========================================================================

    @staticmethod
    def _detect_gene_column(df: pd.DataFrame) -> Optional[str]:
        """Detect the gene identifier column."""
        # Exact matches in priority order (Symbol first, JCP2022 last)
        for c in [
            "Metadata_Symbol",
            "Metadata_Gene",
            "Metadata_gene",
            "Metadata_pert_iname",
            "Metadata_broad_sample",
            "Metadata_JCP2022",
        ]:
            if c in df.columns:
                return c

        # Heuristic: any metadata column containing "symbol" or "gene"
        for c in df.columns:
            cl = c.lower()
            if cl.startswith("metadata_") and ("symbol" in cl or "gene" in cl):
                return c

        return None

    # =========================================================================
    # Gene sampling strategies
    # =========================================================================

    def _sample_genes(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Apply diversity or coverage sampling to reduce the gene set."""
        gene_col = self._detect_gene_column(df)
        if gene_col is None:
            logger.warning("Cannot detect gene column; skipping sampling")
            return df

        target = self.config.target_n_genes
        n_available = df[gene_col].nunique()

        if target >= n_available:
            logger.info(
                f"Target ({target}) ≥ available ({n_available}); keeping all"
            )
            return df

        if self.config.sampling_strategy == "diversity":
            return self._diversity_sample(df, gene_col, feature_cols, target)
        elif self.config.sampling_strategy == "coverage":
            return self._coverage_sample(df, gene_col, target)
        else:
            return df

    def _diversity_sample(
        self,
        df: pd.DataFrame,
        gene_col: str,
        feature_cols: List[str],
        target: int,
    ) -> pd.DataFrame:
        """
        Select genes that maximise phenotypic diversity.

        Uses a greedy farthest-point sampling approach on the
        treatment-level feature vectors.
        """
        logger.info("Diversity sampling via farthest-point selection")

        genes = df[gene_col].unique()
        gene_profiles = (
            df.groupby(gene_col)[feature_cols].mean().loc[genes]
        )
        profile_matrix = gene_profiles.values.astype(np.float32)

        # Greedy farthest-point sampling
        n = len(genes)
        selected_mask = np.zeros(n, dtype=bool)

        # Start with the gene closest to the overall centroid
        centroid = profile_matrix.mean(axis=0, keepdims=True)
        dists_to_centroid = np.linalg.norm(profile_matrix - centroid, axis=1)
        first = int(np.argmin(dists_to_centroid))
        selected_mask[first] = True

        # Track minimum distance to any selected point
        min_dist_to_selected = np.full(n, np.inf)
        min_dist_to_selected = np.minimum(
            min_dist_to_selected,
            np.linalg.norm(profile_matrix - profile_matrix[first], axis=1),
        )

        for _ in range(target - 1):
            candidates = np.where(~selected_mask)[0]
            farthest = candidates[np.argmax(min_dist_to_selected[candidates])]
            selected_mask[farthest] = True

            new_dists = np.linalg.norm(
                profile_matrix - profile_matrix[farthest], axis=1
            )
            min_dist_to_selected = np.minimum(min_dist_to_selected, new_dists)

        selected_genes = set(genes[selected_mask])
        result = df[df[gene_col].isin(selected_genes)].copy()

        logger.info(
            f"Diversity sampling: selected {len(selected_genes)} genes "
            f"({len(result)} rows)"
        )
        return result

    def _coverage_sample(
        self,
        df: pd.DataFrame,
        gene_col: str,
        target: int,
    ) -> pd.DataFrame:
        """
        Select genes to maximise coverage across plates.

        Genes that appear on more plates are preferred, then random
        tie-breaking up to *target*.
        """
        logger.info("Coverage sampling (plate-diversity priority)")

        plate_col = None
        for c in ["Metadata_Plate", "Metadata_plate"]:
            if c in df.columns:
                plate_col = c
                break

        if plate_col is None:
            genes = df[gene_col].unique()
            rng = np.random.default_rng(42)
            selected = rng.choice(genes, size=min(target, len(genes)), replace=False)
            return df[df[gene_col].isin(selected)].copy()

        plate_coverage = (
            df.groupby(gene_col)[plate_col]
            .nunique()
            .reset_index(name="n_plates")
            .sort_values("n_plates", ascending=False)
        )

        selected_genes = set(plate_coverage[gene_col].iloc[:target].tolist())
        result = df[df[gene_col].isin(selected_genes)].copy()

        logger.info(
            f"Coverage sampling: selected {len(selected_genes)} genes "
            f"({len(result)} rows)"
        )
        return result

    def _limit_genes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limit to max_genes, preferring those with more replicates."""
        gene_col = self._detect_gene_column(df)
        if gene_col is None:
            return df

        rep_col = "_n_replicates"
        if rep_col in df.columns:
            ranking = (
                df[[gene_col, rep_col]]
                .drop_duplicates()
                .sort_values(rep_col, ascending=False)
            )
        else:
            ranking = pd.DataFrame({gene_col: df[gene_col].unique()})

        keep_genes = set(ranking[gene_col].iloc[: self.config.max_genes])
        result = df[df[gene_col].isin(keep_genes)].copy()
        logger.info(f"Limited to {len(keep_genes)} genes ({len(result)} rows)")
        return result

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        name: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Save the curated dataset and associated artefacts.

        Creates:
        - ``{name}.parquet``: The curated profiles.
        - ``{name}_report.json``: Curation statistics.
        - ``{name}_features.json``: Selected feature names.

        Args:
            df: Curated DataFrame from :meth:`build_pretraining_set`.
            output_dir: Output directory.
            name: Dataset name (defaults to ``config.output_name``).

        Returns:
            Dictionary of output file paths.
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = name or self.config.output_name

        paths: Dict[str, Path] = {}

        # Save profiles
        profile_path = output_dir / f"{name}.parquet"
        df.to_parquet(profile_path, index=False)
        paths["profiles"] = profile_path
        logger.info(f"Saved profiles to {profile_path}")

        # Also store in the JUMP-CP cache
        self.access.cache.store_curated_dataset(name, df, source="curation_pipeline")

        # Save report
        report_path = output_dir / f"{name}_report.json"
        with open(report_path, "w") as f:
            serialisable_report = {}
            for k, v in self._curation_report.items():
                if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                    serialisable_report[k] = v
                else:
                    serialisable_report[k] = str(v)
            json.dump(serialisable_report, f, indent=2)
        paths["report"] = report_path

        # Save feature names
        feature_path = output_dir / f"{name}_features.json"
        feature_cols = self.loader.feature_columns or []
        with open(feature_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        paths["features"] = feature_path

        logger.info(f"Curation artefacts saved to {output_dir}")
        return paths

    @classmethod
    def load_curated(
        cls,
        path: Union[str, Path],
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        """
        Load a previously curated dataset.

        Args:
            path: Path to the ``.parquet`` file.

        Returns:
            Tuple of (DataFrame, report_dict, feature_names).
        """
        import json

        path = Path(path)
        df = pd.read_parquet(path)

        report_path = path.with_name(path.stem + "_report.json")
        report = {}
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)

        features_path = path.with_name(path.stem + "_features.json")
        features: List[str] = []
        if features_path.exists():
            with open(features_path) as f:
                features = json.load(f)

        logger.info(
            f"Loaded curated dataset from {path}: "
            f"{len(df)} treatments, {len(features)} features"
        )
        return df, report, features

    # =========================================================================
    # Conversion to PhenotypeDataset
    # =========================================================================

    def to_phenotype_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> PhenotypeDataset:
        """
        Convenience wrapper around :meth:`ProfileLoader.to_phenotype_dataset`.

        Args:
            df: Curated DataFrame.
            dataset_name: Name for the dataset.

        Returns:
            :class:`PhenotypeDataset` instance.
        """
        return self.loader.to_phenotype_dataset(
            df,
            dataset_name=dataset_name or self.config.output_name,
        )

    # =========================================================================
    # Report
    # =========================================================================

    @property
    def curation_report(self) -> Dict[str, Any]:
        """Return the report from the most recent curation run."""
        return self._curation_report

    def print_report(self) -> None:
        """Pretty-print the curation report."""
        if not self._curation_report:
            logger.info("No curation report available (run build_pretraining_set first)")
            return

        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="JUMP-CP Curation Report")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in self._curation_report.items():
                if key == "config":
                    continue
                table.add_row(str(key), str(value))

            console.print(table)
        except ImportError:
            for key, value in self._curation_report.items():
                if key == "config":
                    continue
                logger.info(f"  {key}: {value}")

    def __repr__(self) -> str:
        return (
            f"DataCurator(perturbation_types={self.config.perturbation_types}, "
            f"strategy='{self.config.sampling_strategy}')"
        )