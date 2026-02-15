"""
Tests for JUMP-CP data access and curation.

Tests are organised in tiers:
- Unit tests: No network access, test logic with mocked data.
- Integration tests (marked @pytest.mark.slow): Require network access to S3/HTTPS.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from protophen.data.phenotype import PhenotypeDataset

from protophen.data.jumpcp.access import JUMPCPAccess, JUMPCPConfig
from protophen.data.jumpcp.cache import CacheEntry, JUMPCPCache
from protophen.data.jumpcp.curation import CurationConfig, DataCurator, QualityController
from protophen.data.jumpcp.metadata import JUMPCPMetadata
from protophen.data.jumpcp.profiles import (
    ProfileLoader,
    identify_feature_columns,
    identify_metadata_columns,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    return tmp_path / "jumpcp_cache"


@pytest.fixture
def cache(tmp_cache_dir):
    """Provide a JUMPCPCache instance."""
    return JUMPCPCache(cache_dir=tmp_cache_dir)


@pytest.fixture
def config(tmp_cache_dir):
    """Provide a JUMPCPConfig with temp cache."""
    return JUMPCPConfig(cache_dir=str(tmp_cache_dir))


@pytest.fixture
def mock_plate_df():
    """Create a mock JUMP-CP plate profile DataFrame."""
    n_wells = 384
    n_features = 50

    # Metadata columns
    data = {
        "Metadata_Plate": ["TEST_PLATE"] * n_wells,
        "Metadata_Well": [
            f"{chr(65 + i // 24)}{(i % 24) + 1:02d}" for i in range(n_wells)
        ],
        "Metadata_Source": ["source_4"] * n_wells,
        "Metadata_Batch": ["2021_08_test"] * n_wells,
        "Metadata_JCP2022": [f"JCP2022_{i % 50:06d}" for i in range(n_wells)],
        "Metadata_Count_Cells": np.random.randint(20, 500, size=n_wells),
    }

    # Feature columns across compartments and categories
    compartments = ["Cells", "Cytoplasm", "Nuclei"]
    categories = ["AreaShape", "Intensity", "Texture"]
    feature_names = []
    for comp in compartments:
        for cat in categories:
            for j in range(n_features // (len(compartments) * len(categories)) + 1):
                fname = f"{comp}_{cat}_Feature_{j}"
                feature_names.append(fname)
                if len(feature_names) >= n_features:
                    break
            if len(feature_names) >= n_features:
                break
        if len(feature_names) >= n_features:
            break

    feature_names = feature_names[:n_features]
    rng = np.random.default_rng(42)
    for fname in feature_names:
        data[fname] = rng.normal(0, 1, size=n_wells).astype(np.float32)

    # Inject a few NaNs
    data[feature_names[0]][0] = np.nan

    return pd.DataFrame(data)


@pytest.fixture
def mock_orf_metadata():
    """Create mock ORF metadata."""
    rng = np.random.default_rng(123)
    genes = [f"GENE_{i}" for i in range(100)]
    records = []
    for gene in genes:
        n_reps = rng.integers(1, 6)
        for rep in range(n_reps):
            records.append({
                "Metadata_JCP2022": f"JCP2022_{genes.index(gene):06d}",
                "Metadata_Symbol": gene,
                "Metadata_Plate": f"PLATE_{rep % 3}",
                "Metadata_Well": f"A{rep + 1:02d}",
                "Metadata_Source": "source_4",
                "Metadata_Batch": f"batch_{rep % 2}",
            })
    return pd.DataFrame(records)


@pytest.fixture
def mock_plate_metadata():
    """Create mock plate metadata."""
    records = []
    for i in range(10):
        records.append({
            "Metadata_Plate": f"PLATE_{i}",
            "Metadata_Source": "source_4",
            "Metadata_Batch": f"batch_{i % 2}",
            "Metadata_Perturbation_Type": "orf" if i < 7 else "crispr",
        })
    return pd.DataFrame(records)


@pytest.fixture
def mock_well_metadata():
    """Create mock well-level metadata."""
    records = []
    for plate_i in range(3):
        for well_i in range(96):
            records.append({
                "Metadata_Plate": f"PLATE_{plate_i}",
                "Metadata_Well": f"{chr(65 + well_i // 12)}{(well_i % 12) + 1:02d}",
                "Metadata_JCP2022": f"JCP2022_{well_i % 50:06d}",
                "Metadata_Source": "source_4",
            })
    return pd.DataFrame(records)


@pytest.fixture
def multi_plate_df():
    """Create a DataFrame spanning multiple plates for batch correction tests."""
    rng = np.random.default_rng(99)
    n_wells_per_plate = 96
    n_plates = 3
    n_features = 20

    compartments = ["Cells", "Nuclei"]
    feature_names = []
    for comp in compartments:
        for j in range(n_features // len(compartments)):
            feature_names.append(f"{comp}_Intensity_Feature_{j}")

    records = []
    for p in range(n_plates):
        plate_shift = rng.normal(0, 2, size=len(feature_names))
        for w in range(n_wells_per_plate):
            row = {
                "Metadata_Plate": f"PLATE_{p}",
                "Metadata_Well": f"{chr(65 + w // 12)}{(w % 12) + 1:02d}",
                "Metadata_JCP2022": f"JCP2022_{w % 30:06d}",
                "Metadata_Source": "source_4",
                "Metadata_Batch": f"batch_{p % 2}",
                "Metadata_Count_Cells": rng.integers(50, 500),
            }
            features = rng.normal(0, 1, size=len(feature_names)) + plate_shift
            for fname, val in zip(feature_names, features):
                row[fname] = float(val)
            records.append(row)

    return pd.DataFrame(records)


# =============================================================================
# Cache Tests
# =============================================================================

class TestJUMPCPCache:
    """Tests for the local caching system."""

    def test_init(self, cache, tmp_cache_dir):
        assert cache.cache_dir == tmp_cache_dir
        assert len(cache) == 0

    def test_store_and_get_metadata(self, cache):
        df = pd.DataFrame({"gene": ["A", "B"], "value": [1, 2]})
        cache.store_metadata("test_table", df, source="test")

        assert cache.has_metadata("test_table")
        loaded = cache.get_metadata("test_table")
        assert loaded is not None
        assert len(loaded) == 2
        assert list(loaded.columns) == ["gene", "value"]

    def test_store_and_get_profiles(self, cache):
        df = pd.DataFrame({"well": ["A01"], "feature_1": [0.5]})
        cache.store_profiles("plate_001", df)

        assert cache.has_profiles("plate_001")
        loaded = cache.get_profiles("plate_001")
        assert loaded is not None
        assert len(loaded) == 1

    def test_missing_key_returns_none(self, cache):
        assert cache.get_metadata("nonexistent") is None
        assert cache.get_profiles("nonexistent") is None

    def test_curated_dataset_roundtrip(self, cache):
        df = pd.DataFrame({"gene": ["X", "Y"], "feat": [1.0, 2.0]})
        cache.store_curated_dataset("v1", df)

        loaded = cache.get_curated_dataset("v1")
        assert loaded is not None
        assert len(loaded) == 2
        assert list(loaded.columns) == ["gene", "feat"]

    def test_curated_dataset_missing(self, cache):
        assert cache.get_curated_dataset("nonexistent") is None

    def test_cache_info(self, cache):
        df = pd.DataFrame({"a": [1]})
        cache.store_metadata("m1", df)
        cache.store_profiles("p1", df)

        info = cache.get_cache_info()
        assert info["total_entries"] == 2
        assert info["n_metadata"] == 1
        assert info["n_profiles"] == 1
        assert info["total_size_gb"] >= 0

    def test_clear(self, cache):
        df = pd.DataFrame({"a": [1]})
        cache.store_metadata("m1", df)
        cache.store_profiles("p1", df)
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert cache.get_metadata("m1") is None
        assert cache.get_profiles("p1") is None

    def test_evict_lru(self, cache):
        # Store several entries
        for i in range(5):
            df = pd.DataFrame({"col": np.random.randn(1000)})
            cache.store_profiles(f"plate_{i}", df)

        initial_count = len(cache)
        assert initial_count == 5

        # Evict to very small target
        evicted = cache.evict_lru(target_size_gb=0.0)
        assert evicted > 0
        assert len(cache) < initial_count

    def test_evict_lru_noop_when_under_limit(self, cache):
        df = pd.DataFrame({"a": [1]})
        cache.store_metadata("m1", df)
        evicted = cache.evict_lru(target_size_gb=100.0)
        assert evicted == 0
        assert len(cache) == 1

    def test_cache_entry_serialisation(self):
        entry = CacheEntry(
            key="test",
            path="/tmp/test.parquet",
            size_bytes=1024,
            n_rows=10,
            n_cols=5,
            created_at=1000.0,
            accessed_at=1001.0,
            source="s3://test",
        )
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        assert restored.key == entry.key
        assert restored.size_bytes == entry.size_bytes
        assert restored.source == entry.source

    def test_repr(self, cache):
        repr_str = repr(cache)
        assert "JUMPCPCache" in repr_str
        assert "entries=" in repr_str

    def test_deleted_file_handled_gracefully(self, cache):
        """Cache returns None if underlying file has been deleted."""
        df = pd.DataFrame({"a": [1]})
        cache.store_metadata("ephemeral", df)
        assert cache.has_metadata("ephemeral")

        # Delete the underlying file
        entry = cache._index[cache._make_key("metadata", "ephemeral")]
        Path(entry.path).unlink()

        # Should return None and clean up the index
        result = cache.get_metadata("ephemeral")
        assert result is None
        assert not cache.has_metadata("ephemeral")


# =============================================================================
# Feature Identification Tests
# =============================================================================

class TestFeatureIdentification:
    """Tests for profile column detection."""

    def test_identify_feature_columns(self, mock_plate_df):
        features = identify_feature_columns(mock_plate_df)
        assert len(features) > 0
        for f in features:
            assert f.startswith(("Cells_", "Cytoplasm_", "Nuclei_"))

    def test_identify_feature_columns_empty_df(self):
        df = pd.DataFrame({"Metadata_Plate": ["A"], "Other_Col": [1]})
        features = identify_feature_columns(df)
        assert features == []

    def test_identify_metadata_columns(self, mock_plate_df):
        meta_cols = identify_metadata_columns(mock_plate_df)
        assert "Metadata_Plate" in meta_cols
        assert "Metadata_Well" in meta_cols
        assert all(c.startswith("Metadata_") for c in meta_cols)

    def test_identify_metadata_columns_empty(self):
        df = pd.DataFrame({"feature_1": [1], "feature_2": [2]})
        meta_cols = identify_metadata_columns(df)
        assert meta_cols == []

    def test_blocklist_removal(self, mock_plate_df):
        # Add a blocklisted column
        mock_plate_df["Cells_Location_Center_X"] = 0.0

        features_with_blocklist = identify_feature_columns(
            mock_plate_df, remove_blocklist=False
        )
        features_without_blocklist = identify_feature_columns(
            mock_plate_df, remove_blocklist=True
        )
        assert len(features_without_blocklist) < len(features_with_blocklist)
        assert "Cells_Location_Center_X" not in features_without_blocklist
        assert "Cells_Location_Center_X" in features_with_blocklist

    def test_compartment_filtering(self, mock_plate_df):
        features_all = identify_feature_columns(mock_plate_df)
        features_cells = identify_feature_columns(
            mock_plate_df, compartments=["Cells"]
        )
        features_nuclei = identify_feature_columns(
            mock_plate_df, compartments=["Nuclei"]
        )

        assert len(features_cells) < len(features_all)
        assert len(features_nuclei) < len(features_all)
        assert all(f.startswith("Cells_") for f in features_cells)
        assert all(f.startswith("Nuclei_") for f in features_nuclei)

    def test_features_are_sorted(self, mock_plate_df):
        features = identify_feature_columns(mock_plate_df)
        assert features == sorted(features)


# =============================================================================
# Quality Control Tests
# =============================================================================

class TestQualityController:
    """Tests for well-level QC."""

    def test_flag_wells_all_pass(self, mock_plate_df):
        features = identify_feature_columns(mock_plate_df)
        qc = QualityController(max_nan_fraction=0.5, min_cell_count=10)
        mask = qc.flag_wells(mock_plate_df, features)
        assert mask.sum() > 0

    def test_flag_wells_cell_count(self, mock_plate_df):
        features = identify_feature_columns(mock_plate_df)
        # Set a very high threshold to fail most wells
        qc = QualityController(min_cell_count=1000)
        mask = qc.flag_wells(mock_plate_df, features)
        # Some wells should fail
        assert mask.sum() < len(mock_plate_df)

    def test_flag_wells_nan_fraction(self):
        df = pd.DataFrame({
            "Cells_Feature_1": [1.0, np.nan, np.nan],
            "Cells_Feature_2": [1.0, np.nan, np.nan],
            "Cells_Feature_3": [1.0, 1.0, np.nan],
        })
        features = ["Cells_Feature_1", "Cells_Feature_2", "Cells_Feature_3"]
        qc = QualityController(max_nan_fraction=0.5)
        mask = qc.flag_wells(df, features)
        # Row 0: 0/3 NaN → pass
        # Row 1: 2/3 NaN → fail (0.67 > 0.5)
        # Row 2: 2/3 NaN → fail (0.67 > 0.5)
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == False

    def test_flag_wells_no_cell_count_column(self):
        """QC should still work when no cell-count column exists."""
        df = pd.DataFrame({
            "Cells_Feature_1": [1.0, 2.0, 3.0],
            "Metadata_Plate": ["P1", "P1", "P1"],
        })
        features = ["Cells_Feature_1"]
        qc = QualityController(min_cell_count=100)
        mask = qc.flag_wells(df, features)
        # Without the cell count column, all should pass that check
        assert mask.sum() == 3

    def test_flag_wells_returns_correct_length(self, mock_plate_df):
        features = identify_feature_columns(mock_plate_df)
        qc = QualityController()
        mask = qc.flag_wells(mock_plate_df, features)
        assert len(mask) == len(mock_plate_df)
        
    def test_flag_wells_norm_outlier(self):
        """Wells with extreme profile norms should be flagged."""
        n_features = 10
        rng = np.random.default_rng(42)
        
        # 98 normal wells + 2 extreme outliers
        normal = rng.normal(0, 1, size=(98, n_features))
        outlier_high = np.full((1, n_features), 100.0)
        outlier_low = np.full((1, n_features), 0.0)
        
        features = np.vstack([normal, outlier_high, outlier_low])
        cols = [f"Cells_Feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(features, columns=cols)
        
        qc = QualityController(norm_outlier_mad=3.0)
        mask = qc.flag_wells(df, cols)
        
        # The extreme wells should be flagged
        assert mask.sum() < 100
        # Specifically, the high-norm outlier (index 98) should fail
        assert mask.iloc[98] == False

    def test_flag_wells_no_cell_count_no_crash(self):
        """QC should work fine with aggregated profiles lacking cell counts."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Cells_Feature_1": rng.normal(0, 1, 50),
            "Cells_Feature_2": rng.normal(0, 1, 50),
            "Metadata_Plate": ["P1"] * 50,
            "Metadata_Well": [f"A{i:02d}" for i in range(50)],
        })
        features = ["Cells_Feature_1", "Cells_Feature_2"]
        
        qc = QualityController(min_cell_count=100)
        mask = qc.flag_wells(df, features)
        
        # Should not crash; all should pass (no cell count col, norms are normal)
        assert len(mask) == 50
        assert mask.sum() > 0


# =============================================================================
# Config Tests
# =============================================================================

class TestJUMPCPConfig:
    """Tests for configuration loading and serialisation."""

    def test_defaults(self):
        config = JUMPCPConfig()
        assert config.s3_bucket == "cellpainting-gallery"
        assert config.s3_region == "us-east-1"
        assert "orf" in config.perturbation_types
        assert config.organism_id == "9606"
        assert config.min_replicates == 2
        assert config.min_cell_count == 50

    def test_to_dict(self):
        config = JUMPCPConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "s3_bucket" in d
        assert "cache_dir" in d
        assert "metadata_urls" in d

    def test_from_yaml(self, tmp_path):
        yaml_content = """
s3_bucket: "test-bucket"
cache_dir: "/tmp/test_cache"
min_replicates: 3
max_retries: 5
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        config = JUMPCPConfig.from_yaml(yaml_path)
        assert config.s3_bucket == "test-bucket"
        assert config.cache_dir == "/tmp/test_cache"
        assert config.min_replicates == 3
        assert config.max_retries == 5
        # Defaults preserved for non-specified fields
        assert config.s3_region == "us-east-1"
        assert config.organism_id == "9606"

    def test_from_yaml_empty_file(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        config = JUMPCPConfig.from_yaml(yaml_path)
        # Should return all defaults
        assert config.s3_bucket == "cellpainting-gallery"

    def test_from_yaml_ignores_unknown_keys(self, tmp_path):
        yaml_content = """
s3_bucket: "test-bucket"
unknown_key: "should_be_ignored"
"""
        yaml_path = tmp_path / "extra_keys.yaml"
        yaml_path.write_text(yaml_content)
        config = JUMPCPConfig.from_yaml(yaml_path)
        assert config.s3_bucket == "test-bucket"
        assert not hasattr(config, "unknown_key")


class TestCurationConfig:
    """Tests for curation configuration."""

    def test_defaults(self):
        config = CurationConfig()
        assert config.perturbation_types == ["orf"]
        assert config.min_replicates == 2
        assert config.normalisation_method == "robust_mad"
        assert config.batch_correction is True
        assert config.sampling_strategy == "all"
        assert config.output_name == "pretraining_v1"

    def test_custom_values(self):
        config = CurationConfig(
            perturbation_types=["orf", "crispr"],
            min_replicates=3,
            max_genes=5000,
            sampling_strategy="diversity",
            target_n_genes=3000,
        )
        assert config.perturbation_types == ["orf", "crispr"]
        assert config.min_replicates == 3
        assert config.max_genes == 5000
        assert config.target_n_genes == 3000


# =============================================================================
# Metadata Tests (with mocks)
# =============================================================================

class TestJUMPCPMetadata:
    """Tests for metadata parsing and filtering (mocked data)."""

    def test_get_orf_genes(self, mock_orf_metadata, config):
        access = JUMPCPAccess(config=config)
        meta = JUMPCPMetadata(access=access)
        meta._tables["orf"] = mock_orf_metadata

        genes = meta.get_orf_genes()
        assert len(genes) == 100
        assert all(g.startswith("GENE_") for g in genes)
        # Should be sorted
        assert genes == sorted(genes)

    def test_count_replicates(self, mock_orf_metadata, config):
        access = JUMPCPAccess(config=config)
        meta = JUMPCPMetadata(access=access)
        meta._tables["orf"] = mock_orf_metadata

        counts = meta.count_replicates("orf")
        assert "gene" in counts.columns
        assert "n_replicates" in counts.columns
        assert len(counts) == 100
        # Should be sorted descending
        reps = counts["n_replicates"].values
        assert reps[0] >= reps[-1]

    def test_filter_genes_by_replicates(self, mock_orf_metadata, config):
        access = JUMPCPAccess(config=config)
        meta = JUMPCPMetadata(access=access)
        meta._tables["orf"] = mock_orf_metadata

        all_genes = meta.get_orf_genes()
        filtered_2 = meta.filter_genes_by_replicates("orf", min_replicates=2)
        filtered_5 = meta.filter_genes_by_replicates("orf", min_replicates=5)

        assert len(filtered_2) <= len(all_genes)
        assert len(filtered_5) <= len(filtered_2)

    def test_get_plates_for_perturbation(self, mock_plate_metadata, config):
        access = JUMPCPAccess(config=config)
        meta = JUMPCPMetadata(access=access)
        meta._tables["plate"] = mock_plate_metadata

        orf_plates = meta.get_plates_for_perturbation("orf")
        assert len(orf_plates) == 7

        crispr_plates = meta.get_plates_for_perturbation("crispr")
        assert len(crispr_plates) == 3

    def test_get_well_map(self, mock_well_metadata, config):
        access = JUMPCPAccess(config=config)
        meta = JUMPCPMetadata(access=access)
        meta._tables["well"] = mock_well_metadata

        wells = meta.get_well_map("PLATE_0")
        assert len(wells) == 96
        assert all(wells["Metadata_Plate"] == "PLATE_0")

    def test_get_well_map_nonexistent_plate(self, mock_well_metadata, config):
        access = JUMPCPAccess(config=config)
        meta = JUMPCPMetadata(access=access)
        meta._tables["well"] = mock_well_metadata

        wells = meta.get_well_map("NONEXISTENT")
        assert len(wells) == 0

    def test_summary(self, mock_orf_metadata, mock_plate_metadata, config):
        access = JUMPCPAccess(config=config)
        meta = JUMPCPMetadata(access=access)
        meta._tables["orf"] = mock_orf_metadata
        meta._tables["plate"] = mock_plate_metadata

        info = meta.summary()
        assert info["n_orf_genes"] == 100
        assert info["n_plates"] == 10

    def test_count_replicates_invalid_type(self, config):
        access = JUMPCPAccess(config=config)
        meta = JUMPCPMetadata(access=access)

        with pytest.raises(ValueError, match="Unsupported perturbation type"):
            meta.count_replicates("compound")

    def test_repr(self, config):
        meta = JUMPCPMetadata(config=config)
        assert "JUMPCPMetadata" in repr(meta)


# =============================================================================
# Profile Loader Tests
# =============================================================================

class TestProfileLoader:
    """Tests for profile loading and aggregation."""

    def test_to_phenotype_dataset(self, mock_plate_df, config):
        access = JUMPCPAccess(config=config)
        loader = ProfileLoader(access=access)
        loader._feature_columns = identify_feature_columns(mock_plate_df)

        dataset = loader.to_phenotype_dataset(
            mock_plate_df,
            gene_col="Metadata_JCP2022",
        )
        assert len(dataset) == len(mock_plate_df)
        assert dataset.feature_names is not None
        assert len(dataset.feature_names) > 0
        assert dataset.name == "jumpcp"

    def test_to_phenotype_dataset_custom_name(self, mock_plate_df, config):
        access = JUMPCPAccess(config=config)
        loader = ProfileLoader(access=access)
        loader._feature_columns = identify_feature_columns(mock_plate_df)

        dataset = loader.to_phenotype_dataset(
            mock_plate_df,
            gene_col="Metadata_JCP2022",
            dataset_name="my_dataset",
        )
        assert dataset.name == "my_dataset"

    def test_to_phenotype_dataset_features_valid(self, mock_plate_df, config):
        access = JUMPCPAccess(config=config)
        loader = ProfileLoader(access=access)
        loader._feature_columns = identify_feature_columns(mock_plate_df)

        dataset = loader.to_phenotype_dataset(
            mock_plate_df,
            gene_col="Metadata_JCP2022",
        )
        # Check that feature vectors are the right shape
        first = dataset[0]
        assert first.n_features == len(loader._feature_columns)
        assert not np.any(np.isnan(first.features))

    def test_aggregate_to_treatments(self, mock_plate_df, config):
        access = JUMPCPAccess(config=config)
        loader = ProfileLoader(access=access)
        loader._feature_columns = identify_feature_columns(mock_plate_df)

        agg = loader.aggregate_to_treatments(
            mock_plate_df,
            group_by=["Metadata_JCP2022"],
            min_replicates=1,
        )
        n_unique_genes = mock_plate_df["Metadata_JCP2022"].nunique()
        assert len(agg) <= n_unique_genes
        assert "_n_replicates" in agg.columns

    def test_aggregate_min_replicates(self, mock_plate_df, config):
        access = JUMPCPAccess(config=config)
        loader = ProfileLoader(access=access)
        loader._feature_columns = identify_feature_columns(mock_plate_df)

        agg_loose = loader.aggregate_to_treatments(
            mock_plate_df,
            group_by=["Metadata_JCP2022"],
            min_replicates=1,
        )
        agg_strict = loader.aggregate_to_treatments(
            mock_plate_df,
            group_by=["Metadata_JCP2022"],
            min_replicates=5,
        )
        assert len(agg_strict) <= len(agg_loose)

    def test_aggregate_method_mean_vs_median(self, mock_plate_df, config):
        access = JUMPCPAccess(config=config)
        loader = ProfileLoader(access=access)
        loader._feature_columns = identify_feature_columns(mock_plate_df)

        agg_mean = loader.aggregate_to_treatments(
            mock_plate_df,
            group_by=["Metadata_JCP2022"],
            method="mean",
            min_replicates=1,
        )
        agg_median = loader.aggregate_to_treatments(
            mock_plate_df,
            group_by=["Metadata_JCP2022"],
            method="median",
            min_replicates=1,
        )
        # Both should have the same number of treatments
        assert len(agg_mean) == len(agg_median)
        # But different values (generally)
        feat_col = loader._feature_columns[0]
        # Not guaranteed to differ for all, but shapes should match
        assert agg_mean[feat_col].shape == agg_median[feat_col].shape

    def test_aggregate_no_grouping_column_raises(self, config):
        access = JUMPCPAccess(config=config)
        loader = ProfileLoader(access=access)
        loader._feature_columns = ["Cells_Feature_1"]

        df = pd.DataFrame({
            "Cells_Feature_1": [1.0, 2.0],
            "Metadata_Plate": ["P1", "P1"],
        })
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            loader.aggregate_to_treatments(df, min_replicates=1)

    def test_repr(self, config):
        loader = ProfileLoader(config=config)
        assert "ProfileLoader" in repr(loader)


# =============================================================================
# Access Tests (unit-level, no network)
# =============================================================================

class TestJUMPCPAccess:
    """Tests for access utilities that don't require network."""

    def test_init(self, config):
        access = JUMPCPAccess(config=config)
        assert access.config.s3_bucket == "cellpainting-gallery"

    def test_get_profile_s3_path(self, config):
        access = JUMPCPAccess(config=config)
        path = access.get_profile_s3_path("source_4", "batch_1", "plate_A")
        assert "source_4" in path
        assert "batch_1" in path
        assert "plate_A" in path
        assert path.endswith(".parquet")

    def test_fetch_metadata_from_cache(self, config, cache):
        # Pre-populate cache
        df = pd.DataFrame({"gene": ["A", "B"]})
        cache.store_metadata("orf", df)

        access = JUMPCPAccess(config=config, cache=cache)
        loaded = access.fetch_metadata_table("orf")
        assert len(loaded) == 2

    def test_fetch_plate_profiles_from_cache(self, config, cache, mock_plate_df):
        cache_key = "source_4__batch_1__plate_A"
        cache.store_profiles(cache_key, mock_plate_df)

        access = JUMPCPAccess(config=config, cache=cache)
        loaded = access.fetch_plate_profiles(
            "source_4", "batch_1", "plate_A"
        )
        assert len(loaded) == len(mock_plate_df)

    def test_repr(self, config):
        access = JUMPCPAccess(config=config)
        assert "JUMPCPAccess" in repr(access)


# =============================================================================
# Data Curator Tests
# =============================================================================

class TestDataCurator:
    """Tests for the full curation pipeline (mocked data)."""

    def _make_curator_with_mocks(
        self,
        config: JUMPCPConfig,
        mock_plate_df: pd.DataFrame,
        mock_orf_metadata: pd.DataFrame,
        mock_plate_metadata: pd.DataFrame,
    ) -> DataCurator:
        """Create a DataCurator with mocked data access."""
        access = JUMPCPAccess(config=config)
        metadata = JUMPCPMetadata(access=access)

        # Inject mock metadata tables
        metadata._tables["orf"] = mock_orf_metadata
        metadata._tables["plate"] = mock_plate_metadata

        curation_config = CurationConfig(
            perturbation_types=["orf"],
            max_plates=1,  # Only process mock data
            min_replicates=1,
            min_cell_count=10,
        )

        curator = DataCurator(
            config=curation_config,
            access=access,
        )
        curator.metadata = metadata

        # Mock the loader to return our mock plate data
        original_load_plates = curator.loader.load_plates

        def mock_load_plates(plate_info, **kwargs):
            curator.loader._feature_columns = identify_feature_columns(
                mock_plate_df
            )
            curator.loader._metadata_columns = [
                c for c in mock_plate_df.columns if c.startswith("Metadata_")
            ]
            return mock_plate_df.copy()

        curator.loader.load_plates = mock_load_plates

        return curator

    def test_build_pretraining_set(
        self, config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
    ):
        curator = self._make_curator_with_mocks(
            config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
        )
        result = curator.build_pretraining_set()

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_curation_report(
        self, config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
    ):
        curator = self._make_curator_with_mocks(
            config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
        )
        curator.build_pretraining_set()

        report = curator.curation_report
        assert "n_wells_raw" in report
        assert "n_wells_post_qc" in report
        assert "n_features_selected" in report
        assert "n_treatments_final" in report
        assert report["n_wells_raw"] > 0

    def test_save_and_load(
        self,
        config,
        mock_plate_df,
        mock_orf_metadata,
        mock_plate_metadata,
        tmp_path,
    ):
        curator = self._make_curator_with_mocks(
            config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
        )
        result = curator.build_pretraining_set()

        output_dir = tmp_path / "curated_output"
        paths = curator.save(result, output_dir, name="test_v1")

        assert paths["profiles"].exists()
        assert paths["report"].exists()
        assert paths["features"].exists()

        # Verify report is valid JSON
        with open(paths["report"]) as f:
            report = json.load(f)
        assert isinstance(report, dict)

        # Verify features file
        with open(paths["features"]) as f:
            features = json.load(f)
        assert isinstance(features, list)

        # Reload
        loaded_df, loaded_report, loaded_features = DataCurator.load_curated(
            paths["profiles"]
        )
        assert len(loaded_df) == len(result)
        assert isinstance(loaded_report, dict)
        assert isinstance(loaded_features, list)

    def test_to_phenotype_dataset(
        self, config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
    ):
        curator = self._make_curator_with_mocks(
            config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
        )
        result = curator.build_pretraining_set()
        dataset = curator.to_phenotype_dataset(result)

        assert isinstance(dataset, PhenotypeDataset)
        assert len(dataset) == len(result)

    def test_diversity_sampling(
        self, config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
    ):
        curator = self._make_curator_with_mocks(
            config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
        )
        curator.config.sampling_strategy = "diversity"
        curator.config.target_n_genes = 10

        result = curator.build_pretraining_set()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Detect gene column
        gene_col = curator._detect_gene_column(result)
        if gene_col is not None:
            n_genes = result[gene_col].nunique()
            assert n_genes <= 10

    def test_coverage_sampling(
        self, config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
    ):
        curator = self._make_curator_with_mocks(
            config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
        )
        curator.config.sampling_strategy = "coverage"
        curator.config.target_n_genes = 10

        result = curator.build_pretraining_set()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_max_genes_limit(
        self, config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
    ):
        curator = self._make_curator_with_mocks(
            config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
        )
        curator.config.max_genes = 5

        result = curator.build_pretraining_set()
        gene_col = curator._detect_gene_column(result)
        if gene_col is not None:
            assert result[gene_col].nunique() <= 5

    def test_no_batch_correction(
        self, config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
    ):
        curator = self._make_curator_with_mocks(
            config, mock_plate_df, mock_orf_metadata, mock_plate_metadata
        )
        curator.config.batch_correction = False

        result = curator.build_pretraining_set()
        assert curator.curation_report.get("batch_correction_applied") is False

    def test_detect_gene_column(self):
        df = pd.DataFrame({"Metadata_JCP2022": ["A"], "other": [1]})
        assert DataCurator._detect_gene_column(df) == "Metadata_JCP2022"

        df2 = pd.DataFrame({"Metadata_Symbol": ["A"], "other": [1]})
        assert DataCurator._detect_gene_column(df2) == "Metadata_Symbol"

        df3 = pd.DataFrame({"unrelated": [1]})
        assert DataCurator._detect_gene_column(df3) is None

    def test_repr(self, config):
        curator = DataCurator(jumpcp_config=config)
        assert "DataCurator" in repr(curator)

    def test_print_report_no_run(self, config):
        """print_report should not raise when called before build."""
        curator = DataCurator(jumpcp_config=config)
        curator.print_report()  # Should just log a message, not raise


# =============================================================================
# Integration Tests (require network)
# =============================================================================

@pytest.mark.slow
class TestJUMPCPIntegration:
    """
    Integration tests requiring network access.

    Run with: pytest -m slow
    """

    def test_check_connectivity(self):
        access = JUMPCPAccess()
        result = access.check_connectivity()
        # At least HTTPS should work in most environments
        assert isinstance(result, dict)
        assert "s3" in result
        assert "https" in result

    def test_fetch_orf_metadata(self):
        access = JUMPCPAccess()
        df = access.fetch_metadata_table("orf")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert any("gene" in c.lower() or "symbol" in c.lower() for c in df.columns)

    def test_fetch_plate_metadata(self):
        access = JUMPCPAccess()
        df = access.fetch_metadata_table("plate")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_metadata_summary(self):
        meta = JUMPCPMetadata()
        info = meta.summary()
        assert info.get("n_orf_genes") is not None
        assert info["n_orf_genes"] > 0

    def test_list_sources(self):
        access = JUMPCPAccess()
        conn = access.check_connectivity()
        if not conn.get("s3"):
            pytest.skip("S3 not available")
        sources = access.list_sources()
        assert len(sources) > 0

    def test_fetch_uniprot_sequences_small(self):
        """Test UniProt fetching with a small gene set."""
        access = JUMPCPAccess()
        genes = ["TP53", "BRCA1", "EGFR"]
        sequences = access.fetch_uniprot_sequences(genes)
        # At least one should resolve
        assert len(sequences) > 0
        for gene, seq in sequences.items():
            assert len(seq) > 0
            assert all(c.isalpha() for c in seq)