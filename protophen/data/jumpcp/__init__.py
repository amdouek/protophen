"""
JUMP-CP data access and curation for ProToPhen.

This sub-package provides infrastructure for interfacing with the
JUMP Cell Painting dataset (Chandrasekaran et al., 2024), including:

- **access**: S3/HTTPS data retrieval with anonymous access.
- **metadata**: Parsing well, plate, ORF, and CRISPR metadata tables.
- **profiles**: Loading and processing morphological profiles.
- **curation**: Intelligent subset selection, QC, and pre-training set construction.
- **cache**: Parquet-based local caching with LRU eviction.

Typical usage::

    from protophen.data.jumpcp import DataCurator, CurationConfig

    curator = DataCurator(config=CurationConfig(
        perturbation_types=["orf"],
        max_plates=10,  # quick test
    ))
    curated_df = curator.build_pretraining_set()
    curator.save(curated_df, "data/processed/pretraining")
"""

from protophen.data.jumpcp.access import JUMPCPAccess, JUMPCPConfig
from protophen.data.jumpcp.cache import JUMPCPCache
from protophen.data.jumpcp.curation import CurationConfig, DataCurator, QualityController
from protophen.data.jumpcp.metadata import JUMPCPMetadata
from protophen.data.jumpcp.profiles import ProfileLoader

__all__ = [
    # Access
    "JUMPCPAccess",
    "JUMPCPConfig",
    # Cache
    "JUMPCPCache",
    # Metadata
    "JUMPCPMetadata",
    # Profiles
    "ProfileLoader",
    # Curation
    "CurationConfig",
    "DataCurator",
    "QualityController",
]