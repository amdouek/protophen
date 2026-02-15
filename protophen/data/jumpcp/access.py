"""
S3 and HTTP access utilities for JUMP-CP data.

This module provides connectivity to the Cell Painting Gallery S3 bucket
and the JUMP-CP datasets GitHub repository, with automatic fallback from
S3 to HTTPS when boto3/s3fs are unavailable.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from protophen.data.jumpcp.cache import JUMPCPCache
from protophen.utils.io import ensure_dir
from protophen.utils.logging import logger

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False

try:
    import s3fs

    _HAS_S3FS = True
except ImportError:
    _HAS_S3FS = False

try:
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


# =============================================================================
# Configuration
# =============================================================================

# Default S3 paths
S3_BUCKET = "cellpainting-gallery"
S3_PREFIX = "cpg0016-jump"
S3_REGION = "us-east-1"

# GitHub metadata URLs (raw)
_GITHUB_BASE = (
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata"
)

METADATA_URLS: Dict[str, str] = {
    "well": f"{_GITHUB_BASE}/well.csv.gz",
    "plate": f"{_GITHUB_BASE}/plate.csv.gz",
    "orf": f"{_GITHUB_BASE}/orf.csv.gz",
    "crispr": f"{_GITHUB_BASE}/crispr.csv.gz",
    "compound": f"{_GITHUB_BASE}/compound.csv.gz",
}


@dataclass
class JUMPCPConfig:
    """
    Configuration for JUMP-CP data access.

    Loaded from ``configs/jumpcp.yaml``.
    """

    # S3 settings
    s3_bucket: str = S3_BUCKET
    s3_prefix: str = S3_PREFIX
    s3_region: str = S3_REGION

    # Metadata URLs (overridable)
    metadata_urls: Dict[str, str] = field(default_factory=lambda: dict(METADATA_URLS))

    # Cache
    cache_dir: str = "./data/raw/jumpcp"
    max_cache_size_gb: float = 50.0

    # Download behaviour
    request_timeout: int = 120
    max_retries: int = 3

    # Perturbation selection
    perturbation_types: List[str] = field(
        default_factory=lambda: ["orf", "crispr"]
    )

    # Curation defaults
    max_genes_orf: int = 15000
    max_genes_crispr: int = 17000
    min_replicates: int = 2
    min_cell_count: int = 50
    aggregation_method: str = "median"

    # Gene → sequence mapping
    uniprot_batch_size: int = 200
    organism_id: str = "9606"  # Homo sapiens

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "JUMPCPConfig":
        """Load from YAML file."""
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


# =============================================================================
# Access Class
# =============================================================================

class JUMPCPAccess:
    """
    Interface for accessing JUMP-CP data on AWS S3 and GitHub.

    Provides:
    * Anonymous S3 reads (no credentials needed).
    * HTTPS fallback for metadata when ``boto3`` is absent.
    * Transparent local caching via :class:`JUMPCPCache`.

    Attributes:
        config: Access configuration.
        cache: Local cache instance.

    Example:
        >>> access = JUMPCPAccess()
        >>> access.check_connectivity()
        {'s3': True, 'https': True}
        >>> plates = access.list_plates("source_4")
    """

    def __init__(
        self,
        config: Optional[JUMPCPConfig] = None,
        cache: Optional[JUMPCPCache] = None,
    ):
        self.config = config or JUMPCPConfig()
        self.cache = cache or JUMPCPCache(
            cache_dir=self.config.cache_dir,
            max_size_gb=self.config.max_cache_size_gb,
        )

        # Lazy-initialised clients
        self._s3_client = None
        self._s3fs = None

        logger.info("Initialised JUMPCPAccess")

    # =========================================================================
    # S3 client helpers
    # =========================================================================

    def _get_s3_client(self):
        """Get (or create) an anonymous boto3 S3 client."""
        if self._s3_client is None:
            if not _HAS_BOTO3:
                raise ImportError(
                    "boto3 is required for S3 access. "
                    "Install with: pip install boto3"
                )
            self._s3_client = boto3.client(
                "s3",
                region_name=self.config.s3_region,
                config=BotoConfig(signature_version=UNSIGNED),
            )
        return self._s3_client

    def _get_s3fs(self):
        """Get (or create) an anonymous s3fs filesystem."""
        if self._s3fs is None:
            if not _HAS_S3FS:
                raise ImportError(
                    "s3fs is required for streaming profile access. "
                    "Install with: pip install s3fs"
                )
            self._s3fs = s3fs.S3FileSystem(anon=True)
        return self._s3fs

    # =========================================================================
    # Connectivity
    # =========================================================================

    def check_connectivity(self) -> Dict[str, bool]:
        """
        Check connectivity to S3 and GitHub.

        Returns:
            Dictionary with ``"s3"`` and ``"https"`` boolean flags.
        """
        result = {"s3": False, "https": False}

        # S3
        if _HAS_BOTO3:
            try:
                client = self._get_s3_client()
                client.head_bucket(Bucket=self.config.s3_bucket)
                result["s3"] = True
            except Exception as exc:
                logger.warning(f"S3 connectivity check failed: {exc}")

        # HTTPS (GitHub)
        try:
            url = self.config.metadata_urls.get(
                "plate", list(METADATA_URLS.values())[0]
            )
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=10)
            result["https"] = True
        except Exception as exc:
            logger.warning(f"HTTPS connectivity check failed: {exc}")

        logger.info(f"Connectivity: S3={result['s3']}, HTTPS={result['https']}")
        return result

    # =========================================================================
    # S3 listing
    # =========================================================================

    def list_sources(self) -> List[str]:
        """List available JUMP-CP data sources in the S3 bucket."""
        client = self._get_s3_client()
        prefix = f"{self.config.s3_prefix}/"
        resp = client.list_objects_v2(
            Bucket=self.config.s3_bucket,
            Prefix=prefix,
            Delimiter="/",
        )
        sources = []
        for cp in resp.get("CommonPrefixes", []):
            name = cp["Prefix"].rstrip("/").split("/")[-1]
            sources.append(name)
        return sorted(sources)

    def list_batches(self, source: str) -> List[str]:
        """
        List batch IDs under a given source.

        Args:
            source: Source identifier (e.g. ``"source_4"``).

        Returns:
            Sorted list of batch identifiers.
        """
        client = self._get_s3_client()
        prefix = f"{self.config.s3_prefix}/{source}/workspace/profiles/"
        batches: List[str] = []
        paginator = client.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=self.config.s3_bucket,
            Prefix=prefix,
            Delimiter="/",
        ):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/").split("/")[-1]
                batches.append(name)

        logger.info(f"Found {len(batches)} batches for source '{source}'")
        return sorted(batches)

    def list_plates(self, source: str) -> List[str]:
        """
        List plate IDs available for a given source.

        Performs a recursive prefix scan under
        ``{s3_prefix}/{source}/workspace/profiles/``.
        """
        client = self._get_s3_client()
        prefix = f"{self.config.s3_prefix}/{source}/workspace/profiles/"
        plates: List[str] = []
        paginator = client.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=self.config.s3_bucket,
            Prefix=prefix,
            Delimiter="/",
        ):
            for batch_prefix in page.get("CommonPrefixes", []):
                batch_path = batch_prefix["Prefix"]
                for page2 in paginator.paginate(
                    Bucket=self.config.s3_bucket,
                    Prefix=batch_path,
                    Delimiter="/",
                ):
                    for plate_prefix in page2.get("CommonPrefixes", []):
                        plate_id = plate_prefix["Prefix"].rstrip("/").split("/")[-1]
                        plates.append(plate_id)

        logger.info(f"Found {len(plates)} plates for source '{source}'")
        return sorted(plates)

    def get_profile_s3_path(
        self,
        source: str,
        batch: str,
        plate: str,
    ) -> str:
        """Construct the S3 key for a plate's profile parquet file."""
        return (
            f"{self.config.s3_prefix}/{source}/workspace/profiles/"
            f"{batch}/{plate}/{plate}.parquet"
        )

    # =========================================================================
    # File downloads
    # =========================================================================

    def download_file(
        self,
        s3_key: str,
        local_path: Union[str, Path],
    ) -> Path:
        """
        Download a single file from S3 to a local path.

        Args:
            s3_key: Object key in the S3 bucket.
            local_path: Destination on the local filesystem.

        Returns:
            Path to the downloaded file.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client = self._get_s3_client()

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(
                    f"Downloading s3://{self.config.s3_bucket}/{s3_key} "
                    f"→ {local_path} (attempt {attempt})"
                )
                client.download_file(
                    self.config.s3_bucket, s3_key, str(local_path)
                )
                logger.debug(f"Downloaded {local_path.stat().st_size / 1e6:.1f} MB")
                return local_path
            except Exception as exc:
                logger.warning(f"Download attempt {attempt} failed: {exc}")
                if attempt == self.config.max_retries:
                    raise RuntimeError(
                        f"Failed to download s3://{self.config.s3_bucket}/{s3_key} "
                        f"after {self.config.max_retries} attempts"
                    ) from exc

        return local_path  # unreachable; satisfies type checker

    def read_parquet_from_s3(
        self,
        s3_key: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read a Parquet file from S3 directly into a DataFrame.

        Uses ``s3fs`` for streaming reads—no local copy required.

        Args:
            s3_key: Object key in the S3 bucket.
            columns: If provided, only read these columns.

        Returns:
            DataFrame with the parquet content.
        """
        if not _HAS_PYARROW:
            raise ImportError(
                "pyarrow is required to read Parquet files. "
                "Install with: pip install pyarrow"
            )

        fs = self._get_s3fs()
        full_path = f"{self.config.s3_bucket}/{s3_key}"
        logger.debug(f"Streaming parquet from s3://{full_path}")

        table = pq.read_table(
            full_path,
            filesystem=fs,
            columns=columns,
        )
        return table.to_pandas()

    # =========================================================================
    # Metadata downloads (HTTPS with S3 fallback)
    # =========================================================================

    def fetch_metadata_table(
        self,
        table_name: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch a JUMP-CP metadata table (e.g. ``"orf"``, ``"plate"``).

        Tries (in order):
        1. Local cache.
        2. HTTPS download from GitHub.
        3. S3 fallback (if boto3 is available and the file lives there).

        Args:
            table_name: One of ``"well"``, ``"plate"``, ``"orf"``,
                ``"crispr"``, ``"compound"``.
            force_refresh: Re-download even if cached.

        Returns:
            DataFrame with the metadata.
        """
        # 1. Cache
        if not force_refresh and self.cache.has_metadata(table_name):
            logger.debug(f"Loading metadata '{table_name}' from cache")
            return self.cache.get_metadata(table_name)

        # 2. HTTPS
        url = self.config.metadata_urls.get(table_name)
        if url is not None:
            try:
                logger.info(f"Downloading metadata '{table_name}' from {url}")
                df = self._download_csv_from_url(url)
                self.cache.store_metadata(table_name, df, source=url)
                return df
            except Exception as exc:
                logger.warning(
                    f"HTTPS download failed for '{table_name}': {exc}. "
                    "Attempting S3 fallback."
                )

        # 3. S3 fallback
        if _HAS_BOTO3:
            s3_key = f"{self.config.s3_prefix}/metadata/{table_name}.csv.gz"
            try:
                client = self._get_s3_client()
                obj = client.get_object(
                    Bucket=self.config.s3_bucket, Key=s3_key
                )
                import gzip

                with gzip.open(obj["Body"]) as gz:
                    df = pd.read_csv(gz)
                self.cache.store_metadata(
                    table_name, df, source=f"s3://{self.config.s3_bucket}/{s3_key}"
                )
                return df
            except Exception as exc:
                raise RuntimeError(
                    f"Could not fetch metadata '{table_name}' from HTTPS or S3"
                ) from exc

        raise RuntimeError(
            f"Could not fetch metadata '{table_name}'. "
            "Ensure network connectivity and/or install boto3."
        )

    def _download_csv_from_url(self, url: str) -> pd.DataFrame:
        """Download a (possibly gzipped) CSV from a URL."""
        import gzip as gzip_mod

        req = urllib.request.Request(url)
        with urllib.request.urlopen(
            req, timeout=self.config.request_timeout
        ) as resp:
            raw = resp.read()

        # Decompress if gzipped
        if url.endswith(".gz"):
            raw = gzip_mod.decompress(raw)

        return pd.read_csv(io.BytesIO(raw))

    # =========================================================================
    # Profile downloads (per-plate)
    # =========================================================================

    def fetch_plate_profiles(
        self,
        source: str,
        batch: str,
        plate: str,
        columns: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch well-level morphological profiles for a single plate.

        Downloads from S3 (streaming if possible) and caches locally.

        Args:
            source: Source identifier (e.g. ``"source_4"``).
            batch: Batch identifier.
            plate: Plate identifier.
            columns: Optional subset of columns to load.
            force_refresh: Re-download even if cached.

        Returns:
            DataFrame with well-level profiles.
        """
        cache_key = f"{source}__{batch}__{plate}"

        if not force_refresh and self.cache.has_profiles(cache_key):
            logger.debug(f"Loading profiles for plate '{plate}' from cache")
            df = self.cache.get_profiles(cache_key)
            if columns is not None and df is not None:
                available = [c for c in columns if c in df.columns]
                return df[available]
            return df

        s3_key = self.get_profile_s3_path(source, batch, plate)
        logger.info(f"Fetching profiles for plate '{plate}' from S3")

        # Prefer streaming read
        if _HAS_S3FS and _HAS_PYARROW:
            try:
                df = self.read_parquet_from_s3(s3_key, columns=None)
                self.cache.store_profiles(
                    cache_key,
                    df,
                    source=f"s3://{self.config.s3_bucket}/{s3_key}",
                )
                if columns is not None:
                    available = [c for c in columns if c in df.columns]
                    return df[available]
                return df
            except Exception as exc:
                logger.warning(
                    f"Streaming read failed for plate '{plate}': {exc}. "
                    "Falling back to download."
                )

        # Fall back to download + local read
        local_path = (
            Path(self.config.cache_dir) / "downloads" / f"{cache_key}.parquet"
        )
        self.download_file(s3_key, local_path)
        df = pd.read_parquet(local_path, columns=columns)

        # Store in normalised cache and remove the download
        self.cache.store_profiles(
            cache_key,
            df,
            source=f"s3://{self.config.s3_bucket}/{s3_key}",
        )
        if local_path.exists():
            local_path.unlink()

        return df

    # =========================================================================
    # UniProt gene→sequence mapping
    # =========================================================================

    def fetch_uniprot_sequences(
        self,
        gene_symbols: List[str],
        organism_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Map gene symbols to canonical UniProt protein sequences.

        Uses the UniProt REST API (HTTPS) in batches.

        Args:
            gene_symbols: List of HGNC gene symbols.
            organism_id: NCBI taxonomy ID (defaults to ``"9606"`` for human).

        Returns:
            Dictionary mapping gene symbols to amino-acid sequences.
            Genes that could not be resolved are silently omitted.
        """
        organism_id = organism_id or self.config.organism_id
        batch_size = self.config.uniprot_batch_size
        result: Dict[str, str] = {}

        # Check cache first
        cache_name = f"uniprot_sequences_{organism_id}"
        cached = self.cache.get_metadata(cache_name)
        if cached is not None:
            cached_map = dict(zip(cached["gene_symbol"], cached["sequence"]))
            remaining = [g for g in gene_symbols if g not in cached_map]
            result.update(
                {g: cached_map[g] for g in gene_symbols if g in cached_map}
            )
            if not remaining:
                logger.info(
                    f"All {len(gene_symbols)} gene sequences loaded from cache"
                )
                return result
            gene_symbols = remaining
            logger.info(
                f"Loaded {len(result)} sequences from cache, "
                f"{len(remaining)} remaining to fetch"
            )

        logger.info(
            f"Fetching sequences for {len(gene_symbols)} genes from UniProt"
        )

        for i in range(0, len(gene_symbols), batch_size):
            batch = gene_symbols[i : i + batch_size]
            query = "+OR+".join(
                f"(gene_exact:{g}+AND+organism_id:{organism_id})" for g in batch
            )
            url = (
                f"https://rest.uniprot.org/uniprotkb/stream?"
                f"query={query}"
                f"&format=tsv"
                f"&fields=gene_primary,sequence"
                f"&compressed=false"
            )

            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(
                    req, timeout=self.config.request_timeout
                ) as resp:
                    tsv_text = resp.read().decode("utf-8")

                if not tsv_text.strip():
                    continue

                lines = tsv_text.strip().split("\n")
                if len(lines) <= 1:
                    continue

                for line in lines[1:]:  # skip header
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        gene = parts[0].strip()
                        seq = parts[1].strip()
                        # Take first (canonical) hit per gene
                        if gene in batch and gene not in result:
                            result[gene] = seq

                logger.debug(
                    f"UniProt batch {i // batch_size + 1}: "
                    f"resolved {sum(1 for g in batch if g in result)}/{len(batch)}"
                )

            except Exception as exc:
                logger.warning(f"UniProt batch request failed: {exc}")
                continue

        # Update cache
        if result:
            cache_df = pd.DataFrame(
                [
                    {"gene_symbol": g, "sequence": s}
                    for g, s in result.items()
                ]
            )
            # Merge with existing cache
            if cached is not None:
                cache_df = pd.concat([cached, cache_df]).drop_duplicates(
                    subset="gene_symbol", keep="last"
                )
            self.cache.store_metadata(cache_name, cache_df, source="uniprot_api")

        logger.info(
            f"Resolved {len(result)}/{len(gene_symbols) + len(result)} "
            f"gene symbols to sequences"
        )
        return result

    # =========================================================================
    # Dunder
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"JUMPCPAccess(bucket='{self.config.s3_bucket}', "
            f"cache={self.cache})"
        )