"""
Clustering analysis for phenotypic profiles.

This module provides tools for clustering Cell Painting and other
phenotypic data to identify groups of proteins with similar effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import fisher_exact, hypergeom
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

from protophen.utils.logging import logger

# Try to import HDBSCAN (optional)
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# =============================================================================
# Clustering Result
# =============================================================================

@dataclass
class ClusteringResult:
    """Container for clustering results."""
    
    # Cluster assignments
    labels: np.ndarray  # Shape: (n_samples,)
    
    # Number of clusters
    n_clusters: int
    
    # Clustering method used
    method: str
    
    # Cluster centers (if applicable)
    centers: Optional[np.ndarray] = None  # Shape: (n_clusters, n_features)
    
    # Hierarchical clustering linkage matrix
    linkage_matrix: Optional[np.ndarray] = None
    
    # Quality metrics
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    davies_bouldin: Optional[float] = None
    
    # Sample information
    sample_ids: Optional[List[str]] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cluster_sizes(self) -> Dict[int, int]:
        """Get size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """Get indices of samples in a cluster."""
        return np.where(self.labels == cluster_id)[0]
    
    def get_cluster_member_ids(self, cluster_id: int) -> List[str]:
        """Get sample IDs for a cluster."""
        if self.sample_ids is None:
            return []
        indices = self.get_cluster_members(cluster_id)
        return [self.sample_ids[i] for i in indices]
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Clustering Result ({self.method})",
            f"  N clusters: {self.n_clusters}",
            f"  Cluster sizes: {self.cluster_sizes}",
        ]
        if self.silhouette is not None:
            lines.append(f"  Silhouette score: {self.silhouette:.4f}")
        if self.calinski_harabasz is not None:
            lines.append(f"  Calinski-Harabasz: {self.calinski_harabasz:.2f}")
        if self.davies_bouldin is not None:
            lines.append(f"  Davies-Bouldin: {self.davies_bouldin:.4f}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "labels": self.labels.tolist(),
            "n_clusters": self.n_clusters,
            "method": self.method,
            "cluster_sizes": self.cluster_sizes,
            "silhouette": self.silhouette,
            "calinski_harabasz": self.calinski_harabasz,
            "davies_bouldin": self.davies_bouldin,
            "sample_ids": self.sample_ids,
        }


# =============================================================================
# Clustering Methods
# =============================================================================

def hierarchical_clustering(
    features: np.ndarray,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    method: str = "ward",
    metric: str = "euclidean",
    sample_ids: Optional[List[str]] = None,
) -> ClusteringResult:
    """
    Perform hierarchical/agglomerative clustering.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters (mutually exclusive with distance_threshold)
        distance_threshold: Distance threshold for cutting dendrogram
        method: Linkage method ('ward', 'complete', 'average', 'single')
        metric: Distance metric
        sample_ids: Optional sample identifiers
        
    Returns:
        ClusteringResult with cluster assignments
        
    Example:
        >>> result = hierarchical_clustering(phenotypes, n_clusters=5)
        >>> print(result.summary())
    """
    # Standardise features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute linkage matrix
    if method == "ward":
        # Ward requires euclidean distance
        linkage_matrix = linkage(features_scaled, method="ward")
    else:
        distances = pdist(features_scaled, metric=metric)
        linkage_matrix = linkage(distances, method=method)
    
    # Get cluster labels
    if n_clusters is not None:
        labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1
    elif distance_threshold is not None:
        labels = fcluster(linkage_matrix, distance_threshold, criterion="distance") - 1
    else:
        # Default to cutting at 70% of max distance
        max_dist = linkage_matrix[-1, 2]
        labels = fcluster(linkage_matrix, 0.7 * max_dist, criterion="distance") - 1
    
    n_clusters_found = len(np.unique(labels))
    
    # Compute quality metrics
    if n_clusters_found > 1:
        silhouette = silhouette_score(features_scaled, labels)
        calinski = calinski_harabasz_score(features_scaled, labels)
        davies = davies_bouldin_score(features_scaled, labels)
    else:
        silhouette = calinski = davies = None
    
    # Compute cluster centers
    centers = np.array([
        features[labels == i].mean(axis=0)
        for i in range(n_clusters_found)
    ])
    
    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters_found,
        method=f"hierarchical_{method}",
        centers=centers,
        linkage_matrix=linkage_matrix,
        silhouette=silhouette,
        calinski_harabasz=calinski,
        davies_bouldin=davies,
        sample_ids=sample_ids,
    )


def kmeans_clustering(
    features: np.ndarray,
    n_clusters: int = 5,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    sample_ids: Optional[List[str]] = None,
) -> ClusteringResult:
    """
    Perform k-means clustering.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters
        n_init: Number of initialisations
        max_iter: Maximum iterations
        random_state: Random seed
        sample_ids: Optional sample identifiers
        
    Returns:
        ClusteringResult with cluster assignments
    """
    # Standardise features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Fit k-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(features_scaled)
    
    # Get centers in original space
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Compute quality metrics
    if n_clusters > 1:
        silhouette = silhouette_score(features_scaled, labels)
        calinski = calinski_harabasz_score(features_scaled, labels)
        davies = davies_bouldin_score(features_scaled, labels)
    else:
        silhouette = calinski = davies = None
    
    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        method="kmeans",
        centers=centers,
        silhouette=silhouette,
        calinski_harabasz=calinski,
        davies_bouldin=davies,
        sample_ids=sample_ids,
        metadata={"inertia": kmeans.inertia_},
    )


def hdbscan_clustering(
    features: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    sample_ids: Optional[List[str]] = None,
) -> ClusteringResult:
    """
    Perform HDBSCAN clustering.
    
    HDBSCAN is a density-based clustering algorithm that can find
    clusters of varying densities and automatically determines the
    number of clusters.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples in neighborhood (default: min_cluster_size)
        metric: Distance metric
        sample_ids: Optional sample identifiers
        
    Returns:
        ClusteringResult with cluster assignments (-1 = noise)
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError(
            "HDBSCAN not available. Install with: pip install hdbscan"
        )
    
    # Standardise features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Fit HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(features_scaled)
    
    # Count clusters (excluding noise = -1)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    
    # Compute centers for non-noise clusters
    if n_clusters > 0:
        centers = np.array([
            features[labels == i].mean(axis=0)
            for i in range(n_clusters)
        ])
    else:
        centers = None
    
    # Compute quality metrics (excluding noise points)
    non_noise_mask = labels >= 0
    if n_clusters > 1 and non_noise_mask.sum() > n_clusters:
        silhouette = silhouette_score(
            features_scaled[non_noise_mask],
            labels[non_noise_mask],
        )
    else:
        silhouette = None
    
    n_noise = (labels == -1).sum()
    
    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        method="hdbscan",
        centers=centers,
        silhouette=silhouette,
        sample_ids=sample_ids,
        metadata={
            "n_noise_points": int(n_noise),
            "noise_fraction": float(n_noise / len(labels)),
            "probabilities": clusterer.probabilities_,
        },
    )


# =============================================================================
# Main Clustering Class
# =============================================================================

class PhenotypeClustering:
    """
    High-level interface for clustering phenotypic profiles.
    
    Example:
        >>> clustering = PhenotypeClustering(method="hierarchical")
        >>> result = clustering.fit(phenotype_features)
        >>> 
        >>> # Find optimal number of clusters
        >>> optimal_k = clustering.find_optimal_k(phenotype_features, k_range=(2, 15))
        >>> 
        >>> # Get cluster assignments
        >>> labels = result.labels
    """
    
    def __init__(
        self,
        method: Literal["hierarchical", "kmeans", "hdbscan"] = "hierarchical",
        **kwargs,
    ):
        """
        Initialise clustering.
        
        Args:
            method: Clustering method
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.result: Optional[ClusteringResult] = None
    
    def fit(
        self,
        features: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> ClusteringResult:
        """
        Fit clustering to features.
        
        Args:
            features: Feature matrix
            sample_ids: Optional sample identifiers
            **kwargs: Override parameters
            
        Returns:
            ClusteringResult
        """
        params = {**self.kwargs, **kwargs}
        
        if self.method == "hierarchical":
            self.result = hierarchical_clustering(
                features,
                sample_ids=sample_ids,
                **params,
            )
        elif self.method == "kmeans":
            self.result = kmeans_clustering(
                features,
                sample_ids=sample_ids,
                **params,
            )
        elif self.method == "hdbscan":
            self.result = hdbscan_clustering(
                features,
                sample_ids=sample_ids,
                **params,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(self.result.summary())
        return self.result
    
    def find_optimal_k(
        self,
        features: np.ndarray,
        k_range: Tuple[int, int] = (2, 15),
        criterion: Literal["silhouette", "calinski", "elbow"] = "silhouette",
    ) -> int:
        """
        Find optimal number of clusters.
        
        Args:
            features: Feature matrix
            k_range: Range of k values to try
            criterion: Optimisation criterion
            
        Returns:
            Optimal number of clusters
        """
        k_min, k_max = k_range
        scores = []
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        for k in range(k_min, k_max + 1):
            if self.method == "kmeans":
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_scaled)
                inertia = kmeans.inertia_
            else:
                clustering = AgglomerativeClustering(n_clusters=k)
                labels = clustering.fit_predict(features_scaled)
                inertia = None
            
            if criterion == "silhouette":
                score = silhouette_score(features_scaled, labels)
            elif criterion == "calinski":
                score = calinski_harabasz_score(features_scaled, labels)
            elif criterion == "elbow" and inertia is not None:
                score = -inertia  # Negative because we want to maximise
            else:
                score = silhouette_score(features_scaled, labels)
            
            scores.append((k, score))
        
        # Find best k
        best_k = max(scores, key=lambda x: x[1])[0]
        
        logger.info(f"Optimal k={best_k} (criterion={criterion})")
        
        return best_k
    
    def get_cluster_profiles(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Get mean feature profiles for each cluster.
        
        Args:
            features: Original feature matrix
            feature_names: Feature names
            
        Returns:
            Dictionary mapping cluster ID to feature means
        """
        if self.result is None:
            raise RuntimeError("Must fit clustering first")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        profiles = {}
        for cluster_id in range(self.result.n_clusters):
            mask = self.result.labels == cluster_id
            cluster_mean = features[mask].mean(axis=0)
            profiles[cluster_id] = dict(zip(feature_names, cluster_mean))
        
        return profiles


# =============================================================================
# Cluster Enrichment Analysis
# =============================================================================

def compute_cluster_enrichment(
    cluster_result: ClusteringResult,
    annotations: Dict[str, List[str]],
    background_size: Optional[int] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Compute enrichment of annotations in each cluster.
    
    Uses Fisher's exact test to identify annotations that are
    significantly overrepresented in each cluster.
    
    Args:
        cluster_result: Clustering result
        annotations: Dictionary mapping annotation categories to lists of sample IDs
        background_size: Total background size (default: number of samples)
        
    Returns:
        Nested dictionary: cluster_id -> annotation -> {pvalue, odds_ratio, count}
        
    Example:
        >>> # Check enrichment of protein families
        >>> annotations = {
        ...     "kinases": ["prot_1", "prot_5", "prot_8"],
        ...     "receptors": ["prot_2", "prot_3"],
        ... }
        >>> enrichment = compute_cluster_enrichment(result, annotations)
    """
    if cluster_result.sample_ids is None:
        raise ValueError("ClusteringResult must have sample_ids for enrichment analysis")
    
    n_total = len(cluster_result.sample_ids)
    background_size = background_size or n_total
    
    sample_to_cluster = {
        sid: cluster_result.labels[i]
        for i, sid in enumerate(cluster_result.sample_ids)
    }
    
    enrichments = {}
    
    for cluster_id in range(cluster_result.n_clusters):
        cluster_samples = set(cluster_result.get_cluster_member_ids(cluster_id))
        cluster_size = len(cluster_samples)
        
        enrichments[cluster_id] = {}
        
        for annotation_name, annotation_samples in annotations.items():
            annotation_set = set(annotation_samples)
            
            # Count overlap
            overlap = len(cluster_samples & annotation_set)
            
            # Build contingency table for Fisher's exact test
            # [[in_cluster_in_anno, in_cluster_not_anno],
            #  [not_cluster_in_anno, not_cluster_not_anno]]
            a = overlap
            b = cluster_size - overlap
            c = len(annotation_set) - overlap
            d = background_size - cluster_size - c
            
            # Fisher's exact test
            try:
                odds_ratio, pvalue = fisher_exact([[a, b], [c, d]])
            except ValueError:
                odds_ratio, pvalue = 1.0, 1.0
            
            enrichments[cluster_id][annotation_name] = {
                "pvalue": pvalue,
                "odds_ratio": odds_ratio,
                "count": overlap,
                "expected": cluster_size * len(annotation_set) / background_size,
                "cluster_size": cluster_size,
                "annotation_size": len(annotation_set),
            }
    
    return enrichments


# =============================================================================
# Convenience Functions
# =============================================================================

def cluster_phenotypes(
    features: np.ndarray,
    method: str = "hierarchical",
    n_clusters: Optional[int] = None,
    sample_ids: Optional[List[str]] = None,
    **kwargs,
) -> ClusteringResult:
    """
    Convenience function to cluster phenotype features.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        method: Clustering method
        n_clusters: Number of clusters (auto-determined if None for some methods)
        sample_ids: Optional sample identifiers
        **kwargs: Additional method-specific parameters
        
    Returns:
        ClusteringResult with cluster assignments
    """
    clustering = PhenotypeClustering(method=method, **kwargs)
    
    if n_clusters is None and method in ["hierarchical", "kmeans"]:
        n_clusters = clustering.find_optimal_k(features)
    
    if n_clusters is not None:
        kwargs["n_clusters"] = n_clusters
    
    return clustering.fit(features, sample_ids=sample_ids, **kwargs)