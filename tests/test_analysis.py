"""
Tests for analysis module (clustering, visualisation, interpretation).
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from protophen.analysis.clustering import (
    ClusteringResult,
    PhenotypeClustering,
    cluster_phenotypes,
    compute_cluster_enrichment,
    hierarchical_clustering,
    kmeans_clustering,
    HDBSCAN_AVAILABLE,
)
from protophen.analysis.visualisation import (
    PlotConfig,
    create_figure_grid,
    plot_active_learning_progress,
    plot_clustermap,
    plot_correlation_matrix,
    plot_embedding_space,
    plot_feature_distributions,
    plot_heatmap,
    plot_pca,
    plot_prediction_scatter,
    plot_residuals,
    plot_training_history,
    plot_tsne,
    plot_umap,
    plot_uncertainty_distribution,
    save_figure,
)
from protophen.analysis.interpretation import (
    AttentionAnalyser,
    FeatureAblationInterpreter,
    GradientInterpreter,
    IntegratedGradientsInterpreter,
    InterpretationConfig,
    ModelInterpreter,
    SHAPInterpreter,
    compute_feature_importance,
    explain_prediction,
    get_embedding_contribution,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_features():
    """Create sample feature matrix for clustering."""
    np.random.seed(42)
    # Create 3 clusters
    cluster1 = np.random.randn(20, 50) + np.array([0, 0, 0] + [0] * 47)
    cluster2 = np.random.randn(20, 50) + np.array([5, 5, 5] + [0] * 47)
    cluster3 = np.random.randn(20, 50) + np.array([-5, -5, -5] + [0] * 47)
    return np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)


@pytest.fixture
def sample_ids():
    """Create sample IDs for clustering."""
    return [f"sample_{i}" for i in range(60)]


@pytest.fixture
def clustering_result(sample_features, sample_ids):
    """Create a sample ClusteringResult."""
    return ClusteringResult(
        labels=np.array([0] * 20 + [1] * 20 + [2] * 20),
        n_clusters=3,
        method="test",
        centers=np.random.randn(3, 50).astype(np.float32),
        silhouette=0.5,
        calinski_harabasz=100.0,
        davies_bouldin=0.8,
        sample_ids=sample_ids,
    )


@pytest.fixture
def mock_model():
    """Create a mock model for interpretation tests."""
    class SimpleMockModel(nn.Module):
        def __init__(self, input_dim=128, output_dim=100):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, 64)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(64, output_dim)
        
        def forward(self, x, tasks=None, return_uncertainty=False):
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            tasks = tasks or ["cell_painting"]
            outputs = {task: x for task in tasks}
            if return_uncertainty:
                for task in tasks:
                    outputs[f"{task}_log_var"] = torch.zeros_like(x)
            return outputs
    
    return SimpleMockModel()


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for interpretation."""
    return torch.randn(10, 128)


@pytest.fixture
def background_data():
    """Create background data for SHAP."""
    return torch.randn(50, 128)


# =============================================================================
# Tests for ClusteringResult
# =============================================================================

class TestClusteringResult:
    """Tests for ClusteringResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic creation."""
        labels = np.array([0, 0, 1, 1, 2])
        result = ClusteringResult(
            labels=labels,
            n_clusters=3,
            method="test",
        )
        
        assert result.n_clusters == 3
        assert result.method == "test"
        assert len(result.labels) == 5
    
    def test_cluster_sizes(self, clustering_result):
        """Test cluster_sizes property."""
        sizes = clustering_result.cluster_sizes
        
        assert sizes[0] == 20
        assert sizes[1] == 20
        assert sizes[2] == 20
    
    def test_get_cluster_members(self, clustering_result):
        """Test getting cluster member indices."""
        members = clustering_result.get_cluster_members(0)
        
        assert len(members) == 20
        assert all(clustering_result.labels[i] == 0 for i in members)
    
    def test_get_cluster_member_ids(self, clustering_result):
        """Test getting cluster member IDs."""
        member_ids = clustering_result.get_cluster_member_ids(0)
        
        assert len(member_ids) == 20
        assert member_ids[0] == "sample_0"
    
    def test_get_cluster_member_ids_no_sample_ids(self):
        """Test getting member IDs when sample_ids is None."""
        result = ClusteringResult(
            labels=np.array([0, 0, 1]),
            n_clusters=2,
            method="test",
            sample_ids=None,
        )
        
        member_ids = result.get_cluster_member_ids(0)
        assert member_ids == []
    
    def test_summary(self, clustering_result):
        """Test summary string generation."""
        summary = clustering_result.summary()
        
        assert "Clustering Result" in summary
        assert "N clusters: 3" in summary
        assert "Silhouette" in summary
    
    def test_to_dict(self, clustering_result):
        """Test conversion to dictionary."""
        d = clustering_result.to_dict()
        
        assert "labels" in d
        assert "n_clusters" in d
        assert "method" in d
        assert "cluster_sizes" in d
        assert "silhouette" in d
        assert d["n_clusters"] == 3


# =============================================================================
# Tests for Clustering Functions
# =============================================================================

class TestHierarchicalClustering:
    """Tests for hierarchical_clustering function."""
    
    def test_basic_clustering(self, sample_features):
        """Test basic hierarchical clustering."""
        result = hierarchical_clustering(
            sample_features,
            n_clusters=3,
        )
        
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters == 3
        assert result.method == "hierarchical_ward"
        assert len(result.labels) == 60
    
    def test_with_distance_threshold(self, sample_features):
        """Test clustering with distance threshold."""
        result = hierarchical_clustering(
            sample_features,
            distance_threshold=10.0,
        )
        
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters >= 1
    
    def test_different_methods(self, sample_features):
        """Test different linkage methods."""
        for method in ["ward", "complete", "average", "single"]:
            result = hierarchical_clustering(
                sample_features,
                n_clusters=3,
                method=method,
            )
            assert result.n_clusters == 3
    
    def test_with_sample_ids(self, sample_features, sample_ids):
        """Test clustering with sample IDs."""
        result = hierarchical_clustering(
            sample_features,
            n_clusters=3,
            sample_ids=sample_ids,
        )
        
        assert result.sample_ids == sample_ids
    
    def test_quality_metrics_computed(self, sample_features):
        """Test that quality metrics are computed."""
        result = hierarchical_clustering(
            sample_features,
            n_clusters=3,
        )
        
        assert result.silhouette is not None
        assert result.calinski_harabasz is not None
        assert result.davies_bouldin is not None
    
    def test_linkage_matrix_stored(self, sample_features):
        """Test that linkage matrix is stored."""
        result = hierarchical_clustering(
            sample_features,
            n_clusters=3,
        )
        
        assert result.linkage_matrix is not None
        assert result.linkage_matrix.shape[0] == len(sample_features) - 1
    
    def test_centers_computed(self, sample_features):
        """Test that cluster centers are computed."""
        result = hierarchical_clustering(
            sample_features,
            n_clusters=3,
        )
        
        assert result.centers is not None
        assert result.centers.shape == (3, sample_features.shape[1])


class TestKMeansClustering:
    """Tests for kmeans_clustering function."""
    
    def test_basic_clustering(self, sample_features):
        """Test basic k-means clustering."""
        result = kmeans_clustering(
            sample_features,
            n_clusters=3,
        )
        
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters == 3
        assert result.method == "kmeans"
    
    def test_reproducibility(self, sample_features):
        """Test reproducibility with seed."""
        result1 = kmeans_clustering(sample_features, n_clusters=3, random_state=42)
        result2 = kmeans_clustering(sample_features, n_clusters=3, random_state=42)
        
        np.testing.assert_array_equal(result1.labels, result2.labels)
    
    def test_inertia_stored(self, sample_features):
        """Test that inertia is stored in metadata."""
        result = kmeans_clustering(sample_features, n_clusters=3)
        
        assert "inertia" in result.metadata
        assert result.metadata["inertia"] > 0


class TestHDBSCANClustering:
    """Tests for hdbscan_clustering function."""
    
    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="HDBSCAN not installed")
    def test_basic_clustering(self, sample_features):
        """Test basic HDBSCAN clustering."""
        from protophen.analysis.clustering import hdbscan_clustering
        
        result = hdbscan_clustering(
            sample_features,
            min_cluster_size=5,
        )
        
        assert isinstance(result, ClusteringResult)
        assert result.method == "hdbscan"
    
    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="HDBSCAN not installed")
    def test_noise_points_tracked(self, sample_features):
        """Test that noise points are tracked."""
        from protophen.analysis.clustering import hdbscan_clustering
        
        result = hdbscan_clustering(sample_features, min_cluster_size=5)
        
        assert "n_noise_points" in result.metadata
        assert "noise_fraction" in result.metadata
    
    def test_import_error_when_not_available(self):
        """Test that import error is raised when HDBSCAN not available."""
        if HDBSCAN_AVAILABLE:
            pytest.skip("HDBSCAN is available")
        
        from protophen.analysis.clustering import hdbscan_clustering
        
        with pytest.raises(ImportError, match="HDBSCAN not available"):
            hdbscan_clustering(np.random.randn(10, 5))


# =============================================================================
# Tests for PhenotypeClustering
# =============================================================================

class TestPhenotypeClustering:
    """Tests for PhenotypeClustering class."""
    
    def test_initialization(self):
        """Test initialization."""
        clustering = PhenotypeClustering(method="hierarchical")
        
        assert clustering.method == "hierarchical"
        assert clustering.result is None
    
    def test_fit_hierarchical(self, sample_features):
        """Test fitting with hierarchical method."""
        clustering = PhenotypeClustering(method="hierarchical")
        result = clustering.fit(sample_features, n_clusters=3)
        
        assert isinstance(result, ClusteringResult)
        assert clustering.result is result
    
    def test_fit_kmeans(self, sample_features):
        """Test fitting with k-means method."""
        clustering = PhenotypeClustering(method="kmeans")
        result = clustering.fit(sample_features, n_clusters=3)
        
        assert isinstance(result, ClusteringResult)
        assert result.method == "kmeans"
    
    def test_invalid_method_raises(self, sample_features):
        """Test that invalid method raises error."""
        clustering = PhenotypeClustering(method="invalid")
        
        with pytest.raises(ValueError, match="Unknown method"):
            clustering.fit(sample_features)
    
    def test_find_optimal_k(self, sample_features):
        """Test finding optimal number of clusters."""
        clustering = PhenotypeClustering(method="kmeans")
        optimal_k = clustering.find_optimal_k(
            sample_features,
            k_range=(2, 6),
            criterion="silhouette",
        )
        
        assert 2 <= optimal_k <= 6
    
    def test_find_optimal_k_calinski(self, sample_features):
        """Test finding optimal k with Calinski-Harabasz criterion."""
        clustering = PhenotypeClustering(method="kmeans")
        optimal_k = clustering.find_optimal_k(
            sample_features,
            k_range=(2, 6),
            criterion="calinski",
        )
        
        assert 2 <= optimal_k <= 6
    
    def test_get_cluster_profiles(self, sample_features):
        """Test getting cluster profiles."""
        clustering = PhenotypeClustering(method="kmeans")
        clustering.fit(sample_features, n_clusters=3)
        
        feature_names = [f"feature_{i}" for i in range(sample_features.shape[1])]
        profiles = clustering.get_cluster_profiles(sample_features, feature_names)
        
        assert len(profiles) == 3
        assert all(isinstance(p, dict) for p in profiles.values())
    
    def test_get_cluster_profiles_without_fit_raises(self, sample_features):
        """Test that getting profiles without fitting raises error."""
        clustering = PhenotypeClustering(method="kmeans")
        
        with pytest.raises(RuntimeError, match="Must fit"):
            clustering.get_cluster_profiles(sample_features)


# =============================================================================
# Tests for Cluster Enrichment
# =============================================================================

class TestComputeClusterEnrichment:
    """Tests for compute_cluster_enrichment function."""
    
    def test_basic_enrichment(self, clustering_result):
        """Test basic enrichment analysis."""
        annotations = {
            "group_A": ["sample_0", "sample_1", "sample_2"],
            "group_B": ["sample_20", "sample_21", "sample_22"],
        }
        
        enrichment = compute_cluster_enrichment(clustering_result, annotations)
        
        assert len(enrichment) == 3  # 3 clusters
        assert all("group_A" in enrichment[c] for c in enrichment)
        assert all("group_B" in enrichment[c] for c in enrichment)
    
    def test_enrichment_contains_statistics(self, clustering_result):
        """Test that enrichment contains expected statistics."""
        annotations = {
            "group_A": ["sample_0", "sample_1"],
        }
        
        enrichment = compute_cluster_enrichment(clustering_result, annotations)
        
        stats = enrichment[0]["group_A"]
        assert "pvalue" in stats
        assert "odds_ratio" in stats
        assert "count" in stats
        assert "expected" in stats
    
    def test_enrichment_without_sample_ids_raises(self):
        """Test that enrichment without sample_ids raises error."""
        result = ClusteringResult(
            labels=np.array([0, 1]),
            n_clusters=2,
            method="test",
            sample_ids=None,
        )
        
        with pytest.raises(ValueError, match="sample_ids"):
            compute_cluster_enrichment(result, {"group": ["a"]})


# =============================================================================
# Tests for cluster_phenotypes Convenience Function
# =============================================================================

class TestClusterPhenotypes:
    """Tests for cluster_phenotypes function."""
    
    def test_basic_usage(self, sample_features):
        """Test basic usage."""
        result = cluster_phenotypes(
            sample_features,
            method="kmeans",
            n_clusters=3,
        )
        
        assert isinstance(result, ClusteringResult)
    
    def test_auto_determine_clusters(self, sample_features):
        """Test automatic cluster number determination."""
        result = cluster_phenotypes(
            sample_features,
            method="hierarchical",
            n_clusters=None,  # Auto-determine
        )
        
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters >= 2


# =============================================================================
# Tests for PlotConfig
# =============================================================================

class TestPlotConfig:
    """Tests for PlotConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PlotConfig()
        
        assert config.figsize == (10, 8)
        assert config.dpi == 100
        assert config.style == "whitegrid"
        assert config.cmap == "viridis"
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = PlotConfig(
            figsize=(12, 10),
            dpi=150,
            style="darkgrid",
        )
        
        assert config.figsize == (12, 10)
        assert config.dpi == 150
        assert config.style == "darkgrid"
    
    def test_apply(self):
        """Test applying configuration."""
        config = PlotConfig()
        
        # Should not raise
        config.apply()


# =============================================================================
# Tests for Embedding Visualisation
# =============================================================================

class TestPlotEmbeddingSpace:
    """Tests for plot_embedding_space function."""
    
    def test_pca_method(self, sample_features):
        """Test PCA dimensionality reduction."""
        fig, ax = plot_embedding_space(
            sample_features,
            method="pca",
            title="PCA Test",
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_tsne_method(self, sample_features):
        """Test t-SNE dimensionality reduction."""
        # Use small perplexity for small dataset
        fig, ax = plot_embedding_space(
            sample_features[:20],  # Smaller for speed
            method="tsne",
            perplexity=5,
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @pytest.mark.skipif(True, reason="UMAP may not be installed")
    def test_umap_method(self, sample_features):
        """Test UMAP dimensionality reduction."""
        try:
            fig, ax = plot_embedding_space(
                sample_features,
                method="umap",
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except ImportError:
            pytest.skip("UMAP not installed")
    
    def test_with_labels(self, sample_features):
        """Test plotting with labels."""
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        fig, ax = plot_embedding_space(
            sample_features,
            labels=labels,
            method="pca",
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_color_by(self, sample_features):
        """Test plotting with continuous coloring."""
        color_values = np.random.rand(60)
        
        fig, ax = plot_embedding_space(
            sample_features,
            color_by=color_values,
            method="pca",
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_invalid_method_raises(self, sample_features):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            plot_embedding_space(sample_features, method="invalid")
    
    def test_on_existing_axes(self, sample_features):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        
        fig_out, ax_out = plot_embedding_space(
            sample_features,
            method="pca",
            ax=ax,
        )
        
        assert ax_out is ax
        plt.close(fig)


class TestConveniencePlotFunctions:
    """Tests for convenience plot functions."""
    
    def test_plot_pca(self, sample_features):
        """Test plot_pca function."""
        fig, ax = plot_pca(sample_features)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_tsne(self, sample_features):
        """Test plot_tsne function."""
        fig, ax = plot_tsne(sample_features[:20], perplexity=5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Tests for Heatmap Plots
# =============================================================================

class TestPlotHeatmap:
    """Tests for plot_heatmap function."""
    
    def test_basic_heatmap(self):
        """Test basic heatmap."""
        data = np.random.randn(10, 10)
        
        fig, ax = plot_heatmap(data, title="Test Heatmap")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_labels(self):
        """Test heatmap with labels."""
        data = np.random.randn(5, 5)
        row_labels = [f"row_{i}" for i in range(5)]
        col_labels = [f"col_{i}" for i in range(5)]
        
        fig, ax = plot_heatmap(
            data,
            row_labels=row_labels,
            col_labels=col_labels,
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_center(self):
        """Test heatmap with centered colormap."""
        data = np.random.randn(10, 10)
        
        fig, ax = plot_heatmap(data, center=0)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_annotations(self):
        """Test heatmap with cell annotations."""
        data = np.random.randn(5, 5)
        
        fig, ax = plot_heatmap(data, annot=True)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCorrelationMatrix:
    """Tests for plot_correlation_matrix function."""
    
    def test_basic_correlation(self, sample_features):
        """Test basic correlation matrix."""
        fig, ax = plot_correlation_matrix(
            sample_features[:, :10],  # Use subset for speed
            title="Correlation Test",
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_feature_names(self, sample_features):
        """Test with feature names."""
        feature_names = [f"feat_{i}" for i in range(10)]
        
        fig, ax = plot_correlation_matrix(
            sample_features[:, :10],
            feature_names=feature_names,
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_spearman_method(self, sample_features):
        """Test Spearman correlation."""
        fig, ax = plot_correlation_matrix(
            sample_features[:, :10],
            method="spearman",
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotClustermap:
    """Tests for plot_clustermap function."""
    
    def test_basic_clustermap(self):
        """Test basic clustermap."""
        data = np.random.randn(20, 10)
        
        g = plot_clustermap(data)
        
        assert g is not None
        plt.close('all')
    
    def test_with_row_colors(self):
        """Test clustermap with row colors."""
        data = np.random.randn(20, 10)
        cmap = plt.cm.viridis
        row_color_values = np.random.rand(20)
        row_colors = [cmap(v) for v in row_color_values]
        
        g = plot_clustermap(data, row_colors=row_colors)
        
        assert g is not None
        plt.close('all')


# =============================================================================
# Tests for Distribution Plots
# =============================================================================

class TestPlotFeatureDistributions:
    """Tests for plot_feature_distributions function."""
    
    def test_basic_distributions(self, sample_features):
        """Test basic feature distributions."""
        fig, axes = plot_feature_distributions(
            sample_features,
            n_features=8,
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_feature_names(self, sample_features):
        """Test with feature names."""
        feature_names = [f"feature_{i}" for i in range(sample_features.shape[1])]
        
        fig, axes = plot_feature_distributions(
            sample_features,
            feature_names=feature_names,
            n_features=4,
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotUncertaintyDistribution:
    """Tests for plot_uncertainty_distribution function."""
    
    def test_basic_distribution(self):
        """Test basic uncertainty distribution."""
        scores = np.random.rand(100)
        
        fig, ax = plot_uncertainty_distribution(scores)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_selected_indices(self):
        """Test with highlighted selected samples."""
        scores = np.random.rand(100)
        selected = np.array([90, 95, 99])  # High uncertainty samples
        
        fig, ax = plot_uncertainty_distribution(
            scores,
            selected_indices=selected,
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Tests for Prediction Plots
# =============================================================================

class TestPlotPredictionScatter:
    """Tests for plot_prediction_scatter function."""
    
    def test_basic_scatter(self):
        """Test basic prediction scatter."""
        y_true = np.random.randn(50, 10)
        y_pred = y_true + np.random.randn(50, 10) * 0.1
        
        fig, axes = plot_prediction_scatter(y_true, y_pred, n_features=6)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_feature_names(self):
        """Test with feature names."""
        y_true = np.random.randn(50, 10)
        y_pred = y_true + np.random.randn(50, 10) * 0.1
        feature_names = [f"gene_{i}" for i in range(10)]
        
        fig, axes = plot_prediction_scatter(
            y_true,
            y_pred,
            feature_names=feature_names,
            n_features=4,
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotResiduals:
    """Tests for plot_residuals function."""
    
    def test_basic_residuals(self):
        """Test basic residual plot."""
        y_true = np.random.randn(100, 10)
        y_pred = y_true + np.random.randn(100, 10) * 0.1
        
        fig, axes = plot_residuals(y_true, y_pred, feature_idx=0)
        
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3  # Three diagnostic plots
        plt.close(fig)
    
    def test_1d_arrays(self):
        """Test with 1D arrays."""
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        
        fig, axes = plot_residuals(y_true, y_pred)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Tests for Training Plots
# =============================================================================

class TestPlotTrainingHistory:
    """Tests for plot_training_history function."""
    
    def test_basic_history(self):
        """Test basic training history plot."""
        history = {
            "train_losses": [1.0, 0.5, 0.3, 0.2, 0.1],
            "val_losses": [1.1, 0.6, 0.4, 0.3, 0.2],
        }
        
        fig, axes = plot_training_history(history)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_metrics(self):
        """Test with additional metrics."""
        history = {
            "train_losses": [1.0, 0.5, 0.3],
            "val_losses": [1.1, 0.6, 0.4],
            "r2": [0.5, 0.7, 0.8],
            "pearson": [0.6, 0.8, 0.9],
        }
        
        fig, axes = plot_training_history(history)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotActiveLearningProgress:
    """Tests for plot_active_learning_progress function."""
    
    def test_basic_progress(self):
        """Test basic AL progress plot."""
        metrics = [
            {"r2": 0.5, "mse": 0.5},
            {"r2": 0.6, "mse": 0.4},
            {"r2": 0.7, "mse": 0.3},
        ]
        
        fig, ax = plot_active_learning_progress(metrics, metric_name="r2")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Tests for Utility Functions
# =============================================================================

class TestCreateFigureGrid:
    """Tests for create_figure_grid function."""
    
    def test_basic_grid(self):
        """Test basic figure grid creation."""
        fig, axes = create_figure_grid(6, n_cols=3)
        
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 6
        plt.close(fig)
    
    def test_single_plot(self):
        """Test single plot grid."""
        fig, axes = create_figure_grid(1)
        
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 1
        plt.close(fig)


class TestSaveFigure:
    """Tests for save_figure function."""
    
    def test_save_png(self):
        """Test saving figure as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            save_figure(fig, path)
            
            assert path.exists()
        
        plt.close(fig)
    
    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "test.png"
            save_figure(fig, path)
            
            assert path.exists()
        
        plt.close(fig)


# =============================================================================
# Tests for InterpretationConfig
# =============================================================================

class TestInterpretationConfig:
    """Tests for InterpretationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = InterpretationConfig()
        
        assert config.shap_n_samples == 100
        assert config.ig_n_steps == 50
        assert config.ig_baseline == "zero"
        assert config.gradient_method == "vanilla"
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = InterpretationConfig(
            shap_n_samples=50,
            ig_n_steps=100,
            gradient_method="smooth",
        )
        
        assert config.shap_n_samples == 50
        assert config.ig_n_steps == 100
        assert config.gradient_method == "smooth"


# =============================================================================
# Tests for GradientInterpreter
# =============================================================================

class TestGradientInterpreter:
    """Tests for GradientInterpreter."""
    
    def test_initialization(self, mock_model):
        """Test interpreter initialization."""
        config = InterpretationConfig(device="cpu")
        interpreter = GradientInterpreter(mock_model, config)
        
        assert interpreter.model is mock_model
    
    def test_explain(self, mock_model, sample_embeddings):
        """Test gradient explanation."""
        config = InterpretationConfig(device="cpu")
        interpreter = GradientInterpreter(mock_model, config)
        
        results = interpreter.explain(
            sample_embeddings,
            task="cell_painting",
        )
        
        assert "gradients" in results
        assert "importance" in results
        assert "saliency" in results
        assert results["gradients"].shape == sample_embeddings.shape
    
    def test_explain_with_target_idx(self, mock_model, sample_embeddings):
        """Test gradient explanation for specific target."""
        config = InterpretationConfig(device="cpu")
        interpreter = GradientInterpreter(mock_model, config)
        
        results = interpreter.explain(
            sample_embeddings,
            task="cell_painting",
            target_idx=0,
        )
        
        assert "gradients" in results
    
    def test_smooth_gradients(self, mock_model, sample_embeddings):
        """Test SmoothGrad method."""
        config = InterpretationConfig(
            device="cpu",
            smooth_grad_n_samples=5,  # Few samples for speed
        )
        interpreter = GradientInterpreter(mock_model, config)
        
        results = interpreter.explain(
            sample_embeddings,
            task="cell_painting",
            method="smooth",
        )
        
        assert "gradients" in results
    
    def test_feature_importance(self, mock_model, sample_embeddings):
        """Test aggregated feature importance."""
        config = InterpretationConfig(device="cpu")
        interpreter = GradientInterpreter(mock_model, config)
        
        importance = interpreter.feature_importance(
            sample_embeddings,
            task="cell_painting",
            aggregate="mean",
        )
        
        assert importance.shape == (sample_embeddings.shape[1],)
    
    def test_feature_importance_aggregations(self, mock_model, sample_embeddings):
        """Test different aggregation methods."""
        config = InterpretationConfig(device="cpu")
        interpreter = GradientInterpreter(mock_model, config)
        
        for aggregate in ["mean", "max", "l2"]:
            importance = interpreter.feature_importance(
                sample_embeddings,
                aggregate=aggregate,
            )
            assert importance.shape == (sample_embeddings.shape[1],)


# =============================================================================
# Tests for IntegratedGradientsInterpreter
# =============================================================================

class TestIntegratedGradientsInterpreter:
    """Tests for IntegratedGradientsInterpreter."""
    
    def test_initialization(self, mock_model):
        """Test interpreter initialization."""
        config = InterpretationConfig(device="cpu")
        interpreter = IntegratedGradientsInterpreter(mock_model, config)
        
        assert interpreter.model is mock_model
    
    def test_explain(self, mock_model, sample_embeddings):
        """Test integrated gradients explanation."""
        config = InterpretationConfig(device="cpu", ig_n_steps=10)
        interpreter = IntegratedGradientsInterpreter(mock_model, config)
        
        results = interpreter.explain(
            sample_embeddings,
            task="cell_painting",
        )
        
        assert "attributions" in results
        assert "convergence_delta" in results
        assert results["attributions"].shape == sample_embeddings.shape
    
    def test_different_baselines(self, mock_model, sample_embeddings):
        """Test different baseline methods."""
        config = InterpretationConfig(device="cpu", ig_n_steps=5)
        
        for baseline_method in ["zero", "random", "mean"]:
            config.ig_baseline = baseline_method
            interpreter = IntegratedGradientsInterpreter(mock_model, config)
            
            results = interpreter.explain(sample_embeddings)
            assert "attributions" in results
    
    def test_custom_baseline(self, mock_model, sample_embeddings):
        """Test with custom baseline."""
        config = InterpretationConfig(device="cpu", ig_n_steps=5)
        interpreter = IntegratedGradientsInterpreter(mock_model, config)
        
        custom_baseline = torch.zeros_like(sample_embeddings)
        
        results = interpreter.explain(
            sample_embeddings,
            baseline=custom_baseline,
        )
        
        assert "attributions" in results
    
    def test_explain_batch(self, mock_model, sample_embeddings):
        """Test batch explanation with progress."""
        config = InterpretationConfig(device="cpu", ig_n_steps=5, batch_size=5)
        interpreter = IntegratedGradientsInterpreter(mock_model, config)
        
        results = interpreter.explain_batch(
            sample_embeddings,
            show_progress=False,
        )
        
        assert results["attributions"].shape[0] == sample_embeddings.shape[0]


# =============================================================================
# Tests for FeatureAblationInterpreter
# =============================================================================

class TestFeatureAblationInterpreter:
    """Tests for FeatureAblationInterpreter."""
    
    def test_initialization(self, mock_model):
        """Test interpreter initialization."""
        config = InterpretationConfig(device="cpu")
        interpreter = FeatureAblationInterpreter(mock_model, config)
        
        assert interpreter.model is mock_model
    
    def test_explain(self, mock_model):
        """Test feature ablation explanation."""
        config = InterpretationConfig(device="cpu")
        interpreter = FeatureAblationInterpreter(mock_model, config)
        
        # Use small input for speed
        small_input = torch.randn(5, 128)
        
        results = interpreter.explain(small_input, task="cell_painting")
        
        assert "importance" in results
        assert results["importance"].shape == (128,)
    
    def test_compute_group_importance(self, mock_model, sample_embeddings):
        """Test group importance computation."""
        config = InterpretationConfig(device="cpu")
        interpreter = FeatureAblationInterpreter(mock_model, config)
        
        feature_groups = {
            "group_a": (0, 64),
            "group_b": (64, 128),
        }
        
        importance = interpreter.compute_group_importance(
            sample_embeddings,
            feature_groups=feature_groups,
        )
        
        assert "group_a" in importance
        assert "group_b" in importance
    
    def test_ablation_methods(self, mock_model):
        """Test different ablation methods."""
        small_input = torch.randn(3, 128)
        
        for method in ["zero", "mean", "noise"]:
            config = InterpretationConfig(device="cpu", ablation_method=method)
            interpreter = FeatureAblationInterpreter(mock_model, config)
            
            results = interpreter.explain(small_input)
            assert "importance" in results


# =============================================================================
# Tests for SHAPInterpreter
# =============================================================================

class TestSHAPInterpreter:
    """Tests for SHAPInterpreter."""
    
    def test_initialization(self, mock_model, background_data):
        """Test interpreter initialization."""
        config = InterpretationConfig(device="cpu")
        interpreter = SHAPInterpreter(mock_model, background_data, config)
        
        assert interpreter.model is mock_model
        assert interpreter.background_data is background_data
    
    def test_explain_approximate(self, mock_model, sample_embeddings, background_data):
        """Test approximate SHAP values."""
        config = InterpretationConfig(device="cpu")
        interpreter = SHAPInterpreter(mock_model, background_data, config)
        
        results = interpreter.explain_approximate(
            sample_embeddings[:3],  # Small batch
            task="cell_painting",
            n_samples=5,  # Few samples for speed
        )
        
        assert "shap_values" in results
        assert results["shap_values"].shape == (3, 128)
    
    def test_without_background_data(self, mock_model, sample_embeddings):
        """Test that explain without background raises error."""
        config = InterpretationConfig(device="cpu")
        interpreter = SHAPInterpreter(mock_model, None, config)
        
        with pytest.raises(ValueError, match="Background data required"):
            interpreter.explain(sample_embeddings)


# =============================================================================
# Tests for AttentionAnalyser
# =============================================================================

class TestAttentionAnalyser:
    """Tests for AttentionAnalyser."""
    
    def test_initialization(self, mock_model):
        """Test analyser initialization."""
        analyser = AttentionAnalyser(mock_model)
        
        assert analyser.model is mock_model
    
    def test_get_attention_weights_no_attention(self, mock_model, sample_embeddings):
        """Test getting attention weights from model without attention."""
        analyser = AttentionAnalyser(mock_model)
        
        weights = analyser.get_attention_weights(sample_embeddings)
        
        # Model has no attention layers, should return empty dict
        assert isinstance(weights, dict)
    
    def test_get_fusion_weights_no_attention(self, mock_model, sample_embeddings):
        """Test getting fusion weights from model without attention."""
        analyser = AttentionAnalyser(mock_model)
        
        weights = analyser.get_fusion_weights(sample_embeddings)
        
        # Model has no attention layers, should return empty dict
        assert weights == {}


# =============================================================================
# Tests for ModelInterpreter
# =============================================================================

class TestModelInterpreter:
    """Tests for ModelInterpreter unified interface."""
    
    def test_initialization(self, mock_model, background_data):
        """Test interpreter initialization."""
        config = InterpretationConfig(device="cpu")
        interpreter = ModelInterpreter(mock_model, background_data, config)
        
        assert interpreter.model is mock_model
    
    def test_gradient_importance(self, mock_model, sample_embeddings):
        """Test gradient importance method."""
        config = InterpretationConfig(device="cpu")
        interpreter = ModelInterpreter(mock_model, config=config)
        
        results = interpreter.gradient_importance(sample_embeddings)
        
        assert "gradients" in results
        assert "feature_importance" in results
    
    def test_integrated_gradients(self, mock_model, sample_embeddings):
        """Test integrated gradients method."""
        config = InterpretationConfig(device="cpu", ig_n_steps=5)
        interpreter = ModelInterpreter(mock_model, config=config)
        
        results = interpreter.integrated_gradients(sample_embeddings)
        
        assert "attributions" in results
    
    def test_feature_ablation(self, mock_model):
        """Test feature ablation method."""
        config = InterpretationConfig(device="cpu")
        interpreter = ModelInterpreter(mock_model, config=config)
        
        small_input = torch.randn(3, 128)
        results = interpreter.feature_ablation(small_input)
        
        assert "importance" in results
    
    def test_feature_ablation_with_groups(self, mock_model, sample_embeddings):
        """Test feature ablation with groups."""
        config = InterpretationConfig(device="cpu")
        interpreter = ModelInterpreter(mock_model, config=config)
        
        feature_groups = {"group_a": (0, 64), "group_b": (64, 128)}
        results = interpreter.feature_ablation(
            sample_embeddings,
            feature_groups=feature_groups,
        )
        
        assert "group_a" in results
    
    def test_generate_report(self, mock_model):
        """Test generating comprehensive report."""
        config = InterpretationConfig(device="cpu", ig_n_steps=3)
        interpreter = ModelInterpreter(mock_model, config=config)
        
        small_input = torch.randn(3, 128)
        report = interpreter.generate_report(
            small_input,
            include_shap=False,
        )
        
        assert "task" in report
        assert "gradient" in report
        assert "integrated_gradients" in report
        assert "summary" in report


# =============================================================================
# Tests for Convenience Functions
# =============================================================================

class TestComputeFeatureImportance:
    """Tests for compute_feature_importance function."""
    
    def test_gradient_method(self, mock_model, sample_embeddings):
        """Test gradient method."""
        importance = compute_feature_importance(
            mock_model,
            sample_embeddings,
            method="gradient",
        )
        
        assert importance.shape == (sample_embeddings.shape[1],)
    
    def test_ig_method(self, mock_model, sample_embeddings):
        """Test integrated gradients method."""
        with patch.object(
            IntegratedGradientsInterpreter,
            "__init__",
            lambda self, model, config=None: setattr(self, "config", config or InterpretationConfig(device="cpu", ig_n_steps=3)) or setattr(self, "model", model.to("cpu")) or setattr(self, "device", torch.device("cpu")),
        ):
            importance = compute_feature_importance(
                mock_model,
                sample_embeddings,
                method="ig",
            )
            
            assert importance.shape == (sample_embeddings.shape[1],)
    
    def test_invalid_method_raises(self, mock_model, sample_embeddings):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_feature_importance(
                mock_model,
                sample_embeddings,
                method="invalid",
            )


class TestExplainPrediction:
    """Tests for explain_prediction function."""
    
    def test_basic_explanation(self, mock_model, sample_embeddings):
        """Test basic prediction explanation."""
        explanation = explain_prediction(
            mock_model,
            sample_embeddings,
            task="cell_painting",
        )
        
        assert "gradient" in explanation
        assert "integrated_gradients" in explanation


class TestGetEmbeddingContribution:
    """Tests for get_embedding_contribution function."""
    
    def test_embedding_contribution(self, mock_model, sample_embeddings):
        """Test getting embedding contributions."""
        embedding_ranges = {
            "esm2": (0, 64),
            "physicochemical": (64, 128),
        }
        
        contributions = get_embedding_contribution(
            mock_model,
            sample_embeddings,
            embedding_ranges,
        )
        
        assert "esm2" in contributions
        assert "physicochemical" in contributions


# =============================================================================
# Integration Tests
# =============================================================================

class TestAnalysisIntegration:
    """Integration tests for analysis module."""
    
    def test_clustering_and_visualisation(self, sample_features, sample_ids):
        """Test clustering followed by visualisation."""
        # Cluster
        result = hierarchical_clustering(
            sample_features,
            n_clusters=3,
            sample_ids=sample_ids,
        )
        
        # Visualise with cluster labels
        fig, ax = plot_embedding_space(
            sample_features,
            labels=result.labels,
            method="pca",
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_interpretation_workflow(self, mock_model, sample_embeddings):
        """Test complete interpretation workflow."""
        config = InterpretationConfig(device="cpu", ig_n_steps=3)
        
        # Gradient interpretation
        grad_interpreter = GradientInterpreter(mock_model, config)
        grad_results = grad_interpreter.explain(sample_embeddings)
        
        # Get top important features
        importance = grad_results["importance"].mean(dim=0).numpy()
        top_features = np.argsort(importance)[-10:]
        
        assert len(top_features) == 10
    
    def test_full_analysis_pipeline(self, mock_model, sample_features):
        """Test full analysis pipeline from clustering to interpretation."""
        # Step 1: Cluster samples
        cluster_result = kmeans_clustering(sample_features, n_clusters=3)
        
        # Step 2: Get model predictions (mock)
        embeddings = torch.randn(sample_features.shape[0], 128)
        
        # Step 3: Interpret predictions
        config = InterpretationConfig(device="cpu", ig_n_steps=3)
        interpreter = ModelInterpreter(mock_model, config=config)
        
        grad_results = interpreter.gradient_importance(embeddings[:5])
        
        # Step 4: Visualise
        fig, ax = plot_embedding_space(
            sample_features,
            labels=cluster_result.labels,
            method="pca",
        )
        
        assert cluster_result.n_clusters == 3
        assert "feature_importance" in grad_results
        plt.close(fig)