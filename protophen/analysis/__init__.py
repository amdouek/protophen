"""
Analysis and interpretation modules for ProToPhen.

This package provides tools for:
- Clustering phenotypes and proteins
- Visualising embeddings and predictions
- Interpreting model predictions
"""

from protophen.analysis.clustering import (
    PhenotypeClustering,
    ClusteringResult,
    hierarchical_clustering,
    kmeans_clustering,
    cluster_phenotypes,
    compute_cluster_enrichment,
    HDBSCAN_AVAILABLE,
)

# Conditionally import hdbscan_clustering
if HDBSCAN_AVAILABLE:
    from protophen.analysis.clustering import hdbscan_clustering

from protophen.analysis.visualisation import (
    PlotConfig,
    plot_embedding_space,
    plot_umap,
    plot_tsne,
    plot_pca,
    plot_heatmap,
    plot_correlation_matrix,
    plot_clustermap,
    plot_feature_distributions,
    plot_uncertainty_distribution,
    plot_prediction_scatter,
    plot_residuals,
    plot_training_history,
    plot_active_learning_progress,
    create_figure_grid,
    save_figure,
)

from protophen.analysis.interpretation import (
    ModelInterpreter,
    GradientInterpreter,
    IntegratedGradientsInterpreter,
    FeatureAblationInterpreter,
    SHAPInterpreter,
    AttentionAnalyser,
    InterpretationConfig,
    compute_feature_importance,
    explain_prediction,
    get_embedding_contribution,
)

__all__ = [
    # Clustering
    "PhenotypeClustering",
    "ClusteringResult",
    "hierarchical_clustering",
    "kmeans_clustering",
    "cluster_phenotypes",
    "compute_cluster_enrichment",
    "HDBSCAN_AVAILABLE",
    # Visualisation
    "PlotConfig",
    "plot_embedding_space",
    "plot_umap",
    "plot_tsne",
    "plot_pca",
    "plot_heatmap",
    "plot_correlation_matrix",
    "plot_clustermap",
    "plot_feature_distributions",
    "plot_uncertainty_distribution",
    "plot_prediction_scatter",
    "plot_residuals",
    "plot_training_history",
    "plot_active_learning_progress",
    "create_figure_grid",
    "save_figure",
    # Interpretation
    "ModelInterpreter",
    "GradientInterpreter",
    "IntegratedGradientsInterpreter",
    "FeatureAblationInterpreter",
    "SHAPInterpreter",
    "AttentionAnalyser",
    "InterpretationConfig",
    "compute_feature_importance",
    "explain_prediction",
    "get_embedding_contribution",
]