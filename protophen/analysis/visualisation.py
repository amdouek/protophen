"""
Visualisation utilities for ProToPhen.

This module provides plotting functions for visualising embeddings,
predictions, and analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from protophen.utils.logging import logger


# =============================================================================
# Plot Configuration
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration for plots."""
    
    # Figure size
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 100
    
    # Style
    style: str = "whitegrid"
    context: str = "notebook"
    palette: str = "husl"
    
    # Font sizes
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    
    # Colors
    cmap: str = "viridis"
    diverging_cmap: str = "RdBu_r"
    
    # Save options
    save_format: str = "png"
    transparent: bool = False
    
    def apply(self) -> None:
        """Apply configuration to matplotlib/seaborn."""
        sns.set_style(self.style)
        sns.set_context(self.context)
        plt.rcParams["figure.figsize"] = self.figsize
        plt.rcParams["figure.dpi"] = self.dpi
        plt.rcParams["axes.titlesize"] = self.title_size
        plt.rcParams["axes.labelsize"] = self.label_size
        plt.rcParams["xtick.labelsize"] = self.tick_size
        plt.rcParams["ytick.labelsize"] = self.tick_size
        plt.rcParams["legend.fontsize"] = self.legend_size


# Default configuration
DEFAULT_CONFIG = PlotConfig()


def _setup_plot(config: Optional[PlotConfig] = None) -> PlotConfig:
    """Setup plot configuration."""
    config = config or DEFAULT_CONFIG
    config.apply()
    return config


# =============================================================================
# Embedding Visualisations
# =============================================================================

def plot_embedding_space(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: Literal["umap", "tsne", "pca"] = "umap",
    color_by: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    config: Optional[PlotConfig] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot embedding space with dimensionality reduction.
    
    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features)
        labels: Optional cluster or group labels for coloring
        method: Dimensionality reduction method
        color_by: Alternative continuous values for coloring
        title: Plot title
        ax: Existing axes to plot on
        config: Plot configuration
        **kwargs: Additional arguments for reduction method
        
    Returns:
        Tuple of (Figure, Axes)
        
    Example:
        >>> fig, ax = plot_embedding_space(
        ...     embeddings=protein_embeddings,
        ...     labels=cluster_labels,
        ...     method="umap",
        ...     title="Protein Embedding Space",
        ... )
    """
    config = _setup_plot(config)
    
    # Reduce dimensions
    if method == "umap":
        coords = _reduce_umap(embeddings, **kwargs)
    elif method == "tsne":
        coords = _reduce_tsne(embeddings, **kwargs)
    elif method == "pca":
        coords = _reduce_pca(embeddings, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure
    
    # Plot
    if color_by is not None:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_by,
            cmap=config.cmap,
            alpha=0.7,
            s=30,
        )
        plt.colorbar(scatter, ax=ax)
    elif labels is not None:
        unique_labels = np.unique(labels)
        colors = sns.color_palette(config.palette, n_colors=len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[colors[i]],
                label=f"Cluster {label}" if isinstance(label, int) else str(label),
                alpha=0.7,
                s=30,
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30)
    
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def _reduce_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Reduce dimensions using UMAP."""
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        **kwargs,
    )
    return reducer.fit_transform(embeddings)


def _reduce_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Reduce dimensions using t-SNE."""
    from sklearn.manifold import TSNE
    
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        **kwargs,
    )
    return reducer.fit_transform(embeddings)


def _reduce_pca(
    embeddings: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Reduce dimensions using PCA."""
    from sklearn.decomposition import PCA
    
    reducer = PCA(n_components=2, **kwargs)
    return reducer.fit_transform(embeddings)


def plot_umap(embeddings: np.ndarray, **kwargs) -> Tuple[Figure, Axes]:
    """Convenience function for UMAP plot."""
    return plot_embedding_space(embeddings, method="umap", **kwargs)


def plot_tsne(embeddings: np.ndarray, **kwargs) -> Tuple[Figure, Axes]:
    """Convenience function for t-SNE plot."""
    return plot_embedding_space(embeddings, method="tsne", **kwargs)


def plot_pca(embeddings: np.ndarray, **kwargs) -> Tuple[Figure, Axes]:
    """Convenience function for PCA plot."""
    return plot_embedding_space(embeddings, method="pca", **kwargs)


# =============================================================================
# Heatmaps and Matrices
# =============================================================================

def plot_heatmap(
    data: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    annot: bool = False,
    config: Optional[PlotConfig] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot a heatmap.
    
    Args:
        data: 2D array to visualise
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Plot title
        cmap: Colormap
        center: Center value for diverging colormap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        figsize: Figure size
        annot: Whether to annotate cells with values
        config: Plot configuration
        
    Returns:
        Tuple of (Figure, Axes)
    """
    config = _setup_plot(config)
    
    figsize = figsize or config.figsize
    cmap = cmap or (config.diverging_cmap if center is not None else config.cmap)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=".2f" if annot else None,
        xticklabels=col_labels if col_labels else False,
        yticklabels=row_labels if row_labels else False,
    )
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def plot_correlation_matrix(
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
    method: str = "pearson",
    figsize: Optional[Tuple[float, float]] = None,
    config: Optional[PlotConfig] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot correlation matrix of features.
    
    Args:
        data: Feature matrix of shape (n_samples, n_features)
        feature_names: Names of features
        title: Plot title
        method: Correlation method ('pearson', 'spearman')
        figsize: Figure size
        config: Plot configuration
        
    Returns:
        Tuple of (Figure, Axes)
    """
    import pandas as pd
    
    config = _setup_plot(config)
    
    # Compute correlation
    df = pd.DataFrame(data, columns=feature_names)
    corr = df.corr(method=method)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    figsize = figsize or (12, 10)
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=config.diverging_cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.5},
    )
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig, ax


def plot_clustermap(
    data: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    row_colors: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
    config: Optional[PlotConfig] = None,
) -> sns.matrix.ClusterGrid:
    """
    Plot hierarchically clustered heatmap.
    
    Args:
        data: 2D array to visualise
        row_labels: Labels for rows
        col_labels: Labels for columns
        row_colors: Colors for row annotations
        title: Plot title
        figsize: Figure size
        config: Plot configuration
        
    Returns:
        ClusterGrid object
    """
    config = _setup_plot(config)
    
    g = sns.clustermap(
        data,
        figsize=figsize,
        cmap=config.cmap,
        row_colors=row_colors,
        xticklabels=col_labels if col_labels else False,
        yticklabels=row_labels if row_labels else False,
        dendrogram_ratio=(0.1, 0.1),
    )
    
    if title:
        g.fig.suptitle(title, y=1.02)
    
    return g


# =============================================================================
# Distribution Plots
# =============================================================================

def plot_feature_distributions(
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_features: int = 20,
    figsize: Optional[Tuple[float, float]] = None,
    config: Optional[PlotConfig] = None,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot distributions of features.
    
    Args:
        data: Feature matrix of shape (n_samples, n_features)
        feature_names: Names of features
        n_features: Number of features to plot
        figsize: Figure size
        config: Plot configuration
        
    Returns:
        Tuple of (Figure, array of Axes)
    """
    config = _setup_plot(config)
    
    n_features = min(n_features, data.shape[1])
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    figsize = figsize or (4 * n_cols, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        values = data[:, i]
        
        sns.histplot(values, ax=ax, kde=True)
        
        if feature_names:
            ax.set_title(feature_names[i][:30], fontsize=9)
        else:
            ax.set_title(f"Feature {i}", fontsize=9)
        
        ax.set_xlabel("")
    
    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def plot_uncertainty_distribution(
    uncertainty_scores: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
    title: str = "Uncertainty Distribution",
    figsize: Tuple[float, float] = (10, 6),
    config: Optional[PlotConfig] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot distribution of uncertainty scores.
    
    Args:
        uncertainty_scores: Array of uncertainty scores
        selected_indices: Indices of selected samples to highlight
        title: Plot title
        figsize: Figure size
        config: Plot configuration
        
    Returns:
        Tuple of (Figure, Axes)
    """
    config = _setup_plot(config)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    sns.histplot(uncertainty_scores, ax=ax, kde=True, alpha=0.7, label="All samples")
    
    # Highlight selected samples
    if selected_indices is not None:
        selected_scores = uncertainty_scores[selected_indices]
        ax.axvline(
            selected_scores.min(),
            color="red",
            linestyle="--",
            label=f"Selected (n={len(selected_indices)})",
        )
        
        for score in selected_scores:
            ax.axvline(score, color="red", alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel("Uncertainty Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig, ax


# =============================================================================
# Prediction Plots
# =============================================================================

def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_features: int = 9,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    config: Optional[PlotConfig] = None,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot scatter plots of predictions vs true values.
    
    Args:
        y_true: True values of shape (n_samples, n_features)
        y_pred: Predicted values of shape (n_samples, n_features)
        feature_names: Names of features
        n_features: Number of features to plot
        title: Overall plot title
        figsize: Figure size
        config: Plot configuration
        
    Returns:
        Tuple of (Figure, array of Axes)
    """
    from scipy import stats
    
    config = _setup_plot(config)
    
    n_features = min(n_features, y_true.shape[1])
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    figsize = figsize or (4 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        # Scatter plot
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20)
        
        # Add diagonal line
        lims = [
            min(true_vals.min(), pred_vals.min()),
            max(true_vals.max(), pred_vals.max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.75, label="y=x")
        
        # Compute correlation
        r, _ = stats.pearsonr(true_vals, pred_vals)
        
        # Set labels
        if feature_names:
            ax.set_title(f"{feature_names[i][:25]}\nr={r:.3f}", fontsize=9)
        else:
            ax.set_title(f"Feature {i}\nr={r:.3f}", fontsize=9)
        
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    
    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    if title:
        fig.suptitle(title, y=1.02, fontsize=14)
    
    plt.tight_layout()
    return fig, axes


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_idx: int = 0,
    feature_name: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    config: Optional[PlotConfig] = None,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot residual diagnostics for a single feature.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        feature_idx: Feature index to analyse
        feature_name: Feature name
        figsize: Figure size
        config: Plot configuration
        
    Returns:
        Tuple of (Figure, Axes)
    """
    config = _setup_plot(config)
    
    if y_true.ndim > 1:
        true_vals = y_true[:, feature_idx]
        pred_vals = y_pred[:, feature_idx]
    else:
        true_vals = y_true
        pred_vals = y_pred
    
    residuals = true_vals - pred_vals
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Residual vs predicted
    axes[0].scatter(pred_vals, residuals, alpha=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    
    # Residual histogram
    sns.histplot(residuals, ax=axes[1], kde=True)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot")
    
    feature_name = feature_name or f"Feature {feature_idx}"
    fig.suptitle(f"Residual Diagnostics: {feature_name}", y=1.02)
    
    plt.tight_layout()
    return fig, axes


# =============================================================================
# Training Plots
# =============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[float, float] = (12, 5),
    config: Optional[PlotConfig] = None,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary with 'train_losses', 'val_losses', and optional metrics
        title: Plot title
        figsize: Figure size
        config: Plot configuration
        
    Returns:
        Tuple of (Figure, Axes)
    """
    config = _setup_plot(config)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    ax = axes[0]
    if "train_losses" in history:
        ax.plot(history["train_losses"], label="Train Loss")
    if "val_losses" in history:
        ax.plot(history["val_losses"], label="Val Loss")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.set_yscale("log")
    
    # Metrics plot
    ax = axes[1]
    metric_keys = [k for k in history.keys() if k not in ["train_losses", "val_losses"]]
    
    for key in metric_keys[:5]:  # Plot up to 5 metrics
        if isinstance(history[key], list) and len(history[key]) > 0:
            ax.plot(history[key], label=key)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    
    return fig, axes


def plot_active_learning_progress(
    iteration_metrics: List[Dict[str, float]],
    metric_name: str = "r2",
    figsize: Tuple[float, float] = (10, 6),
    config: Optional[PlotConfig] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot active learning progress over iterations.
    
    Args:
        iteration_metrics: List of metric dictionaries per iteration
        metric_name: Which metric to plot
        figsize: Figure size
        config: Plot configuration
        
    Returns:
        Tuple of (Figure, Axes)
    """
    config = _setup_plot(config)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = range(1, len(iteration_metrics) + 1)
    
    # Extract metric values
    values = [m.get(metric_name, np.nan) for m in iteration_metrics]
    
    ax.plot(iterations, values, "o-", linewidth=2, markersize=8)
    
    ax.set_xlabel("AL Iteration")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"Active Learning Progress: {metric_name.upper()}")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


# =============================================================================
# Utility Functions
# =============================================================================

def create_figure_grid(
    n_plots: int,
    n_cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (4, 4),
) -> Tuple[Figure, np.ndarray]:
    """
    Create a grid of subplots.
    
    Args:
        n_plots: Number of plots
        n_cols: Number of columns
        figsize_per_plot: Size of each subplot
        
    Returns:
        Tuple of (Figure, flattened array of Axes containing exactly n_plots axes)
    """
    # Adjust columns if fewer plots than default columns
    n_cols = min(n_cols, n_plots)
    
    n_rows = (n_plots + n_cols - 1) // n_cols
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Hide any extra axes beyond n_plots
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    
    return fig, axes[:n_plots]


def save_figure(
    fig: Figure,
    path: Union[str, Path],
    dpi: int = 150,
    transparent: bool = False,
    bbox_inches: str = "tight",
) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Figure to save
        path: Output path
        dpi: Resolution
        transparent: Transparent background
        bbox_inches: Bounding box setting
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        path,
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
    )
    logger.info(f"Saved figure to {path}")