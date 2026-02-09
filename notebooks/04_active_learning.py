# %% [markdown]
# # 04: Active Learning for Intelligent Experiment Selection
#
# **ProToPhen: Protein-to-Phenotype Foundation Model**
#
# This notebook demonstrates how to use active learning to intelligently select
# the next proteins to test experimentally. Active learning is crucial for efficient
# exploration of the vast protein design space, allowing us to maximise information
# gain while minimising costly wet-lab experiments.
#
# ## Overview
#
# In this notebook, we will:
#
# 1. **Understand Active Learning** - Why it matters for protein design
# 2. **Quantify Uncertainty** - MC Dropout, ensembles, and heteroscedastic methods
# 3. **Apply Acquisition Functions** - Strategies for selecting informative samples
# 4. **Select Experiments** - Use the `ExperimentSelector` for batch selection
# 5. **Simulate an AL Loop** - Watch model performance improve iteratively
# 6. **Interpret Selections** - Understand *why* proteins were selected
# 7. **Visualise Results** - Uncertainty distributions and embedding spaces
#
# ---

# %% [markdown]
# ## 1. Introduction
#
# ### 1.1 Why Active Learning for Protein Design?
#
# De novo protein design presents a fundamental challenge: the space of possible
# proteins is astronomically large (20^100 for a 100-residue protein), yet each
# experimental validation is expensive and time-consuming. Traditional approaches
# either:
#
# - **Random sampling**: Inefficient, may miss interesting regions
# - **Exhaustive screening**: Impossible at scale
# - **Expert intuition**: Biased, not scalable
#
# **Active learning** provides a principled framework for selecting the most
# informative experiments, balancing:
#
# - **Exploration**: Testing proteins in uncertain regions of the prediction space
# - **Exploitation**: Focusing on proteins predicted to have desirable properties
#
# ### 1.2 The Explore-Exploit Trade-off
#
# ```
#                     High Uncertainty
#                           │
#              ┌────────────┼────────────┐
#              │            │            │
#              │  EXPLORE   │  EXPLORE   │
#              │  (unknown) │  + EXPLOIT │
#              │            │            │
# Low Value ───┼────────────┼────────────┼─── High Value
#              │            │            │
#              │   IGNORE   │  EXPLOIT   │
#              │            │  (greedy)  │
#              │            │            │
#              └────────────┼────────────┘
#                           │
#                     Low Uncertainty
# ```
#
# Different acquisition functions navigate this trade-off differently:
# - **Uncertainty Sampling**: Pure exploration
# - **Expected Improvement**: Balanced exploration-exploitation
# - **Diversity Sampling**: Coverage of the design space
# - **Hybrid Methods**: Combine multiple objectives
#
# ### 1.3 Active Learning Loop Overview
#
# ```
# ┌─────────────────────────────────────────────────────────────┐
# │                    ACTIVE LEARNING LOOP                      │
# │                                                              │
# │   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
# │   │  Train   │───▶│  Select  │───▶│   Label  │              │
# │   │  Model   │    │  Samples │    │  (Expt)  │              │
# │   └──────────┘    └──────────┘    └──────────┘              │
# │        ▲                               │                     │
# │        │                               │                     │
# │        └───────────────────────────────┘                     │
# │                                                              │
# │   Iteration 1 → Iteration 2 → ... → Convergence             │
# └─────────────────────────────────────────────────────────────┘
# ```

# %%
# =============================================================================
# 2. Setup
# =============================================================================

# Standard library imports
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Scientific computing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configure plotting
sns.set_style("whitegrid")
sns.set_context("notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

print("✓ Libraries imported successfully")

# %%
# ProToPhen imports
from protophen.models.protophen import (
    ProToPhenModel,
    ProToPhenConfig,
    create_protophen_model,
)
from protophen.data.dataset import (
    ProtoPhenDataset,
    ProteinInferenceDataset,
    DatasetConfig,
    ProtoPhenSample,
)
from protophen.data.loaders import (
    create_dataloader,
    create_dataloaders,
)
from protophen.active_learning.uncertainty import (
    UncertaintyEstimate,
    UncertaintyType,
    MCDropoutEstimator,
    EnsembleEstimator,
    HeteroscedasticEstimator,
    estimate_uncertainty,
    get_uncertainty_ranking,
)
from protophen.active_learning.acquisition import (
    AcquisitionFunction,
    UncertaintySampling,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    DiversitySampling,
    HybridAcquisition,
    BatchAcquisition,
    compute_acquisition_scores,
)
from protophen.active_learning.selection import (
    SelectionConfig,
    SelectionResult,
    ExperimentSelector,
    ActiveLearningLoop,
    select_next_experiments,
    rank_by_uncertainty,
    select_diverse_subset,
)
from protophen.analysis.interpretation import (
    GradientInterpreter,
    IntegratedGradientsInterpreter,
    FeatureAblationInterpreter,
    ModelInterpreter,
    compute_feature_importance,
    get_embedding_contribution,
)
from protophen.analysis.visualisation import (
    plot_embedding_space,
    plot_uncertainty_distribution,
    plot_active_learning_progress,
    plot_heatmap,
    PlotConfig,
)
from protophen.analysis.clustering import (
    PhenotypeClustering,
    cluster_phenotypes,
    ClusteringResult,
)

print("✓ ProToPhen modules imported successfully")

# %%
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ### 2.1 Load Trained Model
#
# We'll load the model trained in Notebook 3. For this demonstration, we'll
# create a model and simulate training if the checkpoint isn't available.

# %%
# Configuration matching Notebook 3
PROTEIN_EMBEDDING_DIM = 1280 + 439  # ESM-2 (1280) + Physicochemical (439)
CELL_PAINTING_DIM = 1500
LATENT_DIM = 256

# Model configuration
model_config = ProToPhenConfig(
    protein_embedding_dim=PROTEIN_EMBEDDING_DIM,
    encoder_hidden_dims=[1024, 512],
    encoder_output_dim=LATENT_DIM,
    decoder_hidden_dims=[512, 1024],
    cell_painting_dim=CELL_PAINTING_DIM,
    predict_viability=True,
    mc_dropout=True,  # Enable MC Dropout for uncertainty
    encoder_dropout=0.1,
    decoder_dropout=0.1,
)

# Create model
model = ProToPhenModel(model_config)
model = model.to(device)

print(f"Model created: {model}")
print(f"\nModel summary:")
for key, value in model.summary().items():
    print(f"  {key}: {value}")

# %%
# Try to load pre-trained weights, or use randomly initialised model for demo
MODEL_PATH = Path("checkpoints/protophen_best.pt")

if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded model from {MODEL_PATH}")
else:
    print("! No checkpoint found - using randomly initialised model for demonstration")
    print("  In practice, load your trained model from Notebook 3")

model.eval()

# %% [markdown]
# ### 2.2 Generate Synthetic Data for Demonstration
#
# We'll create synthetic data to demonstrate the active learning pipeline.
# In practice, you would load your actual protein embeddings and experimental data.

# %%
def generate_synthetic_data(
    n_train: int = 100,
    n_pool: int = 500,
    embedding_dim: int = PROTEIN_EMBEDDING_DIM,
    phenotype_dim: int = CELL_PAINTING_DIM,
    n_clusters: int = 5,
    noise_level: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Generate synthetic protein-phenotype data for demonstration.
    
    Creates data with cluster structure to simulate different protein families
    having different phenotypic effects.
    """
    np.random.seed(seed)
    
    # Generate cluster centers in embedding space
    cluster_centers_emb = np.random.randn(n_clusters, embedding_dim) * 2
    
    # Generate cluster centers in phenotype space (mapping)
    cluster_centers_phen = np.random.randn(n_clusters, phenotype_dim)
    
    def generate_samples(n_samples, prefix="protein"):
        """Generate samples distributed across clusters."""
        # Assign samples to clusters
        cluster_assignments = np.random.randint(0, n_clusters, n_samples)
        
        # Generate embeddings around cluster centers
        embeddings = np.zeros((n_samples, embedding_dim))
        phenotypes = np.zeros((n_samples, phenotype_dim))
        
        for i in range(n_samples):
            cluster = cluster_assignments[i]
            # Embedding with cluster-specific variation
            embeddings[i] = cluster_centers_emb[cluster] + np.random.randn(embedding_dim) * 0.5
            # Phenotype with noise
            phenotypes[i] = cluster_centers_phen[cluster] + np.random.randn(phenotype_dim) * noise_level
        
        # Create sample IDs
        ids = [f"{prefix}_{i:04d}" for i in range(n_samples)]
        
        return {
            "embeddings": embeddings.astype(np.float32),
            "phenotypes": phenotypes.astype(np.float32),
            "ids": ids,
            "clusters": cluster_assignments,
        }
    
    # Generate training data (labeled)
    train_data = generate_samples(n_train, prefix="train")
    
    # Generate pool data (unlabeled candidates)
    pool_data = generate_samples(n_pool, prefix="pool")
    
    # Ground truth function for "labeling" in simulation
    # This simulates what we'd get from wet-lab experiments
    ground_truth_mapping = np.random.randn(embedding_dim, phenotype_dim) * 0.1
    
    return train_data, pool_data, ground_truth_mapping


# Generate data
train_data, pool_data, ground_truth = generate_synthetic_data()

print("Generated synthetic data:")
print(f"  Training samples: {len(train_data['ids'])}")
print(f"  Pool (candidate) samples: {len(pool_data['ids'])}")
print(f"  Embedding dimension: {train_data['embeddings'].shape[1]}")
print(f"  Phenotype dimension: {train_data['phenotypes'].shape[1]}")

# %%
# Create PyTorch datasets
train_samples = [
    ProtoPhenSample(
        protein_id=train_data["ids"][i],
        protein_embedding=train_data["embeddings"][i],
        phenotypes={"cell_painting": train_data["phenotypes"][i]},
        metadata={"cluster": int(train_data["clusters"][i])},
    )
    for i in range(len(train_data["ids"]))
]

train_dataset = ProtoPhenDataset(samples=train_samples)
print(f"Training dataset: {train_dataset}")

# Create pool dataset (for inference - no labels)
pool_inference_dataset = ProteinInferenceDataset(
    protein_embeddings=pool_data["embeddings"],
    protein_ids=pool_data["ids"],
    protein_names=pool_data["ids"],  # Use IDs as names for simplicity
)
print(f"Pool dataset: {pool_inference_dataset}")

# Create DataLoaders
train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
pool_loader = create_dataloader(pool_inference_dataset, batch_size=32, shuffle=False)

print(f"\nDataLoaders created:")
print(f"  Train: {len(train_loader)} batches")
print(f"  Pool: {len(pool_loader)} batches")

# %% [markdown]
# ---
# ## 3. Uncertainty Quantification
#
# Uncertainty quantification is fundamental to active learning. We need to know
# *where* our model is uncertain to prioritise those regions for exploration.
#
# ### 3.1 Types of Uncertainty
#
# | Type | Description | Source | Reducible? |
# |------|-------------|--------|------------|
# | **Epistemic** | Model uncertainty | Limited training data | Yes (with more data) |
# | **Aleatoric** | Data uncertainty | Inherent noise in measurements | No |
# | **Total** | Combined uncertainty | Both sources | Partially |
#
# For active learning, we primarily care about **epistemic uncertainty** - regions
# where more data would help the model learn better.

# %%
# Demonstrate the UncertaintyType enum
print("Available uncertainty types:")
for unc_type in UncertaintyType:
    print(f"  - {unc_type.value}: {unc_type.name}")

# %% [markdown]
# ### 3.2 MC Dropout Estimation
#
# Monte Carlo Dropout is a simple yet effective method for uncertainty estimation.
# By keeping dropout enabled during inference and running multiple forward passes,
# we can estimate epistemic uncertainty through the variance of predictions.
#
# **Key idea**: Different dropout masks → different predictions → variance ≈ uncertainty

# %%
# Create MC Dropout estimator
mc_estimator = MCDropoutEstimator(
    n_samples=20,  # Number of forward passes
    tasks=["cell_painting"],
    device=str(device),
)

print("MC Dropout Estimator Configuration:")
print(f"  Number of MC samples: {mc_estimator.n_samples}")
print(f"  Tasks: {mc_estimator.tasks}")

# %%
# Estimate uncertainty for the pool
print("Estimating uncertainty with MC Dropout...")
mc_uncertainty = mc_estimator.estimate(
    model=model,
    dataloader=pool_loader,
    show_progress=True,
    return_samples=True,  # Keep all MC samples for analysis
)

print(f"\nUncertainty estimation complete:")
print(f"  Number of samples: {mc_uncertainty.n_samples}")
print(f"  Number of features: {mc_uncertainty.n_features}")
print(f"  Mean prediction shape: {mc_uncertainty.mean.shape}")
print(f"  Epistemic uncertainty shape: {mc_uncertainty.epistemic.shape}")
if mc_uncertainty.samples is not None:
    print(f"  MC samples shape: {mc_uncertainty.samples.shape}")

# %%
# Examine the uncertainty estimates
epistemic_per_sample = mc_uncertainty.get_uncertainty(
    uncertainty_type=UncertaintyType.EPISTEMIC,
    reduction="mean",  # Average across phenotype features
)

print("Epistemic uncertainty statistics:")
print(f"  Min: {epistemic_per_sample.min():.4f}")
print(f"  Max: {epistemic_per_sample.max():.4f}")
print(f"  Mean: {epistemic_per_sample.mean():.4f}")
print(f"  Std: {epistemic_per_sample.std():.4f}")

# %%
# Visualise MC Dropout predictions for a single sample
sample_idx = 0
if mc_uncertainty.samples is not None:
    mc_samples = mc_uncertainty.samples[:, sample_idx, :10]  # First 10 features
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.hist(mc_samples[:, i], bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(mc_uncertainty.mean[sample_idx, i], color='red', 
                   linestyle='--', label='Mean')
        ax.set_title(f"Feature {i}")
        ax.set_xlabel("Prediction")
        if i == 0:
            ax.legend()
    
    plt.suptitle(f"MC Dropout Predictions for Sample {sample_idx}\n"
                 f"(20 forward passes, showing first 10 features)", y=1.02)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 3.3 Ensemble Estimation
#
# Deep ensembles use multiple models trained with different random seeds.
# The disagreement between ensemble members indicates uncertainty.
#
# **Advantages over MC Dropout:**
# - Often better calibrated
# - Each member can learn different aspects
#
# **Disadvantages:**
# - Requires training multiple models
# - Higher computational cost

# %%
# Create a small ensemble for demonstration
print("Creating model ensemble...")
n_ensemble = 3
ensemble_models = []

for i in range(n_ensemble):
    # Create model with different random initialisation
    torch.manual_seed(42 + i)
    ensemble_model = ProToPhenModel(model_config)
    ensemble_model = ensemble_model.to(device)
    ensemble_model.eval()
    ensemble_models.append(ensemble_model)
    
print(f"Created ensemble with {n_ensemble} models")

# %%
# Estimate uncertainty using ensemble
ensemble_estimator = EnsembleEstimator(
    tasks=["cell_painting"],
    device=str(device),
)

print("Estimating uncertainty with ensemble...")
ensemble_uncertainty = ensemble_estimator.estimate(
    model=ensemble_models,
    dataloader=pool_loader,
    show_progress=True,
)

print(f"\nEnsemble uncertainty estimation complete:")
print(f"  Epistemic uncertainty shape: {ensemble_uncertainty.epistemic.shape}")

# Compare with MC Dropout
ensemble_epistemic = ensemble_uncertainty.get_uncertainty(
    uncertainty_type=UncertaintyType.EPISTEMIC,
    reduction="mean",
)

print(f"\nComparison of uncertainty methods:")
print(f"  MC Dropout - Mean epistemic: {epistemic_per_sample.mean():.4f}")
print(f"  Ensemble   - Mean epistemic: {ensemble_epistemic.mean():.4f}")

# %% [markdown]
# ### 3.4 Heteroscedastic Estimation (Aleatoric Uncertainty)
#
# If your model predicts both mean and variance (heteroscedastic), you can
# estimate aleatoric (data) uncertainty directly from the predicted variance.
#
# Combined with MC Dropout, this gives both uncertainty types:
# - **Epistemic**: From MC sample variance
# - **Aleatoric**: From predicted variance
# - **Total**: √(epistemic² + aleatoric²)

# %%
# Note: This requires a model that predicts variance
# For demonstration, we'll show the API

print("Heteroscedastic uncertainty estimation:")
print("  Requires model with predict_uncertainty=True")
print("  Model predicts: (mean, log_variance)")
print("  Aleatoric uncertainty = sqrt(exp(log_variance))")
print("\nExample configuration:")

hetero_config = ProToPhenConfig(
    protein_embedding_dim=PROTEIN_EMBEDDING_DIM,
    cell_painting_dim=CELL_PAINTING_DIM,
    predict_uncertainty=True,  # Enable variance prediction
)
print(f"  predict_uncertainty: {hetero_config.predict_uncertainty}")

# %%
# Convenience function for uncertainty estimation
print("Using convenience function estimate_uncertainty()...")

quick_uncertainty = estimate_uncertainty(
    model=model,
    dataloader=pool_loader,
    method="mc_dropout",
    n_samples=10,  # Fewer samples for quick estimation
    tasks=["cell_painting"],
    show_progress=True,
)

print(f"Quick uncertainty estimation complete:")
print(f"  Shape: {quick_uncertainty.total.shape}")

# %% [markdown]
# ### 3.5 Visualising Uncertainty
#
# Let's visualise the uncertainty distribution across our candidate pool.

# %%
# Plot uncertainty distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Histogram of uncertainty scores
ax = axes[0]
ax.hist(epistemic_per_sample, bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel("Epistemic Uncertainty (mean across features)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Uncertainty Scores")
ax.axvline(np.percentile(epistemic_per_sample, 90), color='red', 
           linestyle='--', label='90th percentile')
ax.legend()

# 2. Uncertainty vs sample index (to check for patterns)
ax = axes[1]
ax.scatter(range(len(epistemic_per_sample)), epistemic_per_sample, alpha=0.5, s=10)
ax.set_xlabel("Sample Index")
ax.set_ylabel("Epistemic Uncertainty")
ax.set_title("Uncertainty by Sample")

# 3. Uncertainty by cluster (using ground truth clusters)
ax = axes[2]
clusters = pool_data["clusters"]
for cluster_id in np.unique(clusters):
    mask = clusters == cluster_id
    cluster_unc = epistemic_per_sample[mask]
    ax.scatter(
        np.random.normal(cluster_id, 0.1, len(cluster_unc)),
        cluster_unc,
        alpha=0.5,
        label=f"Cluster {cluster_id}",
        s=20,
    )
ax.set_xlabel("Cluster")
ax.set_ylabel("Epistemic Uncertainty")
ax.set_title("Uncertainty by Protein Cluster")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 4. Uncertainty Analysis
#
# Now let's dive deeper into understanding and using uncertainty estimates.
#
# ### 4.1 The UncertaintyEstimate Object

# %%
# Examine the UncertaintyEstimate object
print("UncertaintyEstimate attributes:")
print(f"  mean: Predictions averaged over MC samples - shape {mc_uncertainty.mean.shape}")
print(f"  epistemic: Model uncertainty - shape {mc_uncertainty.epistemic.shape}")
print(f"  aleatoric: Data uncertainty - {mc_uncertainty.aleatoric}")
print(f"  total: Combined uncertainty - shape {mc_uncertainty.total.shape}")
print(f"  sample_ids: {mc_uncertainty.sample_ids[:5] if mc_uncertainty.sample_ids else None}...")

# %%
# Convert to dictionary for storage
uncertainty_dict = mc_uncertainty.to_dict()
print("Uncertainty estimate as dictionary:")
for key, value in uncertainty_dict.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: array of shape {value.shape}")
    elif isinstance(value, list):
        print(f"  {key}: list of length {len(value)}")
    else:
        print(f"  {key}: {value}")

# %% [markdown]
# ### 4.2 Aggregation Methods
#
# Uncertainty is computed per-feature. We need to aggregate across features
# to get a single score per sample for ranking.

# %%
# Compare different aggregation methods
aggregation_methods = ["mean", "sum", "max"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, method in zip(axes, aggregation_methods):
    scores = mc_uncertainty.get_uncertainty(
        uncertainty_type=UncertaintyType.EPISTEMIC,
        reduction=method,
    )
    ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(f"Uncertainty ({method})")
    ax.set_ylabel("Count")
    ax.set_title(f"Aggregation: {method.upper()}")
    
    # Show statistics
    textstr = f'Mean: {scores.mean():.3f}\nStd: {scores.std():.3f}'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Effect of Aggregation Method on Uncertainty Scores", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Correlation between aggregation methods
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

scores_mean = mc_uncertainty.get_uncertainty(reduction="mean")
scores_sum = mc_uncertainty.get_uncertainty(reduction="sum")
scores_max = mc_uncertainty.get_uncertainty(reduction="max")

pairs = [
    (scores_mean, scores_sum, "Mean", "Sum"),
    (scores_mean, scores_max, "Mean", "Max"),
    (scores_sum, scores_max, "Sum", "Max"),
]

for ax, (s1, s2, n1, n2) in zip(axes, pairs):
    ax.scatter(s1, s2, alpha=0.3, s=10)
    r = np.corrcoef(s1, s2)[0, 1]
    ax.set_xlabel(f"Uncertainty ({n1})")
    ax.set_ylabel(f"Uncertainty ({n2})")
    ax.set_title(f"r = {r:.3f}")

plt.suptitle("Correlation Between Aggregation Methods", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 Ranking by Uncertainty

# %%
# Rank samples by uncertainty
ranked_indices, ranked_scores = get_uncertainty_ranking(
    mc_uncertainty,
    uncertainty_type=UncertaintyType.TOTAL,
    reduction="mean",
    ascending=False,  # Highest uncertainty first
)

print("Top 10 most uncertain samples:")
print("-" * 50)
for i in range(10):
    sample_idx = ranked_indices[i]
    sample_id = mc_uncertainty.sample_ids[sample_idx] if mc_uncertainty.sample_ids else f"sample_{sample_idx}"
    print(f"  Rank {i+1}: {sample_id} (uncertainty: {ranked_scores[i]:.4f})")

# %%
# Visualise top uncertain vs low uncertain samples in embedding space
n_top = 50
n_bottom = 50

top_uncertain_idx = ranked_indices[:n_top]
bottom_uncertain_idx = ranked_indices[-n_bottom:]
middle_idx = ranked_indices[len(ranked_indices)//2 - 25: len(ranked_indices)//2 + 25]

# Reduce embeddings for visualisation
from sklearn.decomposition import PCA

embeddings = pool_data["embeddings"]
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 8))

# Plot all points
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
            c='lightgray', alpha=0.3, s=20, label='All candidates')

# Highlight high/low uncertainty
plt.scatter(embeddings_2d[bottom_uncertain_idx, 0], embeddings_2d[bottom_uncertain_idx, 1],
            c='blue', alpha=0.7, s=40, label=f'Low uncertainty (n={n_bottom})')
plt.scatter(embeddings_2d[top_uncertain_idx, 0], embeddings_2d[top_uncertain_idx, 1],
            c='red', alpha=0.7, s=40, label=f'High uncertainty (n={n_top})')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Protein Embedding Space: Uncertainty Regions")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 Identifying High-Uncertainty Regions
#
# Let's understand what characterises high-uncertainty proteins.

# %%
# Analyse high vs low uncertainty samples
high_unc_embeddings = embeddings[top_uncertain_idx]
low_unc_embeddings = embeddings[bottom_uncertain_idx]

# Compare feature statistics
high_unc_mean = high_unc_embeddings.mean(axis=0)
low_unc_mean = low_unc_embeddings.mean(axis=0)

# Find most discriminative features
feature_diff = np.abs(high_unc_mean - low_unc_mean)
top_diff_features = np.argsort(feature_diff)[-20:]

print("Top 20 most discriminative embedding features (high vs low uncertainty):")
for i, feat_idx in enumerate(top_diff_features[::-1]):
    print(f"  Feature {feat_idx}: diff = {feature_diff[feat_idx]:.4f}")

# %%
# Cluster the high-uncertainty samples
print("Clustering high-uncertainty samples...")

high_unc_clustering = cluster_phenotypes(
    high_unc_embeddings,
    method="kmeans",
    n_clusters=3,
    sample_ids=[mc_uncertainty.sample_ids[i] for i in top_uncertain_idx] 
               if mc_uncertainty.sample_ids else None,
)

print(high_unc_clustering.summary())

# %% [markdown]
# ---
# ## 5. Acquisition Functions
#
# Acquisition functions score samples for selection. Different functions
# embody different exploration-exploitation strategies.
#
# ### 5.1 Uncertainty Sampling
#
# The simplest approach: select samples where the model is most uncertain.

# %%
# Create uncertainty sampling acquisition function
uncertainty_acq = UncertaintySampling(
    uncertainty_type=UncertaintyType.EPISTEMIC,
    reduction="mean",
)

# Score all samples
uncertainty_scores = uncertainty_acq.score(mc_uncertainty)

# Select top samples
n_select = 20
uncertainty_selected = uncertainty_acq.select(mc_uncertainty, n_select=n_select)

print(f"Uncertainty Sampling - Selected {len(uncertainty_selected)} samples")
print(f"Selected indices: {uncertainty_selected[:10]}...")
print(f"Score range: {uncertainty_scores[uncertainty_selected].min():.4f} - "
      f"{uncertainty_scores[uncertainty_selected].max():.4f}")

# %% [markdown]
# ### 5.2 Expected Improvement (EI)
#
# EI balances exploration and exploitation by computing the expected improvement
# over the current best observation.

# %%
# Create Expected Improvement acquisition function
ei_acq = ExpectedImprovement(
    target_feature_idx=0,  # Optimise first phenotype feature
    maximise=True,  # We want to maximise this feature
    xi=0.01,  # Exploration-exploitation trade-off
)

# Set best observed value (from training data)
best_observed = train_data["phenotypes"][:, 0].max()
ei_acq.set_best_value(best_observed)
print(f"Best observed value (feature 0): {best_observed:.4f}")

# Score samples
ei_scores = ei_acq.score(mc_uncertainty, best_value=best_observed)

# Select top samples
ei_selected = ei_acq.select(mc_uncertainty, n_select=n_select)

print(f"\nExpected Improvement - Selected {len(ei_selected)} samples")
print(f"Score range: {ei_scores[ei_selected].min():.4f} - "
      f"{ei_scores[ei_selected].max():.4f}")

# %% [markdown]
# ### 5.3 Probability of Improvement (PI)
#
# PI computes the probability that a sample will improve upon the best observation.
# Simpler than EI but can be too greedy.

# %%
# Create Probability of Improvement acquisition function
pi_acq = ProbabilityOfImprovement(
    target_feature_idx=0,
    maximise=True,
    xi=0.01,
)

# Score samples
pi_scores = pi_acq.score(mc_uncertainty, best_value=best_observed)

# Select top samples
pi_selected = pi_acq.select(mc_uncertainty, n_select=n_select)

print(f"Probability of Improvement - Selected {len(pi_selected)} samples")
print(f"Score range: {pi_scores[pi_selected].min():.4f} - "
      f"{pi_scores[pi_selected].max():.4f}")

# %% [markdown]
# ### 5.4 Diversity Sampling
#
# Pure diversity sampling selects samples that are spread out in embedding space,
# ensuring good coverage of the design space.
#
# **Methods:**
# - **k-means++**: Samples proportional to squared distance
# - **maxmin**: Greedily maximises minimum distance
# - **DPP**: Determinantal Point Process (probabilistic diversity)

# %%
# Create diversity sampling with different methods
methods = ["kmeans++", "maxmin"]

diversity_results = {}
for method in methods:
    div_acq = DiversitySampling(method=method, metric="euclidean")
    selected = div_acq.select(
        mc_uncertainty,
        n_select=n_select,
        embeddings=pool_data["embeddings"],
    )
    diversity_results[method] = selected
    print(f"Diversity ({method}) - Selected {len(selected)} samples")

# %%
# Visualise diversity selection vs uncertainty selection
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Reduce to 2D for visualisation
embeddings_2d = pca.transform(pool_data["embeddings"])

# Plot 1: Uncertainty sampling
ax = axes[0]
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgray', alpha=0.3, s=10)
ax.scatter(embeddings_2d[uncertainty_selected, 0], embeddings_2d[uncertainty_selected, 1],
           c='red', s=50, label='Selected')
ax.set_title("Uncertainty Sampling")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()

# Plot 2: k-means++ diversity
ax = axes[1]
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgray', alpha=0.3, s=10)
ax.scatter(embeddings_2d[diversity_results["kmeans++"], 0], 
           embeddings_2d[diversity_results["kmeans++"], 1],
           c='green', s=50, label='Selected')
ax.set_title("Diversity Sampling (k-means++)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()

# Plot 3: maxmin diversity
ax = axes[2]
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgray', alpha=0.3, s=10)
ax.scatter(embeddings_2d[diversity_results["maxmin"], 0],
           embeddings_2d[diversity_results["maxmin"], 1],
           c='purple', s=50, label='Selected')
ax.set_title("Diversity Sampling (maxmin)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()

plt.suptitle("Comparison of Selection Strategies", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.5 Hybrid Acquisition (Uncertainty + Diversity)
#
# The hybrid approach combines uncertainty and diversity to select samples
# that are both informative AND diverse. This is often the best strategy
# for batch active learning.

# %%
# Create hybrid acquisition function
hybrid_acq = HybridAcquisition(
    uncertainty_weight=0.7,
    diversity_weight=0.3,
    uncertainty_type=UncertaintyType.EPISTEMIC,
    diversity_method="kmeans++",
)

# Select samples
hybrid_selected = hybrid_acq.select(
    mc_uncertainty,
    n_select=n_select,
    embeddings=pool_data["embeddings"],
)

print(f"Hybrid Acquisition - Selected {len(hybrid_selected)} samples")
print(f"  Uncertainty weight: {hybrid_acq.uncertainty_weight}")
print(f"  Diversity weight: {hybrid_acq.diversity_weight}")

# %%
# Compare selections across strategies
print("\nOverlap analysis between selection strategies:")
print("-" * 50)

strategies = {
    "Uncertainty": set(uncertainty_selected),
    "EI": set(ei_selected),
    "Diversity (kmeans++)": set(diversity_results["kmeans++"]),
    "Hybrid": set(hybrid_selected),
}

# Create overlap matrix
strategy_names = list(strategies.keys())
overlap_matrix = np.zeros((len(strategy_names), len(strategy_names)))

for i, name1 in enumerate(strategy_names):
    for j, name2 in enumerate(strategy_names):
        overlap = len(strategies[name1] & strategies[name2])
        overlap_matrix[i, j] = overlap

print("Selection overlap (number of shared samples):")
overlap_df = pd.DataFrame(overlap_matrix, index=strategy_names, columns=strategy_names)
print(overlap_df.to_string())

# %%
# Visualise overlap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(overlap_matrix, annot=True, fmt=".0f", cmap="YlOrRd",
            xticklabels=strategy_names, yticklabels=strategy_names, ax=ax)
ax.set_title(f"Selection Overlap (n_select={n_select})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.6 Comparing Acquisition Strategies
#
# Let's compare the properties of samples selected by different strategies.

# %%
# Analyse properties of selected samples
def analyse_selection(name, indices, embeddings, uncertainty_scores, clusters):
    """Analyse properties of selected samples."""
    selected_emb = embeddings[indices]
    selected_unc = uncertainty_scores[indices]
    selected_clusters = clusters[indices]
    
    # Compute pairwise distances (diversity measure)
    if len(indices) > 1:
        distances = cdist(selected_emb, selected_emb)
        avg_distance = distances[np.triu_indices(len(indices), k=1)].mean()
    else:
        avg_distance = 0
    
    # Cluster coverage
    cluster_coverage = len(np.unique(selected_clusters)) / len(np.unique(clusters))
    
    return {
        "name": name,
        "mean_uncertainty": selected_unc.mean(),
        "std_uncertainty": selected_unc.std(),
        "avg_pairwise_distance": avg_distance,
        "cluster_coverage": cluster_coverage,
        "n_clusters_covered": len(np.unique(selected_clusters)),
    }


# Analyse all strategies
analyses = []
all_strategies = {
    "Uncertainty": uncertainty_selected,
    "Expected Improvement": ei_selected,
    "Diversity (kmeans++)": diversity_results["kmeans++"],
    "Diversity (maxmin)": diversity_results["maxmin"],
    "Hybrid": hybrid_selected,
}

for name, indices in all_strategies.items():
    analysis = analyse_selection(
        name, indices,
        pool_data["embeddings"],
        epistemic_per_sample,
        pool_data["clusters"],
    )
    analyses.append(analysis)

analysis_df = pd.DataFrame(analyses)
analysis_df = analysis_df.set_index("name")
print("\nSelection Strategy Analysis:")
print("=" * 80)
print(analysis_df.to_string())

# %%
# Visualise strategy comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Uncertainty distribution
ax = axes[0]
for name, indices in all_strategies.items():
    selected_unc = epistemic_per_sample[indices]
    ax.hist(selected_unc, bins=15, alpha=0.5, label=name)
ax.set_xlabel("Epistemic Uncertainty")
ax.set_ylabel("Count")
ax.set_title("Uncertainty of Selected Samples")
ax.legend()

# Plot 2: Diversity (pairwise distance)
ax = axes[1]
names = list(all_strategies.keys())
distances = [analysis_df.loc[n, "avg_pairwise_distance"] for n in names]
bars = ax.bar(range(len(names)), distances, color=sns.color_palette("husl", len(names)))
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel("Avg Pairwise Distance")
ax.set_title("Diversity of Selections")

# Plot 3: Cluster coverage
ax = axes[2]
coverage = [analysis_df.loc[n, "cluster_coverage"] * 100 for n in names]
bars = ax.bar(range(len(names)), coverage, color=sns.color_palette("husl", len(names)))
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel("Cluster Coverage (%)")
ax.set_title("Coverage of Protein Clusters")
ax.set_ylim(0, 100)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 6. Experiment Selection
#
# The `ExperimentSelector` class provides a high-level interface for selecting
# experiments, coordinating uncertainty estimation and acquisition functions.
#
# ### 6.1 SelectionConfig Options

# %%
# Explore selection configuration options
print("SelectionConfig options:")
print("-" * 50)

# Create a default config
default_config = SelectionConfig()

for field_name in default_config.__dataclass_fields__:
    value = getattr(default_config, field_name)
    print(f"  {field_name}: {value}")

# %%
# Create a custom configuration
custom_config = SelectionConfig(
    n_select=15,
    uncertainty_method="mc_dropout",
    n_mc_samples=20,
    acquisition_method="hybrid",
    uncertainty_weight=0.6,
    diversity_weight=0.4,
    uncertainty_type=UncertaintyType.EPISTEMIC,
    tasks=["cell_painting"],
)

print("Custom SelectionConfig:")
print(f"  Batch size: {custom_config.n_select}")
print(f"  Uncertainty method: {custom_config.uncertainty_method}")
print(f"  Acquisition method: {custom_config.acquisition_method}")
print(f"  Uncertainty weight: {custom_config.uncertainty_weight}")
print(f"  Diversity weight: {custom_config.diversity_weight}")

# %% [markdown]
# ### 6.2 ExperimentSelector Class

# %%
# Create experiment selector
selector = ExperimentSelector(
    model=model,
    config=custom_config,
    device=str(device),
)

print("ExperimentSelector created:")
print(f"  Uncertainty estimator: {type(selector.uncertainty_estimator).__name__}")
print(f"  Acquisition function: {type(selector.acquisition_fn).__name__}")

# %% [markdown]
# ### 6.3 Single-Round Selection

# %%
# Perform selection
print("Performing experiment selection...")
selection_result = selector.select(
    dataloader=pool_loader,
    embeddings=pool_data["embeddings"],
    show_progress=True,
)

print("\n" + selection_result.summary())

# %%
# Examine selection result
print("\nSelectionResult attributes:")
print(f"  selected_indices: {selection_result.selected_indices}")
print(f"  selected_ids: {selection_result.selected_ids}")
print(f"  acquisition_scores: {selection_result.acquisition_scores}")

# %%
# Get detailed information about selected proteins
selected_proteins = selection_result.get_selected_proteins()

print("\nSelected proteins (detailed):")
print("-" * 70)
for protein in selected_proteins[:10]:
    print(f"  Rank {protein['rank']:2d}: {protein['id']} | "
          f"Acq Score: {protein['acquisition_score']:.4f} | "
          f"Uncertainty: {protein['uncertainty']:.4f}")

# %%
# Visualise selected samples in embedding space
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Selected samples
ax = axes[0]
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
           c='lightgray', alpha=0.3, s=20, label='Pool')
ax.scatter(embeddings_2d[selection_result.selected_indices, 0],
           embeddings_2d[selection_result.selected_indices, 1],
           c='red', s=80, edgecolors='black', linewidths=1.5,
           label=f'Selected (n={len(selection_result.selected_indices)})')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Selected Experiments in Embedding Space")
ax.legend()

# Plot 2: Acquisition scores
ax = axes[1]
scatter = ax.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1],
    c=selection_result.all_scores,
    cmap='YlOrRd', alpha=0.7, s=20,
)
ax.scatter(embeddings_2d[selection_result.selected_indices, 0],
           embeddings_2d[selection_result.selected_indices, 1],
           facecolors='none', edgecolors='black', s=100, linewidths=2)
plt.colorbar(scatter, ax=ax, label='Acquisition Score')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Acquisition Scores (selected highlighted)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.4 Iterative Selection with Exclusion
#
# When running multiple selection rounds, we need to exclude previously selected
# samples. The `ExperimentSelector` handles this automatically.

# %%
# Reset selector for demonstration
selector.reset_exclusions()

# Perform multiple selection rounds
n_rounds = 3
n_per_round = 10

print(f"Performing {n_rounds} rounds of selection ({n_per_round} samples each)...")
print("=" * 60)

round_results = []
for round_num in range(n_rounds):
    result = selector.select(
        dataloader=pool_loader,
        embeddings=pool_data["embeddings"],
        n_select=n_per_round,
        show_progress=False,
    )
    round_results.append(result)
    
    print(f"\nRound {round_num + 1}:")
    print(f"  Selected IDs: {result.selected_ids[:5]}...")
    print(f"  Excluded so far: {len(selector.config.exclude_ids)}")

# %%
# Get selection summary
summary = selector.get_selection_summary()
print("\nSelection Summary:")
print(f"  Total selections: {summary['n_selections']}")
print(f"  Total samples selected: {summary['total_selected']}")
print(f"  Mean acquisition score: {summary['mean_acquisition_score']:.4f}")

# %%
# Visualise iterative selection
fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.viridis(np.linspace(0, 1, n_rounds))

ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
           c='lightgray', alpha=0.2, s=10, label='Pool')

for i, result in enumerate(round_results):
    ax.scatter(
        embeddings_2d[result.selected_indices, 0],
        embeddings_2d[result.selected_indices, 1],
        c=[colors[i]], s=80, edgecolors='black', linewidths=1,
        label=f'Round {i+1}',
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Iterative Selection Across Rounds")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.5 Selection Results Interpretation

# %%
# Convert result to dictionary for storage
result_dict = selection_result.to_dict()
print("Selection result as dictionary:")
for key, value in result_dict.items():
    if isinstance(value, list) and len(value) > 5:
        print(f"  {key}: {value[:5]}... (length {len(value)})")
    else:
        print(f"  {key}: {value}")

# %%
# Analyse what makes selected samples special
selected_idx = selection_result.selected_indices
unselected_idx = np.array([i for i in range(len(pool_data["ids"])) 
                           if i not in selected_idx])

print("Comparison: Selected vs Unselected")
print("-" * 50)

# Uncertainty comparison
selected_unc = epistemic_per_sample[selected_idx]
unselected_unc = epistemic_per_sample[unselected_idx]
print(f"Mean uncertainty - Selected: {selected_unc.mean():.4f}, "
      f"Unselected: {unselected_unc.mean():.4f}")

# Embedding statistics
selected_emb = pool_data["embeddings"][selected_idx]
unselected_emb = pool_data["embeddings"][unselected_idx]
print(f"Mean embedding norm - Selected: {np.linalg.norm(selected_emb, axis=1).mean():.2f}, "
      f"Unselected: {np.linalg.norm(unselected_emb, axis=1).mean():.2f}")

# Cluster distribution
selected_clusters = pool_data["clusters"][selected_idx]
print(f"Cluster distribution - Selected: {np.bincount(selected_clusters, minlength=5)}")

# %% [markdown]
# ---
# ## 7. Simulating Active Learning Loop
#
# Let's simulate a complete active learning loop to see how model performance
# improves with intelligently selected data.
#
# ### 7.1 Setup: Initial Training Set and Pool

# %%
# Create a larger pool for simulation
sim_train_data, sim_pool_data, sim_ground_truth = generate_synthetic_data(
    n_train=50,     # Small initial training set
    n_pool=500,     # Larger pool
    seed=123,
)

print("Simulation data:")
print(f"  Initial training: {len(sim_train_data['ids'])} samples")
print(f"  Pool: {len(sim_pool_data['ids'])} samples")

# %% [markdown]
# ### 7.2 Oracle Function (Simulated Wet-Lab)
#
# In real active learning, selected samples would be tested in the wet-lab.
# Here, we simulate this with a "ground truth" function.

# %%
def oracle_label_fn(
    sample_ids: List[str],
    pool_data: Dict,
    ground_truth: np.ndarray,
    noise_std: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Simulate wet-lab experiments by generating labels for selected proteins.
    
    In reality, this would be months of actual experiments!
    """
    # Find indices of selected samples
    id_to_idx = {pid: i for i, pid in enumerate(pool_data["ids"])}
    indices = [id_to_idx[sid] for sid in sample_ids if sid in id_to_idx]
    
    # Generate "experimental results" (phenotypes)
    embeddings = pool_data["embeddings"][indices]
    
    # Simple linear transformation plus noise
    phenotypes = embeddings @ ground_truth + np.random.randn(
        len(indices), ground_truth.shape[1]
    ) * noise_std
    
    return {
        "cell_painting": phenotypes.astype(np.float32),
        "indices": indices,
    }


# Test oracle
test_ids = sim_pool_data["ids"][:3]
test_labels = oracle_label_fn(test_ids, sim_pool_data, sim_ground_truth)
print(f"Oracle test - Generated labels for {len(test_ids)} samples")
print(f"  Phenotype shape: {test_labels['cell_painting'].shape}")

# %% [markdown]
# ### 7.3 Active Learning Loop Implementation

# %%
def run_al_simulation(
    initial_train_embeddings: np.ndarray,
    initial_train_phenotypes: np.ndarray,
    pool_embeddings: np.ndarray,
    pool_ids: List[str],
    pool_data: Dict,
    ground_truth: np.ndarray,
    n_iterations: int = 10,
    n_per_iteration: int = 10,
    acquisition_method: str = "hybrid",
    val_embeddings: Optional[np.ndarray] = None,
    val_phenotypes: Optional[np.ndarray] = None,
) -> Dict[str, List]:
    """
    Run active learning simulation.
    
    Returns history of metrics and selections.
    """
    history = {
        "iteration": [],
        "train_size": [],
        "train_mse": [],
        "val_mse": [],
        "val_r2": [],
        "selected_ids": [],
        "mean_uncertainty": [],
    }
    
    # Copy data to avoid modifying originals
    current_train_emb = initial_train_embeddings.copy()
    current_train_phen = initial_train_phenotypes.copy()
    remaining_pool_mask = np.ones(len(pool_ids), dtype=bool)
    
    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")
        
        # Create training dataset
        train_samples = [
            ProtoPhenSample(
                protein_id=f"train_{i}",
                protein_embedding=current_train_emb[i],
                phenotypes={"cell_painting": current_train_phen[i]},
            )
            for i in range(len(current_train_emb))
        ]
        train_dataset = ProtoPhenDataset(samples=train_samples)
        train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
        
        # Create new model and train
        iter_model = ProToPhenModel(model_config).to(device)
        optimiser = torch.optim.Adam(iter_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Simple training loop
        iter_model.train()
        for epoch in range(50):  # Quick training
            epoch_loss = 0
            for batch in train_loader:
                optimiser.zero_grad()
                emb = batch["protein_embedding"].to(device)
                target = batch["cell_painting"].to(device)
                
                output = iter_model(emb, tasks=["cell_painting"])
                loss = criterion(output["cell_painting"], target)
                
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
        
        train_mse = epoch_loss / len(train_loader)
        print(f"Training MSE: {train_mse:.4f}")
        
        # Evaluate on validation set if provided
        val_mse = None
        val_r2 = None
        if val_embeddings is not None:
            iter_model.eval()
            with torch.no_grad():
                val_emb = torch.from_numpy(val_embeddings).float().to(device)
                val_pred = iter_model(val_emb, tasks=["cell_painting"])["cell_painting"]
                val_pred = val_pred.cpu().numpy()
                
                val_mse = ((val_pred - val_phenotypes) ** 2).mean()
                
                # Compute R² per feature and average
                r2_scores = []
                for j in range(val_phenotypes.shape[1]):
                    ss_res = ((val_phenotypes[:, j] - val_pred[:, j]) ** 2).sum()
                    ss_tot = ((val_phenotypes[:, j] - val_phenotypes[:, j].mean()) ** 2).sum()
                    r2 = 1 - ss_res / (ss_tot + 1e-8)
                    r2_scores.append(r2)
                val_r2 = np.mean(r2_scores)
                
            print(f"Validation MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        
        # Create pool loader for remaining samples
        remaining_indices = np.where(remaining_pool_mask)[0]
        remaining_emb = pool_embeddings[remaining_indices]
        remaining_ids = [pool_ids[i] for i in remaining_indices]
        
        pool_inference = ProteinInferenceDataset(
            protein_embeddings=remaining_emb,
            protein_ids=remaining_ids,
        )
        pool_loader = create_dataloader(pool_inference, batch_size=32, shuffle=False)
        
        # Select samples
        selector_config = SelectionConfig(
            n_select=n_per_iteration,
            uncertainty_method="mc_dropout",
            n_mc_samples=15,
            acquisition_method=acquisition_method,
        )
        selector = ExperimentSelector(iter_model, selector_config, device=str(device))
        
        selection = selector.select(pool_loader, embeddings=remaining_emb, show_progress=False)
        
        # Get mean uncertainty of selected samples
        mean_unc = selection.uncertainty_estimates.get_uncertainty(reduction="mean").mean()
        
        # "Label" selected samples (simulate wet-lab)
        new_labels = oracle_label_fn(selection.selected_ids, pool_data, ground_truth)
        
        # Find original pool indices of selected samples
        id_to_orig_idx = {pid: i for i, pid in enumerate(pool_ids)}
        selected_orig_indices = [id_to_orig_idx[sid] for sid in selection.selected_ids]
        
        # Add to training set
        new_embeddings = pool_embeddings[selected_orig_indices]
        current_train_emb = np.vstack([current_train_emb, new_embeddings])
        current_train_phen = np.vstack([current_train_phen, new_labels["cell_painting"]])
        
        # Remove from pool
        remaining_pool_mask[selected_orig_indices] = False
        
        # Record history
        history["iteration"].append(iteration + 1)
        history["train_size"].append(len(current_train_emb))
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_r2"].append(val_r2)
        history["selected_ids"].append(selection.selected_ids)
        history["mean_uncertainty"].append(mean_unc)
        
        print(f"Selected {len(selection.selected_ids)} samples, "
              f"new training size: {len(current_train_emb)}")
    
    return history

# %% [markdown]
# ### 7.4 Run the Simulation

# %%
# Create validation set (subset of pool with known labels)
val_indices = np.random.choice(len(sim_pool_data["ids"]), size=100, replace=False)
val_embeddings = sim_pool_data["embeddings"][val_indices]
val_phenotypes = (val_embeddings @ sim_ground_truth + 
                  np.random.randn(100, sim_ground_truth.shape[1]) * 0.05).astype(np.float32)

print(f"Validation set: {len(val_embeddings)} samples")

# %%
# Run active learning simulation with hybrid acquisition
print("Running Active Learning Simulation (Hybrid Strategy)")
print("=" * 60)

al_history_hybrid = run_al_simulation(
    initial_train_embeddings=sim_train_data["embeddings"],
    initial_train_phenotypes=sim_train_data["phenotypes"],
    pool_embeddings=sim_pool_data["embeddings"],
    pool_ids=sim_pool_data["ids"],
    pool_data=sim_pool_data,
    ground_truth=sim_ground_truth,
    n_iterations=8,
    n_per_iteration=15,
    acquisition_method="hybrid",
    val_embeddings=val_embeddings,
    val_phenotypes=val_phenotypes,
)

# %%
# Run comparison with uncertainty-only acquisition
print("\n" + "=" * 60)
print("Running Active Learning Simulation (Uncertainty-Only Strategy)")
print("=" * 60)

al_history_uncertainty = run_al_simulation(
    initial_train_embeddings=sim_train_data["embeddings"],
    initial_train_phenotypes=sim_train_data["phenotypes"],
    pool_embeddings=sim_pool_data["embeddings"],
    pool_ids=sim_pool_data["ids"],
    pool_data=sim_pool_data,
    ground_truth=sim_ground_truth,
    n_iterations=8,
    n_per_iteration=15,
    acquisition_method="uncertainty",
    val_embeddings=val_embeddings,
    val_phenotypes=val_phenotypes,
)

# %%
# Run comparison with random acquisition (baseline)
print("\n" + "=" * 60)
print("Running Active Learning Simulation (Random Baseline)")
print("=" * 60)

# For random, we'll manually implement since it's not in acquisition functions
def run_random_baseline(
    initial_train_embeddings: np.ndarray,
    initial_train_phenotypes: np.ndarray,
    pool_embeddings: np.ndarray,
    pool_ids: List[str],
    pool_data: Dict,
    ground_truth: np.ndarray,
    n_iterations: int = 8,
    n_per_iteration: int = 15,
    val_embeddings: Optional[np.ndarray] = None,
    val_phenotypes: Optional[np.ndarray] = None,
) -> Dict[str, List]:
    """Run random selection baseline."""
    history = {
        "iteration": [],
        "train_size": [],
        "train_mse": [],
        "val_mse": [],
        "val_r2": [],
        "selected_ids": [],
        "mean_uncertainty": [],
    }
    
    current_train_emb = initial_train_embeddings.copy()
    current_train_phen = initial_train_phenotypes.copy()
    remaining_indices = list(range(len(pool_ids)))
    
    for iteration in range(n_iterations):
        print(f"\nRandom Baseline - Iteration {iteration + 1}/{n_iterations}")
        
        # Create and train model (same as before)
        train_samples = [
            ProtoPhenSample(
                protein_id=f"train_{i}",
                protein_embedding=current_train_emb[i],
                phenotypes={"cell_painting": current_train_phen[i]},
            )
            for i in range(len(current_train_emb))
        ]
        train_dataset = ProtoPhenDataset(samples=train_samples)
        train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
        
        iter_model = ProToPhenModel(model_config).to(device)
        optimiser = torch.optim.Adam(iter_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        iter_model.train()
        for epoch in range(50):
            epoch_loss = 0
            for batch in train_loader:
                optimiser.zero_grad()
                emb = batch["protein_embedding"].to(device)
                target = batch["cell_painting"].to(device)
                output = iter_model(emb, tasks=["cell_painting"])
                loss = criterion(output["cell_painting"], target)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
        
        train_mse = epoch_loss / len(train_loader)
        
        # Evaluate
        val_mse = None
        val_r2 = None
        if val_embeddings is not None:
            iter_model.eval()
            with torch.no_grad():
                val_emb = torch.from_numpy(val_embeddings).float().to(device)
                val_pred = iter_model(val_emb, tasks=["cell_painting"])["cell_painting"]
                val_pred = val_pred.cpu().numpy()
                
                val_mse = ((val_pred - val_phenotypes) ** 2).mean()
                r2_scores = []
                for j in range(val_phenotypes.shape[1]):
                    ss_res = ((val_phenotypes[:, j] - val_pred[:, j]) ** 2).sum()
                    ss_tot = ((val_phenotypes[:, j] - val_phenotypes[:, j].mean()) ** 2).sum()
                    r2 = 1 - ss_res / (ss_tot + 1e-8)
                    r2_scores.append(r2)
                val_r2 = np.mean(r2_scores)
        
        # Random selection
        n_select = min(n_per_iteration, len(remaining_indices))
        selected_local_indices = np.random.choice(len(remaining_indices), size=n_select, replace=False)
        selected_pool_indices = [remaining_indices[i] for i in selected_local_indices]
        selected_ids = [pool_ids[i] for i in selected_pool_indices]
        
        # Get labels
        new_labels = oracle_label_fn(selected_ids, pool_data, ground_truth)
        
        # Update training set
        new_embeddings = pool_embeddings[selected_pool_indices]
        current_train_emb = np.vstack([current_train_emb, new_embeddings])
        current_train_phen = np.vstack([current_train_phen, new_labels["cell_painting"]])
        
        # Remove from pool
        for idx in sorted(selected_local_indices, reverse=True):
            remaining_indices.pop(idx)
        
        # Record
        history["iteration"].append(iteration + 1)
        history["train_size"].append(len(current_train_emb))
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_r2"].append(val_r2)
        history["selected_ids"].append(selected_ids)
        history["mean_uncertainty"].append(0)  # Not computed for random
        
        print(f"Val R²: {val_r2:.4f}, Training size: {len(current_train_emb)}")
    
    return history


al_history_random = run_random_baseline(
    initial_train_embeddings=sim_train_data["embeddings"],
    initial_train_phenotypes=sim_train_data["phenotypes"],
    pool_embeddings=sim_pool_data["embeddings"],
    pool_ids=sim_pool_data["ids"],
    pool_data=sim_pool_data,
    ground_truth=sim_ground_truth,
    n_iterations=8,
    n_per_iteration=15,
    val_embeddings=val_embeddings,
    val_phenotypes=val_phenotypes,
)

# %% [markdown]
# ### 7.5 Tracking Performance Improvement

# %%
# Compare strategies
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Validation R² over iterations
ax = axes[0]
ax.plot(al_history_hybrid["iteration"], al_history_hybrid["val_r2"], 
        'o-', linewidth=2, markersize=8, label='Hybrid', color='blue')
ax.plot(al_history_uncertainty["iteration"], al_history_uncertainty["val_r2"], 
        's-', linewidth=2, markersize=8, label='Uncertainty', color='green')
ax.plot(al_history_random["iteration"], al_history_random["val_r2"], 
        '^-', linewidth=2, markersize=8, label='Random', color='red')
ax.set_xlabel("AL Iteration")
ax.set_ylabel("Validation R²")
ax.set_title("Model Performance Over AL Iterations")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Validation MSE over iterations
ax = axes[1]
ax.plot(al_history_hybrid["iteration"], al_history_hybrid["val_mse"], 
        'o-', linewidth=2, markersize=8, label='Hybrid', color='blue')
ax.plot(al_history_uncertainty["iteration"], al_history_uncertainty["val_mse"], 
        's-', linewidth=2, markersize=8, label='Uncertainty', color='green')
ax.plot(al_history_random["iteration"], al_history_random["val_mse"], 
        '^-', linewidth=2, markersize=8, label='Random', color='red')
ax.set_xlabel("AL Iteration")
ax.set_ylabel("Validation MSE")
ax.set_title("Prediction Error Over AL Iterations")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Performance vs training size
ax = axes[2]
ax.plot(al_history_hybrid["train_size"], al_history_hybrid["val_r2"], 
        'o-', linewidth=2, markersize=8, label='Hybrid', color='blue')
ax.plot(al_history_uncertainty["train_size"], al_history_uncertainty["val_r2"], 
        's-', linewidth=2, markersize=8, label='Uncertainty', color='green')
ax.plot(al_history_random["train_size"], al_history_random["val_r2"], 
        '^-', linewidth=2, markersize=8, label='Random', color='red')
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Validation R²")
ax.set_title("Sample Efficiency Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Summary statistics
print("\nActive Learning Simulation Summary")
print("=" * 60)

strategies = {
    "Hybrid": al_history_hybrid,
    "Uncertainty": al_history_uncertainty,
    "Random": al_history_random,
}

summary_data = []
for name, history in strategies.items():
    final_r2 = history["val_r2"][-1]
    initial_r2 = history["val_r2"][0]
    improvement = final_r2 - initial_r2
    
    summary_data.append({
        "Strategy": name,
        "Initial R²": initial_r2,
        "Final R²": final_r2,
        "Improvement": improvement,
        "Final MSE": history["val_mse"][-1],
        "Final Train Size": history["train_size"][-1],
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# %%
# Calculate area under the learning curve (AULC) - higher is better
print("\nArea Under Learning Curve (AULC):")
for name, history in strategies.items():
    aulc = np.trapz(history["val_r2"], history["train_size"])
    print(f"  {name}: {aulc:.2f}")

# %% [markdown]
# ---
# ## 8. Model Interpretation
#
# Understanding *why* certain proteins were selected helps validate the active
# learning strategy and can provide biological insights.
#
# ### 8.1 Gradient-Based Interpretation

# %%
# Create interpreter for the trained model
interpreter = GradientInterpreter(model, config=None)

# Get embeddings of selected proteins from our earlier selection
selected_embeddings = torch.from_numpy(
    pool_data["embeddings"][selection_result.selected_indices]
).float()

print(f"Interpreting {len(selected_embeddings)} selected proteins...")

# %%
# Compute gradient-based attributions
gradient_results = interpreter.explain(
    selected_embeddings,
    task="cell_painting",
    method="vanilla",
)

print("Gradient interpretation results:")
print(f"  Gradients shape: {gradient_results['gradients'].shape}")
print(f"  Importance shape: {gradient_results['importance'].shape}")
print(f"  Saliency shape: {gradient_results['saliency'].shape}")

# %%
# Analyse feature importance
feature_importance = interpreter.feature_importance(
    selected_embeddings,
    task="cell_painting",
    aggregate="mean",
)

print(f"\nFeature importance shape: {feature_importance.shape}")
print(f"Top 10 most important features:")
top_features = np.argsort(feature_importance)[-10:][::-1]
for rank, feat_idx in enumerate(top_features):
    print(f"  {rank+1}. Feature {feat_idx}: importance = {feature_importance[feat_idx]:.4f}")

# %%
# Visualise feature importance distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram of feature importance
ax = axes[0]
ax.hist(feature_importance, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.percentile(feature_importance, 95), color='red', 
           linestyle='--', label='95th percentile')
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Count")
ax.set_title("Distribution of Feature Importance")
ax.legend()

# Plot 2: Top features bar plot
ax = axes[1]
top_n = 20
top_idx = np.argsort(feature_importance)[-top_n:][::-1]
ax.barh(range(top_n), feature_importance[top_idx], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"Feature {i}" for i in top_idx])
ax.set_xlabel("Importance")
ax.set_title(f"Top {top_n} Most Important Features")
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.2 Integrated Gradients

# %%
# Create Integrated Gradients interpreter
ig_interpreter = IntegratedGradientsInterpreter(model)

# Compute attributions for a subset (IG is computationally intensive)
subset_embeddings = selected_embeddings[:5]

print("Computing Integrated Gradients...")
ig_results = ig_interpreter.explain(
    subset_embeddings,
    task="cell_painting",
    target_idx=0,  # Explain first phenotype feature
    n_steps=50,
)

print(f"\nIntegrated Gradients results:")
print(f"  Attributions shape: {ig_results['attributions'].shape}")
print(f"  Convergence delta: {ig_results['convergence_delta'].mean():.6f}")

# %%
# Visualise IG attributions for a single sample
sample_idx = 0
attributions = ig_results['attributions'][sample_idx].numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Attribution distribution
ax = axes[0]
ax.hist(attributions, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel("Attribution Value")
ax.set_ylabel("Count")
ax.set_title(f"IG Attribution Distribution (Sample {sample_idx})")

# Plot 2: Attribution by feature position
ax = axes[1]
# Show first 200 features for clarity
n_show = 200
ax.bar(range(n_show), attributions[:n_show], width=1.0, alpha=0.7)
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel("Feature Index")
ax.set_ylabel("Attribution")
ax.set_title(f"IG Attributions by Feature Position (first {n_show})")

# Highlight ESM-2 vs physicochemical boundary
esm_dim = 1280
if n_show > esm_dim:
    ax.axvline(esm_dim, color='green', linestyle='--', 
               label=f'ESM-2|Physicochemical boundary')
    ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.3 Embedding Type Contribution
#
# Understand how much ESM-2 embeddings vs physicochemical features contribute
# to predictions for selected proteins.

# %%
# Define embedding ranges
ESM2_DIM = 1280
PHYSICOCHEMICAL_DIM = 439

embedding_ranges = {
    "ESM-2": (0, ESM2_DIM),
    "Physicochemical": (ESM2_DIM, ESM2_DIM + PHYSICOCHEMICAL_DIM),
}

# Create feature ablation interpreter
ablation_interpreter = FeatureAblationInterpreter(model)

# Compute group importance
print("Computing embedding type contributions via ablation...")
group_importance = ablation_interpreter.compute_group_importance(
    selected_embeddings,
    feature_groups=embedding_ranges,
    task="cell_painting",
)

print("\nEmbedding Type Contributions:")
print("-" * 40)
total_importance = sum(group_importance.values())
for name, importance in group_importance.items():
    pct = (importance / total_importance) * 100 if total_importance > 0 else 0
    print(f"  {name}: {importance:.4f} ({pct:.1f}%)")

# %%
# Visualise contribution
fig, ax = plt.subplots(figsize=(8, 6))

names = list(group_importance.keys())
values = list(group_importance.values())
colors = ['#3498db', '#e74c3c']

bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.5)

# Add percentage labels
total = sum(values)
for bar, value in zip(bars, values):
    height = bar.get_height()
    pct = (value / total) * 100 if total > 0 else 0
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel("Ablation Importance")
ax.set_title("Contribution of Embedding Types to Predictions\n(for Selected Proteins)")
ax.set_ylim(0, max(values) * 1.2)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.4 Understanding Why Proteins Were Selected
#
# Compare feature attributions between selected and unselected proteins.

# %%
# Get attributions for unselected proteins (sample)
unselected_idx = np.array([i for i in range(len(pool_data["ids"])) 
                           if i not in selection_result.selected_indices])
unselected_sample_idx = np.random.choice(unselected_idx, size=min(15, len(unselected_idx)), replace=False)
unselected_embeddings = torch.from_numpy(
    pool_data["embeddings"][unselected_sample_idx]
).float()

# Compute importance for both groups
selected_importance = interpreter.feature_importance(
    selected_embeddings, task="cell_painting", aggregate="mean"
)
unselected_importance = interpreter.feature_importance(
    unselected_embeddings, task="cell_painting", aggregate="mean"
)

# Compare
importance_diff = selected_importance - unselected_importance

print("Comparison of feature importance: Selected vs Unselected")
print("-" * 60)

# Features more important for selected proteins
more_important_for_selected = np.argsort(importance_diff)[-10:][::-1]
print("\nFeatures MORE important for selected proteins:")
for feat_idx in more_important_for_selected:
    print(f"  Feature {feat_idx}: diff = +{importance_diff[feat_idx]:.4f}")

# Features more important for unselected proteins
more_important_for_unselected = np.argsort(importance_diff)[:10]
print("\nFeatures MORE important for unselected proteins:")
for feat_idx in more_important_for_unselected:
    print(f"  Feature {feat_idx}: diff = {importance_diff[feat_idx]:.4f}")

# %%
# Visualise the comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Scatter of importance values
ax = axes[0]
ax.scatter(unselected_importance, selected_importance, alpha=0.3, s=10)
ax.plot([0, max(selected_importance.max(), unselected_importance.max())],
        [0, max(selected_importance.max(), unselected_importance.max())],
        'r--', label='y=x')
ax.set_xlabel("Importance (Unselected)")
ax.set_ylabel("Importance (Selected)")
ax.set_title("Feature Importance: Selected vs Unselected")
ax.legend()

# Plot 2: Difference distribution
ax = axes[1]
ax.hist(importance_diff, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel("Importance Difference (Selected - Unselected)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Feature Importance Differences")

# Add annotations
positive_pct = (importance_diff > 0).mean() * 100
ax.text(0.95, 0.95, f'{positive_pct:.1f}% features\nmore important\nfor selected',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 9. Visualisation
#
# Let's create comprehensive visualisations of the active learning process.
#
# ### 9.1 Uncertainty Distributions

# %%
# Plot uncertainty distribution with selected samples highlighted
fig = plot_uncertainty_distribution(
    uncertainty_scores=epistemic_per_sample,
    selected_indices=selection_result.selected_indices,
    title="Uncertainty Distribution with Selected Samples",
    figsize=(12, 6),
)
plt.show()

# %%
# Compare uncertainty distributions across acquisition strategies
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

strategies_to_plot = {
    "Uncertainty Sampling": uncertainty_selected,
    "Expected Improvement": ei_selected,
    "Diversity (kmeans++)": diversity_results["kmeans++"],
    "Hybrid": hybrid_selected,
}

for ax, (name, indices) in zip(axes.flatten(), strategies_to_plot.items()):
    # Background distribution
    ax.hist(epistemic_per_sample, bins=50, alpha=0.3, color='gray', 
            label='All candidates', density=True)
    
    # Selected distribution
    selected_unc = epistemic_per_sample[indices]
    ax.hist(selected_unc, bins=20, alpha=0.7, color='red',
            label='Selected', density=True)
    
    ax.set_xlabel("Epistemic Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title(f"{name}\n(mean unc: {selected_unc.mean():.3f})")
    ax.legend()

plt.suptitle("Uncertainty of Selected Samples by Strategy", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 9.2 Selected vs Unselected in Embedding Space

# %%
# Create comprehensive embedding space visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Reduce dimensions using UMAP for better visualisation
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_umap = reducer.fit_transform(pool_data["embeddings"])
    dim_red_name = "UMAP"
except ImportError:
    # Fall back to PCA
    embeddings_umap = embeddings_2d
    dim_red_name = "PCA"

# Plot 1: Colored by uncertainty
ax = axes[0, 0]
scatter = ax.scatter(
    embeddings_umap[:, 0], embeddings_umap[:, 1],
    c=epistemic_per_sample, cmap='viridis', alpha=0.6, s=20
)
plt.colorbar(scatter, ax=ax, label='Uncertainty')
ax.scatter(
    embeddings_umap[selection_result.selected_indices, 0],
    embeddings_umap[selection_result.selected_indices, 1],
    facecolors='none', edgecolors='red', s=100, linewidths=2,
    label='Selected'
)
ax.set_xlabel(f"{dim_red_name} 1")
ax.set_ylabel(f"{dim_red_name} 2")
ax.set_title("Embedding Space Colored by Uncertainty")
ax.legend()

# Plot 2: Colored by cluster
ax = axes[0, 1]
scatter = ax.scatter(
    embeddings_umap[:, 0], embeddings_umap[:, 1],
    c=pool_data["clusters"], cmap='tab10', alpha=0.6, s=20
)
ax.scatter(
    embeddings_umap[selection_result.selected_indices, 0],
    embeddings_umap[selection_result.selected_indices, 1],
    facecolors='none', edgecolors='black', s=100, linewidths=2,
    label='Selected'
)
ax.set_xlabel(f"{dim_red_name} 1")
ax.set_ylabel(f"{dim_red_name} 2")
ax.set_title("Embedding Space Colored by Protein Cluster")
ax.legend()

# Plot 3: Colored by acquisition score
ax = axes[1, 0]
scatter = ax.scatter(
    embeddings_umap[:, 0], embeddings_umap[:, 1],
    c=selection_result.all_scores, cmap='YlOrRd', alpha=0.6, s=20
)
plt.colorbar(scatter, ax=ax, label='Acquisition Score')
ax.scatter(
    embeddings_umap[selection_result.selected_indices, 0],
    embeddings_umap[selection_result.selected_indices, 1],
    facecolors='none', edgecolors='blue', s=100, linewidths=2,
    label='Selected'
)
ax.set_xlabel(f"{dim_red_name} 1")
ax.set_ylabel(f"{dim_red_name} 2")
ax.set_title("Embedding Space Colored by Acquisition Score")
ax.legend()

# Plot 4: Selected vs unselected
ax = axes[1, 1]
unselected_mask = np.ones(len(pool_data["ids"]), dtype=bool)
unselected_mask[selection_result.selected_indices] = False

ax.scatter(
    embeddings_umap[unselected_mask, 0], embeddings_umap[unselected_mask, 1],
    c='lightgray', alpha=0.4, s=15, label='Unselected'
)
ax.scatter(
    embeddings_umap[selection_result.selected_indices, 0],
    embeddings_umap[selection_result.selected_indices, 1],
    c='red', s=60, edgecolors='black', linewidths=1,
    label=f'Selected (n={len(selection_result.selected_indices)})'
)
ax.set_xlabel(f"{dim_red_name} 1")
ax.set_ylabel(f"{dim_red_name} 2")
ax.set_title("Selected vs Unselected Samples")
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 9.3 Active Learning Progress Curves

# %%
# Comprehensive progress visualisation
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: R² improvement
ax = axes[0, 0]
for name, history, color in [
    ("Hybrid", al_history_hybrid, "blue"),
    ("Uncertainty", al_history_uncertainty, "green"),
    ("Random", al_history_random, "red"),
]:
    ax.plot(history["iteration"], history["val_r2"], 
            'o-', linewidth=2, markersize=8, label=name, color=color)
    ax.fill_between(history["iteration"], 
                    np.array(history["val_r2"]) - 0.02,
                    np.array(history["val_r2"]) + 0.02,
                    alpha=0.1, color=color)
ax.set_xlabel("AL Iteration")
ax.set_ylabel("Validation R²")
ax.set_title("Prediction Quality Over Iterations")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: MSE reduction
ax = axes[0, 1]
for name, history, color in [
    ("Hybrid", al_history_hybrid, "blue"),
    ("Uncertainty", al_history_uncertainty, "green"),
    ("Random", al_history_random, "red"),
]:
    ax.plot(history["iteration"], history["val_mse"],
            'o-', linewidth=2, markersize=8, label=name, color=color)
ax.set_xlabel("AL Iteration")
ax.set_ylabel("Validation MSE")
ax.set_title("Prediction Error Over Iterations")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 3: Sample efficiency
ax = axes[0, 2]
for name, history, color in [
    ("Hybrid", al_history_hybrid, "blue"),
    ("Uncertainty", al_history_uncertainty, "green"),
    ("Random", al_history_random, "red"),
]:
    ax.plot(history["train_size"], history["val_r2"],
            'o-', linewidth=2, markersize=8, label=name, color=color)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Validation R²")
ax.set_title("Sample Efficiency")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: R² improvement rate
ax = axes[1, 0]
for name, history, color in [
    ("Hybrid", al_history_hybrid, "blue"),
    ("Uncertainty", al_history_uncertainty, "green"),
    ("Random", al_history_random, "red"),
]:
    r2_values = np.array(history["val_r2"])
    improvement_rate = np.diff(r2_values)
    ax.plot(history["iteration"][1:], improvement_rate,
            'o-', linewidth=2, markersize=8, label=name, color=color)
ax.axhline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel("AL Iteration")
ax.set_ylabel("R² Improvement")
ax.set_title("Per-Iteration Improvement Rate")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Cumulative improvement
ax = axes[1, 1]
for name, history, color in [
    ("Hybrid", al_history_hybrid, "blue"),
    ("Uncertainty", al_history_uncertainty, "green"),
    ("Random", al_history_random, "red"),
]:
    r2_values = np.array(history["val_r2"])
    cumulative_improvement = r2_values - r2_values[0]
    ax.plot(history["iteration"], cumulative_improvement,
            'o-', linewidth=2, markersize=8, label=name, color=color)
ax.set_xlabel("AL Iteration")
ax.set_ylabel("Cumulative R² Improvement")
ax.set_title("Cumulative Improvement from Initial Model")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Training set growth
ax = axes[1, 2]
for name, history, color in [
    ("Hybrid", al_history_hybrid, "blue"),
    ("Uncertainty", al_history_uncertainty, "green"),
    ("Random", al_history_random, "red"),
]:
    ax.plot(history["iteration"], history["train_size"],
            'o-', linewidth=2, markersize=8, label=name, color=color)
ax.set_xlabel("AL Iteration")
ax.set_ylabel("Training Set Size")
ax.set_title("Training Data Accumulation")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Active Learning Progress Analysis", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Create summary heatmap of selections across iterations
fig, ax = plt.subplots(figsize=(12, 6))

# Get cluster distribution of selections per iteration for hybrid
cluster_selections = np.zeros((len(al_history_hybrid["iteration"]), 5))  # 5 clusters

for i, selected_ids in enumerate(al_history_hybrid["selected_ids"]):
    # Map selected IDs back to cluster assignments
    id_to_idx = {pid: idx for idx, pid in enumerate(sim_pool_data["ids"])}
    for sid in selected_ids:
        if sid in id_to_idx:
            cluster = sim_pool_data["clusters"][id_to_idx[sid]]
            cluster_selections[i, cluster] += 1

sns.heatmap(cluster_selections.T, annot=True, fmt=".0f", cmap="YlOrRd",
            xticklabels=[f"Iter {i+1}" for i in range(len(al_history_hybrid["iteration"]))],
            yticklabels=[f"Cluster {i}" for i in range(5)],
            ax=ax)
ax.set_xlabel("AL Iteration")
ax.set_ylabel("Protein Cluster")
ax.set_title("Distribution of Selected Samples Across Protein Clusters\n(Hybrid Strategy)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 10. Practical Considerations
#
# ### 10.1 Batch Size for Wet-Lab Experiments

# %%
print("Practical Considerations for Batch Size")
print("=" * 60)

print("""
The optimal batch size depends on several factors:

1. **Wet-Lab Throughput**
   - Cell Painting: ~384-1536 wells per plate
   - Typical batch: 96-384 proteins per round
   
2. **Model Uncertainty Decay**
   - Larger batches → faster uncertainty reduction
   - But: diminishing returns within a batch
   
3. **Experimental Cost**
   - Reagents: ~\$5-50 per protein
   - Time: 2-4 weeks per round
   - Personnel: Fixed overhead per round

4. **Recommendation**
   - Start with smaller batches (10-50) to validate pipeline
   - Scale to 96-384 for production runs
   - Consider plate layout constraints
""")

# Simulate effect of batch size
batch_sizes = [5, 10, 20, 50, 100]
iterations_per_budget = [100 // bs for bs in batch_sizes]

fig, ax = plt.subplots(figsize=(10, 6))

# This is illustrative - in practice you'd run simulations
# Assume diminishing returns: each sample adds less information
cumulative_info = []
for bs, n_iter in zip(batch_sizes, iterations_per_budget):
    info = 0
    info_history = [0]
    for i in range(n_iter):
        # Simplified model: info gain decreases as we add more data
        batch_info = bs * np.exp(-info * 0.01)
        info += batch_info
        info_history.append(info)
    cumulative_info.append((bs, info_history))

for bs, info_history in cumulative_info:
    samples_seen = [i * bs for i in range(len(info_history))]
    ax.plot(samples_seen, info_history, 'o-', label=f'Batch size = {bs}', markersize=4)

ax.set_xlabel("Total Samples Labeled")
ax.set_ylabel("Cumulative Information Gain (illustrative)")
ax.set_title("Effect of Batch Size on Information Gain")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 10.2 Balancing Exploration vs Exploitation

# %%
print("Balancing Exploration vs Exploitation")
print("=" * 60)

print("""
The explore-exploit balance should change over time:

**Early stages (iterations 1-3):**
- Prioritise EXPLORATION
- High uncertainty weight (0.8-1.0)
- Goal: Learn the landscape
- Recommended: Pure uncertainty sampling or diversity

**Middle stages (iterations 4-7):**
- BALANCED approach
- Hybrid acquisition (0.5-0.7 uncertainty, 0.3-0.5 diversity)
- Goal: Refine model while maintaining coverage

**Late stages (iterations 8+):**
- Prioritise EXPLOITATION
- Expected Improvement or UCB
- Goal: Optimise for desired phenotypes
- Consider target-specific acquisition
""")

# Visualise recommended weights over iterations
iterations = np.arange(1, 11)
exploration_weight = 0.9 * np.exp(-0.2 * (iterations - 1)) + 0.2
exploitation_weight = 1 - exploration_weight

fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(iterations, exploration_weight, exploitation_weight,
             labels=['Exploration (Uncertainty)', 'Exploitation (EI/UCB)'],
             colors=['#3498db', '#e74c3c'], alpha=0.8)
ax.set_xlabel("AL Iteration")
ax.set_ylabel("Weight")
ax.set_title("Recommended Explore-Exploit Balance Over Time")
ax.legend(loc='center right')
ax.set_xlim(1, 10)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 10.3 When to Stop Active Learning

# %%
print("Stopping Criteria for Active Learning")
print("=" * 60)

print("""
Consider stopping when:

1. **Performance Plateau**
   - Validation R² improvement < 0.01 for 2+ iterations
   - Test uncertainty stops decreasing
   
2. **Budget Exhaustion**
   - Fixed experimental budget reached
   - Time constraints (paper deadline, grant period)
   
3. **Sufficient Coverage**
   - All protein clusters represented
   - Design space well-explored
   
4. **Uncertainty Threshold**
   - Max uncertainty < acceptable threshold
   - Model confident on all candidates

5. **Diminishing Returns**
   - Information gain per sample too low
   - AULC improvement minimal
""")

# %%
# Demonstrate stopping criteria evaluation
def evaluate_stopping_criteria(history: Dict) -> Dict[str, bool]:
    """Evaluate various stopping criteria."""
    criteria = {}
    
    # 1. Performance plateau
    r2_values = np.array(history["val_r2"])
    if len(r2_values) >= 3:
        recent_improvement = r2_values[-1] - r2_values[-3]
        criteria["performance_plateau"] = recent_improvement < 0.02
    else:
        criteria["performance_plateau"] = False
    
    # 2. Diminishing returns
    if len(r2_values) >= 2:
        improvements = np.diff(r2_values)
        criteria["diminishing_returns"] = improvements[-1] < 0.005
    else:
        criteria["diminishing_returns"] = False
    
    # 3. Training size threshold (example: 200 samples)
    criteria["size_threshold"] = history["train_size"][-1] >= 200
    
    return criteria


print("\nStopping Criteria Evaluation (Hybrid Strategy):")
print("-" * 50)
stopping_eval = evaluate_stopping_criteria(al_history_hybrid)
for criterion, triggered in stopping_eval.items():
    status = "✓ TRIGGERED" if triggered else "✗ Not triggered"
    print(f"  {criterion}: {status}")

# %% [markdown]
# ---
# ## 11. Summary & Conclusion
#
# ### Key Takeaways

# %%
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ACTIVE LEARNING SUMMARY                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. UNCERTAINTY QUANTIFICATION                                               ║
║     • MC Dropout: Simple, effective, single model                            ║
║     • Ensembles: Better calibrated, higher cost                              ║
║     • Heteroscedastic: Captures aleatoric uncertainty                        ║
║                                                                              ║
║  2. ACQUISITION FUNCTIONS                                                    ║
║     • Uncertainty Sampling: Pure exploration                                 ║
║     • Expected Improvement: Balanced (recommended for optimisation)          ║
║     • Diversity Sampling: Coverage-focused                                   ║
║     • Hybrid: Best for batch selection (uncertainty + diversity)             ║
║                                                                              ║
║  3. KEY FINDINGS FROM SIMULATION                                             ║
║     • Active learning outperforms random selection                           ║
║     • Hybrid strategy provides best sample efficiency                        ║
║     • Early iterations show largest gains                                    ║
║                                                                              ║
║  4. PRACTICAL RECOMMENDATIONS                                                ║
║     • Use MC Dropout (n=20) for single-model uncertainty                     ║
║     • Start with hybrid acquisition (0.7 unc, 0.3 div)                       ║
║     • Batch size: 10-50 for validation, 96-384 for production                ║
║     • Monitor validation R² and stop when plateau detected                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# %%
# Final summary statistics
print("\nSimulation Results Summary")
print("=" * 60)

final_summary = pd.DataFrame({
    "Metric": ["Initial R²", "Final R²", "Improvement", "Final MSE", "Samples Used"],
    "Hybrid": [
        f"{al_history_hybrid['val_r2'][0]:.4f}",
        f"{al_history_hybrid['val_r2'][-1]:.4f}",
        f"{al_history_hybrid['val_r2'][-1] - al_history_hybrid['val_r2'][0]:.4f}",
        f"{al_history_hybrid['val_mse'][-1]:.4f}",
        f"{al_history_hybrid['train_size'][-1]}",
    ],
    "Uncertainty": [
        f"{al_history_uncertainty['val_r2'][0]:.4f}",
        f"{al_history_uncertainty['val_r2'][-1]:.4f}",
        f"{al_history_uncertainty['val_r2'][-1] - al_history_uncertainty['val_r2'][0]:.4f}",
        f"{al_history_uncertainty['val_mse'][-1]:.4f}",
        f"{al_history_uncertainty['train_size'][-1]}",
    ],
    "Random": [
        f"{al_history_random['val_r2'][0]:.4f}",
        f"{al_history_random['val_r2'][-1]:.4f}",
        f"{al_history_random['val_r2'][-1] - al_history_random['val_r2'][0]:.4f}",
        f"{al_history_random['val_mse'][-1]:.4f}",
        f"{al_history_random['train_size'][-1]}",
    ],
})

print(final_summary.to_string(index=False))

# %% [markdown]
# ### Next Steps
#
# After completing this notebook, you should:
#
# 1. **Apply to Real Data**: Replace synthetic data with your actual protein
#    embeddings and experimental results
#
# 2. **Integrate with Wet-Lab**: Set up the feedback loop with your experimental
#    collaborators
#
# 3. **Monitor and Adapt**: Track model performance and adjust acquisition
#    strategy as needed
#
# 4. **Scale Up**: Once validated, increase batch sizes for production runs
#
# ### Related Notebooks
#
# - `01_protein_embeddings.ipynb`: Extract protein embeddings
# - `02_phenotype_exploration.ipynb`: Explore phenotype data
# - `03_model_training.ipynb`: Train the ProToPhen model
# - `05_deployment.ipynb` (future): Deploy for production use

# %%
print("=" * 60)
print(f"\nTotal proteins in pool: {len(pool_data['ids'])}")
print(f"Total selections made: {len(selector.config.exclude_ids)}")
print(f"Remaining candidates: {len(pool_data['ids']) - len(selector.config.exclude_ids)}")
print("\nReady for wet-lab validation!")
