# %% [markdown]
# # 03: Model Training
#
# **ProToPhen: Protein-to-Phenotype Foundation Model**
#
# This notebook demonstrates how to train the ProToPhen model for predicting cellular
# phenotypes from protein embeddings. We cover the complete training workflow including:
#
# - Dataset preparation and data loading
# - Model architecture configuration
# - Training loop with callbacks
# - Evaluation and visualisation
# - Model checkpointing and transfer learning
#
# ## Prerequisites
#
# Before running this notebook, ensure you have:
# 1. Completed **Notebook 1** (protein embeddings extracted)
# 2. Completed **Notebook 2** (phenotype data processed)
# 3. Installed all required dependencies (`pip install -e .`)

# %% [markdown]
# ---
# ## 1. Introduction
#
# ### 1.1 ProToPhen Model Architecture (NOTE: REPLACE WITH ACTUAL FIGURE)
#
# ProToPhen uses an encoder-decoder architecture to map protein embeddings to cellular phenotypes:
#
# ```
# Protein Embedding (ESM-2 + Physicochemical)
#          │
#          ▼
# ┌─────────────────────┐
# │   Protein Encoder   │  MLP with residual connections
# │   (1719 → 256 dim)  │  LayerNorm, GELU, Dropout
# └─────────────────────┘
#          │
#          ▼
#    Latent Space (256)
#          │
#     ┌────┴────┬────────────┐
#     ▼         ▼            ▼
# ┌────────┐ ┌────────┐ ┌──────────┐
# │Cell    │ │Viabil- │ │Transcrip-│
# │Painting│ │ity     │ │tomics    │
# │(~1500) │ │(1)     │ │(978)     │
# └────────┘ └────────┘ └──────────┘
# ```
#
# **Key Components:**
# - **Encoder**: Transforms high-dimensional protein embeddings into a compact latent representation
# - **Decoders**: Task-specific heads for predicting different phenotypic readouts
# - **Multi-task Learning**: Joint training on multiple phenotype types improves generalisation
#
# ### 1.2 Training Workflow Overview
#
# ```
# 1. Data Preparation
#    └─ Load embeddings & phenotypes → Create Dataset → Split → DataLoaders
#
# 2. Model Setup
#    └─ Configure architecture → Initialise model → Move to GPU
#
# 3. Training Configuration
#    └─ Loss function → Optimiser → LR Scheduler → Callbacks
#
# 4. Training Loop
#    └─ Forward pass → Compute loss → Backward pass → Update weights
#
# 5. Evaluation
#    └─ Predict on test set → Compute metrics → Visualise results
#
# 6. Save & Deploy
#    └─ Checkpoint model → Load for inference
# ```

# %% [markdown]
# ---
# ## 2. Setup

# %%
# Core imports
import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Scientific computing
import numpy as np
import pandas as pd
from scipy import stats

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bars
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# %% [markdown]
# ### 2.1 ProToPhen Imports

# %%
# Data structures and loading
from protophen.data.dataset import (
    DatasetConfig,
    ProtoPhenSample,
    ProtoPhenDataset,
    ProteinInferenceDataset,
)
from protophen.data.loaders import (
    create_dataloader,
    create_dataloaders,
    create_balanced_sampler,
    split_by_protein,
    split_by_plate,
    protophen_collate_fn,
)

# Model architecture
from protophen.models.protophen import (
    ProToPhenConfig,
    ProToPhenModel,
    create_protophen_model,
    create_lightweight_model,
)
from protophen.models.encoders import ProteinEncoder, ProteinEncoderConfig
from protophen.models.decoders import CellPaintingHead, ViabilityHead, MultiTaskHead
from protophen.models.losses import (
    CombinedLoss,
    CellPaintingLoss,
    MultiTaskLoss,
    create_loss_function,
)

# Training infrastructure
from protophen.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingState,
    create_trainer,
)
from protophen.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    ProgressCallback,
    TensorBoardCallback,
)
from protophen.training.metrics import (
    MetricCollection,
    MultiTaskMetricCollection,
    MSEMetric,
    MAEMetric,
    R2Metric,
    PearsonCorrelationMetric,
    CosineSimilarityMetric,
    create_default_metrics,
    create_multitask_metrics,
    compute_regression_metrics,
    compute_per_feature_metrics,
    summarise_per_feature_metrics,
)

# Visualisation
from protophen.analysis.visualisation import (
    plot_training_history,
    plot_prediction_scatter,
    plot_residuals,
    plot_embedding_space,
    plot_heatmap,
)

print("All imports successful.")

# %% [markdown]
# ### 2.2 Device Configuration

# %%
def setup_device(prefer_gpu: bool = True) -> torch.device:
    """
    Configure compute device with informative output.
    
    Args:
        prefer_gpu: Whether to use GPU if available
        
    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        
        # Print GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"   Using GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    elif prefer_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("   Using Apple Silicon GPU (MPS)")
        
    else:
        device = torch.device("cpu")
        print(f"   Using CPU ({os.cpu_count()} cores)")
        
        if prefer_gpu:
            print("   Note: GPU requested but not available")
    
    return device

# Configure device
DEVICE = setup_device(prefer_gpu=True)

# %%
# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic operations (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(42)
print("Random seeds set for reproducibility")

# %% [markdown]
# ### 2.3 Directory Setup

# %%
# Define paths
PROJECT_ROOT = Path(".")  # Adjust as needed
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PHENOTYPE_DIR = DATA_DIR / "phenotypes"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for dir_path in [CHECKPOINT_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print(f"Results directory: {RESULTS_DIR}")

# %% [markdown]
# ---
# ## 3. Preparing Data
#
# In this section, we'll load the protein embeddings and phenotype features created
# in the previous notebooks and combine them into a training dataset.

# %% [markdown]
# ### 3.1 Load Protein Embeddings (from Notebook 1)
#
# We assume you have saved protein embeddings as numpy arrays or within the `ProteinLibrary`.

# %%
# Option 1: Load from saved numpy arrays
def load_embeddings_from_numpy(
    embeddings_path: Path,
    protein_ids_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load protein embeddings from numpy files.
    
    Args:
        embeddings_path: Path to .npy file with embeddings
        protein_ids_path: Optional path to protein IDs
        
    Returns:
        Tuple of (embeddings array, protein IDs list)
    """
    embeddings = np.load(embeddings_path)
    
    if protein_ids_path and protein_ids_path.exists():
        with open(protein_ids_path, 'r') as f:
            protein_ids = json.load(f)
    else:
        protein_ids = [f"protein_{i}" for i in range(len(embeddings))]
    
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"  Dimension: {embeddings.shape[1]}")
    print(f"  Number of proteins: {embeddings.shape[0]}")
    
    return embeddings, protein_ids

# %%
# Option 2: Load from ProteinLibrary (recommended)
def load_embeddings_from_library(
    library_path: Path,
    embedding_key: str = "fused",
) -> Tuple[np.ndarray, List[str]]:
    """
    Load protein embeddings from a ProteinLibrary JSON file.
    
    Args:
        library_path: Path to protein library JSON
        embedding_key: Key for the embedding to extract
        
    Returns:
        Tuple of (embeddings array, protein IDs list)
    """
    from protophen.data.protein import ProteinLibrary
    
    library = ProteinLibrary.from_json(library_path)
    
    embeddings = []
    protein_ids = []
    
    for protein in library:
        if embedding_key in protein.embeddings:
            embeddings.append(protein.embeddings[embedding_key])
            protein_ids.append(protein.hash)
        elif "esm2" in protein.embeddings:
            # Fallback to ESM-2 only
            embeddings.append(protein.embeddings["esm2"])
            protein_ids.append(protein.hash)
    
    embeddings = np.stack(embeddings)
    
    print(f"Loaded {len(embeddings)} proteins from library")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, protein_ids

# %% [markdown]
# ### 3.2 Load Phenotype Features (from Notebook 2)

# %%
def load_phenotype_features(
    phenotype_path: Path,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load phenotype features from processed data.
    
    Args:
        phenotype_path: Path to phenotype CSV or numpy file
        
    Returns:
        Tuple of (features array, sample IDs, feature names)
    """
    if phenotype_path.suffix == '.csv':
        df = pd.read_csv(phenotype_path, index_col=0)
        features = df.values
        sample_ids = df.index.tolist()
        feature_names = df.columns.tolist()
    else:
        features = np.load(phenotype_path)
        sample_ids = [f"sample_{i}" for i in range(len(features))]
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
    
    print(f"Loaded phenotype features: {features.shape}")
    print(f"  Number of samples: {features.shape[0]}")
    print(f"  Number of features: {features.shape[1]}")
    
    return features, sample_ids, feature_names

# %% [markdown]
# ### 3.3 Create Synthetic Demo Data
#
# For demonstration purposes, we'll create synthetic data that mimics the structure
# of real protein-phenotype data. **Replace this with your actual data loading.**

# %%
def create_demo_data(
    n_samples: int = 500,
    embedding_dim: int = 1719,  # ESM-2 (1280) + Physicochemical (439)
    phenotype_dim: int = 1500,   # Cell Painting features
    noise_level: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Create synthetic protein-phenotype data for demonstration.
    
    The synthetic data has a learnable structure:
    phenotype = f(protein_embedding) + noise
    
    Args:
        n_samples: Number of samples
        embedding_dim: Protein embedding dimension
        phenotype_dim: Phenotype feature dimension
        noise_level: Amount of noise to add
        seed: Random seed
        
    Returns:
        Tuple of (embeddings, phenotypes, protein_ids, metadata)
    """
    np.random.seed(seed)
    
    # Generate protein embeddings (simulating fused ESM-2 + physicochemical)
    protein_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Create a "true" mapping from embeddings to phenotypes
    # This simulates the biological relationship we want to learn
    latent_dim = 128
    W1 = np.random.randn(embedding_dim, latent_dim) * 0.1
    W2 = np.random.randn(latent_dim, phenotype_dim) * 0.1
    
    # Generate phenotypes with non-linear transformation
    latent = np.tanh(protein_embeddings @ W1)
    phenotypes = latent @ W2
    
    # Add noise
    phenotypes += np.random.randn(n_samples, phenotype_dim) * noise_level
    phenotypes = phenotypes.astype(np.float32)
    
    # Standardise phenotypes (as would be done in real preprocessing)
    phenotypes = (phenotypes - phenotypes.mean(axis=0)) / (phenotypes.std(axis=0) + 1e-8)
    
    # Generate protein IDs and metadata
    protein_ids = [f"protein_{i:04d}" for i in range(n_samples)]
    
    # Simulate plate-based experimental design (5 plates)
    n_plates = 5
    plate_ids = [f"plate_{i % n_plates:02d}" for i in range(n_samples)]
    
    # Simulate viability scores (0-1)
    viability = np.clip(0.5 + latent[:, 0] * 0.3 + np.random.randn(n_samples) * 0.1, 0, 1)
    
    metadata = {
        "plate_ids": plate_ids,
        "viability": viability.astype(np.float32),
        "cell_counts": np.random.randint(100, 500, n_samples),
    }
    
    print(f"Created synthetic dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Phenotype dimension: {phenotype_dim}")
    print(f"  Plates: {n_plates}")
    
    return protein_embeddings, phenotypes, protein_ids, metadata

# Create demo data
protein_embeddings, phenotype_features, protein_ids, metadata = create_demo_data(
    n_samples=500,
    embedding_dim=1719,
    phenotype_dim=1500,
)

# %% [markdown]
# ### 3.4 Create ProtoPhenSample Objects

# %%
def create_samples(
    protein_embeddings: np.ndarray,
    phenotype_features: np.ndarray,
    protein_ids: List[str],
    metadata: Optional[Dict] = None,
    viability: Optional[np.ndarray] = None,
) -> List[ProtoPhenSample]:
    """
    Create ProtoPhenSample objects from arrays.
    
    Args:
        protein_embeddings: Embedding matrix (n_samples, embed_dim)
        phenotype_features: Phenotype matrix (n_samples, n_features)
        protein_ids: List of protein identifiers
        metadata: Optional metadata dictionary
        viability: Optional viability scores
        
    Returns:
        List of ProtoPhenSample objects
    """
    samples = []
    metadata = metadata or {}
    
    for i in range(len(protein_ids)):
        # Build sample metadata
        sample_meta = {
            "sample_id": f"sample_{i}",
            "protein_name": protein_ids[i],
        }
        
        # Add plate information if available
        if "plate_ids" in metadata:
            sample_meta["plate_id"] = metadata["plate_ids"][i]
        
        if "cell_counts" in metadata:
            sample_meta["cell_count"] = int(metadata["cell_counts"][i])
        
        # Build phenotype dictionary
        phenotypes = {"cell_painting": phenotype_features[i]}
        
        # Add viability if available
        if viability is not None:
            phenotypes["viability"] = np.array([viability[i]], dtype=np.float32)
        
        sample = ProtoPhenSample(
            protein_id=protein_ids[i],
            protein_embedding=protein_embeddings[i],
            phenotypes=phenotypes,
            metadata=sample_meta,
        )
        samples.append(sample)
    
    print(f"Created {len(samples)} ProtoPhenSample objects")
    print(f"  Phenotype tasks: {list(samples[0].phenotypes.keys())}")
    
    return samples

# Create samples
samples = create_samples(
    protein_embeddings=protein_embeddings,
    phenotype_features=phenotype_features,
    protein_ids=protein_ids,
    metadata=metadata,
    viability=metadata["viability"],
)

# %% [markdown]
# ### 3.5 Build ProtoPhenDataset

# %%
# Configure dataset
dataset_config = DatasetConfig(
    protein_embedding_key="fused",
    fallback_embedding_keys=["esm2", "physicochemical"],
    phenotype_tasks=["cell_painting", "viability"],
    embedding_noise_std=0.0,  # Set > 0 for training augmentation
    feature_dropout=0.0,       # Set > 0 for training augmentation
    require_qc_passed=True,
    min_cell_count=None,
)

# Create dataset
dataset = ProtoPhenDataset(
    samples=samples,
    config=dataset_config,
)

print(f"\nDataset created: {dataset}")
print(f"  Number of samples: {len(dataset)}")
print(f"  Embedding dimension: {dataset.embedding_dim}")
print(f"  Phenotype dimensions: {dataset.phenotype_dims}")

# %% [markdown]
# ### 3.6 Inspect Dataset Statistics

# %%
# Get dataset statistics
stats = dataset.get_statistics()

print("Dataset Statistics:")
print("-" * 40)
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# %%
# Examine a single sample
sample = dataset[0]

print("\nSample structure:")
print("-" * 40)
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: Tensor{tuple(value.shape)}, dtype={value.dtype}")
    elif isinstance(value, dict):
        print(f"  {key}: Dict with {len(value)} keys")
    else:
        print(f"  {key}: {type(value).__name__}")

# %% [markdown]
# ### 3.7 Data Augmentation Options
#
# ProToPhen supports on-the-fly data augmentation during training:

# %%
# Demonstrate augmentation settings
print("Data Augmentation Options:")
print("-" * 40)
print("""
1. Embedding Noise (embedding_noise_std):
   - Adds Gaussian noise to protein embeddings
   - Helps prevent overfitting to exact embedding values
   - Recommended: 0.01 - 0.1
   
2. Feature Dropout (feature_dropout):
   - Randomly zeros phenotype features during training
   - Encourages learning robust representations
   - Recommended: 0.0 - 0.1
   
3. Balanced Sampling:
   - Weights samples to balance across plates/batches
   - Useful when plate sizes are imbalanced
""")

# Example: Create augmented training config
train_config = DatasetConfig(
    protein_embedding_key="fused",
    phenotype_tasks=["cell_painting", "viability"],
    embedding_noise_std=0.02,  # Add small noise during training
    feature_dropout=0.05,      # Drop 5% of features randomly
)

print(f"Training config with augmentation:")
print(f"  embedding_noise_std: {train_config.embedding_noise_std}")
print(f"  feature_dropout: {train_config.feature_dropout}")

# %% [markdown]
# ---
# ## 4. Data Splitting
#
# Proper data splitting is crucial for reliable model evaluation. We provide two
# main strategies designed to avoid data leakage.

# %% [markdown]
# ### 4.1 Split by Protein (Recommended)
#
# This ensures that the same protein never appears in both training and test sets,
# which is essential for evaluating generalisation to **new proteins**.

# %%
# Split by protein
train_dataset, val_dataset, test_dataset = split_by_protein(
    dataset=dataset,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    seed=42,
)

print("\nSplit by Protein:")
print("-" * 40)
print(f"  Training:   {len(train_dataset):4d} samples")
print(f"  Validation: {len(val_dataset):4d} samples")
print(f"  Test:       {len(test_dataset):4d} samples")

# Verify no overlap
train_proteins = set(s.protein_id for s in train_dataset.samples)
val_proteins = set(s.protein_id for s in val_dataset.samples)
test_proteins = set(s.protein_id for s in test_dataset.samples)

print(f"\n  Unique proteins in train: {len(train_proteins)}")
print(f"  Unique proteins in val:   {len(val_proteins)}")
print(f"  Unique proteins in test:  {len(test_proteins)}")
print(f"  Overlap train-test: {len(train_proteins & test_proteins)}")

# %% [markdown]
# ### 4.2 Split by Plate (Experimental Design)
#
# Use this when you want to test generalisation across experimental batches,
# which is important for assessing robustness to batch effects.

# %%
# Get unique plates
all_plates = list(set(s.metadata.get("plate_id", "unknown") for s in dataset.samples))
print(f"Available plates: {all_plates}")

# Example plate-based split
if len(all_plates) >= 3:
    train_plates = all_plates[:3]
    val_plates = all_plates[3:4] if len(all_plates) > 3 else all_plates[:1]
    test_plates = all_plates[4:] if len(all_plates) > 4 else all_plates[1:2]
    
    train_plate, val_plate, test_plate = split_by_plate(
        dataset=dataset,
        train_plates=train_plates,
        val_plates=val_plates,
        test_plates=test_plates,
    )
    
    print(f"\nSplit by Plate:")
    print(f"  Training plates:   {train_plates} → {len(train_plate)} samples")
    print(f"  Validation plates: {val_plates} → {len(val_plate)} samples")
    if test_plate:
        print(f"  Test plates:       {test_plates} → {len(test_plate)} samples")

# %% [markdown]
# ### 4.3 Creating DataLoaders

# %%
# Create DataLoaders with optimal settings
BATCH_SIZE = 32
NUM_WORKERS = 4  # Adjust based on your CPU cores

loaders = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

print(f"\nDataLoaders created:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training batches: {len(loaders['train'])}")
print(f"  Validation batches: {len(loaders['val'])}")
print(f"  Test batches: {len(loaders['test'])}")

# %% [markdown]
# ### 4.4 Balanced Sampling Strategies
#
# When dealing with imbalanced experimental designs (e.g., different numbers of
# samples per plate), balanced sampling can improve training stability.

# %%
# Create balanced sampler
balanced_sampler = create_balanced_sampler(
    dataset=train_dataset,
    balance_by="plate_id",
)

# Create DataLoader with balanced sampling
balanced_train_loader = create_dataloader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    sampler=balanced_sampler,
    num_workers=NUM_WORKERS,
)

print("Balanced sampling configured")
print("  Samples are weighted inversely to plate frequency")

# %%
# Inspect a batch
batch = next(iter(loaders['train']))

print("\nBatch structure:")
print("-" * 40)
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {tuple(value.shape)}")
    elif isinstance(value, list):
        print(f"  {key}: List[{len(value)}]")
    else:
        print(f"  {key}: {type(value).__name__}")

# %% [markdown]
# ---
# ## 5. Model Architecture
#
# Now we'll configure and create the ProToPhen model.

# %% [markdown]
# ### 5.1 ProToPhenConfig Options

# %%
# Display all configuration options
print("ProToPhenConfig Options:")
print("=" * 60)
print("""
INPUT DIMENSIONS
----------------
protein_embedding_dim : int (default: 1280)
    Dimension of input protein embeddings.
    - ESM-2 8M:   320
    - ESM-2 35M:  480
    - ESM-2 150M: 640
    - ESM-2 650M: 1280
    - Fused (ESM-2 + physicochemical): 1280 + 439 = 1719

ENCODER SETTINGS
----------------
encoder_hidden_dims : List[int] (default: [1024, 512])
    Hidden layer dimensions for the protein encoder.
    
encoder_output_dim : int (default: 256)
    Dimension of the latent representation.
    
encoder_dropout : float (default: 0.1)
    Dropout rate in encoder layers.
    
encoder_activation : str (default: "gelu")
    Activation function: "relu", "gelu", "silu", "tanh"

DECODER SETTINGS
----------------
decoder_hidden_dims : List[int] (default: [512, 1024])
    Hidden layer dimensions for decoder heads.
    
decoder_dropout : float (default: 0.1)
    Dropout rate in decoder layers.

TASK-SPECIFIC SETTINGS
----------------------
cell_painting_dim : int (default: 1500)
    Number of Cell Painting features to predict.
    
predict_viability : bool (default: True)
    Whether to include viability prediction head.
    
predict_transcriptomics : bool (default: False)
    Whether to include transcriptomics prediction head.
    
transcriptomics_dim : int (default: 978)
    Number of genes (L1000 = 978).

UNCERTAINTY & REGULARISATION
----------------------------
predict_uncertainty : bool (default: False)
    Whether to predict aleatoric uncertainty.
    
mc_dropout : bool (default: True)
    Enable MC Dropout for epistemic uncertainty.
    
use_spectral_norm : bool (default: False)
    Apply spectral normalisation to layers.
""")

# %% [markdown]
# ### 5.2 Create ProToPhenModel

# %%
# Get dimensions from our data
embedding_dim = dataset.embedding_dim
cell_painting_dim = dataset.phenotype_dims.get("cell_painting", 1500)

print(f"Detected dimensions:")
print(f"  Embedding: {embedding_dim}")
print(f"  Cell Painting: {cell_painting_dim}")

# %%
# Create model configuration
model_config = ProToPhenConfig(
    # Input
    protein_embedding_dim=embedding_dim,
    
    # Encoder architecture
    encoder_hidden_dims=[1024, 512],
    encoder_output_dim=256,
    encoder_dropout=0.1,
    encoder_activation="gelu",
    
    # Decoder architecture
    decoder_hidden_dims=[512, 1024],
    decoder_dropout=0.1,
    
    # Tasks
    cell_painting_dim=cell_painting_dim,
    predict_viability=True,
    predict_transcriptomics=False,
    
    # Uncertainty
    predict_uncertainty=False,
    mc_dropout=True,
)

# Create model
model = ProToPhenModel(model_config)

print(f"\nModel created: {model}")

# %% [markdown]
# ### 5.3 Multi-task Configuration

# %%
# Example: Creating a model with multiple output tasks
multitask_config = ProToPhenConfig(
    protein_embedding_dim=embedding_dim,
    
    # Encoder
    encoder_hidden_dims=[1024, 512, 256],
    encoder_output_dim=256,
    
    # Tasks
    cell_painting_dim=cell_painting_dim,
    predict_viability=True,
    predict_transcriptomics=True,  # Enable transcriptomics
    transcriptomics_dim=978,       # Assumes L1000 assay - set to actual number if using RNA-seq data. This default behaviour will probably change when I have a clearer experimental pipeline in place lol
    
    # Uncertainty
    predict_uncertainty=True,  # Enable uncertainty estimation
)

multitask_model = ProToPhenModel(multitask_config)

print("Multi-task model:")
print(f"  Tasks: {multitask_model.task_names}")
print(f"  Latent dim: {multitask_model.latent_dim}")

# %% [markdown]
# ### 5.4 Adding Custom Tasks

# %%
# You can add new tasks to an existing model
model.add_task(
    task_name="custom_phenotype",
    output_dim=100,
    hidden_dims=[256, 256],
)

print(f"Tasks after adding custom: {model.task_names}")

# Remove for this demo
del model.decoders["custom_phenotype"]

# %% [markdown]
# ### 5.5 Model Summary and Parameter Counts

# %%
# Get detailed model summary
summary = model.summary()

print("\nModel Summary:")
print("=" * 50)
for key, value in summary.items():
    print(f"  {key}: {value}")

# %%
# Detailed parameter count by component
def count_parameters(model: nn.Module, verbose: bool = True) -> Dict[str, int]:
    """Count parameters by model component."""
    counts = {}
    
    # Encoder
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    counts["encoder"] = encoder_params
    
    # Each decoder
    for name, decoder in model.decoders.items():
        decoder_params = sum(p.numel() for p in decoder.parameters())
        counts[f"decoder_{name}"] = decoder_params
    
    # Total
    counts["total"] = sum(p.numel() for p in model.parameters())
    counts["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print("\nParameter Counts:")
        print("-" * 40)
        for name, count in counts.items():
            print(f"  {name:20s}: {count:>10,}")
    
    return counts

param_counts = count_parameters(model)

# %%
# Move model to device
model = model.to(DEVICE)
print(f"\nModel moved to {DEVICE}")

# %% [markdown]
# ### 5.6 Lightweight Model for Quick Experiments

# %%
# For rapid prototyping, use a lighter model
light_model = create_lightweight_model(
    protein_embedding_dim=embedding_dim,
    cell_painting_dim=cell_painting_dim,
)

light_params = sum(p.numel() for p in light_model.parameters())
full_params = sum(p.numel() for p in model.parameters())

print(f"Lightweight model: {light_params:,} parameters")
print(f"Full model:        {full_params:,} parameters")
print(f"Reduction:         {(1 - light_params/full_params)*100:.1f}%")

# %% [markdown]
# ---
# ## 6. Training Configuration
#
# Configure the training loop with optimiser, scheduler, and other settings.

# %% [markdown]
# ### 6.1 TrainerConfig Options

# %%
print("TrainerConfig Options:")
print("=" * 60)
print("""
TRAINING LOOP
-------------
epochs : int (default: 100)
    Number of training epochs.

OPTIMISATION
------------
learning_rate : float (default: 1e-4)
    Initial learning rate.
    
weight_decay : float (default: 0.01)
    L2 regularisation strength.
    
optimiser : str (default: "adamw")
    Optimiser type: "adamw", "adam", "sgd"

LEARNING RATE SCHEDULE
----------------------
scheduler : str (default: "cosine")
    LR scheduler: "cosine", "linear", "constant", "plateau", "none"
    
warmup_steps : int (default: 100)
    Number of warmup steps.
    
warmup_ratio : float (default: 0.0)
    Alternative: fraction of total steps for warmup.
    
min_lr : float (default: 1e-6)
    Minimum learning rate.

GRADIENT HANDLING
-----------------
gradient_accumulation_steps : int (default: 1)
    Accumulate gradients over N steps (effective batch = batch_size * N).
    
max_grad_norm : float (default: 1.0)
    Maximum gradient norm for clipping.

MIXED PRECISION
---------------
use_amp : bool (default: True)
    Use automatic mixed precision (FP16) training.

EVALUATION
----------
eval_every_n_epochs : int (default: 1)
    Run validation every N epochs.

TASKS
-----
tasks : List[str]
    List of prediction tasks.
    
task_weights : Dict[str, float]
    Weight for each task in the loss function.
""")

# %% [markdown]
# ### 6.2 Create Training Configuration

# %%
# Configure training
trainer_config = TrainerConfig(
    # Training loop
    epochs=50,
    
    # Optimisation
    learning_rate=1e-4,
    weight_decay=0.01,
    optimiser="adamw",
    
    # Learning rate schedule
    scheduler="cosine",
    warmup_steps=50,
    min_lr=1e-6,
    
    # Gradient handling
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    
    # Mixed precision
    use_amp=torch.cuda.is_available(),
    
    # Evaluation
    eval_every_n_epochs=1,
    
    # Tasks
    tasks=["cell_painting", "viability"],
    task_weights={"cell_painting": 1.0, "viability": 0.5},
    
    # Device
    device=str(DEVICE),
    
    # Reproducibility
    seed=42,
)

print("Training Configuration:")
print("-" * 40)
for field_name, field_value in trainer_config.__dict__.items():
    print(f"  {field_name}: {field_value}")

# %% [markdown]
# ### 6.3 Optimiser Selection

# %%
# The Trainer creates the optimiser automatically, but here's how it works:

def demonstrate_optimiser_creation(model: nn.Module, config: TrainerConfig):
    """Show how optimiser is created with proper weight decay."""
    
    # Separate parameters: apply weight decay only to weights, not biases/norms
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "bias" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    print(f"Parameter groups:")
    print(f"  With weight decay:    {sum(p.numel() for p in decay_params):,} params")
    print(f"  Without weight decay: {sum(p.numel() for p in no_decay_params):,} params")
    
    # Create optimiser
    if config.optimiser == "adamw":
        optimiser = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif config.optimiser == "adam":
        optimiser = torch.optim.Adam(
            param_groups,
            lr=config.learning_rate,
        )
    elif config.optimiser == "sgd":
        optimiser = torch.optim.SGD(
            param_groups,
            lr=config.learning_rate,
            momentum=0.9,
        )
    
    return optimiser

demo_optimiser = demonstrate_optimiser_creation(model, trainer_config)
print(f"\nOptimiser: {type(demo_optimiser).__name__}")

# %% [markdown]
# ### 6.4 Learning Rate Schedules

# %%
def visualise_lr_schedule(
    config: TrainerConfig,
    n_batches_per_epoch: int,
    figsize: Tuple[float, float] = (12, 4),
):
    """Visualise different learning rate schedules."""
    
    total_steps = n_batches_per_epoch * config.epochs
    warmup_steps = config.warmup_steps
    
    schedules = {}
    
    # Cosine schedule
    cosine_lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = config.learning_rate * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            lr = max(config.min_lr, config.learning_rate * cosine_decay)
        cosine_lrs.append(lr)
    schedules["cosine"] = cosine_lrs
    
    # Linear schedule
    linear_lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = config.learning_rate * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = max(config.min_lr, config.learning_rate * (1 - progress))
        linear_lrs.append(lr)
    schedules["linear"] = linear_lrs
    
    # Constant with warmup
    constant_lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = config.learning_rate * step / warmup_steps
        else:
            lr = config.learning_rate
        constant_lrs.append(lr)
    schedules["constant"] = constant_lrs
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for ax, (name, lrs) in zip(axes, schedules.items()):
        ax.plot(lrs)
        ax.axvline(warmup_steps, color='r', linestyle='--', alpha=0.5, label='Warmup end')
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{name.capitalize()} Schedule")
        ax.set_yscale('log')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return schedules

# Visualise schedules
_ = visualise_lr_schedule(trainer_config, n_batches_per_epoch=len(loaders['train']))

# %% [markdown]
# ### 6.5 Mixed Precision Training

# %%
print("Mixed Precision Training (AMP):")
print("-" * 40)
print("""
Benefits:
- ~2x faster training on modern GPUs (V100, A100, RTX 30xx/40xx)
- ~50% memory reduction
- Minimal accuracy impact

How it works:
1. Forward pass in FP16
2. Loss scaled to prevent underflow
3. Backward pass in FP16
4. Weights updated in FP32

Enabled automatically when:
- use_amp=True in TrainerConfig
- CUDA device is available
""")

print(f"\nAMP enabled: {trainer_config.use_amp and torch.cuda.is_available()}")

# %% [markdown]
# ### 6.6 Gradient Accumulation

# %%
print("Gradient Accumulation:")
print("-" * 40)
print("""
Purpose: Simulate larger batch sizes without more GPU memory

Example:
- batch_size = 32
- gradient_accumulation_steps = 4
- effective_batch_size = 32 * 4 = 128

Use when:
- Limited GPU memory
- Want larger effective batch sizes for stability
- Training with very long protein sequences
""")

# Example calculation
batch_size = BATCH_SIZE
accum_steps = trainer_config.gradient_accumulation_steps
effective_batch = batch_size * accum_steps

print(f"\nCurrent settings:")
print(f"  Batch size: {batch_size}")
print(f"  Accumulation steps: {accum_steps}")
print(f"  Effective batch size: {effective_batch}")

# %% [markdown]
# ---
# ## 7. Callbacks
#
# Callbacks provide hooks into the training loop for monitoring, checkpointing,
# and early stopping.

# %% [markdown]
# ### 7.1 CheckpointCallback

# %%
# Configure checkpointing
checkpoint_callback = CheckpointCallback(
    checkpoint_dir=CHECKPOINT_DIR / "protophen_demo",
    save_best=True,
    monitor="val_loss",
    mode="min",
    save_every_n_epochs=10,
    keep_n_checkpoints=3,
    save_weights_only=False,  # Save optimiser state too for resuming
)

print("CheckpointCallback configured:")
print(f"  Directory: {checkpoint_callback.checkpoint_dir}")
print(f"  Monitor: {checkpoint_callback.monitor}")
print(f"  Save best: {checkpoint_callback.save_best}")
print(f"  Keep N checkpoints: {checkpoint_callback.keep_n_checkpoints}")

# %% [markdown]
# ### 7.2 EarlyStoppingCallback

# %%
# Configure early stopping
early_stopping_callback = EarlyStoppingCallback(
    monitor="val_loss",
    patience=10,
    min_delta=1e-4,
    mode="min",
    restore_best_weights=True,
)

print("EarlyStoppingCallback configured:")
print(f"  Monitor: {early_stopping_callback.monitor}")
print(f"  Patience: {early_stopping_callback.patience}")
print(f"  Min delta: {early_stopping_callback.min_delta}")
print(f"  Restore best weights: {early_stopping_callback.restore_best_weights}")

# %% [markdown]
# ### 7.3 LoggingCallback

# %%
# Configure logging
logging_callback = LoggingCallback(
    log_every_n_steps=10,
    log_file=RESULTS_DIR / "training.log",
)

print("LoggingCallback configured:")
print(f"  Log every N steps: {logging_callback.log_every_n_steps}")
print(f"  Log file: {logging_callback.log_file}")

# %% [markdown]
# ### 7.4 Progress and TensorBoard Callbacks

# %%
# Progress bar callback
progress_callback = ProgressCallback()

# TensorBoard callback (optional)
tensorboard_callback = TensorBoardCallback(
    log_dir=RESULTS_DIR / "tensorboard",
    log_every_n_steps=10,
    log_histograms=False,  # Set True to log weight distributions
)

print("Additional callbacks configured:")
print("  - ProgressCallback (tqdm progress bars)")
print("  - TensorBoardCallback (run `tensorboard --logdir results/tensorboard`)")

# %% [markdown]
# ### 7.5 Custom Callbacks

# %%
class GradientMonitorCallback(Callback):
    """
    Custom callback to monitor gradient statistics.
    
    Useful for debugging training issues like vanishing/exploding gradients.
    """
    
    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self.gradient_norms: List[float] = []
    
    def on_batch_end(
        self,
        state: TrainingState,
        batch: Dict,
        outputs: Dict,
    ) -> None:
        if state.global_step % self.log_every_n_steps != 0:
            return
        
        # Compute total gradient norm
        total_norm = 0.0
        for param in self.trainer.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        self.gradient_norms.append(total_norm)
        
        # Check for issues
        if total_norm > 100:
            print(f"   Large gradient norm: {total_norm:.2f} at step {state.global_step}")
        elif total_norm < 1e-7:
            print(f"   Very small gradient norm: {total_norm:.2e} at step {state.global_step}")

# Create gradient monitor
gradient_monitor = GradientMonitorCallback(log_every_n_steps=50)

# %%
# Another custom callback: Learning rate logging
class LRLoggerCallback(Callback):
    """Log learning rate at each epoch."""
    
    def __init__(self):
        self.lr_history: List[float] = []
    
    def on_epoch_end(self, state: TrainingState) -> None:
        lr = self.trainer.optimiser.param_groups[0]["lr"]
        self.lr_history.append(lr)
        print(f"  LR at epoch {state.epoch}: {lr:.2e}")

lr_logger = LRLoggerCallback()

# %% [markdown]
# ### 7.6 Combining Callbacks

# %%
# Assemble all callbacks
callbacks = [
    logging_callback,
    progress_callback,
    checkpoint_callback,
    early_stopping_callback,
    gradient_monitor,
    lr_logger,
    # tensorboard_callback,  # Uncomment if you want TensorBoard logging
]

print(f"Total callbacks: {len(callbacks)}")
for cb in callbacks:
    print(f"  - {type(cb).__name__}")

# %% [markdown]
# ---
# ## 8. Training Loop
#
# Now we'll initialise the Trainer and run training.

# %% [markdown]
# ### 8.1 Initialise Trainer

# %%
# Create loss function
loss_fn = create_loss_function(
    tasks=trainer_config.tasks,
    task_weights=trainer_config.task_weights,
    use_uncertainty_weighting=False,
    predict_aleatoric=model_config.predict_uncertainty,
    cell_painting_config={
        "mse_weight": 1.0,
        "correlation_weight": 0.1,  # Encourage correlated predictions
        "cosine_weight": 0.0,
    },
)

print(f"Loss function: {type(loss_fn).__name__}")
print(f"  Tasks: {trainer_config.tasks}")
print(f"  Weights: {trainer_config.task_weights}")

# %%
# Create metrics collection - one per task for proper handling of different output shapes
metrics = create_multitask_metrics(
    tasks=trainer_config.tasks,
    include_correlation=True,
)

print(f"Multi-task metrics created: {metrics}")
for task in trainer_config.tasks:
    task_metrics = metrics.get_task_metrics(task)
    if task_metrics:
        print(f"  {task}: {[m.name for m in task_metrics.metrics]}")

# %%
# Initialise trainer
trainer = Trainer(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    config=trainer_config,
    loss_fn=loss_fn,
    callbacks=callbacks,
    metrics=metrics,
)

print("\nTrainer initialised successfully.")
print(f"  Device: {trainer.device}")
print(f"  Epochs: {trainer.config.epochs}")
print(f"  Callbacks: {len(trainer.callbacks)}")

# %% [markdown]
# ### 8.2 Alternative: Quick Trainer Creation

# %%
# For simple cases, use the convenience function
quick_trainer = create_trainer(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    learning_rate=1e-4,
    epochs=50,
    device=str(DEVICE),
    checkpoint_dir=CHECKPOINT_DIR / "quick_run",
    early_stopping_patience=10,
    use_tensorboard=False,
)

print("Quick trainer created with defaults")

# %% [markdown]
# ### 8.3 Run Training

# %%
# Train the model
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = trainer.train(epochs=trainer_config.epochs)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

# %% [markdown]
# ### 8.4 Training History Visualisation

# %%
# Plot training history
# Plot per-task metrics from validation history
if history['val_metrics']:
    # Extract metric names from first validation result
    first_metrics = history['val_metrics'][0]
    
    # Group metrics by task
    task_metric_names = {}
    for task in trainer_config.tasks:
        task_metric_names[task] = [
            key for key in first_metrics.keys() 
            if key.startswith(f"{task}_")
        ]
    
    # Create subplot for each task
    n_tasks = len(trainer_config.tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]
    
    for ax, task in zip(axes, trainer_config.tasks):
        metric_names = task_metric_names.get(task, [])
        
        for metric_name in metric_names:
            if metric_name == "val_loss":
                continue
            values = [m.get(metric_name, 0) for m in history['val_metrics']]
            # Shorten the label by removing task prefix
            short_name = metric_name.replace(f"{task}_", "")
            ax.plot(values, label=short_name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'{task.replace("_", " ").title()} Metrics')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
# Plot loss curves in detail
fig, ax = plt.subplots(figsize=(10, 6))

epochs = range(1, len(history['train_losses']) + 1)
ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)

# Mark best epoch
best_epoch = np.argmin(history['val_losses']) + 1
best_val_loss = min(history['val_losses'])
ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
ax.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Log scale for better visualisation
ax.set_yscale('log')

plt.tight_layout()
plt.show()

print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

# %%
# Plot learning rate history
if hasattr(lr_logger, 'lr_history') and lr_logger.lr_history:
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(range(1, len(lr_logger.lr_history) + 1), lr_logger.lr_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
# Plot gradient norms
if hasattr(gradient_monitor, 'gradient_norms') and gradient_monitor.gradient_norms:
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(gradient_monitor.gradient_norms)
    ax.set_xlabel('Step (every 50)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm During Training')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Mean gradient norm: {np.mean(gradient_monitor.gradient_norms):.4f}")
    print(f"Max gradient norm:  {np.max(gradient_monitor.gradient_norms):.4f}")

# %% [markdown]
# ---
# ## 9. Evaluation
#
# Now we'll evaluate the trained model on the held-out test set.

# %% [markdown]
# ### 9.1 Predict on Test Set

# %%
# Get predictions on test set
results = trainer.predict(
    dataloader=loaders['test'],
    return_targets=True,
)

print("Prediction results:")
for key in results:
    if isinstance(results[key], np.ndarray):
        print(f"  {key}: {results[key].shape}")
    else:
        print(f"  {key}: {type(results[key]).__name__}[{len(results[key])}]")

# %%
# Extract predictions and targets
cell_painting_pred = results['cell_painting_predictions']
cell_painting_true = results['cell_painting_targets']

print(f"\nCell Painting predictions: {cell_painting_pred.shape}")
print(f"Cell Painting targets:     {cell_painting_true.shape}")

# %% [markdown]
# ### 9.2 Compute Metrics

# %%
# Compute overall metrics
overall_metrics = compute_regression_metrics(
    predictions=cell_painting_pred,
    targets=cell_painting_true,
)

print("\nOverall Test Metrics:")
print("-" * 40)
for name, value in overall_metrics.items():
    print(f"  {name:20s}: {value:.4f}")

# %% [markdown]
# ### 9.3 Per-Feature Metrics

# %%
# Compute metrics for each feature
feature_metrics = compute_per_feature_metrics(
    predictions=cell_painting_pred,
    targets=cell_painting_true,
)

# Summarise
summary = summarise_per_feature_metrics(feature_metrics)

print("\nPer-Feature Metrics Summary:")
print("-" * 40)
for name, value in summary.items():
    print(f"  {name:20s}: {value:.4f}")

# %%
# Distribution of R² across features
r2_values = [m['r2'] for m in feature_metrics.values()]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(r2_values, bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(np.mean(r2_values), color='r', linestyle='--', label=f'Mean: {np.mean(r2_values):.3f}')
axes[0].axvline(np.median(r2_values), color='g', linestyle='--', label=f'Median: {np.median(r2_values):.3f}')
axes[0].set_xlabel('R² Score')
axes[0].set_ylabel('Number of Features')
axes[0].set_title('Distribution of R² Across Features')
axes[0].legend()

# Sorted R² values
sorted_r2 = np.sort(r2_values)[::-1]
axes[1].plot(sorted_r2)
axes[1].axhline(0, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Feature Rank')
axes[1].set_ylabel('R² Score')
axes[1].set_title('R² Scores (Sorted)')
axes[1].fill_between(range(len(sorted_r2)), sorted_r2, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistics
n_positive = sum(1 for r2 in r2_values if r2 > 0)
n_high = sum(1 for r2 in r2_values if r2 > 0.5)

print(f"\nFeature statistics:")
print(f"  Features with R² > 0:   {n_positive}/{len(r2_values)} ({100*n_positive/len(r2_values):.1f}%)")
print(f"  Features with R² > 0.5: {n_high}/{len(r2_values)} ({100*n_high/len(r2_values):.1f}%)")

# %% [markdown]
# ### 9.4 Visualise Predictions vs Targets

# %%
# Select features to visualise (best performing)
best_features = sorted(feature_metrics.items(), key=lambda x: x[1]['r2'], reverse=True)[:9]
best_feature_indices = [int(name.split('_')[1]) for name, _ in best_features]

fig, axes = plot_prediction_scatter(
    y_true=cell_painting_true,
    y_pred=cell_painting_pred,
    feature_names=[f"Feature {i}" for i in best_feature_indices],
    n_features=9,
    title="Best Predicted Features (by R²)",
    figsize=(12, 12),
)
plt.show()

# %%
# Also show some poorly predicted features
worst_features = sorted(feature_metrics.items(), key=lambda x: x[1]['r2'])[:9]
worst_feature_indices = [int(name.split('_')[1]) for name, _ in worst_features]

fig, axes = plot_prediction_scatter(
    y_true=cell_painting_true[:, worst_feature_indices],
    y_pred=cell_painting_pred[:, worst_feature_indices],
    feature_names=[f"Feature {i}" for i in worst_feature_indices],
    n_features=9,
    title="Worst Predicted Features (by R²)",
    figsize=(12, 12),
)
plt.show()

# %% [markdown]
# ### 9.5 Residual Analysis

# %%
# Analyse residuals for a well-predicted feature
best_idx = best_feature_indices[0]

fig, axes = plot_residuals(
    y_true=cell_painting_true,
    y_pred=cell_painting_pred,
    feature_idx=best_idx,
    feature_name=f"Feature {best_idx} (Best)",
    figsize=(14, 4),
)
plt.show()

# %% [markdown]
# ### 9.6 Viability Prediction (if trained)

# %%
# Check if viability predictions are available
if 'viability_predictions' in results and 'viability_targets' in results:
    viability_pred = results['viability_predictions'].flatten()
    viability_true = results['viability_targets'].flatten()
    
    # Compute metrics
    viability_metrics = compute_regression_metrics(
        predictions=torch.tensor(viability_pred),
        targets=torch.tensor(viability_true),
    )
    
    print("Viability Prediction Metrics:")
    print("-" * 40)
    for name, value in viability_metrics.items():
        print(f"  {name:20s}: {value:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    axes[0].scatter(viability_true, viability_pred, alpha=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0].set_xlabel('True Viability')
    axes[0].set_ylabel('Predicted Viability')
    axes[0].set_title(f'Viability Prediction (R²={viability_metrics["r2"]:.3f})')
    axes[0].legend()
    axes[0].set_xlim(-0.1, 1.1)
    axes[0].set_ylim(-0.1, 1.1)
    
    # Residual histogram
    residuals = viability_true - viability_pred
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Residual (True - Predicted)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residual Distribution (MAE={viability_metrics["mae"]:.3f})')
    
    plt.tight_layout()
    plt.show()
else:
    print("Viability predictions not available in results")

# %% [markdown]
# ### 9.7 Latent Space Analysis

# %%
# Extract latent representations for test set
model.eval()
latent_representations = []
protein_ids_test = []

with torch.no_grad():
    for batch in loaders['test']:
        embeddings = batch['protein_embedding'].to(DEVICE)
        latent = model.get_latent(embeddings)
        latent_representations.append(latent.cpu().numpy())
        protein_ids_test.extend(batch['protein_id'])

latent_representations = np.concatenate(latent_representations, axis=0)
print(f"Latent representations shape: {latent_representations.shape}")

# %%
# Visualise latent space with UMAP
try:
    fig, ax = plot_embedding_space(
        embeddings=latent_representations,
        method="umap",
        color_by=cell_painting_true[:, 0],  # Color by first phenotype feature
        title="Latent Space (colored by Feature 0)",
    )
    plt.show()
except ImportError:
    print("UMAP not available. Install with: pip install umap-learn")
    
    # Fallback to PCA
    fig, ax = plot_embedding_space(
        embeddings=latent_representations,
        method="pca",
        color_by=cell_painting_true[:, 0],
        title="Latent Space - PCA (colored by Feature 0)",
    )
    plt.show()

# %% [markdown]
# ---
# ## 10. Saving and Loading Models
#
# Proper model checkpointing is essential for reproducibility and deployment.

# %% [markdown]
# ### 10.1 Save Checkpoint

# %%
# Save full training checkpoint (includes optimiser state)
checkpoint_path = CHECKPOINT_DIR / "protophen_demo" / "final_model.pt"
trainer.save_checkpoint(checkpoint_path)

print(f"Checkpoint saved to: {checkpoint_path}")

# %%
# Alternative: Save only model weights (smaller file)
weights_path = CHECKPOINT_DIR / "protophen_demo" / "model_weights.pt"
weights_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), weights_path)
print(f"Model weights saved to: {weights_path}")
print(f"  File size: {weights_path.stat().st_size / 1e6:.2f} MB")

# %%
# Save model configuration for reproducibility
config_path = CHECKPOINT_DIR / "protophen_demo" / "model_config.json"

from dataclasses import asdict

config_dict = asdict(model_config)
# Convert any non-serialisable items
config_dict = {k: v if isinstance(v, (int, float, str, bool, list, dict, type(None))) 
               else str(v) for k, v in config_dict.items()}

with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"Model config saved to: {config_path}")

# %% [markdown]
# ### 10.2 Load for Inference

# %%
def load_model_for_inference(
    checkpoint_path: Path,
    config: Optional[ProToPhenConfig] = None,
    device: str = "cuda",
) -> ProToPhenModel:
    """
    Load a trained model for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration (if not in checkpoint)
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Reconstruct config if saved in checkpoint
    if config is None and 'config' in checkpoint:
        saved_config = checkpoint['config']
        # Handle TrainerConfig vs ProToPhenConfig
        if 'protein_embedding_dim' in saved_config:
            config = ProToPhenConfig(**saved_config)
        else:
            # Need to create config manually
            raise ValueError("Model config not found in checkpoint. Please provide config.")
    
    # Create model
    model = ProToPhenModel(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model

# Load the saved model
loaded_model = load_model_for_inference(
    checkpoint_path=checkpoint_path,
    config=model_config,
    device=str(DEVICE),
)

# %% [markdown]
# ### 10.3 Simple Weights Loading

# %%
# For simpler cases, just load weights directly
def load_weights_only(
    model: ProToPhenModel,
    weights_path: Path,
    device: str = "cuda",
) -> ProToPhenModel:
    """Load only model weights (not optimiser state)."""
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

# Example usage
fresh_model = ProToPhenModel(model_config)
fresh_model = load_weights_only(fresh_model, weights_path, str(DEVICE))
print("Weights loaded successfully")

# %% [markdown]
# ### 10.4 Verify Loaded Model

# %%
# Verify loaded model produces same predictions
loaded_model.eval()
model.eval()

with torch.no_grad():
    test_batch = next(iter(loaders['test']))
    test_embeddings = test_batch['protein_embedding'].to(DEVICE)
    
    original_pred = model(test_embeddings)
    loaded_pred = loaded_model(test_embeddings)
    
    # Compare predictions
    for task in original_pred:
        diff = (original_pred[task] - loaded_pred[task]).abs().max().item()
        print(f"  {task} max difference: {diff:.2e}")
        
    if all((original_pred[t] - loaded_pred[t]).abs().max().item() < 1e-5 
           for t in original_pred):
        print("\n✓ Loaded model produces identical predictions")
    else:
        print("\n⚠ Warning: Small differences detected (may be due to floating point)")

# %% [markdown]
# ### 10.5 Transfer Learning Setup

# %%
def setup_transfer_learning(
    pretrained_model: ProToPhenModel,
    new_output_dim: int,
    freeze_encoder: bool = True,
    new_task_name: str = "new_phenotype",
) -> ProToPhenModel:
    """
    Setup model for transfer learning to a new task.
    
    Args:
        pretrained_model: Pretrained ProToPhen model
        new_output_dim: Output dimension for new task
        freeze_encoder: Whether to freeze encoder weights
        new_task_name: Name for the new task
        
    Returns:
        Model configured for transfer learning
    """
    # Freeze encoder if requested
    if freeze_encoder:
        pretrained_model.freeze_encoder()
        print("Encoder frozen - only decoder will be trained")
    
    # Add new task head
    pretrained_model.add_task(
        task_name=new_task_name,
        output_dim=new_output_dim,
        hidden_dims=[512, 256],
    )
    
    # Report trainable parameters
    total_params = sum(p.numel() for p in pretrained_model.parameters())
    trainable_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    
    print(f"\nTransfer learning setup complete:")
    print(f"  New task: {new_task_name} ({new_output_dim} outputs)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    return pretrained_model

# Example: Transfer to a new phenotype assay
transfer_model = setup_transfer_learning(
    pretrained_model=loaded_model,
    new_output_dim=500,  # e.g., different phenotype readout
    freeze_encoder=True,
    new_task_name="new_assay",
)

# %%
# Fine-tuning strategy: Gradually unfreeze layers
def gradual_unfreezing(
    model: ProToPhenModel,
    unfreeze_encoder_after_epochs: int = 5,
    current_epoch: int = 0,
) -> None:
    """
    Gradually unfreeze encoder layers during training.
    
    Call this at the start of each epoch.
    """
    if current_epoch >= unfreeze_encoder_after_epochs:
        model.unfreeze_encoder()
        print(f"Epoch {current_epoch}: Encoder unfrozen for fine-tuning")

print("\nGradual Unfreezing Strategy:")
print("-" * 40)
print("""
1. Train with frozen encoder (epochs 1-5)
   - Only new decoder learns
   - Faster convergence
   - Prevents catastrophic forgetting

2. Unfreeze encoder (epochs 6+)
   - Fine-tune entire model
   - Use lower learning rate (1e-5)
   - Adapt representations to new task
""")

# %% [markdown]
# ---
# ## 11. Hyperparameter Tuning Tips
#
# Suggestions for optimising ProToPhen model performance.

# %% [markdown]
# ### 11.1 Key Hyperparameters to Tune

# %%
print("=" * 60)
print("KEY HYPERPARAMETERS TO TUNE")
print("=" * 60)

print("""
LEARNING RATE (most important)
------------------------------
- Start with: 1e-4
- Range to try: 1e-5 to 1e-3
- Use learning rate finder or grid search
- Lower LR if training is unstable
- Higher LR if training is too slow

Tips:
- With AdamW, 1e-4 is often optimal
- Reduce by 10x when fine-tuning pretrained models
- Use warmup to stabilise early training
""")

# %%
print("""
BATCH SIZE
----------
- Start with: 32
- Range: 16 to 128
- Larger = more stable gradients, but needs more memory
- Use gradient accumulation for effective larger batches

Trade-offs:
- Batch 16:  More noisy, may need lower LR
- Batch 32:  Good balance (recommended)
- Batch 64+: Stable training, may need LR scaling
""")

# %%
print("""
MODEL ARCHITECTURE
------------------
Encoder hidden dims:
- Default: [1024, 512]
- Lighter: [512, 256]
- Deeper:  [1024, 512, 256]

Latent dimension:
- Default: 256
- Range: 64 to 512
- Larger = more capacity, risk of overfitting

Dropout:
- Default: 0.1
- Range: 0.0 to 0.3
- Increase if overfitting
""")

# %%
print("""
REGULARISATION
--------------
Weight decay:
- Default: 0.01
- Range: 1e-4 to 0.1
- Increase if overfitting

Gradient clipping:
- Default: 1.0
- Range: 0.5 to 5.0
- Lower if training is unstable

Data augmentation:
- embedding_noise_std: 0.01-0.1
- feature_dropout: 0.0-0.1
""")

# %% [markdown]
# ### 11.2 Common Pitfalls and Solutions

# %%
print("=" * 60)
print("COMMON PITFALLS AND SOLUTIONS")
print("=" * 60)

print("""
PITFALL 1: Training Loss Not Decreasing
---------------------------------------
Symptoms:
- Loss stays flat or oscillates wildly
- Model outputs constant values

Solutions:
- Reduce learning rate (try 1e-5)
- Check data normalisation (should be ~N(0,1))
- Verify data loading (print batch samples)
- Use gradient clipping (max_grad_norm=1.0)
- Add warmup steps (100-500)
""")

# %%
print("""
PITFALL 2: Overfitting (Val Loss Increases)
-------------------------------------------
Symptoms:
- Training loss decreases
- Validation loss increases after some epochs
- Large gap between train/val metrics

Solutions:
- Increase dropout (0.1 → 0.2)
- Increase weight decay (0.01 → 0.05)
- Reduce model size
- Add data augmentation
- Use early stopping (patience=10)
- Get more training data
""")

# %%
print("""
PITFALL 3: Vanishing/Exploding Gradients
----------------------------------------
Symptoms:
- NaN losses
- Very small or very large gradient norms
- Training suddenly diverges

Solutions:
- Use gradient clipping (essential!)
- Use LayerNorm (already in ProToPhen)
- Reduce learning rate
- Check for data issues (NaN, Inf values)
- Use GELU activation (more stable than ReLU)
""")

# %%
print("""
PITFALL 4: Poor Generalisation to New Proteins
----------------------------------------------
Symptoms:
- Good validation metrics
- Poor test metrics on truly held-out proteins
- Model memorises training proteins

Solutions:
- Use split_by_protein() not random split
- Increase regularisation
- Use larger/more diverse training set
- Reduce model capacity
- Add embedding noise augmentation
""")

# %%
print("""
PITFALL 5: Slow Training
------------------------
Symptoms:
- Each epoch takes too long
- GPU utilisation is low

Solutions:
- Enable mixed precision (use_amp=True)
- Increase num_workers in DataLoader
- Use pin_memory=True for GPU
- Reduce model size for experiments
- Use persistent_workers=True
""")

# %% [markdown]
# ### 11.3 Hyperparameter Search Strategy

# %%
print("=" * 60)
print("RECOMMENDED HYPERPARAMETER SEARCH STRATEGY")
print("=" * 60)

print("""
STAGE 1: Quick Exploration (1-2 hours)
--------------------------------------
- Use 10-20% of data
- Train for 10-20 epochs
- Coarse grid search:
  - learning_rate: [1e-3, 1e-4, 1e-5]
  - weight_decay: [0.0, 0.01, 0.1]

STAGE 2: Architecture Search (2-4 hours)
----------------------------------------
- Use full training data
- Train for 30 epochs
- Compare:
  - Encoder depths: [[512], [1024, 512], [1024, 512, 256]]
  - Latent dims: [128, 256, 512]

STAGE 3: Fine-tuning (4-8 hours)
--------------------------------
- Best architecture from Stage 2
- Train for 100 epochs with early stopping
- Fine grid search:
  - learning_rate: log-uniform [5e-5, 5e-4]
  - dropout: [0.05, 0.1, 0.15, 0.2]
  - warmup_steps: [50, 100, 200]

STAGE 4: Final Training
-----------------------
- Best hyperparameters
- Full data
- Multiple random seeds (3-5)
- Report mean ± std of metrics
""")

# %%
# Example: Simple hyperparameter search
def hyperparameter_search_example():
    """Example of a simple hyperparameter search."""
    
    learning_rates = [1e-3, 1e-4, 1e-5]
    weight_decays = [0.0, 0.01, 0.1]
    
    results = []
    
    for lr in learning_rates:
        for wd in weight_decays:
            print(f"\nTrying lr={lr}, weight_decay={wd}")
            
            # Create config
            config = TrainerConfig(
                epochs=10,  # Short for search
                learning_rate=lr,
                weight_decay=wd,
                device=str(DEVICE),
                tasks=["cell_painting"],
            )
            
            # Create fresh model
            search_model = ProToPhenModel(model_config).to(DEVICE)
            
            # Create trainer
            search_trainer = Trainer(
                model=search_model,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                config=config,
                callbacks=[],  # Minimal callbacks for speed
            )
            
            # Train
            history = search_trainer.train()
            
            # Record results
            best_val_loss = min(history['val_losses'])
            results.append({
                'lr': lr,
                'weight_decay': wd,
                'best_val_loss': best_val_loss,
            })
            
            print(f"  Best val loss: {best_val_loss:.4f}")
    
    # Find best
    best = min(results, key=lambda x: x['best_val_loss'])
    print(f"\nBest configuration:")
    print(f"  lr={best['lr']}, weight_decay={best['weight_decay']}")
    print(f"  val_loss={best['best_val_loss']:.4f}")
    
    return results

# Uncomment to run hyperparameter search:
# search_results = hyperparameter_search_example()

# %% [markdown]
# ---
# ## 12. Summary & Next Steps

# %% [markdown]
# ### 12.1 What We Covered

# %%
print("=" * 60)
print("NOTEBOOK 3 SUMMARY: MODEL TRAINING")
print("=" * 60)

print("""
  DATA PREPARATION
  - Created ProtoPhenSample objects from embeddings and phenotypes
  - Built ProtoPhenDataset with configurable augmentation
  - Implemented proper data splitting (by protein, by plate)
  - Created efficient DataLoaders with balanced sampling

  MODEL ARCHITECTURE
  - Configured ProToPhenModel with encoder-decoder architecture
  - Set up multi-task learning (Cell Painting + viability)
  - Explored model variants (lightweight, with uncertainty)
  - Counted and analysed model parameters

  TRAINING CONFIGURATION
  - Configured TrainerConfig (optimiser, scheduler, etc.)
  - Set up mixed precision training for efficiency
  - Implemented gradient accumulation for larger effective batches
  - Visualised learning rate schedules

  CALLBACKS
  - CheckpointCallback for saving best models
  - EarlyStoppingCallback to prevent overfitting
  - LoggingCallback for training progress
  - Custom callbacks (gradient monitoring, LR logging)

  TRAINING LOOP
  - Executed training with Trainer class
  - Monitored training and validation losses
  - Visualised training history

  EVALUATION
  - Computed comprehensive metrics (MSE, R², Pearson, etc.)
  - Analysed per-feature prediction quality
  - Visualised predictions vs targets
  - Performed residual analysis
  - Examined latent space structure

  MODEL PERSISTENCE
  - Saved and loaded checkpoints
  - Configured transfer learning
  - Verified loaded model predictions

  HYPERPARAMETER TUNING
  - Identified key hyperparameters
  - Discussed common pitfalls and solutions
  - Outlined systematic search strategy
""")

# %% [markdown]
# ### 12.2 Key Takeaways

# %%
print("=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. DATA SPLITTING MATTERS
   - Always use split_by_protein() for evaluating generalisation
   - Random splits lead to overly optimistic metrics
   - Consider plate-based splits for batch effect robustness

2. START SIMPLE
   - Begin with default hyperparameters
   - Use lightweight model for initial experiments
   - Add complexity only if needed

3. MONITOR TRAINING CAREFULLY
   - Watch for overfitting (val loss increasing)
   - Check gradient norms for stability
   - Use early stopping to save compute

4. EVALUATION IS MULTI-FACETED
   - Global metrics hide per-feature variation
   - Some features are inherently harder to predict
   - Correlation metrics complement MSE/R²

5. CHECKPOINTING IS ESSENTIAL
   - Save best models during training
   - Include config for reproducibility
   - Verify loaded models produce same predictions
""")

# %% [markdown]
# ### 12.3 Next Steps: Active Learning (Notebook 4)

# %%
print("=" * 60)
print("NEXT: NOTEBOOK 4 - ACTIVE LEARNING")
print("=" * 60)

print("""
In the next notebook, we'll explore active learning for intelligent
experiment selection:

1. UNCERTAINTY QUANTIFICATION
   - MC Dropout for epistemic uncertainty
   - Ensemble methods
   - Calibration analysis

2. ACQUISITION FUNCTIONS
   - Uncertainty sampling
   - Expected improvement
   - Diversity sampling (DPP)
   - Hybrid strategies

3. EXPERIMENT SELECTION
   - Ranking candidate proteins
   - Batch selection strategies
   - Budget-aware selection

4. ACTIVE LEARNING LOOP
   - Iterative model improvement
   - Simulated AL experiments
   - Performance vs. data efficiency

This enables efficient use of expensive experimental resources by
prioritising the most informative proteins to test.
""")

# %% [markdown]
# ### 12.4 Cleanup

# %%
# Clean up GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("GPU memory cleared")

# %%
# Final summary statistics
print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

# Per-task test metrics
for task in trainer_config.tasks:
    print(f"\n{task.replace('_', ' ').title()} Task:")
    print("-" * 40)
    
    task_prefix = f"{task}_"
    task_metrics = {
        k.replace(task_prefix, ""): v 
        for k, v in overall_metrics.items() 
        if k.startswith(task_prefix)
    }
    
    for name, value in task_metrics.items():
        if isinstance(value, float):
            print(f"  {name:15s}: {value:.4f}")

print(f"""
Training Summary:
  Final epoch:    {history.get('final_epoch', 'N/A')}
  Best val loss:  {history.get('best_val_loss', 'N/A'):.4f}
  
Model saved to: {checkpoint_path}
""")
