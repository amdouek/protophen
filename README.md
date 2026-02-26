# ProToPhen

**A foundation model for predicting cellular responses to de novo-designed proteins.**

## Overview

ProToPhen is an experimental-computational platform for:

1. **Extracting protein embeddings** from de novo designed sequences using ESM-2 and physicochemical features
2. **Processing Cell Painting phenotypic data** from high-content imaging experiments
3. **Training predictive models** that map protein sequence/structure to cellular phenotype
4. **Active learning** to intelligently select proteins for experimental characterisation


## Installation

```bash
# Clone the repository
git clone https://github.com/amdouek/protophen.git
cd protophen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Optional extras
pip install -e ".[serving]"     # REST API and deployment
pip install -e ".[jumpcp]"      # JUMP-CP data access
pip install -e ".[full]"        # Everything!
```

## Quick Start
```python
from protophen import Protein, ProteinLibrary, ProtoPhenConfig

# Create a protein
protein = Protein(
    sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
    name="my_designed_protein",
    source="de_novo",
    metadata={"design_method": "RFdiffusion"}
)

print(f"Length: {protein.length}")
print(f"MW: {protein.molecular_weight:.1f} Da")

# Create a library
library = ProteinLibrary(name="my_designs")
library.add(protein)
library.add_from_fasta("additional_designs.fasta")

# Load configuration
config = ProtoPhenConfig.from_yaml("configs/default.yaml")
```

## Tutorials

Detailed tutorials are available as Jupyter notebooks in the `notebooks/` directory:

| Notebook | Description |
|----------|-------------|
| `00_jumpcp_exploration.ipynb` | Examine and curate the JUMP-CP pretraining data |
| `01_protein_embeddings.ipynb` | Extract ESM-2 and physicochemical embeddings |
| `02_phenotype_exploration.ipynb` | Process and explore Cell Painting data |
| `03_model_training.ipynb` | Train protein-to-phenotype prediction models |
| `04_active_learning.ipynb` | Intelligent experiment selection |
| `05_deployment.ipynb` | Serving and Deployment Infrastructure |


## Core Workflows

### Extract Protein Embeddings

```python
from protophen.data.protein import Protein, ProteinLibrary
from protophen.embeddings.esm import ESMEmbedder
from protophen.embeddings.physicochemical import PhysicochemicalCalculator
from protophen.embeddings.fusion import EmbeddingFusion

# Load proteins
library = ProteinLibrary(name="my_designs")
library.add_from_fasta("proteins.fasta", source="de_novo")

# Extract ESM-2 embeddings
embedder = ESMEmbedder(model_name="esm2_t33_650M_UR50D")
embedder.embed_library(library, embedding_key="esm2")

# Extract physicochemical features
calc = PhysicochemicalCalculator()
calc.calculate_for_library(library, embedding_key="physicochemical")

# Fuse embeddings
fusion = EmbeddingFusion(method="concatenate", embedding_names=["esm2", "physicochemical"])
fusion.fuse_library(library, embedding_key="fused")

# Save
library.to_json("library_with_embeddings.json", include_embeddings=True)
```

### Training a model

```python
from protophen.models.protophen import ProToPhenModel, ProToPhenConfig
from protophen.training.trainer import Trainer, TrainerConfig

# Configure model
model_config = ProToPhenConfig(
    protein_embedding_dim=1719,  # ESM-2 (1280) + physicochemical (439)
    cell_painting_dim=1500,
    predict_viability=True,
)

model = ProToPhenModel(model_config)

# Train
trainer = Trainer(model, train_loader, val_loader, TrainerConfig(epochs=100))
history = trainer.train()
```

### Active Learning
```python
from protophen.active_learning.selection import ExperimentSelector, SelectionConfig

selector = ExperimentSelector(
    model=trained_model,
    config=SelectionConfig(n_select=20, acquisition_method="hybrid"),
)

result = selector.select(candidate_loader, embeddings=candidate_embeddings)
print(f"Selected proteins: {result.selected_ids}")
```

### Command-Line Scripts
```bash
# Extract embeddings for a FASTA file
python scripts/extract_embeddings.py --input proteins.fasta --output embeddings/

# Train a model
python scripts/train_model.py --config configs/experiment.yaml

# Run active learning selection
python scripts/run_active_learning.py --model checkpoints/best.pt --candidates pool.json
```

### Serving
```bash
# Install serving dependencies
pip install 'protophen[serving]'

# Serve from a checkpoint
python scripts/serve.py --checkpoint checkpoints/best.pt

# Serve from the model registry
python scripts/serve.py --reigstry ./model_registry

# Serve with full configuration
python scripts/serve.py --checkpoint checkpoints/best.pt --config configs/deployment.yaml

# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

```bash
# Batch inference (FASTA or CSV input, Parquet or CSV output)
python scripts/batch_inference.py \
    --input proteins.fasta \
    --checkpoint checkpoints/best.pt \
    --output predictions.parquet \
    --uncertainty

# Docker deployment
docker build -t protophen:latest -f docker/Dockerfile .
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints:ro \
    protophen:latest python scripts/serve.py --checkpoint /app/checkpoints/best.pt
```

## Project Structure
```markdown
protophen/
├── __init__.py
├── data/
├── embeddings/
├── phenotype/
├── models/
├── training/
├── active_learning/
├── analysis/
├── serving/
└── utils/

scripts/                  # Top-level CLI scripts
├── serve.py
├── batch_inference.py
├── extract_embeddings.py
├── train_model.py
├── run_active_learning.py
├── download_jumpcp.py
├── curate_pretraining.py
├── pretrain_mapping.py
└── pretrain_phenotype.py

configs/                  # Configuration files
├── default.yaml
├── experiment.yaml
├── deployment.yaml
├── jumpcp.yaml
└── pretraining.yaml

docker/                   # Container definitions
├── Dockerfile
├── Dockerfile.gpu
└── docker-compose.yml
```