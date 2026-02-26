"""
End-to-end inference pipeline for ProToPhen.

The InferencePipeline encapsulates the full journey from raw amino-acid
sequence to phenotype prediction:

    sequence ──► ESM-2 embedding ──► physicochemical features ──► fusion
         ──► ProToPhen model ──► task predictions (+ optional uncertainty)

It manages device placement, lazy model loading, batching, and
checkpoint restoration so that downstream consumers (REST API, CLI,
notebooks) need only supply sequences.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from protophen.data.protein import Protein, compute_sequence_hash
from protophen.embeddings.esm import ESMEmbedder, ESM2_MODELS
from protophen.embeddings.physicochemical import PhysicochemicalCalculator
from protophen.embeddings.fusion import EmbeddingFusion
from protophen.models.protophen import ProToPhenModel, ProToPhenConfig
from protophen.serving.schemas import (
    PredictionResponse,
    TaskPrediction,
    UncertaintyOutput,
    ProteinInput,
)
from protophen.utils.logging import logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""

    # Checkpoint
    checkpoint_path: Optional[str] = None

    # ESM-2
    esm_model_name: str = "esm2_t33_650M_UR50D"
    esm_layer: int = -1
    esm_pooling: str = "mean"
    esm_batch_size: int = 8

    # Physicochemical
    include_physicochemical: bool = True
    include_dipeptide: bool = True

    # Fusion
    fusion_method: str = "concatenate"
    fusion_normalise: bool = True

    # Device
    device: str = "auto"  # auto, cuda, cpu, mps
    use_fp16: bool = True

    # Caching
    embedding_cache_dir: Optional[str] = None

    # Batching
    max_batch_size: int = 64
    max_sequence_length: int = 2000

    # Uncertainty
    default_mc_samples: int = 20

    def resolve_device(self) -> str:
        """Resolve 'auto' to a concrete device string."""
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# =============================================================================
# Checkpoint Utilities
# =============================================================================

# Keys that belong to TrainerConfig, NOT ProToPhenConfig
_TRAINER_CONFIG_KEYS = {
    "epochs", "learning_rate", "weight_decay", "optimiser", "scheduler",
    "warmup_steps", "warmup_ratio", "min_lr", "gradient_accumulation_steps",
    "max_grad_norm", "use_amp", "eval_every_n_epochs", "tasks",
    "task_weights", "seed",
}

# Keys that belong to ProToPhenConfig
_MODEL_CONFIG_KEYS = {
    "protein_embedding_dim", "encoder_hidden_dims", "encoder_output_dim",
    "encoder_dropout", "encoder_activation", "decoder_hidden_dims",
    "decoder_dropout", "cell_painting_dim", "predict_viability",
    "predict_transcriptomics", "transcriptomics_dim",
    "predict_uncertainty", "use_spectral_norm", "mc_dropout", "autoencoder_latent_dim", "use_pretrained_decoder", "freeze_decoder", "autoencoder_hidden_dims", "variational"
}


def _is_trainer_config(config_dict: dict) -> bool:
    """Heuristic: check whether a config dict looks like TrainerConfig."""
    trainer_keys = config_dict.keys() & _TRAINER_CONFIG_KEYS
    model_keys = config_dict.keys() & _MODEL_CONFIG_KEYS
    return len(trainer_keys) > len(model_keys)


def _infer_model_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> ProToPhenConfig:
    """
    Infer ``ProToPhenConfig`` from the shapes of a model state dict.

    This handles the common case where the Trainer (Session 6) or
    CheckpointCallback saves ``TrainerConfig`` under the ``config`` key
    rather than ``ProToPhenConfig``.

    Strategy:
        - ``encoder.encoder.0.linear.weight`` → (hidden_0, input_dim)
        - ``encoder.output_proj.weight`` → (encoder_output_dim, last_hidden)
        - ``decoders.cell_painting.mean_head.weight`` → (cp_dim, ...)
        - ``decoders.viability.*`` presence → predict_viability
        - ``decoders.transcriptomics.*`` presence → predict_transcriptomics
        
    Also detects autoencoder decoder layers (``decoders.cell_painting.decoder_input_proj.*``,
        ``decoders.cell_painting.decoder_layers.*``,
        ``decoders.cell_painting.output_proj.*``) which indicate a Phase 2 checkpoint with ``AutoencoderDecoderHead`` replacing ``CellPaintingHead``.
    """
    config_kwargs: Dict[str, Any] = {}

    # --- protein_embedding_dim (input to first encoder layer) ---------------
    for key in sorted(state_dict.keys()):
        if re.match(r"encoder\.encoder\.0\.linear\.weight", key):
            config_kwargs["protein_embedding_dim"] = state_dict[key].shape[1]
            break

    # --- encoder_output_dim -------------------------------------------------
    if "encoder.output_proj.weight" in state_dict:
        config_kwargs["encoder_output_dim"] = state_dict[
            "encoder.output_proj.weight"
        ].shape[0]

    # --- encoder hidden dims ------------------------------------------------
    hidden_dims: List[int] = []
    layer_idx = 0
    while True:
        key = f"encoder.encoder.{layer_idx}.linear.weight"
        if key in state_dict:
            hidden_dims.append(state_dict[key].shape[0])
            layer_idx += 1
        else:
            break
    if hidden_dims:
        config_kwargs["encoder_hidden_dims"] = hidden_dims
        
    # --- detect autoencoder decoder head --------------------------------------------------
     # Phase 2 checkpoints have AutoencoderDecoderHead as the cell_painting
    # decoder.  Its state-dict keys differ from CellPaintingHead:
    #   CellPaintingHead:        decoders.cell_painting.shared.*, .mean_head.*
    #   AutoencoderDecoderHead:  decoders.cell_painting.decoder_input_proj.*,
    #                            .decoder_layers.*, .output_proj.*
    
    has_autoencoder_decoder = any(
        k.startswith("decoders.cell_painting.decoder_input_proj.")
        for k in state_dict
    )

    if has_autoencoder_decoder:
        logger.info(
            "Detected AutoencoderDecoderHead in state dict "
            "(Phase 2 checkpoint)"
        )

        # Infer autoencoder latent dim from decoder_input_proj weight
        # decoder_input_proj is nn.Sequential: [0]=Linear, [1]=LayerNorm, ...
        dip_key = "decoders.cell_painting.decoder_input_proj.0.weight"
        if dip_key in state_dict:
            first_dec_hidden = state_dict[dip_key].shape[0]
            autoencoder_latent_dim = state_dict[dip_key].shape[1]
            config_kwargs["encoder_output_dim"] = autoencoder_latent_dim

        # Infer cell_painting_dim from output_proj
        op_key = "decoders.cell_painting.output_proj.weight"
        if op_key in state_dict:
            config_kwargs["cell_painting_dim"] = state_dict[op_key].shape[0]

        # Infer autoencoder decoder hidden dims
        ae_dec_hidden: List[int] = []
        ae_layer_idx = 0
        while True:
            ae_key = f"decoders.cell_painting.decoder_layers.{ae_layer_idx}.linear.weight"
            if ae_key in state_dict:
                ae_dec_hidden.append(state_dict[ae_key].shape[0])
                ae_layer_idx += 1
            else:
                break

        # Store decoder hidden dims for the model config
        if ae_dec_hidden:
            config_kwargs["decoder_hidden_dims"] = ae_dec_hidden

    else:
        # --- decoder hidden dims (from cell_painting shared layers) -------------
        decoder_hidden: List[int] = []
        dec_idx = 0
        while True:
            # CellPaintingHead uses nn.Sequential with groups of 4 (Linear, LN, GELU, Dropout)
            linear_key = f"decoders.cell_painting.shared.{dec_idx * 4}.weight"
            if linear_key in state_dict:
                decoder_hidden.append(state_dict[linear_key].shape[0])
                dec_idx += 1
            else:
                break
        if decoder_hidden:
            config_kwargs["decoder_hidden_dims"] = decoder_hidden

        # --- cell_painting_dim --------------------------------------------------
        if "decoders.cell_painting.mean_head.weight" in state_dict:
            config_kwargs["cell_painting_dim"] = state_dict[
                "decoders.cell_painting.mean_head.weight"
            ].shape[0]

    # --- predict_viability --------------------------------------------------
    config_kwargs["predict_viability"] = any(
        k.startswith("decoders.viability.") for k in state_dict
    )

    # --- predict_transcriptomics -------------------------------------------
    config_kwargs["predict_transcriptomics"] = any(
        k.startswith("decoders.transcriptomics.") for k in state_dict
    )

    # --- predict_uncertainty ------------------------------------------------
    config_kwargs["predict_uncertainty"] = (
        "decoders.cell_painting.log_var_head.weight" in state_dict
    )

    # --- mc_dropout (default True, can't infer from weights) ----------------
    config_kwargs.setdefault("mc_dropout", True)

    logger.info(
        f"Inferred ProToPhenConfig from state dict: "
        f"input={config_kwargs.get('protein_embedding_dim', '?')}, "
        f"latent={config_kwargs.get('encoder_output_dim', '?')}, "
        f"cp_dim={config_kwargs.get('cell_painting_dim', '?')}"
        f"autoencoder_decoder={has_autoencoder_decoder}"
    )

    return ProToPhenConfig(**config_kwargs)


def load_checkpoint(
    path: Union[str, Path],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a ProToPhen checkpoint.

    Handles multiple checkpoint formats:

    1. **Trainer.save_checkpoint()** (Session 6) — ``config`` contains
       ``TrainerConfig`` fields; no ``version`` key.
    2. **CheckpointCallback** (Session 6) — similar to above, may include
       ``best_value`` and ``monitor``.
    3. **Pipeline / Registry** — ``config`` contains ``ProToPhenConfig``
       fields; may include ``version`` and ``metrics``.
    4. **Raw state_dict** — a flat ``OrderedDict`` of tensors.

    Args:
        path: Filesystem path to the ``.pt`` / ``.pth`` checkpoint.
        device: Device to map tensors to during load.

    Returns:
        Normalised checkpoint dictionary with at least
        ``model_state_dict`` and ``config``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # --- Format 4: raw state dict -------------------------------------------
    if "model_state_dict" not in checkpoint:
        if isinstance(checkpoint, (dict, OrderedDict)) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            checkpoint = {
                "model_state_dict": checkpoint,
                "config": {},
                "epoch": -1,
            }
        else:
            checkpoint = {
                "model_state_dict": checkpoint,
                "config": {},
                "epoch": -1,
            }

    # --- Normalise config ---------------------------------------------------
    raw_config = checkpoint.get("config", {})

    # If the checkpoint has a dedicated model_config key, prefer it
    if "model_config" in checkpoint:
        checkpoint["config"] = checkpoint["model_config"]
    elif isinstance(raw_config, dict) and _is_trainer_config(raw_config):
        logger.info(
            "Checkpoint 'config' appears to be TrainerConfig; "
            "inferring ProToPhenConfig from state dict shapes."
        )
        checkpoint["_trainer_config"] = raw_config
        checkpoint["config"] = {}  # will be inferred later
        
    # --- Log phase info if present --------------------------------------------------
    phase = checkpoint.get("phase")
    if phase is not None:
        logger.info(
            f"Checkpoint is Phase {phase}"
            + (
                f" (Phase 1 ref: {checkpoint.get('phase1_checkpoint', 'N/A')})"
                if phase == 2
                else ""
            )
        )

    # --- Normalise version --------------------------------------------------
    if "version" not in checkpoint:
        epoch = checkpoint.get("epoch", -1)
        if phase is not None:
            checkpoint["version"] = f"phase{phase}_epoch_{epoch}" if epoch >= 0 else f"phase{phase}_unknown"
        else:
            checkpoint["version"] = f"epoch_{epoch}" if epoch >= 0 else "unknown"

    # --- Normalise metrics --------------------------------------------------
    if "metrics" not in checkpoint:
        metrics: Dict[str, float] = {}
        if "best_val_loss" in checkpoint:
            metrics["best_val_loss"] = float(checkpoint["best_val_loss"])
        if "best_value" in checkpoint:
            monitor = checkpoint.get("monitor", "val_loss")
            metrics[f"best_{monitor}"] = float(checkpoint["best_value"])
        checkpoint["metrics"] = metrics

    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: str = "cpu",
) -> ProToPhenModel:
    """
    Reconstruct a ``ProToPhenModel`` from a checkpoint.

    Handles both checkpoints that store ``ProToPhenConfig`` directly and
    those from the Trainer that store ``TrainerConfig``.
    
    Also handles Phase2 checkpoints containing an ``AutoencoderDecoderHead``. When
    ``autoencoder_config`` is in the checkpoint, the model's Cell Painting decoder 
    is replaced with an ``AutoencoderDecoderHead`` before loading the state dict.

    Args:
        checkpoint: Dictionary returned by :func:`load_checkpoint`.
        device: Target device.

    Returns:
        Model with restored weights in eval mode.
    """
    config_data = checkpoint.get("config", {})
    state_dict = checkpoint["model_state_dict"]

    # Attempt 1: build from config dict if it has model-config keys
    if isinstance(config_data, ProToPhenConfig):
        config = config_data
    elif isinstance(config_data, dict) and (config_data.keys() & _MODEL_CONFIG_KEYS):
        # Filter to only ProToPhenConfig fields
        valid_fields = {
            f.name for f in ProToPhenConfig.__dataclass_fields__.values()
        }
        filtered = {k: v for k, v in config_data.items() if k in valid_fields}
        config = ProToPhenConfig(**filtered)
    else:
        # Attempt 2: infer from state dict
        config = _infer_model_config_from_state_dict(state_dict)

    model = ProToPhenModel(config)
    
    # Replace cell_painting head if checkpoint is Phase 2
    has_autoencoder_decoder = any(
        k.startswith("decoders.cell_painting.decoder_input_proj.")
        for k in state_dict
    )
    
    if has_autoencoder_decoder and "autoencoder_config" in checkpoint:
        from protophen.models.autoencoder import (
            PhenotypeAutoencoder,
            PhenotypeAutoencoderConfig,
        )
        
        ae_config_dict = dict(checkpoint["autoencoder_config"])
        ae_config_dict.pop("effective_decoder_hidden_dims", None)
        ae_config = PhenotypeAutoencoderConfig(**ae_config_dict)
        
        # Build a fresh autoencoder and extract its decoder head
        ae = PhenotypeAutoencoder(ae_config)
        decoder_head = ae.get_decoder_head(freeze=True)
        
        # Replace the default CellPaintingHead so that state_dict keys align
        model.decoders["cell_painting"] = decoder_head
        
        logger.info(
            f"Replaced cell_painting decoder with AutoencoderDecoderHead "
            f"(latent_dim={ae_config.latent_dim}, "
            f"output_dim={ae_config.input_dim})"
        )
    elif has_autoencoder_decoder:
        # No autoencoder_config in checkpoint, so attempt best effort
        # reconstruction from state_dict shapes.
        logger.warning(
            "Phase 2 checkpoint detected by no 'autoencoder_config' found. "
            "Attempting best-effort AutoencoderDecoderHead reconstruction."
        )
        from protophen.models.autoencoder import (
            PhenotypeAutoencoder,
            PhenotypeAutoencoderConfig,
        )
        
        # Infer latent_dim and input_dim from state dict
        dip_key = "decoders.cell_painting.decoder_input_proj.0.weight"
        op_key = "decoders.cell_painting.output_proj.weight"
        ae_latent_dim = state_dict[dip_key].shape[1] if dip_key in state_dict else config.encoder_output_dim
        ae_input_dim = state_dict[op_key].shape[0] if op_key in state_dict else config.cell_painting_dim
        
        # Infer decoder hidden dims
        ae_dec_dims: List[int] = []
        dip_first_dim = state_dict[dip_key].shape[0] if dip_key in state_dict else 512
        ae_dec_dims_list = [dip_first_dim]
        idx = 0
        while True:
            k = f"decoders.cell_painting.decoder_layers.{idx}.linear.weight"
            if k in state_dict:
                ae_dec_dims_list.append(state_dict[k].shape[0])
                idx += 1
            else:
                break
            
        ae_config = PhenotypeAutoencoderConfig(
            input_dim=ae_input_dim,
            latent_dim=ae_latent_dim,
            encoder_hidden_dims=[512,256],  # N.B. this is a placeholder -- the encoder is not used.
            decoder_hidden_dims=ae_dec_dims_list if len(ae_dec_dims_list) > 1 else None,
            use_skip_connections=False,     # Phase 2 decoder has no skips
        )
        ae = PhenotypeAutoencoder(ae_config)
        model.decoders["cell_painting"] = ae.get_decoder_head(freeze=True)

    # Load weights (strict=False tolerates minor mismatches from config evolution)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected[:5]}...")

    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    logger.info(
        f"Model restored (epoch {epoch}), "
        f"{model.n_parameters:,} params on {device}"
    )
    return model


# =============================================================================
# Main Pipeline
# =============================================================================

class InferencePipeline:
    """
    End-to-end inference from amino-acid sequence to phenotype prediction.

    The pipeline lazily initialises heavy components (ESM-2 model weights,
    ProToPhen checkpoint) on first use, so construction is cheap.

    Attributes:
        config: Pipeline configuration.
        model: The loaded ``ProToPhenModel`` (``None`` until first prediction
            or explicit call to :meth:`load_model`).
        device: Resolved device string.

    Example::

        pipeline = InferencePipeline(
            checkpoint_path="checkpoints/best.pt",
            esm_model_name="esm2_t33_650M_UR50D",
        )

        # Single prediction
        result = pipeline.predict("MKFLILLFNILCLFPVLAADNHGVGPQGAS")

        # Batch prediction
        results = pipeline.predict_batch(["MKFLIL...", "ACDEFG..."])

        # With uncertainty
        result = pipeline.predict(
            "MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            return_uncertainty=True,
            n_mc_samples=50,
        )
        
        # With phenotype latent
        result = pipeline.predict(
            "MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            return_phenotype_latent=True,
        )
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[PipelineConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = PipelineConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        if checkpoint_path is not None:
            config.checkpoint_path = str(checkpoint_path)

        self.config = config
        self.device = config.resolve_device()

        # Lazy-loaded components
        self._model: Optional[ProToPhenModel] = None
        self._esm_embedder: Optional[ESMEmbedder] = None
        self._physchem_calc: Optional[PhysicochemicalCalculator] = None
        self._fusion: Optional[EmbeddingFusion] = None

        # State
        self._model_version: str = "unknown"
        self._model_loaded_at: Optional[str] = None
        self._checkpoint_meta: Dict[str, Any] = {}
        self._trainer_config: Optional[Dict[str, Any]] = None
        self._start_time = time.monotonic()

        logger.info(
            f"InferencePipeline initialised "
            f"(device={self.device}, checkpoint={config.checkpoint_path})"
        )

    # =========================================================================
    # Lazy Loaders
    # =========================================================================

    def _ensure_model(self) -> ProToPhenModel:
        if self._model is not None:
            return self._model
        if self.config.checkpoint_path is None:
            raise RuntimeError(
                "No checkpoint_path provided. Supply one at construction "
                "time or call load_model() explicitly."
            )
        self.load_model(self.config.checkpoint_path)
        return self._model

    def _ensure_esm(self) -> ESMEmbedder:
        if self._esm_embedder is not None:
            return self._esm_embedder
        self._esm_embedder = ESMEmbedder(
            model_name=self.config.esm_model_name,
            layer=self.config.esm_layer,
            pooling=self.config.esm_pooling,
            batch_size=self.config.esm_batch_size,
            device=self.device,
            use_fp16=self.config.use_fp16,
            cache_dir=self.config.embedding_cache_dir,
        )
        return self._esm_embedder

    def _ensure_physchem(self) -> PhysicochemicalCalculator:
        if self._physchem_calc is not None:
            return self._physchem_calc
        from protophen.embeddings.physicochemical import PhysicochemicalConfig

        physchem_config = PhysicochemicalConfig(
            include_dipeptide_composition=self.config.include_dipeptide,
        )
        self._physchem_calc = PhysicochemicalCalculator(config=physchem_config)
        return self._physchem_calc

    def _ensure_fusion(self) -> EmbeddingFusion:
        if self._fusion is not None:
            return self._fusion
        embedding_names = ["esm2"]
        if self.config.include_physicochemical:
            embedding_names.append("physicochemical")
        self._fusion = EmbeddingFusion(
            method=self.config.fusion_method,
            embedding_names=embedding_names,
            normalise_inputs=self.config.fusion_normalise,
        )
        return self._fusion

    # =========================================================================
    # Model Loading
    # =========================================================================

    def load_model(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Explicitly load (or reload) a model checkpoint.

        Handles checkpoints produced by both ``Trainer.save_checkpoint()``
        and ``CheckpointCallback`` (Session 6), as well as registry-style
        checkpoints that include ``model_config`` and ``version``.
        
        Also handles Phase 1 and Phase 2 checkpoints from the 2-phase pre-training pipeline.
        """
        checkpoint = load_checkpoint(checkpoint_path, device=self.device)
        self._model = build_model_from_checkpoint(checkpoint, device=self.device)

        # Preserve non-weight metadata for introspection
        self._checkpoint_meta = {
            k: v
            for k, v in checkpoint.items()
            if k not in ("model_state_dict", "optimizer_state_dict",
                         "optimiser_state_dict", "scheduler_state_dict",
                         "scaler_state_dict")
        }

        # Preserve trainer config if present (useful for reproducibility)
        self._trainer_config = checkpoint.get("_trainer_config")

        self._model_version = str(
            checkpoint.get("version", checkpoint.get("epoch", "unknown"))
        )
        self._model_loaded_at = datetime.now(timezone.utc).isoformat()
        self.config.checkpoint_path = str(checkpoint_path)

        logger.info(f"Model version '{self._model_version}' loaded successfully")

    @property
    def model(self) -> ProToPhenModel:
        return self._ensure_model()

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def checkpoint_metrics(self) -> Dict[str, float]:
        """Metrics stored in the loaded checkpoint."""
        return self._checkpoint_meta.get("metrics", {})

    @property
    def trainer_config(self) -> Optional[Dict[str, Any]]:
        """TrainerConfig from the loaded checkpoint (if available)."""
        return self._trainer_config
    
    @property
    def has_autoencoder_decoder(self) -> bool:
        """Whether the loaded model uses an AutoencoderDecoderHead."""
        if self._model is None:
            return False
        from protophen.models.autoencoder import AutoencoderDecoderHead
        cp_head = self._model.decoders.get("cell_painting")
        return isinstance(cp_head, AutoencoderDecoderHead)

    # =========================================================================
    # Embedding
    # =========================================================================

    def _embed_sequence(self, sequence: str) -> np.ndarray:
        esm = self._ensure_esm()
        esm_emb = esm.embed_sequence(sequence)
        embeddings: Dict[str, np.ndarray] = {"esm2": esm_emb}
        if self.config.include_physicochemical:
            calc = self._ensure_physchem()
            physchem_emb = calc.calculate(sequence)
            embeddings["physicochemical"] = physchem_emb
        fusion = self._ensure_fusion()
        return fusion.fuse(embeddings)

    def _embed_sequences(self, sequences: List[str]) -> np.ndarray:
        esm = self._ensure_esm()
        esm_embs = esm.embed_sequences(sequences, show_progress=False)
        if self.config.include_physicochemical:
            calc = self._ensure_physchem()
            physchem_embs = calc.calculate_batch(sequences, show_progress=False)
            fused_list = []
            fusion = self._ensure_fusion()
            for i in range(len(sequences)):
                embs: Dict[str, np.ndarray] = {"esm2": esm_embs[i]}
                embs["physicochemical"] = physchem_embs[i]
                fused_list.append(fusion.fuse(embs))
            return np.stack(fused_list)
        return esm_embs

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict(
        self,
        sequence: str,
        tasks: Optional[List[str]] = None,
        return_latent: bool = False,
        return_uncertainty: bool = False,
        n_mc_samples: Optional[int] = None,
        protein_name: Optional[str] = None,
        return_phenotype_latent: bool = False,
    ) -> PredictionResponse:
        """
        Predict cellular phenotype from a protein sequence.
        
        The ``return_phenotype_latent`` parameter relates to the 2-phase pre-training pipeline. When the model uses and ``AutoencoderDecoderHead`` (Phase 2 checkpoint), the phenotype latent (intermediate autoencoder representation) can be included in the response alongside the reconstructed features.
        
        Args:
            sequence: Amino acid sequence.
            tasks: Tasks to predict (None = all).
            return_latent: Include protein encoder latent in response.
            return_uncertainty: Include MC-Dropout uncertainty estimates.
            n_mc_samples: Number of MC-Dropout forward passes.
            protein_name: Optional protein name.
            return_phenotype_latent: Include phenotype autoencoder latent.
            
        Returns:
            PredictionResponse with predictions and optional extras.
        """
        t0 = time.perf_counter()
        model = self._ensure_model()

        if len(sequence) > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds maximum "
                f"{self.config.max_sequence_length}."
            )

        fused = self._embed_sequence(sequence)
        tensor = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(self.device)

        if return_uncertainty:
            n_samples = n_mc_samples or self.config.default_mc_samples
            raw = model.predict_with_uncertainty(
                tensor, n_samples=n_samples, tasks=tasks
            )
            predictions, uncertainties = self._format_uncertainty(raw, n_samples)
        else:
            raw = model.predict(tensor, tasks=tasks)
            predictions = self._format_predictions(
                raw,
                return_phenotype_latent=return_phenotype_latent,
            )
            uncertainties = None

        latent_list = None
        if return_latent:
            with torch.no_grad():
                latent_vec = model.get_latent(tensor)
            latent_list = latent_vec.squeeze(0).cpu().tolist()
            
        # Retrieve phenotype latent from decoder
        phenotype_latent_list = None
        if return_phenotype_latent:
            phenotype_latent_list = self._get_phenotype_latent()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PredictionResponse(
            protein_name=protein_name,
            protein_hash=compute_sequence_hash(sequence),
            sequence_length=len(sequence),
            predictions=predictions,
            uncertainty=uncertainties,
            latent=latent_list,
            phenotype_latent=phenotype_latent_list,
            model_version=self._model_version,
            inference_time_ms=round(elapsed_ms, 2),
        )

    def predict_batch(
        self,
        sequences: List[str],
        tasks: Optional[List[str]] = None,
        return_latent: bool = False,
        return_uncertainty: bool = False,
        n_mc_samples: Optional[int] = None,
        protein_names: Optional[List[str]] = None,
        return_phenotype_latent: bool = False,
    ) -> List[PredictionResponse]:
        if protein_names is not None and len(protein_names) != len(sequences):
            raise ValueError("protein_names must match sequences in length.")

        results: List[PredictionResponse] = []
        bs = self.config.max_batch_size

        for start in range(0, len(sequences), bs):
            end = min(start + bs, len(sequences))
            chunk_seqs = sequences[start:end]
            chunk_names = (
                protein_names[start:end] if protein_names else [None] * (end - start)
            )
            for seq, name in zip(chunk_seqs, chunk_names):
                resp = self.predict(
                    sequence=seq,
                    tasks=tasks,
                    return_latent=return_latent,
                    return_uncertainty=return_uncertainty,
                    n_mc_samples=n_mc_samples,
                    protein_name=name,
                    return_phenotype_latent=return_phenotype_latent,
                )
                results.append(resp)

        return results

    # ========================================================================
    # Phenotype latent retrieval (Phase 2 pre-training autoencoder decoder)
    # ========================================================================
    
    def _get_phenotype_latent(self) -> Optional[List[float]]:
        """Retrieve the phenotype latent from the AutoencoderDecoderHead cache.
        
        Returns None if the model does not use an autoencoder decoder or if no cached latent is available.
        """
        if self._model is None:
            return None
        
        from protophen.models.autoencoder import AutoencoderDecoderHead
        
        cp_head = self._model.decoders.get("cell_painting")
        if isinstance(cp_head, AutoencoderDecoderHead):
            cached = cp_head.get_last_latent()
            if cached is not None:
                return cached.squeeze(0).cpu().tolist()
            
        return None


    # =========================================================================
    # Formatting helpers
    # =========================================================================

    @staticmethod
    def _format_predictions(
        raw: Dict[str, torch.Tensor],
        return_phenotype_latent: bool = False,
    ) -> List[TaskPrediction]:
        """
        Format raw model output into TaskPrediction objects.
        
        Includes filtering for ``_latent`` suffix keys. 
        Keys ending in ``_latent`` are excluded from predictions unless ``return_phenotype_latent`` is True.
        Keys ending in ``_log_var`` and ``_std`` are always excluded (as they belong to uncertainty output). 
        """
        preds = []
        for task_name, tensor in raw.items():
            # Always filter uncertainty-related keys
            if task_name.endswith("_log_var") or task_name.endswith("_std"):
                continue
            # Filter protein latent (returned via a separate field)
            if task_name == "latent":
                continue
            # Filter phenotype latent (unless explicitly requested)
            if task_name == "phenotype_latent" and not return_phenotype_latent:
                continue
            values = tensor.squeeze(0).cpu().tolist()
            if not isinstance(values, list):
                values = [values]
            preds.append(
                TaskPrediction(
                    task_name=task_name, values=values, dimension=len(values),
                )
            )
        return preds

    @staticmethod
    def _format_uncertainty(
        raw: Dict[str, Dict[str, torch.Tensor]],
        n_samples: int,
    ) -> Tuple[List[TaskPrediction], List[UncertaintyOutput]]:
        predictions: List[TaskPrediction] = []
        uncertainties: List[UncertaintyOutput] = []

        for task_name, stats in raw.items():
            mean_vals = stats["mean"].squeeze(0).cpu().tolist()
            std_vals = stats["std"].squeeze(0).cpu().tolist()
            if not isinstance(mean_vals, list):
                mean_vals = [mean_vals]
            if not isinstance(std_vals, list):
                std_vals = [std_vals]

            predictions.append(
                TaskPrediction(
                    task_name=task_name, values=mean_vals, dimension=len(mean_vals),
                )
            )
            uncertainties.append(
                UncertaintyOutput(
                    task_name=task_name, mean=mean_vals, std=std_vals,
                    n_samples=n_samples,
                )
            )
        return predictions, uncertainties

    # =========================================================================
    # Introspection
    # =========================================================================

    def get_model_info(self) -> Dict[str, Any]:
        model = self._ensure_model()
        summary = model.summary()
        info = {
            "model_version": self._model_version,
            "model_name": "ProToPhen",
            "tasks": summary["tasks"],
            "latent_dim": summary["latent_dim"],
            "protein_embedding_dim": summary["protein_embedding_dim"],
            "n_parameters": summary["n_parameters"],
            "n_trainable_parameters": summary["n_trainable_parameters"],
            "encoder_hidden_dims": summary["encoder_hidden_dims"],
            "decoder_hidden_dims": summary["decoder_hidden_dims"],
            "esm_model": self.config.esm_model_name,
            "fusion_method": self.config.fusion_method,
            "device": self.device,
            "loaded_at": self._model_loaded_at or "",
        }
        
        # Include autoencoder info
        info["has_autoencoder_decoder"] = self.has_autoencoder_decoder
        phase = self._checkpoint_meta.get("phase")
        if phase is not None:
            info["pretraining_phase"] = phase
        ae_config = self._checkpoint_meta.get("autoencoder_config")
        if ae_config is not None:
            info["autoencoder_config"] = ae_config.get("latent_dim")
            info["autoencoder_variational"] = ae_config.get("variational", False)
            
        return info

    def health_check(self) -> Dict[str, Any]:
        checks = {
            "model_loaded": self._model is not None,
            "esm_loaded": (
                self._esm_embedder is not None
                and self._esm_embedder._model_loaded
            ),
            "checkpoint_exists": (
                self.config.checkpoint_path is not None
                and Path(self.config.checkpoint_path).exists()
            ),
        }
        all_ok = all(checks.values()) if checks["model_loaded"] else False

        return {
            "status": "healthy" if all_ok else (
                "degraded" if checks["model_loaded"] else "unhealthy"
            ),
            "model_loaded": checks["model_loaded"],
            "esm_loaded": checks["esm_loaded"],
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "version": self._model_version,
            "device": self.device,
            "checks": checks,
        }

    def __repr__(self) -> str:
        status = "ready" if self.is_ready else "uninitialised"
        return (
            f"InferencePipeline(status={status}, "
            f"device={self.device}, version={self._model_version})"
        )