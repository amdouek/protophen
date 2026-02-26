"""
Phenotype autoencoder for ProToPhen two-phase pre-training.

This module implements the PhenotypeAutoencoder that learns the structure of
Cell Painting phenotype space from JUMP-CP data.  It underpins a two-phase
pre-training strategy:

  Phase 1: Train autoencoder on **all** JUMP-CP plates (ORF + CRISPR + COMPOUND +
            controls) to learn the manifold of biologically meaningful Cell
            Painting responses.
  Phase 2: Freeze the autoencoder decoder, then train a protein encoder
            (ESM-2 -> hidden layers) to predict the phenotype latent
            representation, effectively learning a protein-to-phenotype mapping
            that leverages the structure discovered in Phase 1.

Key components
--------------
PhenotypeAutoencoder
    Encoder-decoder with optional variational bottleneck, residual blocks,
    skip connections, and layer normalisation.
AutoencoderDecoderHead
    Thin wrapper around the autoencoder decoder that exposes the same
    ``forward`` / ``output_dim`` interface as ``CellPaintingHead``, enabling
    drop-in replacement inside ``ProToPhenModel.decoders``.
NTXentLoss
    Normalised-temperature cross-entropy (supervised contrastive) loss
    operating on integer treatment labels.
AutoencoderLoss
    Combined loss (reconstruction + contrastive + KL divergence) with
    configurable component weights and optional per-feature-group weighting.
PretrainingDataset
    ``torch.utils.data.Dataset`` for phenotype-only Phase 1 training.
PretrainingConfig / Phase1Config / Phase2Config
    Dataclass hierarchy loadable from ``configs/pretraining.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import yaml

from protophen.models.encoders import MLPBlock
from protophen.utils.logging import logger


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PhenotypeAutoencoderConfig:
    """Architecture configuration for the phenotype autoencoder.

    Attributes
    ----------
    input_dim : int
        Number of Cell Painting features (after curation / feature selection).
    latent_dim : int
        Dimension of the bottleneck latent space.
    encoder_hidden_dims : list[int]
        Hidden layer widths for the encoder MLP.
    decoder_hidden_dims : list[int] | None
        Hidden layer widths for the decoder MLP.  If *None*, the encoder
        hidden dims are reversed to create a symmetric decoder.
    activation : str
        Activation function name (``"gelu"``, ``"relu"``, ``"silu"``).
    dropout : float
        Dropout probability applied after every hidden layer.
    use_layer_norm : bool
        Whether to apply layer normalisation in each ``MLPBlock``.
    use_batch_norm : bool
        Whether to apply batch normalisation (mutually exclusive with
        layer norm inside ``MLPBlock``; layer norm takes precedence when
        both are *True*).
    use_residual : bool
        Whether to use residual connections when consecutive hidden dims
        are equal.
    use_skip_connections : bool
        Whether to add skip (U-Net-style) connections from encoder hidden
        states to the corresponding decoder layers.  Requires a symmetric
        architecture (i.e. ``decoder_hidden_dims is None``).
    variational : bool
        If *True*, the bottleneck is variational (VAE): the encoder
        produces *μ* and *log σ²*, and the latent is sampled via the
        reparameterisation trick during training.
    weight_init : str
        Weight initialisation strategy (``"kaiming"``, ``"xavier"``,
        ``"normal"``).
    """

    input_dim: int = 1500
    latent_dim: int = 256
    encoder_hidden_dims: List[int] = field(
        default_factory=lambda: [1024, 512],
    )
    decoder_hidden_dims: Optional[List[int]] = None  # symmetric if None

    # Architecture knobs
    activation: str = "gelu"
    dropout: float = 0.1
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    use_residual: bool = True
    use_skip_connections: bool = True
    variational: bool = False
    weight_init: str = "kaiming"

    # -- derived -----------------------------------------------------------

    @property
    def effective_decoder_hidden_dims(self) -> List[int]:
        """Return decoder hidden dims, defaulting to reversed encoder dims."""
        if self.decoder_hidden_dims is not None:
            return self.decoder_hidden_dims
        return list(reversed(self.encoder_hidden_dims))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["effective_decoder_hidden_dims"] = self.effective_decoder_hidden_dims
        return d


# ============================================================================
# Pretraining configuration (YAML-loadable)
# ============================================================================

@dataclass
class Phase1LossConfig:
    """Loss configuration for Phase 1 autoencoder training."""

    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.1
    kl_weight: float = 0.0  # >0 only when variational=True
    reconstruction_type: Literal["mse", "huber"] = "mse"
    huber_delta: float = 1.0
    contrastive_temperature: float = 0.1
    feature_group_weights: Optional[Dict[str, float]] = None


@dataclass
class Phase1DataConfig:
    """Data configuration for Phase 1."""

    plate_types: List[str] = field(
        default_factory=lambda: [
            "ORF", "CRISPR", "COMPOUND", "TARGET1", "TARGET2",
            "DMSO", "COMPOUND_EMPTY", "POSCON8",
        ],
    )
    cache_dir: str = "./data/processed/pretraining"
    max_samples: Optional[int] = None
    min_replicates: int = 2


@dataclass
class Phase1TrainingConfig:
    """Training hyper-parameters for Phase 1."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    optimiser: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    seed: int = 42


@dataclass
class Phase1EvalConfig:
    """Evaluation settings for Phase 1."""

    eval_every_n_epochs: int = 5
    compute_silhouette: bool = True
    compute_replicate_correlation: bool = True


@dataclass
class Phase1Config:
    """Complete Phase 1 configuration."""

    data: Phase1DataConfig = field(default_factory=Phase1DataConfig)
    loss: Phase1LossConfig = field(default_factory=Phase1LossConfig)
    training: Phase1TrainingConfig = field(default_factory=Phase1TrainingConfig)
    evaluation: Phase1EvalConfig = field(default_factory=Phase1EvalConfig)


@dataclass
class Phase2FreezeConfig:
    """Freezing strategy for Phase 2."""

    decoder: bool = True
    encoder: bool = False  # autoencoder encoder is not used in Phase 2
    gradual_unfreeze: bool = False
    unfreeze_after_epochs: int = 20


@dataclass
class Phase2DataConfig:
    """Data configuration for Phase 2."""

    plate_types: List[str] = field(
        default_factory=lambda: ["ORF", "CRISPR", "TARGET1", "TARGET2"],
    )
    normalisation_plates: List[str] = field(
        default_factory=lambda: ["DMSO", "COMPOUND_EMPTY"],
    )
    include_compound_targets: bool = False
    compound_target_weight: float = 0.1
    cache_dir: str = "./data/processed/pretraining"


@dataclass
class Phase2ProteinEncoderConfig:
    """Protein encoder architecture for Phase 2."""

    embedding_dim: int = 1280
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])
    output_dim: int = 256  # must match autoencoder latent_dim
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class Phase2TrainingConfig:
    """Training hyper-parameters for Phase 2."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimiser: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    seed: int = 42


@dataclass
class Phase2Config:
    """Complete Phase 2 configuration."""

    data: Phase2DataConfig = field(default_factory=Phase2DataConfig)
    freeze: Phase2FreezeConfig = field(default_factory=Phase2FreezeConfig)
    protein_encoder: Phase2ProteinEncoderConfig = field(
        default_factory=Phase2ProteinEncoderConfig,
    )
    training: Phase2TrainingConfig = field(default_factory=Phase2TrainingConfig)


@dataclass
class OutputConfig:
    """Output paths for pretraining artefacts."""

    checkpoint_dir: str = "./data/checkpoints/pretraining"
    log_dir: str = "./data/logs/pretraining"


@dataclass
class PretrainingConfig:
    """Master configuration for the two-phase pretraining pipeline.

    Load from YAML::

        config = PretrainingConfig.from_yaml("configs/pretraining.yaml")

    Or construct programmatically and persist::

        config = PretrainingConfig()
        config.save("configs/pretraining.yaml")
    """

    autoencoder: PhenotypeAutoencoderConfig = field(
        default_factory=PhenotypeAutoencoderConfig,
    )
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- persistence -------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.dump(
                asdict(self),
                fh,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PretrainingConfig":
        """Load configuration from a YAML file, merging with defaults."""
        path = Path(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PretrainingConfig":
        """Create configuration from a plain dictionary."""
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "PretrainingConfig":
        """Recursively instantiate the config hierarchy from *d*."""
        ae_d = d.get("autoencoder", {})
        # Remove derived key if accidentally present in YAML
        ae_d.pop("effective_decoder_hidden_dims", None)
        ae = PhenotypeAutoencoderConfig(**ae_d) if ae_d else PhenotypeAutoencoderConfig()

        def _sub(section_key: str, outer_cls, inner_map: Dict[str, type]):
            """Build a nested dataclass from *d[section_key]*."""
            section = d.get(section_key, {})
            kwargs = {}
            for field_name, inner_cls in inner_map.items():
                sub = section.get(field_name, {})
                kwargs[field_name] = inner_cls(**sub) if isinstance(sub, dict) else sub
            return outer_cls(**kwargs)

        p1 = _sub(
            "phase1",
            Phase1Config,
            {
                "data": Phase1DataConfig,
                "loss": Phase1LossConfig,
                "training": Phase1TrainingConfig,
                "evaluation": Phase1EvalConfig,
            },
        )
        p2 = _sub(
            "phase2",
            Phase2Config,
            {
                "data": Phase2DataConfig,
                "freeze": Phase2FreezeConfig,
                "protein_encoder": Phase2ProteinEncoderConfig,
                "training": Phase2TrainingConfig,
            },
        )
        out_d = d.get("output", {})
        out = OutputConfig(**out_d) if isinstance(out_d, dict) else OutputConfig()

        return cls(autoencoder=ae, phase1=p1, phase2=p2, output=out)


# ============================================================================
# PhenotypeAutoencoder
# ============================================================================


class PhenotypeAutoencoder(nn.Module):
    """Encoder-decoder for Cell Painting morphological profiles.

    The autoencoder compresses high-dimensional Cell Painting feature vectors
    into a compact latent space and reconstructs them.  In *variational* mode
    the bottleneck is stochastic (VAE); otherwise it is deterministic (AE).

    Architecture overview::

        features (input_dim)
          ↓  MLPBlock x len(encoder_hidden_dims)
        encoder hidden
          ↓  Linear → [LayerNorm]
        latent (latent_dim)                        ← μ, log σ² if VAE
          ↓  Linear → [LayerNorm] → activation
        decoder hidden  [+ skip from encoder if enabled]
          ↓  MLPBlock x len(decoder_hidden_dims)
        reconstruction (input_dim)

    Parameters
    ----------
    config : PhenotypeAutoencoderConfig, optional
        Architecture configuration.  If *None* defaults are used.
    **kwargs
        Override individual config fields.

    Examples
    --------
    >>> ae = PhenotypeAutoencoder(PhenotypeAutoencoderConfig(
    ...     input_dim=1500, latent_dim=256, variational=False,
    ... ))
    >>> out = ae(torch.randn(32, 1500))
    >>> out["reconstruction"].shape
    torch.Size([32, 1500])
    >>> out["latent"].shape
    torch.Size([32, 256])
    """

    def __init__(
        self,
        config: Optional[PhenotypeAutoencoderConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if config is None:
            config = PhenotypeAutoencoderConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        self.config = config

        # ---- encoder --------------------------------------------------------
        self.encoder_layers = self._build_encoder()

        # ---- latent projection ----------------------------------------------
        last_enc_dim = (
            config.encoder_hidden_dims[-1]
            if config.encoder_hidden_dims
            else config.input_dim
        )

        if config.variational:
            self.latent_mu = nn.Linear(last_enc_dim, config.latent_dim)
            self.latent_log_var = nn.Linear(last_enc_dim, config.latent_dim)
        else:
            self.latent_proj = nn.Linear(last_enc_dim, config.latent_dim)

        self.latent_norm = nn.LayerNorm(config.latent_dim)

        # ---- decoder --------------------------------------------------------
        self.decoder_input_proj, self.decoder_layers, self._decoder_skip_dims = (
            self._build_decoder()
        )

        # ---- output projection ----------------------------------------------
        dec_hidden = config.effective_decoder_hidden_dims
        if dec_hidden:
            last_dec_dim = dec_hidden[-1]
        else:
            # No decoder hidden layers: decoder_input_proj outputs directly
            # to output_proj.  decoder_input_proj projects latent_dim →
            # first_dec_dim, which equals input_dim when dec_hidden is empty.
            last_dec_dim = config.input_dim
        self.output_proj = nn.Linear(last_dec_dim, config.input_dim)

        # ---- bookkeeping ----------------------------------------------------
        self._init_weights()
        self._n_params = sum(p.numel() for p in self.parameters())

        logger.info(
            f"PhenotypeAutoencoder initialised: "
            f"input_dim={config.input_dim}, latent_dim={config.latent_dim}, "
            f"variational={config.variational}, "
            f"skip_connections={config.use_skip_connections}, "
            f"params={self._n_params:,}"
        )

    # --------------------------------------------------------------------- #
    #  Build helpers                                                        #
    # --------------------------------------------------------------------- #

    def _build_encoder(self) -> nn.ModuleList:
        """Construct encoder ``MLPBlock`` layers."""
        cfg = self.config
        layers = nn.ModuleList()
        dims = [cfg.input_dim] + list(cfg.encoder_hidden_dims)

        for i in range(len(dims) - 1):
            layers.append(
                MLPBlock(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    activation=cfg.activation,
                    dropout=cfg.dropout,
                    use_batch_norm=cfg.use_batch_norm,
                    use_layer_norm=cfg.use_layer_norm,
                    use_residual=cfg.use_residual and dims[i] == dims[i + 1],
                )
            )
        return layers

    def _build_decoder(
        self,
    ) -> Tuple[nn.Module, nn.ModuleList, Dict[int, int]]:
        """Construct decoder input projection, ``MLPBlock`` layers, and skip dim map.

        Returns
        -------
        input_proj : nn.Module
        layers : nn.ModuleList
        skip_dims : dict[int, int]
            Mapping from decoder layer index to the dimension of the skip
            connection expected at that layer.  Used for zero-padding when
            encoder hiddens are unavailable (e.g. Phase 2).
        """
        cfg = self.config
        dec_dims = cfg.effective_decoder_hidden_dims

        # Project latent → first decoder hidden dim
        first_dec_dim = dec_dims[0] if dec_dims else cfg.input_dim
        input_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim, first_dec_dim),
            nn.LayerNorm(first_dec_dim),
            MLPBlock._get_activation(cfg.activation),
        )

        layers = nn.ModuleList()
        skip_dims: Dict[int, int] = {}

        for i in range(len(dec_dims) - 1):
            in_features = dec_dims[i]
            if cfg.use_skip_connections and self._can_skip(i):
                skip_dim = dec_dims[i]
                in_features = dec_dims[i] + skip_dim
                skip_dims[i] = skip_dim

            layers.append(
                MLPBlock(
                    in_features=in_features,
                    out_features=dec_dims[i + 1],
                    activation=cfg.activation,
                    dropout=cfg.dropout,
                    use_batch_norm=cfg.use_batch_norm,
                    use_layer_norm=cfg.use_layer_norm,
                    use_residual=(
                        cfg.use_residual and in_features == dec_dims[i + 1]
                    ),
                )
            )

        return input_proj, layers, skip_dims

    def _can_skip(self, decoder_layer_idx: int) -> bool:
        """Return *True* if a skip connection is available for this layer.

        Skip connections map decoder layer *i* to encoder layer
        ``N_enc - 2 - i`` (counting from 0, skipping the final encoder
        hidden that feeds the latent projection).
        """
        n_enc = len(self.config.encoder_hidden_dims)
        n_dec = len(self.config.effective_decoder_hidden_dims) - 1  # transition count
        # We need a matching encoder hidden state
        enc_idx = n_enc - 1 - decoder_layer_idx
        return 0 <= enc_idx < n_enc and decoder_layer_idx < n_dec

    # --------------------------------------------------------------------- #
    #  Weight initialisation                                                #
    # --------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "kaiming":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # --------------------------------------------------------------------- #
    #  Forward helpers                                                      #
    # --------------------------------------------------------------------- #

    def encode(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode features to latent space.

        Parameters
        ----------
        x : Tensor, shape ``(B, input_dim)``

        Returns
        -------
        dict
            ``"latent"``: deterministic or sampled latent, ``(B, latent_dim)``.
            ``"mu"``: mean (present only in variational mode).
            ``"log_var"``: log-variance (present only in variational mode).
            ``"encoder_hiddens"``: list of intermediate encoder hidden states
            (used for skip connections in the decoder).
        """
        encoder_hiddens: List[torch.Tensor] = []
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
            encoder_hiddens.append(h)

        result: Dict[str, Any] = {"encoder_hiddens": encoder_hiddens}

        if self.config.variational:
            mu = self.latent_mu(h)
            log_var = torch.clamp(self.latent_log_var(h), min=-10.0, max=10.0)
            # Reparameterisation trick
            if self.training:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                latent = mu + eps * std
            else:
                latent = mu
            latent = self.latent_norm(latent)
            result.update({"latent": latent, "mu": mu, "log_var": log_var})
        else:
            latent = self.latent_proj(h)
            latent = self.latent_norm(latent)
            result["latent"] = latent

        return result

    def decode(
        self,
        latent: torch.Tensor,
        encoder_hiddens: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Decode latent representation to reconstructed features.

        Parameters
        ----------
        latent : Tensor, shape ``(B, latent_dim)``
        encoder_hiddens : list[Tensor] | None
            Encoder hidden states for skip connections.  If *None* (e.g.
            during Phase 2 where only the decoder is used), skip layers
            receive zero-padding in place of encoder hiddens so that the
            weight dimensions remain compatible.

        Returns
        -------
        Tensor, shape ``(B, input_dim)``
        """
        h = self.decoder_input_proj(latent)

        use_skips = (
            self.config.use_skip_connections and encoder_hiddens is not None
        )
        n_enc = len(self.config.encoder_hidden_dims)

        for i, layer in enumerate(self.decoder_layers):
            if use_skips and self._can_skip(i):
                # Concatenate matching encoder hidden state
                enc_idx = n_enc - 1 - i
                skip = encoder_hiddens[enc_idx]
                h = torch.cat([h, skip], dim=-1)
            elif i in self._decoder_skip_dims:
                # Layer was built expecting a skip connection but encoder
                # hiddens are unavailable — zero-pad to match weight dims
                pad = torch.zeros(
                    h.size(0),
                    self._decoder_skip_dims[i],
                    device=h.device,
                    dtype=h.dtype,
                )
                h = torch.cat([h, pad], dim=-1)
            h = layer(h)

        reconstruction = self.output_proj(h)
        return reconstruction

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full autoencoder forward pass.

        Parameters
        ----------
        x : Tensor, shape ``(B, input_dim)``

        Returns
        -------
        dict
            ``"reconstruction"``: ``(B, input_dim)``
            ``"latent"``: ``(B, latent_dim)``
            ``"mu"``: ``(B, latent_dim)`` *(VAE only)*
            ``"log_var"``: ``(B, latent_dim)`` *(VAE only)*
        """
        enc_out = self.encode(x)
        reconstruction = self.decode(
            enc_out["latent"],
            encoder_hiddens=enc_out.get("encoder_hiddens"),
        )

        outputs: Dict[str, torch.Tensor] = {
            "reconstruction": reconstruction,
            "latent": enc_out["latent"],
        }
        if self.config.variational:
            outputs["mu"] = enc_out["mu"]
            outputs["log_var"] = enc_out["log_var"]
        return outputs

    # --------------------------------------------------------------------- #
    #  Phase 2 integration                                                  #
    # --------------------------------------------------------------------- #

    def get_decoder_head(self, freeze: bool = True) -> "AutoencoderDecoderHead":
        """Return the decoder wrapped as a ``CellPaintingHead``-compatible module.

        This is the primary integration point for Phase 2: the returned head
        can replace ``ProToPhenModel.decoders["cell_painting"]``.

        Parameters
        ----------
        freeze : bool
            If *True* (the default for Phase 2), all decoder parameters are
            frozen (``requires_grad = False``).
        """
        head = AutoencoderDecoderHead(self, freeze=freeze)
        return head

    # --------------------------------------------------------------------- #
    #  Freeze / unfreeze                                                    #
    # --------------------------------------------------------------------- #

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for layer in self.encoder_layers:
            for p in layer.parameters():
                p.requires_grad = False
        if self.config.variational:
            for p in self.latent_mu.parameters():
                p.requires_grad = False
            for p in self.latent_log_var.parameters():
                p.requires_grad = False
        else:
            for p in self.latent_proj.parameters():
                p.requires_grad = False
        for p in self.latent_norm.parameters():
            p.requires_grad = False
        logger.info("Autoencoder encoder frozen")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for layer in self.encoder_layers:
            for p in layer.parameters():
                p.requires_grad = True
        if self.config.variational:
            for p in self.latent_mu.parameters():
                p.requires_grad = True
            for p in self.latent_log_var.parameters():
                p.requires_grad = True
        else:
            for p in self.latent_proj.parameters():
                p.requires_grad = True
        for p in self.latent_norm.parameters():
            p.requires_grad = True
        logger.info("Autoencoder encoder unfrozen")

    def freeze_decoder(self) -> None:
        """Freeze all decoder parameters (typical for Phase 2)."""
        for p in self.decoder_input_proj.parameters():
            p.requires_grad = False
        for layer in self.decoder_layers:
            for p in layer.parameters():
                p.requires_grad = False
        for p in self.output_proj.parameters():
            p.requires_grad = False
        logger.info("Autoencoder decoder frozen")

    def unfreeze_decoder(self) -> None:
        """Unfreeze all decoder parameters."""
        for p in self.decoder_input_proj.parameters():
            p.requires_grad = True
        for layer in self.decoder_layers:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.output_proj.parameters():
            p.requires_grad = True
        logger.info("Autoencoder decoder unfrozen")

    # --------------------------------------------------------------------- #
    #  Properties                                                           #
    # --------------------------------------------------------------------- #

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    @property
    def output_dim(self) -> int:
        """Alias for ``input_dim`` (reconstruction target dimensionality)."""
        return self.config.input_dim

    @property
    def n_parameters(self) -> int:
        return self._n_params

    @property
    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "encoder_hidden_dims": self.config.encoder_hidden_dims,
            "decoder_hidden_dims": self.config.effective_decoder_hidden_dims,
            "variational": self.config.variational,
            "skip_connections": self.config.use_skip_connections,
            "n_parameters": self.n_parameters,
            "n_trainable_parameters": self.n_trainable_parameters,
        }

    def __repr__(self) -> str:
        return (
            f"PhenotypeAutoencoder("
            f"input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"variational={self.config.variational}, "
            f"params={self.n_parameters:,})"
        )


# ============================================================================
# AutoencoderDecoderHead (CellPaintingHead-compatible wrapper)
# ============================================================================


class AutoencoderDecoderHead(nn.Module):
    """Wraps the autoencoder decoder for drop-in use as a decoder head.

    After Phase 1, the autoencoder decoder can replace
    ``ProToPhenModel.decoders["cell_painting"]``.  This wrapper provides
    the same ``forward(latent) → Tensor`` and ``output_dim`` interface
    expected by ``ProToPhenModel.forward()``.

    When ``freeze=True`` all decoder parameters have ``requires_grad=False``
    so that only the protein encoder trains during Phase 2.

    The wrapper also exposes the intermediate phenotype latent so that
    ``ProToPhenModel`` or the serving pipeline can include it in the output
    dict when ``return_phenotype_latent=True``.

    Parameters
    ----------
    autoencoder : PhenotypeAutoencoder
        A fully initialised (and typically Phase-1-trained) autoencoder.
    freeze : bool
        If *True*, all decoder parameters are frozen.
    """

    def __init__(
        self,
        autoencoder: PhenotypeAutoencoder,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        # Store references to the decoder sub-modules rather than the full
        # autoencoder so that state-dict keys are clean and serialisable.
        self.decoder_input_proj = autoencoder.decoder_input_proj
        self.decoder_layers = autoencoder.decoder_layers
        self.output_proj = autoencoder.output_proj

        # Carry skip-dim mapping so we can zero-pad when encoder hiddens
        # are unavailable (which is always the case in Phase 2).
        self._decoder_skip_dims: Dict[int, int] = dict(
            autoencoder._decoder_skip_dims
        )

        # Preserve config for introspection
        self._input_dim = autoencoder.config.latent_dim
        self._output_dim = autoencoder.config.input_dim
        self._latent_dim = autoencoder.config.latent_dim

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            logger.info(
                "AutoencoderDecoderHead: all parameters frozen "
                f"({sum(p.numel() for p in self.parameters()):,} params)"
            )

        # Cache the last latent that passed through (for optional return)
        self._last_latent: Optional[torch.Tensor] = None

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector to reconstructed Cell Painting features.

        This signature matches ``CellPaintingHead.forward()`` so the head
        is a drop-in replacement inside ``ProToPhenModel.decoders``.

        Parameters
        ----------
        latent : Tensor, shape ``(B, latent_dim)``
            Protein encoder output (which, in Phase 2, targets the
            autoencoder latent space).

        Returns
        -------
        Tensor, shape ``(B, output_dim)``
            Reconstructed Cell Painting features.
        """
        # Cache for optional retrieval via get_last_latent()
        self._last_latent = latent.detach()

        h = self.decoder_input_proj(latent)
        for i, layer in enumerate(self.decoder_layers):
            if i in self._decoder_skip_dims:
                # Layer was built expecting a skip connection — zero-pad
                pad = torch.zeros(
                    h.size(0),
                    self._decoder_skip_dims[i],
                    device=h.device,
                    dtype=h.dtype,
                )
                h = torch.cat([h, pad], dim=-1)
            h = layer(h)
        reconstruction = self.output_proj(h)
        return reconstruction

    def get_last_latent(self) -> Optional[torch.Tensor]:
        """Return the latent that produced the most recent reconstruction.

        This allows ``ProToPhenModel`` or the serving pipeline to retrieve
        the phenotype latent without modifying the ``forward()`` signature.
        """
        return self._last_latent

    @property
    def input_dim(self) -> int:
        """Input dimension (autoencoder latent dim)."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Output dimension (number of Cell Painting features)."""
        return self._output_dim

    @property
    def latent_dim(self) -> int:
        """Latent space dimension."""
        return self._latent_dim

    def __repr__(self) -> str:
        frozen = not any(p.requires_grad for p in self.parameters())
        return (
            f"AutoencoderDecoderHead("
            f"latent_dim={self._latent_dim}, "
            f"output_dim={self._output_dim}, "
            f"frozen={frozen})"
        )


# ============================================================================
# Loss functions for autoencoder pre-training
# ============================================================================


class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross-Entropy (NT-Xent) loss.

    Also known as *supervised contrastive loss* when operating on integer
    labels.  Given a batch of latent vectors and their treatment labels,
    the loss pulls together representations that share the same label and
    pushes apart representations with different labels.

    Parameters
    ----------
    temperature : float
        Scaling temperature τ for the softmax. Lower values make the
        distribution sharper.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        latent: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss.

        Parameters
        ----------
        latent : Tensor, shape ``(B, D)``
            Normalised (or un-normalised if L2-normalisation is applied
            internally) latent representations.
        labels : Tensor, shape ``(B,)``
            Integer treatment labels.  Samples sharing a label are
            treated as positives.

        Returns
        -------
        Tensor
            Scalar loss.  Returns ``0`` if no positive pairs exist in the
            batch.
        """
        device = latent.device
        batch_size = latent.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2-normalise
        z = F.normalize(latent, p=2, dim=-1)

        # Cosine similarity matrix scaled by temperature
        sim = z @ z.T / self.temperature  # (B, B)

        # Mask: positive pairs share the same label (excluding self)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = label_eq & ~self_mask

        # If no positive pairs exist, return zero
        n_positives_per_sample = positive_mask.sum(dim=1)
        has_positives = n_positives_per_sample > 0
        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Log-sum-exp over all non-self entries (denominator)
        # Mask out self-similarities by setting them to -inf before exp
        logits = sim.masked_fill(self_mask, float("-inf"))
        log_sum_exp = torch.logsumexp(logits, dim=1)  # (B,)

        # Mean of log-prob over positive pairs for each anchor
        # For numerical stability compute: log(exp(sim_pos)/sum_exp)
        #   = sim_pos - log_sum_exp
        pos_logits = sim.masked_fill(~positive_mask, float("-inf"))
        # For each anchor, average over its positives
        # Use logsumexp - log(n_pos) trick:
        #   mean_i = (1/|P_i|) * sum_{p in P_i} (sim_ip - log_sum_exp_i)
        #          = (1/|P_i|) * (logsumexp(sim_P_i) + log(|P_i|) ... )
        # Simpler direct approach:
        loss_per_anchor = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            if n_positives_per_sample[i] == 0:
                continue
            pos_sims_i = sim[i][positive_mask[i]]  # (n_pos,)
            loss_per_anchor[i] = -(pos_sims_i - log_sum_exp[i]).mean()

        # Average over anchors that have positives
        loss = loss_per_anchor[has_positives].mean()
        return loss


class NTXentLossVectorised(nn.Module):
    """Fully vectorised NT-Xent loss (faster for large batches).

    Semantically identical to :class:`NTXentLoss` but avoids the Python
    loop over batch elements.

    Parameters
    ----------
    temperature : float
        Scaling temperature.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        latent: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = latent.device
        batch_size = latent.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z = F.normalize(latent, p=2, dim=-1)
        sim = z @ z.T / self.temperature

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = label_eq & ~self_mask

        n_positives = positive_mask.sum(dim=1).float()  # (B,)
        has_positives = n_positives > 0

        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Denominator: log-sum-exp over non-self entries
        logits_for_denom = sim.masked_fill(self_mask, float("-inf"))
        log_sum_exp = torch.logsumexp(logits_for_denom, dim=1)  # (B,)

        # Numerator: sum of (sim - log_sum_exp) over positive pairs
        log_prob = sim - log_sum_exp.unsqueeze(1)  # (B, B)
        # Zero out non-positive entries
        pos_log_prob = log_prob * positive_mask.float()  # (B, B)
        # Sum over positives, normalise
        mean_pos_log_prob = pos_log_prob.sum(dim=1) / n_positives.clamp(min=1)

        loss = -mean_pos_log_prob[has_positives].mean()
        return loss


class AutoencoderLoss(nn.Module):
    """Combined loss for Phase 1 autoencoder training.

    Components
    ----------
    1. **Reconstruction loss**: MSE or Huber between input and
       reconstruction.  Optionally weighted per feature group (e.g.
       nucleus vs. cytoplasm).
    2. **Contrastive loss**: NT-Xent on treatment labels, encouraging
       replicates of the same gene / compound to cluster in latent space.
    3. **KL divergence**: standard VAE KL (only when ``variational=True``).

    Parameters
    ----------
    config : Phase1LossConfig
        Loss component weights and settings.
    variational : bool
        Whether the autoencoder uses a variational bottleneck.
    feature_group_indices : dict[str, Tensor] | None
        Mapping from group names to index tensors for per-group weighting.
    """

    def __init__(
        self,
        config: Optional[Phase1LossConfig] = None,
        variational: bool = False,
        feature_group_indices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.config = config or Phase1LossConfig()
        self.variational = variational

        # Reconstruction loss
        if self.config.reconstruction_type == "huber":
            self.recon_fn = nn.SmoothL1Loss(
                reduction="none", beta=self.config.huber_delta,
            )
        else:
            self.recon_fn = nn.MSELoss(reduction="none")

        # Contrastive loss
        self.contrastive_fn = NTXentLossVectorised(
            temperature=self.config.contrastive_temperature,
        )

        # Feature group weighting
        self.feature_group_indices = feature_group_indices
        if (
            feature_group_indices is not None
            and self.config.feature_group_weights is not None
        ):
            self._build_feature_weight_vector(feature_group_indices)
        else:
            self.feature_weights: Optional[torch.Tensor] = None

    def _build_feature_weight_vector(
        self,
        group_indices: Dict[str, torch.Tensor],
    ) -> None:
        """Build a per-feature weight vector from group weights."""
        weights = self.config.feature_group_weights or {}
        # Determine total feature count from max index
        max_idx = max(idx.max().item() for idx in group_indices.values()) + 1
        w = torch.ones(max_idx)
        for group_name, indices in group_indices.items():
            if group_name in weights:
                w[indices] = weights[group_name]
        self.register_buffer("feature_weights", w)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        inputs: torch.Tensor,
        treatment_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined autoencoder loss.

        Parameters
        ----------
        outputs : dict
            Autoencoder output dict containing at least ``"reconstruction"``
            and ``"latent"``.  If variational, also ``"mu"`` and ``"log_var"``.
        inputs : Tensor, shape ``(B, input_dim)``
            Original phenotype features (reconstruction target).
        treatment_labels : Tensor, shape ``(B,)`` | None
            Integer treatment labels for the contrastive component.  If
            *None* the contrastive loss is skipped.

        Returns
        -------
        dict
            ``"total"``: weighted sum of all components.
            ``"reconstruction"``: reconstruction loss.
            ``"contrastive"``: contrastive loss (``0`` if skipped).
            ``"kl"``: KL divergence (``0`` if not variational).
        """
        device = inputs.device
        reconstruction = outputs["reconstruction"]
        latent = outputs["latent"]

        losses: Dict[str, torch.Tensor] = {}

        # ---- Reconstruction -------------------------------------------------
        recon_per_feature = self.recon_fn(reconstruction, inputs)  # (B, D)
        if self.feature_weights is not None:
            recon_per_feature = recon_per_feature * self.feature_weights.to(device)
        recon_loss = recon_per_feature.mean()
        losses["reconstruction"] = recon_loss

        # ---- Contrastive ----------------------------------------------------
        if (
            self.config.contrastive_weight > 0
            and treatment_labels is not None
        ):
            contrastive_loss = self.contrastive_fn(latent, treatment_labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
        losses["contrastive"] = contrastive_loss

        # ---- KL divergence --------------------------------------------------
        if self.variational and "mu" in outputs and "log_var" in outputs:
            mu = outputs["mu"]
            log_var = outputs["log_var"]
            # Standard VAE KL: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1)
            kl_loss = kl.mean()
        else:
            kl_loss = torch.tensor(0.0, device=device)
        losses["kl"] = kl_loss

        # ---- Total ----------------------------------------------------------
        total = (
            self.config.reconstruction_weight * recon_loss
            + self.config.contrastive_weight * contrastive_loss
            + self.config.kl_weight * kl_loss
        )
        losses["total"] = total

        return losses


# ============================================================================
# PretrainingDataset - Phase 1 (phenotype-only)
# ============================================================================


class PretrainingDataset(Dataset):
    """``torch.utils.data.Dataset`` for Phase 1 phenotype-only training.

    Unlike :class:`~protophen.data.dataset.ProtoPhenDataset`, this dataset
    does **not** require protein embeddings.  Each sample is a single
    morphological profile accompanied by a treatment label (for the
    contrastive loss) and optional metadata.

    The dataset supports two loading modes:

    1. **Eager**: pass pre-loaded numpy arrays at construction time.
    2. **Lazy**: pass a path to a curated Parquet directory; profiles are
       memory-mapped and loaded on demand (TODO: future session).

    Parameters
    ----------
    phenotype_features : ndarray, shape ``(N, D)``
        Normalised Cell Painting feature matrix.
    treatment_labels : ndarray | list, shape ``(N,)``
        String or integer treatment identifiers.
    plate_ids : ndarray | list | None, shape ``(N,)``
        Plate identifiers (used for plate-aware batching or logging).
    sample_weights : ndarray | None, shape ``(N,)``
        Per-sample weights (e.g. down-weight compound-only data in
        Phase 2 hybrid mode).
    augmentation_noise_std : float
        Standard deviation of additive Gaussian noise applied to features
        during training (set to 0 for validation / test).

    Examples
    --------
    >>> features = np.random.randn(10_000, 1500).astype(np.float32)
    >>> labels = np.random.randint(0, 500, size=10_000)
    >>> ds = PretrainingDataset(features, labels)
    >>> sample = ds[0]
    >>> sample["phenotype_features"].shape
    torch.Size([1500])
    """

    def __init__(
        self,
        phenotype_features: np.ndarray,
        treatment_labels: Union[np.ndarray, List[Any]],
        plate_ids: Optional[Union[np.ndarray, List[str]]] = None,
        sample_weights: Optional[np.ndarray] = None,
        augmentation_noise_std: float = 0.0,
    ) -> None:
        super().__init__()

        self.phenotype_features = np.ascontiguousarray(
            phenotype_features, dtype=np.float32,
        )
        n_samples = self.phenotype_features.shape[0]

        # Treatment labels (encode strings to integers for contrastive loss)
        if isinstance(treatment_labels, np.ndarray):
            raw_labels = treatment_labels.tolist()
        else:
            raw_labels = list(treatment_labels)

        self.label_to_int: Dict[Any, int] = {}
        self.int_to_label: Dict[int, Any] = {}
        encoded: List[int] = []
        for lbl in raw_labels:
            if lbl not in self.label_to_int:
                idx = len(self.label_to_int)
                self.label_to_int[lbl] = idx
                self.int_to_label[idx] = lbl
            encoded.append(self.label_to_int[lbl])
        self.treatment_label_ints = np.array(encoded, dtype=np.int64)

        # Plate IDs (keep as strings)
        if plate_ids is not None:
            if isinstance(plate_ids, np.ndarray):
                self.plate_ids: List[str] = plate_ids.tolist()
            else:
                self.plate_ids = list(plate_ids)
        else:
            self.plate_ids = ["unknown"] * n_samples

        # Sample weights
        if sample_weights is not None:
            self.sample_weights = np.ascontiguousarray(
                sample_weights, dtype=np.float32,
            )
        else:
            self.sample_weights = np.ones(n_samples, dtype=np.float32)

        self.augmentation_noise_std = augmentation_noise_std

        # Sanity checks
        assert len(self.treatment_label_ints) == n_samples
        assert len(self.plate_ids) == n_samples
        assert len(self.sample_weights) == n_samples

        logger.info(
            f"PretrainingDataset: {n_samples:,} samples, "
            f"{self.phenotype_features.shape[1]} features, "
            f"{len(self.label_to_int):,} unique treatments"
        )

    def __len__(self) -> int:
        return self.phenotype_features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample.

        Returns
        -------
        dict
            ``"phenotype_features"`` — ``Tensor (D,)``
            ``"treatment_label"`` — ``Tensor (scalar, int64)``
            ``"plate_id"`` — ``str``
            ``"sample_weight"`` — ``Tensor (scalar, float32)``
        """
        features = self.phenotype_features[idx].copy()

        # Training-time augmentation
        if self.augmentation_noise_std > 0:
            noise = np.random.randn(*features.shape).astype(np.float32)
            features = features + self.augmentation_noise_std * noise

        return {
            "phenotype_features": torch.from_numpy(features),
            "treatment_label": torch.tensor(
                self.treatment_label_ints[idx], dtype=torch.long,
            ),
            "plate_id": self.plate_ids[idx],
            "sample_weight": torch.tensor(
                self.sample_weights[idx], dtype=torch.float32,
            ),
        }

    @property
    def n_features(self) -> int:
        """Number of phenotype features per sample."""
        return self.phenotype_features.shape[1]

    @property
    def n_treatments(self) -> int:
        """Number of unique treatment labels."""
        return len(self.label_to_int)

    @classmethod
    def from_parquet(
        cls,
        parquet_path: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
        treatment_column: str = "treatment_label",
        plate_column: str = "plate_id",
        weight_column: Optional[str] = None,
        augmentation_noise_std: float = 0.0,
    ) -> "PretrainingDataset":
        """Load dataset from a curated Parquet file.

        Parameters
        ----------
        parquet_path : str | Path
            Path to the Parquet file produced by
            :class:`~protophen.data.jumpcp.curation.DataCurator`.
        feature_columns : list[str] | None
            If provided, restrict to these feature columns.  If *None*,
            all columns that are not metadata are treated as features.
        treatment_column : str
            Column name for treatment labels.
        plate_column : str
            Column name for plate identifiers.
        weight_column : str | None
            Optional column for per-sample weights.
        augmentation_noise_std : float
            Gaussian noise std for training augmentation.

        Returns
        -------
        PretrainingDataset
        """
        import pandas as pd

        parquet_path = Path(parquet_path)
        df = pd.read_parquet(parquet_path)

        # Identify feature columns
        metadata_cols = {treatment_column, plate_column}
        if weight_column:
            metadata_cols.add(weight_column)
        # Also exclude common metadata prefixes
        meta_prefixes = ("Metadata_",)

        if feature_columns is None:
            feature_columns = [
                c for c in df.columns
                if c not in metadata_cols
                and not any(c.startswith(p) for p in meta_prefixes)
            ]

        features = df[feature_columns].values.astype(np.float32)
        labels = df[treatment_column].values if treatment_column in df.columns else np.zeros(len(df))
        plates = (
            df[plate_column].values if plate_column in df.columns else None
        )
        weights = (
            df[weight_column].values.astype(np.float32)
            if weight_column and weight_column in df.columns
            else None
        )

        return cls(
            phenotype_features=features,
            treatment_labels=labels,
            plate_ids=plates,
            sample_weights=weights,
            augmentation_noise_std=augmentation_noise_std,
        )

    def split(
        self,
        train_frac: float = 0.9,
        val_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple["PretrainingDataset", "PretrainingDataset"]:
        """Split into training and validation sets.

        Augmentation noise is disabled for the validation set.

        Parameters
        ----------
        train_frac, val_frac : float
            Must sum to 1.
        seed : int
            Random seed.

        Returns
        -------
        tuple[PretrainingDataset, PretrainingDataset]
        """
        assert abs(train_frac + val_frac - 1.0) < 1e-6
        n = len(self)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        n_train = int(n * train_frac)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        def _subset(idx: np.ndarray, noise: float) -> "PretrainingDataset":
            raw_labels = [self.int_to_label[self.treatment_label_ints[i]] for i in idx]
            return PretrainingDataset(
                phenotype_features=self.phenotype_features[idx],
                treatment_labels=raw_labels,
                plate_ids=[self.plate_ids[i] for i in idx],
                sample_weights=self.sample_weights[idx],
                augmentation_noise_std=noise,
            )

        train_ds = _subset(train_idx, self.augmentation_noise_std)
        val_ds = _subset(val_idx, 0.0)  # No augmentation for validation

        logger.info(
            f"PretrainingDataset split: train={len(train_ds)}, val={len(val_ds)}"
        )
        return train_ds, val_ds

    def __repr__(self) -> str:
        return (
            f"PretrainingDataset("
            f"n_samples={len(self):,}, "
            f"n_features={self.n_features}, "
            f"n_treatments={self.n_treatments:,})"
        )


# ============================================================================
# Latent space quality utilities
# ============================================================================


@torch.no_grad()
def compute_replicate_correlation(
    latent: torch.Tensor,
    labels: torch.Tensor,
    max_treatments: int = 500,
) -> float:
    """Compute mean within-treatment cosine similarity in latent space.

    This measures how consistently the autoencoder maps replicates of the
    same treatment to nearby points. This is a key quality indicator for the
    contrastive component.

    Parameters
    ----------
    latent : Tensor, shape ``(N, D)``
    labels : Tensor, shape ``(N,)``
        Integer treatment labels.
    max_treatments : int
        Subsample treatments for efficiency.

    Returns
    -------
    float
        Mean within-treatment cosine similarity (higher is better).
    """
    z = F.normalize(latent, p=2, dim=-1)
    unique_labels = labels.unique()

    if len(unique_labels) > max_treatments:
        perm = torch.randperm(len(unique_labels))[:max_treatments]
        unique_labels = unique_labels[perm]

    sims: List[float] = []
    for lbl in unique_labels:
        mask = labels == lbl
        n = mask.sum().item()
        if n < 2:
            continue
        z_group = z[mask]
        # Pairwise cosine similarity within group
        sim_matrix = z_group @ z_group.T
        # Upper triangle (excluding diagonal)
        n_pairs = n * (n - 1) / 2
        triu_sum = (sim_matrix.triu(diagonal=1)).sum().item()
        sims.append(triu_sum / max(n_pairs, 1))

    return float(np.mean(sims)) if sims else 0.0


@torch.no_grad()
def compute_latent_silhouette(
    latent: torch.Tensor,
    labels: torch.Tensor,
    sample_size: int = 5000,
) -> float:
    """Compute silhouette score on a subsample of the latent space.

    Parameters
    ----------
    latent : Tensor, shape ``(N, D)``
    labels : Tensor, shape ``(N,)``
    sample_size : int
        Subsample for computational tractability.

    Returns
    -------
    float
        Silhouette score in [-1, 1] (higher is better).
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.warning("scikit-learn not available; skipping silhouette score")
        return 0.0

    z_np = latent.cpu().numpy()
    lbl_np = labels.cpu().numpy()

    # Subsample if necessary
    n = len(z_np)
    if n > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_size, replace=False)
        z_np = z_np[idx]
        lbl_np = lbl_np[idx]

    # Need at least 2 labels with >= 1 sample each
    unique, counts = np.unique(lbl_np, return_counts=True)
    if len(unique) < 2:
        return 0.0

    try:
        score = silhouette_score(z_np, lbl_np, metric="cosine", sample_size=None)
        return float(score)
    except Exception:
        return 0.0


# ============================================================================
# Checkpoint helpers
# ============================================================================


def save_phase1_checkpoint(
    path: Union[str, Path],
    autoencoder: PhenotypeAutoencoder,
    config: PretrainingConfig,
    epoch: int,
    global_step: int = 0,
    best_val_loss: float = float("inf"),
    optimiser_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a Phase 1 checkpoint in the ``Trainer`` format.

    The checkpoint is compatible with ``Trainer._load_checkpoint()``,
    ``load_checkpoint()``, ``build_model_from_checkpoint()``, and
    ``register_from_trainer_checkpoint()`` from serving/deployment infrastructure.

    Parameters
    ----------
    path : str | Path
        Output path.
    autoencoder : PhenotypeAutoencoder
        The trained autoencoder.
    config : PretrainingConfig
        Full pretraining configuration (for provenance).
    epoch, global_step, best_val_loss
        Training state metadata.
    optimiser_state, scheduler_state, scaler_state
        Optional optimiser / scheduler / AMP scaler state dicts.
    metrics : dict | None
        Optional evaluation metrics to include in the checkpoint.
    """
    from protophen.training.trainer import _to_python_types

    checkpoint: Dict[str, Any] = {
        # Standard keys
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": autoencoder.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "tasks": ["cell_painting"],
            "epochs": config.phase1.training.epochs,
            "learning_rate": config.phase1.training.learning_rate,
            "weight_decay": config.phase1.training.weight_decay,
            "optimiser": config.phase1.training.optimiser,
            "scheduler": config.phase1.training.scheduler,
        },
        # Phase 1 extensions
        "phase": 1,
        "autoencoder_config": asdict(config.autoencoder),
        "pretraining_config": asdict(config),
    }

    if optimiser_state is not None:
        checkpoint["optimiser_state_dict"] = optimiser_state
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state
    if scaler_state is not None:
        checkpoint["scaler_state_dict"] = scaler_state
    if metrics is not None:
        checkpoint["metrics"] = metrics

    checkpoint = _to_python_types(checkpoint)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Phase 1 checkpoint saved to {path}")


def save_phase2_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    config: PretrainingConfig,
    phase1_checkpoint_path: Union[str, Path],
    epoch: int,
    global_step: int = 0,
    best_val_loss: float = float("inf"),
    optimiser_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a Phase 2 checkpoint in the ``Trainer`` format.

    The checkpoint contains all keys expected by the serving/deployment infrastructure
    (``load_checkpoint``, ``build_model_from_checkpoint``,
    ``_infer_model_config_from_state_dict``).

    Parameters
    ----------
    path : str | Path
        Output path.
    model : nn.Module
        The full ProToPhen model (protein encoder + autoencoder decoder).
    config : PretrainingConfig
        Full pretraining configuration.
    phase1_checkpoint_path : str | Path
        Path to the Phase 1 checkpoint used for initialisation.
    epoch, global_step, best_val_loss
        Training state metadata.
    """
    from protophen.training.trainer import _to_python_types

    checkpoint: Dict[str, Any] = {
        # Standard keys
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "tasks": ["cell_painting"],
            "epochs": config.phase2.training.epochs,
            "learning_rate": config.phase2.training.learning_rate,
            "weight_decay": config.phase2.training.weight_decay,
            "optimiser": config.phase2.training.optimiser,
            "scheduler": config.phase2.training.scheduler,
        },
        # Phase 2 extensions
        "phase": 2,
        "phase1_checkpoint": str(phase1_checkpoint_path),
        "autoencoder_config": asdict(config.autoencoder),
        "pretraining_config": asdict(config),
    }

    if optimiser_state is not None:
        checkpoint["optimiser_state_dict"] = optimiser_state
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state
    if scaler_state is not None:
        checkpoint["scaler_state_dict"] = scaler_state
    if metrics is not None:
        checkpoint["metrics"] = metrics

    checkpoint = _to_python_types(checkpoint)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Phase 2 checkpoint saved to {path}")


def load_autoencoder_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[PhenotypeAutoencoder, Dict[str, Any]]:
    """Load a Phase 1 autoencoder from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the checkpoint file.
    device : str | torch.device
        Device to load the model onto.

    Returns
    -------
    tuple[PhenotypeAutoencoder, dict]
        The autoencoder instance and the full checkpoint dict.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )

    ae_config_dict = checkpoint.get("autoencoder_config", {})
    # Remove derived keys that may have been saved
    ae_config_dict.pop("effective_decoder_hidden_dims", None)
    ae_config = PhenotypeAutoencoderConfig(**ae_config_dict)

    autoencoder = PhenotypeAutoencoder(ae_config)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.to(device)
    autoencoder.eval()

    logger.info(
        f"Loaded Phase {checkpoint.get('phase', '?')} autoencoder from "
        f"{checkpoint_path} (epoch {checkpoint.get('epoch', '?')})"
    )
    return autoencoder, checkpoint