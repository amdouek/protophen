"""
ESM-2 protein embedding extraction.

This module provides the ESMEmbedder class for extracting embeddings
from protein sequences using Facebook's ESM-2 models.

ESM-2 models available:
    - esm2_t6_8M_UR50D: 6 layers, 8M params (fastest, least accurate)
    - esm2_t12_35M_UR50D: 12 layers, 35M params
    - esm2_t30_150M_UR50D: 30 layers, 150M params
    - esm2_t33_650M_UR50D: 33 layers, 650M params (recommended)
    - esm2_t36_3B_UR50D: 36 layers, 3B params (most accurate, slowest)

References:
    Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein 
    structure with a language model. Science.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from protophen.data.protein import Protein, ProteinLibrary
from protophen.utils.io import EmbeddingCache
from protophen.utils.logging import logger


# =============================================================================
# Constants
# =============================================================================

# Available ESM-2 models with their embedding dimensions
ESM2_MODELS = {
    "esm2_t6_8M_UR50D": {"layers": 6, "dim": 320},
    "esm2_t12_35M_UR50D": {"layers": 12, "dim": 480},
    "esm2_t30_150M_UR50D": {"layers": 30, "dim": 640},
    "esm2_t33_650M_UR50D": {"layers": 33, "dim": 1280},
    "esm2_t36_3B_UR50D": {"layers": 36, "dim": 2560},
}

# Pooling strategies
PoolingStrategy = Literal["mean", "cls", "max", "mean_cls"]


# =============================================================================
# ESM Embedder Configuration
# =============================================================================

@dataclass
class ESMEmbedderConfig:
    """Configuration for ESM-2 embedding extraction."""
    
    model_name: str = "esm2_t33_650M_UR50D"
    layer: int = -1  # -1 = last layer
    pooling: PoolingStrategy = "mean"
    batch_size: int = 8
    max_sequence_length: int = 1022  # ESM-2 limit
    device: str = "cuda"
    use_fp16: bool = True
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.model_name not in ESM2_MODELS:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Available: {list(ESM2_MODELS.keys())}"
            )
        
        if self.pooling not in ("mean", "cls", "max", "mean_cls"):
            raise ValueError(
                f"Unknown pooling strategy: {self.pooling}. "
                f"Available: mean, cls, max, mean_cls"
            )


# =============================================================================
# ESM Embedder Class
# =============================================================================

class ESMEmbedder:
    """
    Extract protein embeddings using ESM-2 models.
    
    This class handles loading ESM-2 models, processing protein sequences,
    and extracting embeddings with various pooling strategies.
    
    Attributes:
        config: Embedding configuration
        model: Loaded ESM-2 model
        alphabet: ESM-2 alphabet (tokeniser)
        embedding_dim: Dimension of output embeddings
        
    Example:
        >>> embedder = ESMEmbedder(model_name="esm2_t33_650M_UR50D")
        >>> 
        >>> # Single sequence
        >>> embedding = embedder.embed_sequence("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        >>> print(embedding.shape)  # (1280,)
        >>> 
        >>> # Batch of sequences
        >>> sequences = ["MKFLIL...", "ACDEFG...", "GHIKLM..."]
        >>> embeddings = embedder.embed_sequences(sequences)
        >>> print(embeddings.shape)  # (3, 1280)
        >>> 
        >>> # With Protein objects
        >>> protein = Protein(sequence="MKFLIL...")
        >>> embedder.embed_protein(protein)
        >>> print(protein.embeddings["esm2"].shape)  # (1280,)
    """
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        layer: int = -1,
        pooling: PoolingStrategy = "mean",
        batch_size: int = 8,
        device: Optional[str] = None,
        use_fp16: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialise ESM-2 embedder.
        
        Args:
            model_name: Name of ESM-2 model to use
            layer: Which layer to extract (-1 = last layer)
            pooling: Pooling strategy for sequence embeddings
            batch_size: Batch size for processing
            device: Device to use (cuda, cpu, mps). Auto-detected if None.
            use_fp16: Whether to use half precision (faster, less memory)
            cache_dir: Directory for caching computed embeddings
        """
        self.config = ESMEmbedderConfig(
            model_name=model_name,
            layer=layer,
            pooling=pooling,
            batch_size=batch_size,
            device=device or self._detect_device(),
            use_fp16=use_fp16,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self._model_loaded = False
        
        # Setup cache if specified
        self.cache = None
        if cache_dir is not None:
            self.cache = EmbeddingCache(cache_dir)
            logger.info(f"Using embedding cache at {cache_dir} ({len(self.cache)} cached)")
        
        # Model info
        model_info = ESM2_MODELS[model_name]
        self.embedding_dim = model_info["dim"]
        self.num_layers = model_info["layers"]
        
        logger.info(
            f"Initialised ESMEmbedder: model={model_name}, "
            f"dim={self.embedding_dim}, device={self.config.device}"
        )
    
    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load the ESM-2 model (lazy loading)."""
        if self._model_loaded:
            return
        
        logger.info(f"Loading ESM-2 model: {self.config.model_name}")
        
        try:
            import esm
        except ImportError:
            raise ImportError(
                "ESM package not found. Install with: pip install fair-esm"
            )
        
        # Load model and alphabet
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            self.config.model_name
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move to device
        self.model = self.model.to(self.config.device)
        self.model.eval()
        
        # Use half precision if specified and on CUDA
        if self.config.use_fp16 and self.config.device == "cuda":
            self.model = self.model.half()
            logger.debug("Using FP16 precision")
        
        self._model_loaded = True
        logger.info(f"Model loaded successfully on {self.config.device}")
    
    def _get_layer_index(self) -> int:
        """Get the actual layer index for extraction."""
        if self.config.layer == -1:
            return self.num_layers
        elif self.config.layer < 0:
            return self.num_layers + self.config.layer + 1
        else:
            return self.config.layer
    
    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token embeddings into sequence embeddings.
        
        Args:
            token_embeddings: (batch, seq_len, dim) tensor
            attention_mask: (batch, seq_len) tensor (1 for real tokens, 0 for padding)
            
        Returns:
            Pooled embeddings (batch, dim)
        """
        # Remove BOS and EOS tokens from consideration
        # ESM-2 adds <cls> at position 0 and <eos> at the end
        
        if self.config.pooling == "cls":
            # Use CLS token (position 0)
            return token_embeddings[:, 0, :]
        
        elif self.config.pooling == "mean":
            # Mean pool over actual sequence tokens (excluding BOS/EOS)
            # Mask: set padding and special tokens to 0
            seq_mask = attention_mask.clone()
            seq_mask[:, 0] = 0  # Exclude BOS
            # Find EOS positions and exclude
            for i in range(seq_mask.size(0)):
                eos_pos = attention_mask[i].sum().item() - 1
                if eos_pos > 0:
                    seq_mask[i, int(eos_pos)] = 0
            
            seq_mask = seq_mask.unsqueeze(-1)  # (batch, seq, 1)
            masked_embeddings = token_embeddings * seq_mask
            summed = masked_embeddings.sum(dim=1)
            counts = seq_mask.sum(dim=1).clamp(min=1)
            return summed / counts
        
        elif self.config.pooling == "max":
            # Max pool over sequence tokens
            seq_mask = attention_mask.clone()
            seq_mask[:, 0] = 0
            for i in range(seq_mask.size(0)):
                eos_pos = attention_mask[i].sum().item() - 1
                if eos_pos > 0:
                    seq_mask[i, int(eos_pos)] = 0
            
            # Set masked positions to very negative value
            mask_expanded = seq_mask.unsqueeze(-1).expand_as(token_embeddings)
            masked_embeddings = token_embeddings.masked_fill(~mask_expanded.bool(), -1e9)
            return masked_embeddings.max(dim=1)[0]
        
        elif self.config.pooling == "mean_cls":
            # Concatenate mean pooling and CLS token
            cls_emb = token_embeddings[:, 0, :]
            
            seq_mask = attention_mask.clone()
            seq_mask[:, 0] = 0
            for i in range(seq_mask.size(0)):
                eos_pos = attention_mask[i].sum().item() - 1
                if eos_pos > 0:
                    seq_mask[i, int(eos_pos)] = 0
            
            seq_mask = seq_mask.unsqueeze(-1)
            masked_embeddings = token_embeddings * seq_mask
            summed = masked_embeddings.sum(dim=1)
            counts = seq_mask.sum(dim=1).clamp(min=1)
            mean_emb = summed / counts
            
            return torch.cat([cls_emb, mean_emb], dim=-1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")
    
    def _truncate_sequence(self, sequence: str) -> str:
        """Truncate sequence to maximum length if necessary."""
        max_len = self.config.max_sequence_length
        if len(sequence) > max_len:
            logger.warning(
                f"Sequence length {len(sequence)} exceeds maximum {max_len}. "
                f"Truncating to {max_len} residues."
            )
            return sequence[:max_len]
        return sequence
    
    @torch.no_grad()
    def _embed_batch(
        self,
        sequences: list[tuple[str, str]],  # List of (id, sequence) tuples
    ) -> dict[str, np.ndarray]:
        """
        Embed a batch of sequences.
        
        Args:
            sequences: List of (id, sequence) tuples
            
        Returns:
            Dictionary mapping IDs to embeddings
        """
        self._load_model()
        
        # Truncate sequences if necessary
        sequences = [(id_, self._truncate_sequence(seq)) for id_, seq in sequences]
        
        # Convert to batch
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(self.config.device)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (batch_tokens != self.alphabet.padding_idx).float()
        
        # Get embeddings
        if self.config.use_fp16 and self.config.device == "cuda":
            with torch.cuda.amp.autocast():
                results = self.model(
                    batch_tokens,
                    repr_layers=[self._get_layer_index()],
                    return_contacts=False,
                )
        else:
            results = self.model(
                batch_tokens,
                repr_layers=[self._get_layer_index()],
                return_contacts=False,
            )
        
        # Extract token embeddings from specified layer
        token_embeddings = results["representations"][self._get_layer_index()]
        
        # Pool to sequence embeddings
        pooled = self._pool_embeddings(token_embeddings, attention_mask)
        
        # Convert to numpy and create output dict
        pooled_np = pooled.float().cpu().numpy()
        
        return {id_: pooled_np[i] for i, (id_, _) in enumerate(sequences)}
    
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Embed a single protein sequence.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Embedding array of shape (embedding_dim,) or (embedding_dim * 2,) for mean_cls
        """
        result = self._embed_batch([("seq", sequence)])
        return result["seq"]
    
    def embed_sequences(
        self,
        sequences: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        show_progress: bool = True,
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """
        Embed multiple protein sequences.
        
        Args:
            sequences: List of amino acid sequences
            ids: Optional list of IDs. If provided, returns dict; otherwise returns array.
            show_progress: Whether to show progress bar
            
        Returns:
            If ids is None: Array of shape (n_sequences, embedding_dim)
            If ids is provided: Dictionary mapping IDs to embeddings
        """
        if ids is not None and len(ids) != len(sequences):
            raise ValueError(
                f"Number of IDs ({len(ids)}) must match number of sequences ({len(sequences)})"
            )
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"seq_{i}" for i in range(len(sequences))]
            return_array = True
        else:
            return_array = False
        
        # Check cache for existing embeddings
        all_embeddings = {}
        sequences_to_embed = []
        
        for id_, seq in zip(ids, sequences):
            cache_key = f"{self.config.model_name}_{self.config.pooling}_{id_}"
            
            if self.cache is not None and self.cache.has(cache_key):
                all_embeddings[id_] = self.cache.get(cache_key)
            else:
                sequences_to_embed.append((id_, seq, cache_key))
        
        if sequences_to_embed:
            logger.info(
                f"Computing embeddings for {len(sequences_to_embed)} sequences "
                f"({len(all_embeddings)} cached)"
            )
            
            # Process in batches
            batches = [
                sequences_to_embed[i:i + self.config.batch_size]
                for i in range(0, len(sequences_to_embed), self.config.batch_size)
            ]
            
            iterator = tqdm(batches, desc="Embedding", disable=not show_progress)
            
            for batch in iterator:
                batch_input = [(id_, seq) for id_, seq, _ in batch]
                batch_embeddings = self._embed_batch(batch_input)
                
                for id_, seq, cache_key in batch:
                    embedding = batch_embeddings[id_]
                    all_embeddings[id_] = embedding
                    
                    # Cache the embedding
                    if self.cache is not None:
                        self.cache.set(cache_key, embedding)
        
        # Return in original order
        if return_array:
            return np.stack([all_embeddings[f"seq_{i}"] for i in range(len(sequences))])
        else:
            return {id_: all_embeddings[id_] for id_ in ids}
    
    def embed_protein(
        self,
        protein: Protein,
        embedding_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Embed a Protein object and store the embedding.
        
        Args:
            protein: Protein object
            embedding_key: Key to store embedding under. Default: "esm2"
            
        Returns:
            Embedding array
        """
        if embedding_key is None:
            embedding_key = "esm2"
        
        # Check if already embedded
        if embedding_key in protein.embeddings:
            return protein.embeddings[embedding_key]
        
        # Compute embedding
        embedding = self.embed_sequence(protein.sequence)
        protein.set_embedding(embedding_key, embedding)
        
        return embedding
    
    def embed_library(
        self,
        library: ProteinLibrary,
        embedding_key: Optional[str] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed all proteins in a library.
        
        Args:
            library: ProteinLibrary object
            embedding_key: Key to store embeddings under. Default: "esm2"
            show_progress: Whether to show progress bar
            
        Returns:
            Stacked embeddings array of shape (n_proteins, embedding_dim)
        """
        if embedding_key is None:
            embedding_key = "esm2"
        
        # Get sequences and hashes
        sequences = []
        hashes = []
        proteins_to_embed = []
        
        for protein in library:
            if embedding_key in protein.embeddings:
                continue
            sequences.append(protein.sequence)
            hashes.append(protein.hash)
            proteins_to_embed.append(protein)
        
        if not sequences:
            logger.info("All proteins already have embeddings")
            return library.get_embedding_matrix(embedding_key)
        
        logger.info(f"Embedding {len(sequences)} proteins from library '{library.name}'")
        
        # Compute embeddings
        embeddings = self.embed_sequences(
            sequences,
            ids=hashes,
            show_progress=show_progress,
        )
        
        # Store embeddings in proteins
        for protein in proteins_to_embed:
            protein.set_embedding(embedding_key, embeddings[protein.hash])
        
        return library.get_embedding_matrix(embedding_key)
    
    @property
    def output_dim(self) -> int:
        """Get the output embedding dimension."""
        if self.config.pooling == "mean_cls":
            return self.embedding_dim * 2
        return self.embedding_dim
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ESMEmbedder("
            f"model={self.config.model_name}, "
            f"dim={self.output_dim}, "
            f"pooling={self.config.pooling}, "
            f"device={self.config.device})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def get_esm_embedder(
    model_name: str = "esm2_t33_650M_UR50D",
    **kwargs,
) -> ESMEmbedder:
    """
    Get an ESM-2 embedder with the specified model.
    
    This is a convenience function for creating ESMEmbedder instances.
    
    Args:
        model_name: Name of ESM-2 model
        **kwargs: Additional arguments passed to ESMEmbedder
        
    Returns:
        ESMEmbedder instance
    """
    return ESMEmbedder(model_name=model_name, **kwargs)


def list_esm_models() -> dict[str, dict]:
    """
    List available ESM-2 models with their specifications.
    
    Returns:
        Dictionary mapping model names to their specifications
    """
    return ESM2_MODELS.copy()