"""
Tests for protein embedding modules.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from protophen.data.protein import Protein, ProteinLibrary
from protophen.embeddings.esm import (
    ESM2_MODELS,
    ESMEmbedder,
    ESMEmbedderConfig,
    list_esm_models,
)
from protophen.utils.io import EmbeddingCache


class TestESMEmbedderConfig:
    """Tests for ESMEmbedderConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ESMEmbedderConfig()
        
        assert config.model_name == "esm2_t33_650M_UR50D"
        assert config.layer == -1
        assert config.pooling == "mean"
        assert config.batch_size == 8
    
    def test_invalid_model_raises(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            ESMEmbedderConfig(model_name="invalid_model")
    
    def test_invalid_pooling_raises(self):
        """Test that invalid pooling raises error."""
        with pytest.raises(ValueError, match="Unknown pooling"):
            ESMEmbedderConfig(pooling="invalid_pooling")


class TestListModels:
    """Tests for model listing."""
    
    def test_list_esm_models(self):
        """Test listing available models."""
        models = list_esm_models()
        
        assert "esm2_t33_650M_UR50D" in models
        assert "esm2_t6_8M_UR50D" in models
        assert models["esm2_t33_650M_UR50D"]["dim"] == 1280
        assert models["esm2_t6_8M_UR50D"]["dim"] == 320


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(tmpdir)
            
            # Initially empty
            assert len(cache) == 0
            assert not cache.has("test_key")
            
            # Set and get
            embedding = np.random.randn(1280).astype(np.float32)
            cache.set("test_key", embedding)
            
            assert cache.has("test_key")
            assert len(cache) == 1
            
            retrieved = cache.get("test_key")
            np.testing.assert_array_almost_equal(retrieved, embedding)
    
    def test_cache_persistence(self):
        """Test that cache persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate cache
            cache1 = EmbeddingCache(tmpdir)
            embedding = np.random.randn(1280).astype(np.float32)
            cache1.set("test_key", embedding)
            
            # Create new cache instance
            cache2 = EmbeddingCache(tmpdir)
            
            assert cache2.has("test_key")
            retrieved = cache2.get("test_key")
            np.testing.assert_array_almost_equal(retrieved, embedding)
    
    def test_get_many(self):
        """Test batch retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(tmpdir)
            
            # Add multiple embeddings
            for i in range(5):
                cache.set(f"key_{i}", np.random.randn(1280).astype(np.float32))
            
            # Get multiple
            results = cache.get_many(["key_0", "key_2", "key_4", "nonexistent"])
            
            assert len(results) == 3
            assert "key_0" in results
            assert "nonexistent" not in results
    
    def test_cache_clear(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(tmpdir)
            cache.set("test_key", np.random.randn(1280).astype(np.float32))
            
            assert len(cache) == 1
            
            cache.clear()
            
            assert len(cache) == 0
            assert not cache.has("test_key")


class TestESMEmbedderMocked:
    """
    Tests for ESMEmbedder using mocked ESM model.
    
    These tests don't require the actual ESM model to be downloaded.
    """
    
    @pytest.fixture
    def mock_esm_model(self):
        """Create a mock ESM model."""
        mock_model = MagicMock()
        mock_alphabet = MagicMock()
        
        # Setup alphabet
        mock_alphabet.padding_idx = 0
        mock_alphabet.get_batch_converter.return_value = self._mock_batch_converter
        
        # Setup model output
        def mock_forward(tokens, repr_layers, return_contacts):
            batch_size, seq_len = tokens.shape
            # Return fake embeddings
            import torch
            embeddings = torch.randn(batch_size, seq_len, 1280)
            return {"representations": {33: embeddings}}
        
        mock_model.side_effect = mock_forward
        mock_model.return_value = {"representations": {33: MagicMock()}}
        
        return mock_model, mock_alphabet
    
    @staticmethod
    def _mock_batch_converter(sequences):
        """Mock batch converter."""
        import torch
        
        labels = [s[0] for s in sequences]
        strs = [s[1] for s in sequences]
        max_len = max(len(s) for s in strs) + 2  # +2 for BOS/EOS
        
        # Create fake tokens
        tokens = torch.ones(len(sequences), max_len, dtype=torch.long)
        
        return labels, strs, tokens
    
    def test_embedder_initialisation(self):
        """Test embedder initialisation without loading model."""
        embedder = ESMEmbedder(
            model_name="esm2_t33_650M_UR50D",
            device="cpu",
        )
        
        assert embedder.embedding_dim == 1280
        assert embedder.num_layers == 33
        assert not embedder._model_loaded
    
    def test_output_dim_mean(self):
        """Test output dimension for mean pooling."""
        embedder = ESMEmbedder(
            model_name="esm2_t33_650M_UR50D",
            pooling="mean",
            device="cpu",
        )
        assert embedder.output_dim == 1280
    
    def test_output_dim_mean_cls(self):
        """Test output dimension for mean_cls pooling."""
        embedder = ESMEmbedder(
            model_name="esm2_t33_650M_UR50D",
            pooling="mean_cls",
            device="cpu",
        )
        assert embedder.output_dim == 2560  # 1280 * 2
    
    def test_truncate_sequence(self):
        """Test sequence truncation."""
        embedder = ESMEmbedder(device="cpu")
        
        short_seq = "MKFL" * 10  # 40 AA
        long_seq = "MKFL" * 500  # 2000 AA (exceeds 1022 limit)
        
        assert embedder._truncate_sequence(short_seq) == short_seq
        assert len(embedder._truncate_sequence(long_seq)) == 1022


class TestESMEmbedderIntegration:
    """
    Integration tests for ESMEmbedder.
    
    These tests require the ESM model to be available.
    Skip if model is not installed or on resource-constrained systems.
    """
    
    @pytest.fixture
    def embedder(self):
        """Create embedder with smallest model for testing."""
        try:
            import esm
            return ESMEmbedder(
                model_name="esm2_t6_8M_UR50D",  # Smallest model
                device="cpu",
                use_fp16=False,
            )
        except ImportError:
            pytest.skip("ESM package not installed")
    
    @pytest.mark.slow
    def test_embed_single_sequence(self, embedder):
        """Test embedding a single sequence."""
        sequence = "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
        
        embedding = embedder.embed_sequence(sequence)
        
        assert embedding.shape == (320,)  # t6 model dim
        assert not np.isnan(embedding).any()
    
    @pytest.mark.slow
    def test_embed_multiple_sequences(self, embedder):
        """Test embedding multiple sequences."""
        sequences = [
            "MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            "ACDEFGHIKLMNPQRSTVWY",
            "MHHHHHHGGGGG",
        ]
        
        embeddings = embedder.embed_sequences(sequences, show_progress=False)
        
        assert embeddings.shape == (3, 320)
        assert not np.isnan(embeddings).any()
    
    @pytest.mark.slow
    def test_embed_with_ids(self, embedder):
        """Test embedding with custom IDs."""
        sequences = ["MKFLIL", "ACDEFG"]
        ids = ["protein_1", "protein_2"]
        
        embeddings = embedder.embed_sequences(sequences, ids=ids, show_progress=False)
        
        assert isinstance(embeddings, dict)
        assert "protein_1" in embeddings
        assert "protein_2" in embeddings
        assert embeddings["protein_1"].shape == (320,)
    
    @pytest.mark.slow
    def test_embed_protein(self, embedder):
        """Test embedding a Protein object."""
        protein = Protein(
            sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            name="test_protein",
        )
        
        embedding = embedder.embed_protein(protein)
        
        assert embedding.shape == (320,)
        assert "esm2" in protein.embeddings
        np.testing.assert_array_equal(protein.embeddings["esm2"], embedding)
    
    @pytest.mark.slow
    def test_embed_library(self, embedder):
        """Test embedding a ProteinLibrary."""
        library = ProteinLibrary(name="test")
        library.add(Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS", name="p1"))
        library.add(Protein(sequence="ACDEFGHIKLMNPQRSTVWY", name="p2"))
        
        embeddings = embedder.embed_library(library, show_progress=False)
        
        assert embeddings.shape == (2, 320)
        assert "esm2" in library[0].embeddings
        assert "esm2" in library[1].embeddings
    
    @pytest.mark.slow
    def test_caching(self, embedder):
        """Test that caching works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create embedder with cache
            cached_embedder = ESMEmbedder(
                model_name="esm2_t6_8M_UR50D",
                device="cpu",
                use_fp16=False,
                cache_dir=tmpdir,
            )
            
            sequence = "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
            
            # First call - should compute
            embedding1 = cached_embedder.embed_sequences(
                [sequence], 
                ids=["test"],
                show_progress=False,
            )
            
            # Second call - should use cache
            embedding2 = cached_embedder.embed_sequences(
                [sequence],
                ids=["test"],
                show_progress=False,
            )
            
            np.testing.assert_array_equal(embedding1["test"], embedding2["test"])