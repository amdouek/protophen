"""
Tests for protein data structures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from protophen.data.protein import (
    Protein,
    ProteinLibrary,
    compute_sequence_hash,
    validate_sequence,
)


class TestValidateSequence:
    """Tests for sequence validation."""
    
    def test_valid_sequence(self):
        """Test validation of a valid sequence."""
        seq = "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
        result = validate_sequence(seq)
        assert result == seq
    
    def test_lowercase_normalisation(self):
        """Test that lowercase is converted to uppercase."""
        seq = "mkflillfnilclfpvlaadnhgvgpqgas"
        result = validate_sequence(seq)
        assert result == seq.upper()
    
    def test_whitespace_removal(self):
        """Test that whitespace is removed."""
        seq = "MKF LIL LFN ILC"
        result = validate_sequence(seq)
        assert result == "MKFLILLFNILC"
    
    def test_empty_sequence_raises(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_sequence("")
    
    def test_invalid_characters_raises(self):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_sequence("MKFL123ABC")
    
    def test_extended_alphabet(self):
        """Test extended alphabet support."""
        seq = "MKFLBXZ"  # B, X, Z are ambiguous codes
        
        # Should fail without extended
        with pytest.raises(ValueError):
            validate_sequence(seq, allow_extended=False)
        
        # Should pass with extended
        result = validate_sequence(seq, allow_extended=True)
        assert result == seq


class TestProtein:
    """Tests for Protein class."""
    
    def test_basic_creation(self):
        """Test basic protein creation."""
        protein = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        assert protein.sequence == "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
        assert protein.length == 30
        assert protein.name is not None  # Auto-generated
    
    def test_with_metadata(self):
        """Test protein with metadata."""
        protein = Protein(
            sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            name="test_protein",
            source="de_novo",
            metadata={"design_method": "RFdiffusion"},
        )
        
        assert protein.name == "test_protein"
        assert protein.source == "de_novo"
        assert protein.metadata["design_method"] == "RFdiffusion"
    
    def test_sequence_hash(self):
        """Test that same sequence produces same hash."""
        protein1 = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        protein2 = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        assert protein1.hash == protein2.hash
    
    def test_different_sequence_different_hash(self):
        """Test that different sequences produce different hashes."""
        protein1 = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        protein2 = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAT")
        
        assert protein1.hash != protein2.hash
    
    def test_molecular_weight(self):
        """Test molecular weight calculation."""
        protein = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        mw = protein.molecular_weight
        
        # Should be roughly in expected range for a 30 AA protein
        assert 2500 < mw < 4000
    
    def test_amino_acid_composition(self):
        """Test amino acid composition calculation."""
        protein = Protein(sequence="AAACCC")
        composition = protein.amino_acid_composition
        
        assert composition["A"] == pytest.approx(0.5)
        assert composition["C"] == pytest.approx(0.5)
        assert composition["D"] == pytest.approx(0.0)
    
    def test_embedding_storage(self):
        """Test embedding storage and retrieval."""
        protein = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        embedding = np.random.randn(1280)
        protein.set_embedding("esm2", embedding)
        
        assert protein.has_embeddings
        assert "esm2" in protein.embedding_types
        
        retrieved = protein.get_embedding("esm2")
        np.testing.assert_array_equal(retrieved, embedding)
    
    def test_to_fasta(self):
        """Test FASTA conversion."""
        protein = Protein(
            sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            name="test_protein",
            source="de_novo",
        )
        
        fasta = protein.to_fasta()
        
        assert fasta.startswith(">test_protein")
        assert "source=de_novo" in fasta
        assert "MKFLILLFNILCLFPVLAADNHGVGPQGAS" in fasta
    
    def test_to_dict_roundtrip(self):
        """Test dictionary conversion roundtrip."""
        protein = Protein(
            sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            name="test_protein",
            source="de_novo",
            metadata={"key": "value"},
        )
        
        data = protein.to_dict()
        restored = Protein.from_dict(data)
        
        assert restored.sequence == protein.sequence
        assert restored.name == protein.name
        assert restored.source == protein.source
        assert restored.metadata == protein.metadata


class TestProteinLibrary:
    """Tests for ProteinLibrary class."""
    
    def test_empty_library(self):
        """Test empty library creation."""
        library = ProteinLibrary(name="test")
        
        assert len(library) == 0
        assert library.name == "test"
    
    def test_add_protein(self):
        """Test adding proteins."""
        library = ProteinLibrary()
        protein = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        library.add(protein)
        
        assert len(library) == 1
        assert protein in library
    
    def test_duplicate_prevention(self):
        """Test that duplicates are prevented."""
        library = ProteinLibrary()
        protein1 = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS", name="p1")
        protein2 = Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS", name="p2")
        
        library.add(protein1)
        library.add(protein2)  # Same sequence, should be skipped
        
        assert len(library) == 1
    
    def test_filter(self):
        """Test filtering proteins."""
        library = ProteinLibrary()
        library.add(Protein(sequence="AAAAA", name="short"))
        library.add(Protein(sequence="A" * 100, name="long"))
        
        filtered = library.filter(lambda p: p.length > 50)
        
        assert len(filtered) == 1
        assert filtered[0].name == "long"
    
    def test_sample(self):
        """Test random sampling."""
        library = ProteinLibrary()
        for i in range(100):
            library.add(Protein(sequence=f"MKFL{'A' * i}", name=f"p{i}"))
        
        sampled = library.sample(10, seed=42)
        
        assert len(sampled) == 10
    
    def test_json_roundtrip(self):
        """Test JSON save/load roundtrip."""
        library = ProteinLibrary(name="test_library")
        library.add(Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS", name="p1"))
        library.add(Protein(sequence="ACDEFGHIKLMNPQRSTVWY", name="p2"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "library.json"
            library.to_json(path)
            
            loaded = ProteinLibrary.from_json(path)
        
        assert len(loaded) == 2
        assert loaded.name == "test_library"
        assert loaded[0].sequence == "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
    
    def test_summary(self):
        """Test summary statistics."""
        library = ProteinLibrary(name="test")
        library.add(Protein(sequence="AAAAA", source="de_novo"))
        library.add(Protein(sequence="A" * 100, source="de_novo"))
        library.add(Protein(sequence="A" * 50, source="literature"))
        
        summary = library.summary()
        
        assert summary["count"] == 3
        assert summary["length_stats"]["min"] == 5
        assert summary["length_stats"]["max"] == 100
        assert summary["sources"]["de_novo"] == 2
        assert summary["sources"]["literature"] == 1