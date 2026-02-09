"""
Core protein data structure for ProToPhen

This module defines the fundamental data classes for representing proteins and collections of proteins throughout the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# ===========
# Constants
# ===========

# Standard amino acid alphabet
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Extended alphabet (ambig. + non-standard amino acids)
AMINO_ACIDS_EXTENDED = AMINO_ACIDS | set("BXZJUO")

# Amino acid properties for basic validation
AA_MOLECULAR_WEIGHTS = {
    "A": 89.1,  "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "E": 147.1, "Q": 146.2, "G": 75.1,  "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

# ====================
# Protein Validation Utilities
# ====================

def validate_sequence(sequence: str, allow_extended: bool = False) -> str:
    """
    Validate and normalise a protein string.

    Args:
        sequence: Amino acid sequence string.
        allow_extended: If True, allow ambiguous/non-standard amino acid codes (B, X, Z, J, U, O).

    Returns:
        Normalised (uppercase, stripped) sequence

    Raises:
        ValueError: If sequence is empty or contains invalid characters.            
    """
    # Normalise sequence
    sequence = sequence.upper().strip()

    # Remove common formatting characters
    sequence = re.sub(r'[\s\-\.\*]', '', sequence)

    if not sequence:
        raise ValueError("Sequence cannot be empty.")
    
    # Check for invalid characters
    valid_chars = AMINO_ACIDS_EXTENDED if allow_extended else AMINO_ACIDS
    invalid_chars = set(sequence) - valid_chars

    if invalid_chars:
        raise ValueError(f"Sequence contains invalid characters: {invalid_chars}."
                         f" Valid amino acids are: {''.join(sorted(valid_chars))}"
                         )
    return sequence

def compute_sequence_hash(sequence: str) -> str:
    """
    Compute a unique hash for a protein sequence.

    Useful for caching embeddings and deduplication.

    Args:
        sequence: Amino acid sequence.

    Returns:
        SHA-256 hash (first 16 chars).    
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]

# ====================
# Core Protein Class (Pydantic for validation)
# ====================

class Protein(BaseModel):
    """
    Core protein representation with validation.

    This class represents a single protein with its sequence, metadata, and optional computed properties (embeddings, structure, etc.).

    Attributes:
        sequence: Amino acid sequence (required).
        name: Protein name or identifier (optional).
        source: Origin of the protein (e.g., "de_novo", "literature", "pdb").
        metadata: Additional metadata dictionary.
        embeddings: Dictionary of computed embeddings (keyed by embedding type).
        structure: Predicted or known 3D structure information.

    Example:
        >>> protein = Protein(
                sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
                name="my_designed_protein",
                source="de_novo",
                metadata={"design_method": "RFdiffusion", "target": "binder"}
            )
        >>> print(protein.length)
        30
        >>> print(protein.hash)
        'a1b2c3d4e5f6g7h8'
    """
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="allow",
    )

    # Required fields
    sequence: str

    # Optional identification
    name: Optional[str] = None
    source: Optional[str] = None

    # Flexible metadata storage
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed properties (populated by embedding pipeline)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    # Structure information (populated by structure prediction)
    structure: Optional[dict[str, Any]] = None

    # ====================
    # Validators
    # ====================

    @field_validator("sequence")
    @classmethod
    def validate_sequence_field(cls, v: str) -> str:
        """Validate and normalise the sequence."""
        return validate_sequence(v, allow_extended=False)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Normalise protein name."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            # Remove potentially problematic characters for filenames
            v = re.sub(r'[<>:"/\\|?*]', '_', v)
        return v
    
    @field_validator("source")
    @classmethod
    def validate_source(cls, v: Optional[str]) -> Optional[str]:
        """Validate source field."""
        valid_sources = {"de_novo", "literature", "pdb", "uniprot", "synthetic", "unknown", None}
        if v is not None:
            v = v.lower().strip()
            if v not in valid_sources:
                # Don't raise error, just keep as-is for flexibility
                pass
        return v
    
    @model_validator(mode="after")
    def set_default_name(self) -> Protein:
        """Set default name based on hash if not provided."""
        if self.name is None:
            self.name = f"protein_{self.hash}"
        return self
    
    # ====================
    # Computed Properties
    # ====================

    @property
    def length(self) -> int:
        """ Return sequence length."""
        return len(self.sequence)
    
    @property
    def hash(self) -> str:
        """ Return unique sequence hash."""
        return compute_sequence_hash(self.sequence)
    
    @property
    def molecular_weight(self) -> float:
        """Calculate approximate molecular weight in Daltons.
        
        Note: This is an approximation - it doesn't account for modifications or terminal groups, etc.
        """
        water_mass = 18.015 # Mass of water lost during peptide bond formation
        total = sum(AA_MOLECULAR_WEIGHTS.get(aa, 110.0) for aa in self.sequence)  # Default avg weight for unknown AAs
        return total - (self.length - 1) * water_mass
    
    @property
    def amino_acid_composition(self) -> dict[str, float]:
        """Calculate amino acid composition (frequencies).
        
        Returns:
            Dictionary mapping amino acids to their frequencies (0-1).
        """
        counts = {aa: 0 for aa in AMINO_ACIDS}
        for aa in self.sequence:
            if aa in counts:
                counts[aa] += 1

        length = self.length
        return {aa: count / length for aa, count in counts.items()}
    
    @property
    def has_embeddings(self) -> bool:
        """Check if any embeddings have been computed."""
        return len(self.embeddings) > 0
    
    @property
    def embedding_types(self) -> list[str]:
        """List available embedding types."""
        return list(self.embeddings.keys())
    
    # ====================
    # Methods
    # ====================

    def get_embedding(self, embedding_type: str) -> np.ndarray:
        """Retrieve a specific embedding.
        
        Args:
            embedding_type: Type of embedding (e.g., "esm2", "physicochemical").
            
        Returns:
            Embedding array.
               
                  
        Raises:
            KeyError: If embedding type not found.
        """
        if embedding_type not in self.embeddings:
            available = self.embedding_types or ["none"]
            raise KeyError(
                f"Embedding type '{embedding_type}' not found. Available types: {available}"
            )
        return self.embeddings[embedding_type]
    
    def set_embedding(self, embedding_type: str, embedding: np.ndarray) -> None:
        """
        Store an embedding.

        Args:
            embedding_type: Type of embedding (e.g., "esm2", "physicochemical").
            embedding: Embedding array.
        """
        self.embeddings[embedding_type] = embedding

    def to_fasta(self) -> str:
        """
        Convert to FASTA format string.

        Returns:
            FASTA formatted string.
        """
        header = f">{self.name}"
        if self.source:
            header += f" source={self.source}"

        # Wrap sequence to 80 chars per line
        wrapped_seq = "\n".join(
            self.sequence[i:i+80] for i in range(0, len(self.sequence), 80)
        )

        return f"{header}\n{wrapped_seq}"
    
    def to_dict(self, include_embeddings: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Args:
            include_embeddings: If True, include embeddings (as lists).

        Returns:
            Dictionary representation.
        """
        data = {
            "sequence": self.sequence,
            "name": self.name,
            "source": self.source,
            "metadata": self.metadata,
            "length": self.length,
            "hash": self.hash,
        }

        if include_embeddings and self.embeddings:
            data["embeddings"] = {
                k: v.tolist() for k, v in self.embeddings.items()
            }

        if self.structure:
            data["structure"] = self.structure

        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Protein:
        """
        Create Protein from dictionary.
        
        Args:
            data: Dictionary with protein data.
            
        Returns:
            Protein instance.
        """
        embeddings = {}
        if "embeddings" in data:
            embeddings = {
                k: np.array(v) for k, v in data["embeddings"].items()
            }

        return cls(
            sequence=data["sequence"],
            name=data.get("name"),
            source=data.get("source"),
            metadata=data.get("metadata", {}),
            embeddings=embeddings,
            structure=data.get("structure"),
        )
    
    def __repr__(self):
        """String representation."""
        seq_preview = self.sequence[:20] + "..." if len(self.sequence) > 20 else self.sequence
        return f"Protein(name='{self.name}', length={self.length}, sequence='{seq_preview}')"
    
    def __hash__(self) -> int:
        """Hash based on sequence."""
        return hash(self.sequence)
    
    def __eq__(self, other: Any) -> bool:
        """Equality based on sequence."""
        if not isinstance(other, Protein):
            return False
        return self.sequence == other.sequence
    
# ====================
# Protein Library (Collection)
# ====================

class ProteinLibrary:
    """
    A collection of proteins with utilities for batch operations.

    This class manages a collection of Protein objects and provides methods for loading, saving, filtering and batch processing.

    Attributes:
        proteins: List of Protein objects.
        name: Optional name for this library.

    Example:
        >>> library = ProteinLibrary(name="de_novo_designs_v1")
        >>> library.add(Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS", name="design_1")
        >>> library.add_from_fasta("designs.fasta")
        >>> print(f"Library contains {len(library)} proteins.")
        >>>
        >>> # Filter by length
        >>> long_proteins = library.filter(lambda p: p.length > 100)
    """
    def __init__(
        self,
        proteins: Optional[Sequence[Protein]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialise protein library.

        Args:
            proteins: Initial list of proteins.
            name: Library name.
        """
        self._proteins: list[Protein] = list(proteins) if proteins else []
        self.name = name
        self._hash_index: dict[str, int] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the hash-to-index mapping."""
        self._hash_index = {p.hash: i for i, p in enumerate(self._proteins)}

    # ====================
    # Collection Operations
    # ====================

    def add(self, protein: Protein) -> None:
        """
        Add a protein to the library.

        Args:
            protein: Protein object to add.

        Note:
            Duplicate sequences (same hash) will be skipped with a warning.
        """
        if protein.hash in self._hash_index:
            # Could use logging here instead - consider later
            return
        
        self._hash_index[protein.hash] = len(self._proteins)
        self._proteins.append(protein)

    def add_many(self, proteins: Sequence[Protein]) -> None:
        """
        Add multiple proteins to the library.

        Args:
            proteins: Proteins to add.

        Returns:
            Number of proteins actually added (excluding duplicates).
        """
        initial_count = len(self.proteins)
        for protein in proteins:
            self.add(protein)
        return len(self._proteins) - initial_count

    def get_by_hash(self, hash_str: str) -> Optional[Protein]:
        """
        Retrieve protein by sequence hash.

        Args:
            hash_str: Sequence hash.

        Returns:
            Protein if found, None otherwise.
        """
        idx = self._hash_index.get(hash_str)
        return self._proteins[idx] if idx is not None else None

    def get_by_name(self, name: str) -> Optional[Protein]:
        """
        Retrieve protein by name.

        Args:
            name: Protein name.

        Returns:
            Protein if found, None otherwise.
        """

        for protein in self._proteins:
            if protein.name == name:
                return protein
        return None

    def filter(self, predicate: callable) -> ProteinLibrary:
        """
        Create a new library with filtered proteins.

        Args:
            predicate: Function that takes a Protein and returns bool

        Returns:
            New ProteinLibrary with matching proteins.
        """
        filtered = [p for p in self._proteins if predicate(p)]
        return ProteinLibrary(proteins=filtered, name=f"{self.name}_filtered")

    def sample(self, n: int, seed: Optional[int] = None) -> ProteinLibrary:
        """
        Randomly sample proteins from the library.

        Args:
            n: Number of proteins to sample.
            seed: Random seed for reproducibility.

        Returns:
            New ProteinLibrary with sampled proteins.
        """
        rng = np.random.default_rng(seed)
        n = min(n, len(self._proteins))
        indices = rng.choice(len(self._proteins), size=n, replace=False)
        sampled = [self._proteins[i] for i in indices]
        return ProteinLibrary(proteins=sampled, name=f"{self.name}_sampled_{n}")

    # ====================
    # I/O Operations
    # ====================

    def add_from_fasta(self, fasta_path: str | Path, source: Optional[str] = None) -> int:
        """
        Load proteins from a FASTA file.

        Args:
            fasta_path: Path to FASTA file.
            source: Source annotation for loaded proteins.

        Returns:
            Number of proteins loaded.
        """

        from Bio import SeqIO

        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        proteins = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            try:
                protein = Protein(
                    sequence=str(record.seq),
                    name=record.id,
                    source=source,
                    metadata={"description": record.description},
                )
                proteins.append(protein)
            except ValueError as e:
                # Skip invalid sequences but log warning
                print(f"Warning: Skipping invalid sequence '{record.id}': {e}")

        return self.add_many(proteins)

    def to_fasta(self, output_path: str | Path) -> None:
        """
        Save library to a FASTA file.

        Args:
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for protein in self._proteins:
                f.write(protein.to_fasta() + "\n")

    def to_json(self, output_path: str | Path, include_embeddings: bool = False) -> None:
        """
        Save library to a JSON file.

        Args:
            output_path: Output file path.
            include_embeddings: Whether to include embeddings in the output.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "name": self.name,
            "count": len(self._proteins),
            "proteins": [p.to_dict(include_embeddings=include_embeddings) for p in self._proteins],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, json_path: str | Path) -> ProteinLibrary:
        """
        Load library from a JSON file.

        Args:
            json_path: Path to JSON file.

        Returns:
            ProteinLibrary instance.
        """
        json_path = Path(json_path)
        
        with open(json_path) as f:
            data = json.load(f)

        proteins = [Protein.from_dict(p) for p in data["proteins"]]
        return cls(proteins=proteins, name=data.get("name"))

    # ====================
    # Batch Operations
    # ====================

    @property
    def sequences(self) -> list[str]:
        """Get all sequences in the library."""
        return [p.sequence for p in self._proteins]

    @property
    def names(self) -> list[str]:
        """Get all names in the library."""
        return [p.name for p in self._proteins]

    @property
    def hashes(self) -> list[str]:
        """Get all hashes in the library."""
        return [p.hash for p in self._proteins]

    def get_embedding_matrix(self, embedding_type: str) -> np.ndarray:
        """
        Get stacked embedding matrix for all proteins.
        
        Args:
            embedding_type: Type of embedding to retrieve.
            
        Returns:
            Array of shape (n_proteins, embedding_dim).
            
        Raises:
            ValueError: If not all proteins have the requested embedding.
        """
        missing = [p.name for p in self._proteins if embedding_type not in p.embeddings]
        if missing:
            raise ValueError(
                f"Embedding '{embedding_type}' missing for {len(missing)} proteins. "
                f"First few: {missing[:5]}"
            )
            
        return np.stack([p.embeddings[embedding_type] for p in self._proteins])

    def summary(self) -> dict[str, Any]:
        """
        Get summary statistics for the library.
        
        Returns:
            Dictionary with summary statistics.
        """
        if not self._proteins:
            return {"count": 0}
        
        lenths = [p.length for p in self._proteins]
        sources = {}
        for p in self._proteins:
            src = p.source or "unknown"
            sources[src] = sources.get(src, 0) + 1
            
        embedding_coverage = {}
        all_embedding_types = set()
        for p in self._proteins:
            all_embedding_types.update(p.embedding_types)
            
        for emb_type in all_embedding_types:
            count = sum(1 for p in self._proteins if emb_type in p.embeddings)
            embedding_coverage[emb_type] = count / len(self._proteins)
            
        return {
            "name": self.name,
            "count": len(self._proteins),
            "length_stats": {
                "min": min(lenths),
                "max": max(lenths),
                "mean": np.mean(lenths),
                "median": np.median(lenths),
            },
            "sources": sources,
            "embedding_coverage": embedding_coverage,
        }
        
    # ====================
    # Dunder Methods
    # ====================

    def __len__(self) -> int:
        """Return number of proteins in the library."""
        return len(self._proteins)
    
    def __iter__(self) -> Iterator[Protein]:
        """Iterate over proteins."""
        return iter(self._proteins)
    
    def __getitem__(self, idx: int | slice) -> Protein | list[Protein]:
        """Index or slice the library."""
        return self._proteins[idx]
    
    def __contains__(self, item: Protein | str) -> bool:
        """Check if a protein or sequence hash is in the library."""
        if isinstance(item, Protein):
            return item.hash in self._hash_index
        return item in self._hash_index
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ProteinLibrary(name='{self.name}', count={len(self._proteins)})"