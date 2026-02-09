"""
ESMFold structure prediction.

This module provides the ESMFoldPredictor class for predicting protein
3D structures using ESMFold, Facebook's end-to-end structure prediction
model based on ESM-2.

Note: ESMFold is computationally intensive and requires significant GPU memory.
For large-scale applications, consider using the ESMFold API or pre-computed
structures from the ESM Atlas.

References:
    Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... & Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637), 1123-1130. https://doi.org/10.1126/science.ade2574
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from protophen.data.protein import Protein, ProteinLibrary
from protophen.utils.logging import logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ESMFoldConfig:
    """Configuration for ESMFold structure prediction."""
    
    # Model settings
    chunk_size: Optional[int] = None  # For long sequences
    
    # Computation
    device: str = "cuda"
    
    # Output options
    return_plddt: bool = True
    return_contacts: bool = False
    
    # Caching
    cache_dir: Optional[str] = None


# =============================================================================
# ESMFold Predictor
# =============================================================================

class ESMFoldPredictor:
    """
    Predict protein structures using ESMFold.
    
    ESMFold is an end-to-end structure prediction model that predicts
    atomic-level protein structures directly from sequence using ESM-2.
    
    Attributes:
        config: Configuration for structure prediction
        model: Loaded ESMFold model (lazy loading)
        
    Example:
        >>> predictor = ESMFoldPredictor()
        >>> 
        >>> # Predict structure
        >>> result = predictor.predict("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        >>> print(result['plddt'].mean())  # Per-residue confidence
        >>> 
        >>> # Save as PDB
        >>> predictor.predict_and_save("MKFLIL...", "structure.pdb")
        >>> 
        >>> # Get structural features for ML
        >>> features = predictor.get_structural_features("MKFLIL...")
    
    Note:
        ESMFold requires significant GPU memory (~8GB for sequences up to 400 AA).
        For longer sequences, use the chunk_size parameter.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        chunk_size: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialise ESMFold predictor.
        
        Args:
            device: Device to use (cuda, cpu). Auto-detected if None.
            chunk_size: Chunk size for long sequences (None = no chunking)
            cache_dir: Directory for caching predicted structures
        """
        self.config = ESMFoldConfig(
            device=device or self._detect_device(),
            chunk_size=chunk_size,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        
        self.model = None
        self._model_loaded = False
        
        logger.info(f"Initialised ESMFoldPredictor on {self.config.device}")
    
    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available device."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load the ESMFold model (lazy loading)."""
        if self._model_loaded:
            return
        
        logger.info("Loading ESMFold model (this may take a moment)...")
        
        try:
            import torch
            import esm
        except ImportError:
            raise ImportError(
                "ESMFold requires 'fair-esm' package. Install with: pip install fair-esm"
            )
        
        # Load ESMFold
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.to(self.config.device)
        self.model.eval()
        
        # Set chunk size if specified
        if self.config.chunk_size is not None:
            self.model.set_chunk_size(self.config.chunk_size)
        
        self._model_loaded = True
        logger.info("ESMFold model loaded successfully")
    
    def predict(
        self,
        sequence: str,
        return_raw: bool = False,
    ) -> Dict[str, Union[str, np.ndarray]]:
        """
        Predict structure for a single sequence.
        
        Args:
            sequence: Amino acid sequence
            return_raw: If True, return raw model output
            
        Returns:
            Dictionary containing:
                - 'pdb': PDB format string
                - 'plddt': Per-residue pLDDT confidence scores
                - 'ptm': Predicted TM-score (overall confidence)
        """
        import torch
        
        self._load_model()
        
        logger.debug(f"Predicting structure for sequence of length {len(sequence)}")
        
        with torch.no_grad():
            output = self.model.infer_pdb(sequence)
        
        # Parse pLDDT from B-factor column
        plddt = self._extract_plddt_from_pdb(output)
        
        # Calculate mean pLDDT
        mean_plddt = np.mean(plddt) if len(plddt) > 0 else 0.0
        
        result = {
            'pdb': output,
            'plddt': plddt,
            'mean_plddt': mean_plddt,
            'sequence_length': len(sequence),
        }
        
        logger.debug(f"Structure predicted, mean pLDDT: {mean_plddt:.2f}")
        
        return result
    
    def _extract_plddt_from_pdb(self, pdb_string: str) -> np.ndarray:
        """Extract pLDDT scores from PDB B-factor column."""
        plddt_scores = []
        seen_residues = set()
        
        for line in pdb_string.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                # Extract residue number and B-factor
                try:
                    res_num = int(line[22:26].strip())
                    b_factor = float(line[60:66].strip())
                    
                    if res_num not in seen_residues:
                        plddt_scores.append(b_factor)
                        seen_residues.add(res_num)
                except (ValueError, IndexError):
                    continue
        
        return np.array(plddt_scores, dtype=np.float32)
    
    def predict_and_save(
        self,
        sequence: str,
        output_path: Union[str, Path],
    ) -> Dict[str, Union[str, np.ndarray]]:
        """
        Predict structure and save to PDB file.
        
        Args:
            sequence: Amino acid sequence
            output_path: Path to save PDB file
            
        Returns:
            Prediction result dictionary
        """
        result = self.predict(sequence)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(result['pdb'])
        
        logger.info(f"Saved structure to {output_path}")
        
        return result
    
    def get_structural_features(self, sequence: str) -> np.ndarray:
        """
        Get structural features suitable for ML.
        
        Returns a feature vector summarising the predicted structure:
        - Mean pLDDT
        - pLDDT standard deviation
        - Fraction of high-confidence residues (pLDDT > 70)
        - Fraction of very high-confidence residues (pLDDT > 90)
        - Fraction of low-confidence residues (pLDDT < 50)
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Feature vector of shape (5,)
        """
        result = self.predict(sequence)
        plddt = result['plddt']
        
        if len(plddt) == 0:
            return np.zeros(5, dtype=np.float32)
        
        features = np.array([
            np.mean(plddt),
            np.std(plddt),
            np.mean(plddt > 70),  # Fraction confident
            np.mean(plddt > 90),  # Fraction very confident
            np.mean(plddt < 50),  # Fraction uncertain
        ], dtype=np.float32)
        
        return features
    
    def predict_for_protein(
        self,
        protein: Protein,
        embedding_key: str = "structure",
        save_pdb: bool = False,
        pdb_dir: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Predict structure for a Protein object.
        
        Args:
            protein: Protein object
            embedding_key: Key to store structural features
            save_pdb: Whether to save PDB file
            pdb_dir: Directory to save PDB files
            
        Returns:
            Structural feature vector
        """
        # Check if already predicted
        if embedding_key in protein.embeddings:
            return protein.embeddings[embedding_key]
        
        # Get structural features
        features = self.get_structural_features(protein.sequence)
        protein.set_embedding(embedding_key, features)
        
        # Optionally save PDB
        if save_pdb and pdb_dir is not None:
            pdb_path = Path(pdb_dir) / f"{protein.name}.pdb"
            self.predict_and_save(protein.sequence, pdb_path)
        
        return features
    
    def predict_for_library(
        self,
        library: ProteinLibrary,
        embedding_key: str = "structure",
        show_progress: bool = True,
        save_pdbs: bool = False,
        pdb_dir: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Predict structures for all proteins in a library.
        
        Args:
            library: ProteinLibrary object
            embedding_key: Key to store structural features
            show_progress: Whether to show progress bar
            save_pdbs: Whether to save PDB files
            pdb_dir: Directory to save PDB files
            
        Returns:
            Feature matrix of shape (n_proteins, 5)
        """
        from tqdm.auto import tqdm
        
        proteins_to_predict = [
            p for p in library if embedding_key not in p.embeddings
        ]
        
        if not proteins_to_predict:
            logger.info("All proteins already have structural features")
            return library.get_embedding_matrix(embedding_key)
        
        logger.info(f"Predicting structures for {len(proteins_to_predict)} proteins")
        
        if save_pdbs and pdb_dir is not None:
            pdb_dir = Path(pdb_dir)
            pdb_dir.mkdir(parents=True, exist_ok=True)
        
        iterator = tqdm(
            proteins_to_predict,
            desc="Predicting structures",
            disable=not show_progress,
        )
        
        for protein in iterator:
            self.predict_for_protein(
                protein,
                embedding_key=embedding_key,
                save_pdb=save_pdbs,
                pdb_dir=pdb_dir,
            )
        
        return library.get_embedding_matrix(embedding_key)
    
    @staticmethod
    def is_available() -> bool:
        """Check if ESMFold is available."""
        try:
            import esm
            return hasattr(esm.pretrained, 'esmfold_v1')
        except ImportError:
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ESMFoldPredictor(device={self.config.device}, loaded={self._model_loaded})"


# =============================================================================
# Convenience Functions
# =============================================================================

def predict_structure(sequence: str, output_path: Optional[str] = None) -> Dict:
    """
    Convenience function to predict structure for a single sequence.
    
    Args:
        sequence: Amino acid sequence
        output_path: Optional path to save PDB file
        
    Returns:
        Prediction result dictionary
    """
    predictor = ESMFoldPredictor()
    
    if output_path:
        return predictor.predict_and_save(sequence, output_path)
    else:
        return predictor.predict(sequence)


def check_esmfold_available() -> bool:
    """Check if ESMFold is available and working."""
    return ESMFoldPredictor.is_available()