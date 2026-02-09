"""
Physicochemical feature extraction from protein sequences.

This module provides the PhysicochemicalCalculator class for computing
various biochemical and biophysical properties from amino acid sequences.

Features include:
- Amino acid composition (20 features)
- Dipeptide composition (400 features)
- Physicochemical properties (MW, pI, charge, etc.)
- Hydrophobicity profiles
- Secondary structure propensities
- Sequence complexity measures

These features complement learned embeddings (ESM-2) with interpretable,
biochemically meaningful descriptors.

References:
    Hydrophobicity scale: Kyte, J. and Doolittle, R.F., 1982. A simple method for displaying the hydropathic character of a protein. Journal of molecular biology, 157(1), pp.105-132. https://doi.org/10.1016/0022-2836(82)90515-0
    Instability index: Ikai, A., 1980. Thermostability and aliphatic index of globular proteins. The Journal of Biochemistry, 88(6), pp.1895-1898. https://doi.org/10.1093/oxfordjournals.jbchem.a133168
    pI calculation: Bjellqvist, B., Hughes, G.J., Pasquali, C., Paquet, N., Ravier, F., Sanchez, J.C., Frutiger, S. and Hochstrasser, D., 1993. The focusing positions of polypeptides in immobilized pH gradients can be predicted from their amino acid sequences. Electrophoresis, 14(1), pp.1023-1031. https://doi.org/10.1002/elps.11501401163
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from scipy import stats

from protophen.data.protein import AMINO_ACIDS, Protein, ProteinLibrary
from protophen.utils.logging import logger


# =============================================================================
# Amino Acid Property Tables
# =============================================================================

# Standard amino acids in alphabetical order
AA_ORDER = tuple(sorted(AMINO_ACIDS))  # ('A', 'C', 'D', 'E', 'F', ...)

# Molecular weights (Da) - monoisotopic
AA_MOLECULAR_WEIGHT: Dict[str, float] = {
    'A': 89.09,  'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
    'E': 147.13, 'Q': 146.15, 'G': 75.07,  'H': 155.16, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
}

# Kyte-Doolittle hydrophobicity scale
AA_HYDROPHOBICITY_KD: Dict[str, float] = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Eisenberg hydrophobicity scale (normalised)
AA_HYDROPHOBICITY_EISENBERG: Dict[str, float] = {
    'A': 0.620,  'R': -2.530, 'N': -0.780, 'D': -0.900, 'C': 0.290,
    'E': -0.740, 'Q': -0.850, 'G': 0.480,  'H': -0.400, 'I': 1.380,
    'L': 1.060,  'K': -1.500, 'M': 0.640,  'F': 1.190,  'P': 0.120,
    'S': -0.180, 'T': -0.050, 'W': 0.810,  'Y': 0.260,  'V': 1.080,
}

# Charge at pH 7.0
AA_CHARGE: Dict[str, float] = {
    'A': 0.0,  'R': 1.0,  'N': 0.0,  'D': -1.0, 'C': 0.0,
    'E': -1.0, 'Q': 0.0,  'G': 0.0,  'H': 0.1,  'I': 0.0,
    'L': 0.0,  'K': 1.0,  'M': 0.0,  'F': 0.0,  'P': 0.0,
    'S': 0.0,  'T': 0.0,  'W': 0.0,  'Y': 0.0,  'V': 0.0,
}

# pKa values for pI calculation
AA_PKA: Dict[str, Dict[str, float]] = {
    'N_terminus': {'pKa': 9.69},
    'C_terminus': {'pKa': 2.34},
    'D': {'pKa': 3.86},
    'E': {'pKa': 4.25},
    'C': {'pKa': 8.33},
    'Y': {'pKa': 10.07},
    'H': {'pKa': 6.00},
    'K': {'pKa': 10.53},
    'R': {'pKa': 12.48},
}

# Instability index weights (Guruprasad et al., 1990)
INSTABILITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    'A': {'A': 1.0, 'C': 44.94, 'E': 1.0, 'D': -7.49, 'G': 1.0, 'F': 1.0, 'I': 1.0, 'H': -7.49, 'K': 1.0, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': 1.0, 'P': 20.26, 'S': 1.0, 'R': 1.0, 'T': 1.0, 'W': 1.0, 'V': 1.0, 'Y': 1.0},
    'C': {'A': 1.0, 'C': 1.0, 'E': 1.0, 'D': 20.26, 'G': 1.0, 'F': 1.0, 'I': 1.0, 'H': 33.60, 'K': 1.0, 'M': 33.60, 'L': 20.26, 'N': 1.0, 'Q': -6.54, 'P': 20.26, 'S': 1.0, 'R': 1.0, 'T': 33.60, 'W': 24.68, 'V': -6.54, 'Y': 1.0},
    'E': {'A': 1.0, 'C': 44.94, 'E': 33.60, 'D': 20.26, 'G': 1.0, 'F': 1.0, 'I': 20.26, 'H': -6.54, 'K': 1.0, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': 20.26, 'P': 20.26, 'S': 20.26, 'R': 1.0, 'T': 1.0, 'W': -14.03, 'V': 1.0, 'Y': 1.0},
    'D': {'A': 1.0, 'C': 1.0, 'E': 1.0, 'D': 1.0, 'G': 1.0, 'F': -6.54, 'I': 1.0, 'H': 1.0, 'K': -7.49, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': 1.0, 'P': 1.0, 'S': 20.26, 'R': -6.54, 'T': -14.03, 'W': 1.0, 'V': 1.0, 'Y': 1.0},
    'G': {'A': -7.49, 'C': 1.0, 'E': -6.54, 'D': 1.0, 'G': 13.34, 'F': 1.0, 'I': -7.49, 'H': 1.0, 'K': -7.49, 'M': 1.0, 'L': 1.0, 'N': -7.49, 'Q': 1.0, 'P': 1.0, 'S': 1.0, 'R': 1.0, 'T': -7.49, 'W': 13.34, 'V': 1.0, 'Y': -7.49},
    'F': {'A': 1.0, 'C': 1.0, 'E': 1.0, 'D': 13.34, 'G': 1.0, 'F': 1.0, 'I': 1.0, 'H': 1.0, 'K': -14.03, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': 1.0, 'P': 20.26, 'S': 1.0, 'R': 1.0, 'T': 1.0, 'W': 1.0, 'V': 1.0, 'Y': 33.60},
    'I': {'A': 1.0, 'C': 1.0, 'E': 44.94, 'D': 1.0, 'G': 1.0, 'F': 1.0, 'I': 1.0, 'H': 13.34, 'K': -7.49, 'M': 1.0, 'L': 20.26, 'N': 1.0, 'Q': 1.0, 'P': -1.88, 'S': 1.0, 'R': 1.0, 'T': 1.0, 'W': 1.0, 'V': -7.49, 'Y': 1.0},
    'H': {'A': 1.0, 'C': 1.0, 'E': 1.0, 'D': 1.0, 'G': -9.37, 'F': -9.37, 'I': 44.94, 'H': 1.0, 'K': 24.68, 'M': 1.0, 'L': 1.0, 'N': 24.68, 'Q': 1.0, 'P': -1.88, 'S': 1.0, 'R': 1.0, 'T': -6.54, 'W': -1.88, 'V': 1.0, 'Y': 44.94},
    'K': {'A': 1.0, 'C': 1.0, 'E': 1.0, 'D': 1.0, 'G': -7.49, 'F': 1.0, 'I': -7.49, 'H': 1.0, 'K': 1.0, 'M': 33.60, 'L': -7.49, 'N': 1.0, 'Q': 24.68, 'P': -6.54, 'S': 1.0, 'R': 33.60, 'T': 1.0, 'W': 1.0, 'V': -7.49, 'Y': 1.0},
    'M': {'A': 13.34, 'C': 1.0, 'E': 1.0, 'D': 1.0, 'G': 1.0, 'F': 1.0, 'I': 1.0, 'H': 58.28, 'K': 1.0, 'M': -1.88, 'L': 1.0, 'N': 1.0, 'Q': -6.54, 'P': 44.94, 'S': 44.94, 'R': -6.54, 'T': -1.88, 'W': 1.0, 'V': 1.0, 'Y': 24.68},
    'L': {'A': 1.0, 'C': 1.0, 'E': 1.0, 'D': 1.0, 'G': 1.0, 'F': 1.0, 'I': 1.0, 'H': 1.0, 'K': -7.49, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': 33.60, 'P': 20.26, 'S': 1.0, 'R': 20.26, 'T': 1.0, 'W': 24.68, 'V': 1.0, 'Y': 1.0},
    'N': {'A': 1.0, 'C': -1.88, 'E': 1.0, 'D': 1.0, 'G': -14.03, 'F': -14.03, 'I': 44.94, 'H': 1.0, 'K': 24.68, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': -6.54, 'P': -1.88, 'S': 1.0, 'R': 1.0, 'T': -7.49, 'W': -9.37, 'V': 1.0, 'Y': 1.0},
    'Q': {'A': 1.0, 'C': -6.54, 'E': 20.26, 'D': 20.26, 'G': 1.0, 'F': -6.54, 'I': 1.0, 'H': 1.0, 'K': 1.0, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': 20.26, 'P': 20.26, 'S': 44.94, 'R': 1.0, 'T': 1.0, 'W': 1.0, 'V': -6.54, 'Y': -6.54},
    'P': {'A': 20.26, 'C': -6.54, 'E': 18.38, 'D': -6.54, 'G': 1.0, 'F': 20.26, 'I': 1.0, 'H': 1.0, 'K': 1.0, 'M': -6.54, 'L': 1.0, 'N': 1.0, 'Q': 20.26, 'P': 20.26, 'S': 20.26, 'R': -6.54, 'T': 1.0, 'W': -1.88, 'V': 20.26, 'Y': 1.0},
    'S': {'A': 1.0, 'C': 33.60, 'E': 20.26, 'D': 1.0, 'G': 1.0, 'F': 1.0, 'I': 1.0, 'H': 1.0, 'K': 1.0, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': 20.26, 'P': 44.94, 'S': 20.26, 'R': 20.26, 'T': 1.0, 'W': 1.0, 'V': 1.0, 'Y': 1.0},
    'R': {'A': 1.0, 'C': 1.0, 'E': 1.0, 'D': 1.0, 'G': -7.49, 'F': 1.0, 'I': 1.0, 'H': 20.26, 'K': 1.0, 'M': 1.0, 'L': 1.0, 'N': 13.34, 'Q': 20.26, 'P': 20.26, 'S': 44.94, 'R': 58.28, 'T': 1.0, 'W': 58.28, 'V': 1.0, 'Y': -6.54},
    'T': {'A': 1.0, 'C': 1.0, 'E': 20.26, 'D': 1.0, 'G': -7.49, 'F': 13.34, 'I': 1.0, 'H': 1.0, 'K': 1.0, 'M': 1.0, 'L': 1.0, 'N': -14.03, 'Q': -6.54, 'P': 1.0, 'S': 1.0, 'R': 1.0, 'T': 1.0, 'W': -14.03, 'V': 1.0, 'Y': 1.0},
    'W': {'A': -14.03, 'C': 1.0, 'E': 1.0, 'D': 1.0, 'G': -9.37, 'F': 1.0, 'I': 1.0, 'H': 24.68, 'K': 1.0, 'M': 24.68, 'L': 13.34, 'N': 13.34, 'Q': 1.0, 'P': 1.0, 'S': 1.0, 'R': 1.0, 'T': -14.03, 'W': 1.0, 'V': -7.49, 'Y': 1.0},
    'V': {'A': 1.0, 'C': 1.0, 'E': 1.0, 'D': -14.03, 'G': -7.49, 'F': 1.0, 'I': 1.0, 'H': 1.0, 'K': -1.88, 'M': 1.0, 'L': 1.0, 'N': 1.0, 'Q': 1.0, 'P': 20.26, 'S': 1.0, 'R': 1.0, 'T': -7.49, 'W': 1.0, 'V': 1.0, 'Y': -6.54},
    'Y': {'A': 24.68, 'C': 1.0, 'E': -6.54, 'D': 24.68, 'G': -7.49, 'F': 1.0, 'I': 1.0, 'H': 13.34, 'K': 1.0, 'M': 44.94, 'L': 1.0, 'N': 1.0, 'Q': 1.0, 'P': 13.34, 'S': 1.0, 'R': -15.91, 'T': -7.49, 'W': -9.37, 'V': 1.0, 'Y': 13.34},
}

# Secondary structure propensities (Chou-Fasman)
AA_HELIX_PROPENSITY: Dict[str, float] = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'E': 1.51, 'Q': 1.11, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06,
}

AA_SHEET_PROPENSITY: Dict[str, float] = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'E': 0.37, 'Q': 1.10, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
}

AA_TURN_PROPENSITY: Dict[str, float] = {
    'A': 0.66, 'R': 0.95, 'N': 1.56, 'D': 1.46, 'C': 1.19,
    'E': 0.74, 'Q': 0.98, 'G': 1.56, 'H': 0.95, 'I': 0.47,
    'L': 0.59, 'K': 1.01, 'M': 0.60, 'F': 0.60, 'P': 1.52,
    'S': 1.43, 'T': 0.96, 'W': 0.96, 'Y': 1.14, 'V': 0.50,
}

# Volume (Å³)
AA_VOLUME: Dict[str, float] = {
    'A': 88.6,  'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'E': 138.4, 'Q': 143.8, 'G': 60.1,  'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0,  'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
}

# Polarity (Grantham)
AA_POLARITY: Dict[str, float] = {
    'A': 8.1,  'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
    'E': 12.3, 'Q': 10.5, 'G': 9.0,  'H': 10.4, 'I': 5.2,
    'L': 4.9,  'K': 11.3, 'M': 5.7,  'F': 5.2,  'P': 8.0,
    'S': 9.2,  'T': 8.6,  'W': 5.4,  'Y': 6.2,  'V': 5.9,
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PhysicochemicalConfig:
    """Configuration for physicochemical feature calculation."""
    
    # Which feature groups to compute
    include_aa_composition: bool = True
    include_dipeptide_composition: bool = True
    include_global_properties: bool = True
    include_hydrophobicity: bool = True
    include_secondary_structure: bool = True
    include_charge_features: bool = True
    include_sequence_complexity: bool = True
    
    # Hydrophobicity scale to use
    hydrophobicity_scale: Literal['kyte_doolittle', 'eisenberg'] = 'kyte_doolittle'
    
    # Window size for local features (if applicable)
    window_size: int = 7
    
    # Whether to normalise features
    normalise: bool = True


# =============================================================================
# Feature Calculation Functions
# =============================================================================

def calculate_aa_composition(sequence: str) -> np.ndarray:
    """
    Calculate amino acid composition (frequency of each AA).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Array of shape (20,) with frequencies for each amino acid
    """
    length = len(sequence)
    counts = np.zeros(20)
    
    for aa in sequence:
        if aa in AA_ORDER:
            idx = AA_ORDER.index(aa)
            counts[idx] += 1
    
    return counts / length if length > 0 else counts


def calculate_dipeptide_composition(sequence: str) -> np.ndarray:
    """
    Calculate dipeptide composition (frequency of each AA pair).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Array of shape (400,) with frequencies for each dipeptide
    """
    if len(sequence) < 2:
        return np.zeros(400)
    
    counts = np.zeros(400)
    n_dipeptides = len(sequence) - 1
    
    for i in range(n_dipeptides):
        aa1, aa2 = sequence[i], sequence[i + 1]
        if aa1 in AA_ORDER and aa2 in AA_ORDER:
            idx1 = AA_ORDER.index(aa1)
            idx2 = AA_ORDER.index(aa2)
            counts[idx1 * 20 + idx2] += 1
    
    return counts / n_dipeptides if n_dipeptides > 0 else counts


def calculate_molecular_weight(sequence: str) -> float:
    """
    Calculate molecular weight of a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Molecular weight in Daltons
    """
    water_mass = 18.015
    total = sum(AA_MOLECULAR_WEIGHT.get(aa, 110.0) for aa in sequence)
    # Subtract water for each peptide bond
    return total - (len(sequence) - 1) * water_mass


def calculate_charge_at_ph(sequence: str, ph: float = 7.0) -> float:
    """
    Calculate net charge of a protein at given pH.
    
    Uses Henderson-Hasselbalch equation.
    
    Args:
        sequence: Amino acid sequence
        ph: pH value (default 7.0)
        
    Returns:
        Net charge
    """
    charge = 0.0
    
    # N-terminus (positive)
    pka = AA_PKA['N_terminus']['pKa']
    charge += 1.0 / (1.0 + 10 ** (ph - pka))
    
    # C-terminus (negative)
    pka = AA_PKA['C_terminus']['pKa']
    charge -= 1.0 / (1.0 + 10 ** (pka - ph))
    
    # Acidic residues (D, E, C, Y) - negative when deprotonated
    for aa in ['D', 'E', 'C', 'Y']:
        count = sequence.count(aa)
        if count > 0:
            pka = AA_PKA[aa]['pKa']
            charge -= count / (1.0 + 10 ** (pka - ph))
    
    # Basic residues (H, K, R) - positive when protonated
    for aa in ['H', 'K', 'R']:
        count = sequence.count(aa)
        if count > 0:
            pka = AA_PKA[aa]['pKa']
            charge += count / (1.0 + 10 ** (ph - pka))
    
    return charge


def calculate_isoelectric_point(sequence: str, precision: float = 0.01) -> float:
    """
    Calculate isoelectric point (pI) using binary search.
    
    Args:
        sequence: Amino acid sequence
        precision: Desired precision for pI
        
    Returns:
        Isoelectric point (pH where net charge is 0)
    """
    ph_low, ph_high = 0.0, 14.0
    
    while (ph_high - ph_low) > precision:
        ph_mid = (ph_low + ph_high) / 2
        charge = calculate_charge_at_ph(sequence, ph_mid)
        
        if charge > 0:
            ph_low = ph_mid
        else:
            ph_high = ph_mid
    
    return (ph_low + ph_high) / 2


def calculate_instability_index(sequence: str) -> float:
    """
    Calculate instability index (Guruprasad et al., 1990).
    
    Proteins with instability index < 40 are considered stable.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Instability index value
    """
    if len(sequence) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(sequence) - 1):
        aa1, aa2 = sequence[i], sequence[i + 1]
        if aa1 in INSTABILITY_WEIGHTS and aa2 in INSTABILITY_WEIGHTS[aa1]:
            total += INSTABILITY_WEIGHTS[aa1][aa2]
    
    return (10.0 / len(sequence)) * total


def calculate_gravy(sequence: str) -> float:
    """
    Calculate GRAVY (Grand Average of Hydropathicity).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        GRAVY score (mean Kyte-Doolittle hydrophobicity)
    """
    if len(sequence) == 0:
        return 0.0
    
    total = sum(AA_HYDROPHOBICITY_KD.get(aa, 0.0) for aa in sequence)
    return total / len(sequence)


def calculate_aromaticity(sequence: str) -> float:
    """
    Calculate aromaticity (fraction of aromatic residues: F, W, Y).
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Fraction of aromatic residues
    """
    if len(sequence) == 0:
        return 0.0
    
    aromatic_count = sum(1 for aa in sequence if aa in 'FWY')
    return aromatic_count / len(sequence)


def calculate_aliphatic_index(sequence: str) -> float:
    """
    Calculate aliphatic index (Ikai, 1980).
    
    Measure of thermostability based on aliphatic side chains.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Aliphatic index
    """
    if len(sequence) == 0:
        return 0.0
    
    length = len(sequence)
    a = sequence.count('A') / length
    v = sequence.count('V') / length
    i = sequence.count('I') / length
    l = sequence.count('L') / length
    
    return 100 * (a + 2.9 * v + 3.9 * (i + l))


def calculate_secondary_structure_fractions(sequence: str) -> Dict[str, float]:
    """
    Calculate predicted secondary structure fractions using Chou-Fasman propensities.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with helix, sheet, turn fractions
    """
    if len(sequence) == 0:
        return {'helix': 0.0, 'sheet': 0.0, 'turn': 0.0}
    
    helix_sum = sum(AA_HELIX_PROPENSITY.get(aa, 1.0) for aa in sequence)
    sheet_sum = sum(AA_SHEET_PROPENSITY.get(aa, 1.0) for aa in sequence)
    turn_sum = sum(AA_TURN_PROPENSITY.get(aa, 1.0) for aa in sequence)
    
    total = helix_sum + sheet_sum + turn_sum
    
    return {
        'helix': helix_sum / total if total > 0 else 0.0,
        'sheet': sheet_sum / total if total > 0 else 0.0,
        'turn': turn_sum / total if total > 0 else 0.0,
    }


def calculate_sequence_entropy(sequence: str) -> float:
    """
    Calculate Shannon entropy of amino acid sequence.
    
    Higher entropy = more diverse sequence composition.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Shannon entropy (bits)
    """
    if len(sequence) == 0:
        return 0.0
    
    # Calculate amino acid frequencies
    aa_counts = {}
    for aa in sequence:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    # Calculate entropy
    length = len(sequence)
    entropy = 0.0
    for count in aa_counts.values():
        p = count / length
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calculate_sequence_complexity(sequence: str, window_size: int = 12) -> float:
    """
    Calculate sequence complexity using Wootton-Federhen algorithm.
    
    Low complexity regions have biased amino acid composition.
    
    Args:
        sequence: Amino acid sequence
        window_size: Window size for complexity calculation
        
    Returns:
        Mean complexity score (0-1, higher = more complex)
    """
    if len(sequence) < window_size:
        return calculate_sequence_entropy(sequence) / np.log2(20)  # Normalise
    
    complexities = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        entropy = calculate_sequence_entropy(window)
        # Normalise by maximum possible entropy
        max_entropy = np.log2(min(window_size, 20))
        complexities.append(entropy / max_entropy if max_entropy > 0 else 0)
    
    return np.mean(complexities)


def calculate_hydrophobic_moment(
    sequence: str,
    angle: float = 100.0,
    window_size: int = 11,
) -> float:
    """
    Calculate maximum hydrophobic moment (Eisenberg et al., 1982).
    
    Measure of amphiphilicity - high values suggest membrane-associating helices.
    
    Args:
        sequence: Amino acid sequence
        angle: Angle between residues (100° for alpha-helix)
        window_size: Window size
        
    Returns:
        Maximum hydrophobic moment
    """
    if len(sequence) < window_size:
        return 0.0
    
    angle_rad = np.radians(angle)
    max_moment = 0.0
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        
        sum_cos = 0.0
        sum_sin = 0.0
        
        for j, aa in enumerate(window):
            h = AA_HYDROPHOBICITY_EISENBERG.get(aa, 0.0)
            sum_cos += h * np.cos(j * angle_rad)
            sum_sin += h * np.sin(j * angle_rad)
        
        moment = np.sqrt(sum_cos**2 + sum_sin**2) / window_size
        max_moment = max(max_moment, moment)
    
    return max_moment


# =============================================================================
# Main Calculator Class
# =============================================================================

class PhysicochemicalCalculator:
    """
    Calculate physicochemical features from protein sequences.
    
    This class computes various biochemical and biophysical properties
    from amino acid sequences, producing a feature vector that complements
    learned embeddings like ESM-2.
    
    Attributes:
        config: Configuration for feature calculation
        feature_names: List of feature names in output order
        
    Example:
        >>> calc = PhysicochemicalCalculator()
        >>> 
        >>> # Single sequence
        >>> features = calc.calculate("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        >>> print(features.shape)  # (n_features,)
        >>> 
        >>> # With Protein object
        >>> protein = Protein(sequence="MKFLIL...")
        >>> calc.calculate_for_protein(protein)
        >>> print(protein.embeddings["physicochemical"].shape)
        >>> 
        >>> # Batch processing
        >>> sequences = ["MKFLIL...", "ACDEFG...", "GHIKLM..."]
        >>> features = calc.calculate_batch(sequences)
        >>> print(features.shape)  # (3, n_features)
    """
    
    def __init__(self, config: Optional[PhysicochemicalConfig] = None):
        """
        Initialise the physicochemical calculator.
        
        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or PhysicochemicalConfig()
        self._feature_names: Optional[List[str]] = None
        
        logger.info(f"Initialised PhysicochemicalCalculator")
        logger.debug(f"Config: {self.config}")
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names in output order."""
        if self._feature_names is None:
            self._feature_names = self._build_feature_names()
        return self._feature_names
    
    @property
    def n_features(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)
    
    def _build_feature_names(self) -> List[str]:
        """Build list of feature names based on configuration."""
        names = []
        
        if self.config.include_aa_composition:
            names.extend([f"aa_comp_{aa}" for aa in AA_ORDER])
        
        if self.config.include_dipeptide_composition:
            names.extend([f"dipep_{aa1}{aa2}" for aa1 in AA_ORDER for aa2 in AA_ORDER])
        
        if self.config.include_global_properties:
            names.extend([
                'molecular_weight',
                'length',
                'instability_index',
                'aliphatic_index',
                'aromaticity',
            ])
        
        if self.config.include_hydrophobicity:
            names.extend([
                'gravy',
                'hydrophobic_moment',
            ])
        
        if self.config.include_secondary_structure:
            names.extend([
                'helix_fraction',
                'sheet_fraction',
                'turn_fraction',
            ])
        
        if self.config.include_charge_features:
            names.extend([
                'isoelectric_point',
                'charge_at_ph7',
                'positive_residue_fraction',
                'negative_residue_fraction',
            ])
        
        if self.config.include_sequence_complexity:
            names.extend([
                'sequence_entropy',
                'sequence_complexity',
            ])
        
        return names
    
    def calculate(self, sequence: str) -> np.ndarray:
        """
        Calculate all physicochemical features for a sequence.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Amino acid composition (20 features)
        if self.config.include_aa_composition:
            features.extend(calculate_aa_composition(sequence))
        
        # Dipeptide composition (400 features)
        if self.config.include_dipeptide_composition:
            features.extend(calculate_dipeptide_composition(sequence))
        
        # Global properties
        if self.config.include_global_properties:
            features.append(calculate_molecular_weight(sequence))
            features.append(len(sequence))
            features.append(calculate_instability_index(sequence))
            features.append(calculate_aliphatic_index(sequence))
            features.append(calculate_aromaticity(sequence))
        
        # Hydrophobicity
        if self.config.include_hydrophobicity:
            features.append(calculate_gravy(sequence))
            features.append(calculate_hydrophobic_moment(sequence))
        
        # Secondary structure
        if self.config.include_secondary_structure:
            ss_fractions = calculate_secondary_structure_fractions(sequence)
            features.append(ss_fractions['helix'])
            features.append(ss_fractions['sheet'])
            features.append(ss_fractions['turn'])
        
        # Charge features
        if self.config.include_charge_features:
            features.append(calculate_isoelectric_point(sequence))
            features.append(calculate_charge_at_ph(sequence, 7.0))
            # Fraction of positive residues (K, R, H)
            pos_frac = sum(1 for aa in sequence if aa in 'KRH') / len(sequence) if sequence else 0
            features.append(pos_frac)
            # Fraction of negative residues (D, E)
            neg_frac = sum(1 for aa in sequence if aa in 'DE') / len(sequence) if sequence else 0
            features.append(neg_frac)
        
        # Sequence complexity
        if self.config.include_sequence_complexity:
            features.append(calculate_sequence_entropy(sequence))
            features.append(calculate_sequence_complexity(sequence))
        
        return np.array(features, dtype=np.float32)
    
    def calculate_batch(
        self,
        sequences: Sequence[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Calculate features for multiple sequences.
        
        Args:
            sequences: List of amino acid sequences
            show_progress: Whether to show progress bar
            
        Returns:
            Feature matrix of shape (n_sequences, n_features)
        """
        from tqdm.auto import tqdm
        
        iterator = tqdm(sequences, desc="Calculating features", disable=not show_progress)
        features = [self.calculate(seq) for seq in iterator]
        
        return np.stack(features)
    
    def calculate_for_protein(
        self,
        protein: Protein,
        embedding_key: str = "physicochemical",
    ) -> np.ndarray:
        """
        Calculate features for a Protein object and store them.
        
        Args:
            protein: Protein object
            embedding_key: Key to store features under in protein.embeddings
            
        Returns:
            Feature vector
        """
        # Check if already calculated
        if embedding_key in protein.embeddings:
            return protein.embeddings[embedding_key]
        
        features = self.calculate(protein.sequence)
        protein.set_embedding(embedding_key, features)
        
        return features
    
    def calculate_for_library(
        self,
        library: ProteinLibrary,
        embedding_key: str = "physicochemical",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Calculate features for all proteins in a library.
        
        Args:
            library: ProteinLibrary object
            embedding_key: Key to store features under
            show_progress: Whether to show progress bar
            
        Returns:
            Feature matrix of shape (n_proteins, n_features)
        """
        # Get sequences for proteins that need calculation
        sequences = []
        proteins_to_calculate = []
        
        for protein in library:
            if embedding_key not in protein.embeddings:
                sequences.append(protein.sequence)
                proteins_to_calculate.append(protein)
        
        if not sequences:
            logger.info("All proteins already have physicochemical features")
            return library.get_embedding_matrix(embedding_key)
        
        logger.info(f"Calculating features for {len(sequences)} proteins")
        
        # Calculate features
        features = self.calculate_batch(sequences, show_progress=show_progress)
        
        # Store in proteins
        for protein, feat_vec in zip(proteins_to_calculate, features):
            protein.set_embedding(embedding_key, feat_vec)
        
        return library.get_embedding_matrix(embedding_key)
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {}
        
        for name in self.feature_names:
            if name.startswith('aa_comp_'):
                aa = name.split('_')[-1]
                descriptions[name] = f"Frequency of amino acid {aa}"
            elif name.startswith('dipep_'):
                dipep = name.split('_')[-1]
                descriptions[name] = f"Frequency of dipeptide {dipep}"
            elif name == 'molecular_weight':
                descriptions[name] = "Molecular weight in Daltons"
            elif name == 'length':
                descriptions[name] = "Sequence length"
            elif name == 'instability_index':
                descriptions[name] = "Instability index (<40 = stable)"
            elif name == 'aliphatic_index':
                descriptions[name] = "Aliphatic index (thermostability)"
            elif name == 'aromaticity':
                descriptions[name] = "Fraction of aromatic residues (F, W, Y)"
            elif name == 'gravy':
                descriptions[name] = "Grand average of hydropathicity"
            elif name == 'hydrophobic_moment':
                descriptions[name] = "Maximum hydrophobic moment (amphiphilicity)"
            elif name == 'helix_fraction':
                descriptions[name] = "Predicted alpha-helix fraction"
            elif name == 'sheet_fraction':
                descriptions[name] = "Predicted beta-sheet fraction"
            elif name == 'turn_fraction':
                descriptions[name] = "Predicted turn fraction"
            elif name == 'isoelectric_point':
                descriptions[name] = "Isoelectric point (pI)"
            elif name == 'charge_at_ph7':
                descriptions[name] = "Net charge at pH 7.0"
            elif name == 'positive_residue_fraction':
                descriptions[name] = "Fraction of positive residues (K, R, H)"
            elif name == 'negative_residue_fraction':
                descriptions[name] = "Fraction of negative residues (D, E)"
            elif name == 'sequence_entropy':
                descriptions[name] = "Shannon entropy of sequence"
            elif name == 'sequence_complexity':
                descriptions[name] = "Sequence complexity score"
            else:
                descriptions[name] = "Unknown feature"
        
        return descriptions
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PhysicochemicalCalculator(n_features={self.n_features})"


# =============================================================================
# Convenience Functions
# =============================================================================

def get_physicochemical_calculator(
    include_dipeptides: bool = True,
    **kwargs,
) -> PhysicochemicalCalculator:
    """
    Get a physicochemical calculator with common settings.
    
    Args:
        include_dipeptides: Whether to include dipeptide composition (400 features)
        **kwargs: Additional config options
        
    Returns:
        PhysicochemicalCalculator instance
    """
    config = PhysicochemicalConfig(
        include_dipeptide_composition=include_dipeptides,
        **kwargs,
    )
    return PhysicochemicalCalculator(config)


def calculate_all_features(sequence: str) -> Dict[str, float]:
    """
    Calculate all physicochemical features and return as dictionary.
    
    Convenience function for exploratory analysis.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary mapping feature names to values
    """
    calc = PhysicochemicalCalculator()
    features = calc.calculate(sequence)
    return dict(zip(calc.feature_names, features))