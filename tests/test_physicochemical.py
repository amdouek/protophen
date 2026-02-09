"""
Tests for physicochemical feature extraction.
"""

import numpy as np
import pytest

from protophen.data.protein import Protein, ProteinLibrary
from protophen.embeddings.physicochemical import (
    PhysicochemicalCalculator,
    PhysicochemicalConfig,
    calculate_aa_composition,
    calculate_dipeptide_composition,
    calculate_molecular_weight,
    calculate_isoelectric_point,
    calculate_gravy,
    calculate_instability_index,
    calculate_aromaticity,
    calculate_aliphatic_index,
    calculate_charge_at_ph,
    calculate_sequence_entropy,
    calculate_sequence_complexity,
    calculate_hydrophobic_moment,
    calculate_secondary_structure_fractions,
    calculate_all_features,
    get_physicochemical_calculator,
)


class TestAminoAcidComposition:
    """Tests for amino acid composition calculation."""
    
    def test_single_aa(self):
        """Test composition of single amino acid repeated."""
        comp = calculate_aa_composition("AAAAA")
        
        # A should be 100%
        assert comp[0] == pytest.approx(1.0)  # A is first in sorted order
        
        # All others should be 0
        assert sum(comp) == pytest.approx(1.0)
    
    def test_equal_distribution(self):
        """Test equal distribution of amino acids."""
        # Sequence with 2 each of A, C, D, E
        sequence = "AACCDDEE"
        comp = calculate_aa_composition(sequence)
        
        # Each should be 0.25
        assert comp[0] == pytest.approx(0.25)  # A
        assert comp[1] == pytest.approx(0.25)  # C
        assert comp[2] == pytest.approx(0.25)  # D
        assert comp[3] == pytest.approx(0.25)  # E
    
    def test_output_shape(self):
        """Test output shape is (20,)."""
        comp = calculate_aa_composition("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert comp.shape == (20,)
    
    def test_sums_to_one(self):
        """Test that composition sums to 1."""
        comp = calculate_aa_composition("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert sum(comp) == pytest.approx(1.0)


class TestDipeptideComposition:
    """Tests for dipeptide composition calculation."""
    
    def test_output_shape(self):
        """Test output shape is (400,)."""
        comp = calculate_dipeptide_composition("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert comp.shape == (400,)
    
    def test_sums_to_one(self):
        """Test that dipeptide composition sums to 1."""
        comp = calculate_dipeptide_composition("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert sum(comp) == pytest.approx(1.0)
    
    def test_single_dipeptide(self):
        """Test sequence with single dipeptide type."""
        comp = calculate_dipeptide_composition("AAA")
        # Should have 2 AA dipeptides out of 2 total
        assert max(comp) == pytest.approx(1.0)
    
    def test_short_sequence(self):
        """Test single amino acid returns zeros."""
        comp = calculate_dipeptide_composition("A")
        assert sum(comp) == pytest.approx(0.0)


class TestMolecularWeight:
    """Tests for molecular weight calculation."""
    
    def test_known_weight(self):
        """Test with a sequence of known approximate weight."""
        # Glycine (G) has MW ~75.07, minus water for peptide bonds
        # 5 glycines: 5 * 75.07 - 4 * 18.015 = 375.35 - 72.06 = 303.29
        mw = calculate_molecular_weight("GGGGG")
        assert 300 < mw < 310
    
    def test_increases_with_length(self):
        """Test that MW increases with length."""
        mw1 = calculate_molecular_weight("GGG")
        mw2 = calculate_molecular_weight("GGGGG")
        mw3 = calculate_molecular_weight("GGGGGGGGGG")
        
        assert mw1 < mw2 < mw3
    
    def test_positive(self):
        """Test that MW is always positive."""
        mw = calculate_molecular_weight("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert mw > 0


class TestIsoelectricPoint:
    """Tests for isoelectric point calculation."""
    
    def test_acidic_protein(self):
        """Test that acidic protein has low pI."""
        # Sequence with many D and E (acidic)
        pI = calculate_isoelectric_point("DDDDEEEE")
        assert pI < 5.0
    
    def test_basic_protein(self):
        """Test that basic protein has high pI."""
        # Sequence with many K and R (basic)
        pI = calculate_isoelectric_point("KKKKRRRR")
        assert pI > 9.0
    
    def test_neutral_range(self):
        """Test that pI is in valid range."""
        pI = calculate_isoelectric_point("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert 0 < pI < 14


class TestCharge:
    """Tests for charge calculation."""
    
    def test_positive_charge(self):
        """Test that basic residues give positive charge."""
        charge = calculate_charge_at_ph("KKKRRR", ph=7.0)
        assert charge > 0
    
    def test_negative_charge(self):
        """Test that acidic residues give negative charge."""
        charge = calculate_charge_at_ph("DDDEE", ph=7.0)
        assert charge < 0
    
    def test_ph_dependence(self):
        """Test that charge changes with pH."""
        seq = "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
        charge_low = calculate_charge_at_ph(seq, ph=2.0)
        charge_high = calculate_charge_at_ph(seq, ph=12.0)
        
        # Should be more positive at low pH
        assert charge_low > charge_high


class TestGRAVY:
    """Tests for GRAVY calculation."""
    
    def test_hydrophobic_sequence(self):
        """Test that hydrophobic sequence has positive GRAVY."""
        # I, L, V are hydrophobic
        gravy = calculate_gravy("IIILLLLVVV")
        assert gravy > 0
    
    def test_hydrophilic_sequence(self):
        """Test that hydrophilic sequence has negative GRAVY."""
        # D, E, K, R are hydrophilic
        gravy = calculate_gravy("DDEEKKRR")
        assert gravy < 0


class TestInstabilityIndex:
    """Tests for instability index calculation."""
    
    def test_returns_float(self):
        """Test that instability index returns a float."""
        ii = calculate_instability_index("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert isinstance(ii, float)
    
    def test_short_sequence(self):
        """Test single amino acid returns 0."""
        ii = calculate_instability_index("M")
        assert ii == 0.0


class TestAromaticity:
    """Tests for aromaticity calculation."""
    
    def test_aromatic_sequence(self):
        """Test sequence with all aromatic residues."""
        arom = calculate_aromaticity("FFFWWWYYY")
        assert arom == pytest.approx(1.0)
    
    def test_non_aromatic_sequence(self):
        """Test sequence with no aromatic residues."""
        arom = calculate_aromaticity("AAAGGGIII")
        assert arom == pytest.approx(0.0)
    
    def test_mixed_sequence(self):
        """Test sequence with some aromatic residues."""
        # 2 aromatic (F, W) out of 6
        arom = calculate_aromaticity("AAFWGG")
        assert arom == pytest.approx(2/6)


class TestAliphaticIndex:
    """Tests for aliphatic index calculation."""
    
    def test_aliphatic_sequence(self):
        """Test that aliphatic sequence has high aliphatic index."""
        # A, V, I, L are aliphatic
        ai = calculate_aliphatic_index("AAAAVVVVIIIILLLL")
        assert ai > 100
    
    def test_returns_float(self):
        """Test that aliphatic index returns a float."""
        ai = calculate_aliphatic_index("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert isinstance(ai, float)


class TestSequenceEntropy:
    """Tests for sequence entropy calculation."""
    
    def test_low_entropy(self):
        """Test that repetitive sequence has low entropy."""
        entropy = calculate_sequence_entropy("AAAAAAAAAA")
        assert entropy == pytest.approx(0.0)
    
    def test_high_entropy(self):
        """Test that diverse sequence has higher entropy."""
        # All different amino acids
        entropy = calculate_sequence_entropy("ACDEFGHIKLMNPQRSTVWY")
        # Max entropy for 20 symbols = log2(20) ≈ 4.32
        assert entropy > 4.0
    
    def test_entropy_increases_with_diversity(self):
        """Test that entropy increases with diversity."""
        entropy_low = calculate_sequence_entropy("AAAA")
        entropy_high = calculate_sequence_entropy("ACDE")
        
        assert entropy_high > entropy_low


class TestSequenceComplexity:
    """Tests for sequence complexity calculation."""
    
    def test_low_complexity(self):
        """Test that repetitive sequence has low complexity."""
        complexity = calculate_sequence_complexity("AAAAAAAAAAAAAAAA")
        assert complexity < 0.3
    
    def test_higher_complexity(self):
        """Test that diverse sequence has higher complexity."""
        complexity = calculate_sequence_complexity("ACDEFGHIKLMNPQRSTVWY")
        assert complexity > 0.5


class TestSecondaryStructure:
    """Tests for secondary structure propensity calculation."""
    
    def test_returns_dict(self):
        """Test that function returns dictionary."""
        ss = calculate_secondary_structure_fractions("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        assert 'helix' in ss
        assert 'sheet' in ss
        assert 'turn' in ss
    
    def test_sums_to_one(self):
        """Test that fractions sum to 1."""
        ss = calculate_secondary_structure_fractions("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        total = ss['helix'] + ss['sheet'] + ss['turn']
        assert total == pytest.approx(1.0)
    
    def test_helix_former(self):
        """Test that helix-forming sequence has high helix fraction."""
        # A, E, L, M are strong helix formers
        ss = calculate_secondary_structure_fractions("AAAEEEELLLMMM")
        assert ss['helix'] > ss['sheet']


class TestHydrophobicMoment:
    """Tests for hydrophobic moment calculation."""
    
    def test_returns_float(self):
        """Test that function returns float."""
        hm = calculate_hydrophobic_moment("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert isinstance(hm, float)
    
    def test_non_negative(self):
        """Test that hydrophobic moment is non-negative."""
        hm = calculate_hydrophobic_moment("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        assert hm >= 0
    
    def test_short_sequence(self):
        """Test that short sequence returns 0."""
        hm = calculate_hydrophobic_moment("MKF")
        assert hm == 0.0


class TestPhysicochemicalCalculator:
    """Tests for PhysicochemicalCalculator class."""
    
    def test_default_config(self):
        """Test calculator with default configuration."""
        calc = PhysicochemicalCalculator()
        
        features = calc.calculate("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        assert features.ndim == 1
        assert len(features) == calc.n_features
        assert not np.isnan(features).any()
    
    def test_feature_names_match(self):
        """Test that feature names match feature count."""
        calc = PhysicochemicalCalculator()
        
        assert len(calc.feature_names) == calc.n_features
    
    def test_without_dipeptides(self):
        """Test calculator without dipeptide features."""
        config = PhysicochemicalConfig(include_dipeptide_composition=False)
        calc = PhysicochemicalCalculator(config)
        
        features = calc.calculate("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        # Should have ~35 features without 400 dipeptide features
        assert calc.n_features < 100
    
    def test_batch_calculation(self):
        """Test batch feature calculation."""
        calc = PhysicochemicalCalculator()
        
        sequences = [
            "MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            "ACDEFGHIKLMNPQRSTVWY",
            "GGGGGGGGGG",
        ]
        
        features = calc.calculate_batch(sequences, show_progress=False)
        
        assert features.shape == (3, calc.n_features)
        assert not np.isnan(features).any()
    
    def test_protein_integration(self):
        """Test integration with Protein class."""
        calc = PhysicochemicalCalculator()
        protein = Protein(
            sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
            name="test_protein",
        )
        
        features = calc.calculate_for_protein(protein)
        
        assert "physicochemical" in protein.embeddings
        assert protein.embeddings["physicochemical"].shape == (calc.n_features,)
    
    def test_library_integration(self):
        """Test integration with ProteinLibrary class."""
        calc = PhysicochemicalCalculator()
        library = ProteinLibrary(name="test")
        library.add(Protein(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS", name="p1"))
        library.add(Protein(sequence="ACDEFGHIKLMNPQRSTVWY", name="p2"))
        
        features = calc.calculate_for_library(library, show_progress=False)
        
        assert features.shape == (2, calc.n_features)
        assert "physicochemical" in library[0].embeddings
        assert "physicochemical" in library[1].embeddings
    
    def test_get_feature_descriptions(self):
        """Test feature description generation."""
        calc = PhysicochemicalCalculator()
        descriptions = calc.get_feature_descriptions()
        
        assert len(descriptions) == calc.n_features
        assert all(isinstance(v, str) for v in descriptions.values())


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_calculate_all_features(self):
        """Test calculate_all_features function."""
        features = calculate_all_features("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        
        assert isinstance(features, dict)
        assert 'molecular_weight' in features
        assert 'gravy' in features
        assert 'isoelectric_point' in features
    
    def test_get_physicochemical_calculator(self):
        """Test get_physicochemical_calculator function."""
        calc = get_physicochemical_calculator(include_dipeptides=False)
        
        assert isinstance(calc, PhysicochemicalCalculator)
        assert calc.config.include_dipeptide_composition is False