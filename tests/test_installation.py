# test_installation.py
"""Quick test to verify ProToPhen installation."""

from protophen import Protein, ProteinLibrary, ProtoPhenConfig

# Test Protein
print("Testing Protein class...")
protein = Protein(
    sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS",
    name="test_protein",
    source="de_novo",
)
print(f"  ✓ Created protein: {protein}")
print(f"  ✓ Length: {protein.length}")
print(f"  ✓ Hash: {protein.hash}")
print(f"  ✓ MW: {protein.molecular_weight:.1f} Da")

# Test ProteinLibrary
print("\nTesting ProteinLibrary class...")
library = ProteinLibrary(name="test_library")
library.add(protein)
library.add(Protein(sequence="ACDEFGHIKLMNPQRSTVWY", name="all_aa"))
print(f"  ✓ Created library: {library}")
print(f"  ✓ Summary: {library.summary()}")

# Test Configuration
print("\nTesting Configuration...")
config = ProtoPhenConfig(experiment_name="test")
print(f"  ✓ Created config: {config}")
print(f"  ✓ ESM model: {config.embedding.esm_model_name}")
print(f"  ✓ Learning rate: {config.training.learning_rate}")

print("\n All tests passed. ProToPhen is installed correctly.")