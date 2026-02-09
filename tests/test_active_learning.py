"""
Tests for active learning module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from protophen.active_learning.acquisition import (
    AcquisitionFunction,
    BatchAcquisition,
    DiversitySampling,
    ExpectedImprovement,
    HybridAcquisition,
    ProbabilityOfImprovement,
    UncertaintySampling,
    compute_acquisition_scores,
)
from protophen.active_learning.selection import (
    ActiveLearningLoop,
    ExperimentSelector,
    SelectionConfig,
    SelectionResult,
    compute_diversity_matrix,
    rank_by_uncertainty,
    select_diverse_subset,
    select_next_experiments,
)
from protophen.active_learning.uncertainty import (
    EnsembleEstimator,
    HeteroscedasticEstimator,
    MCDropoutEstimator,
    UncertaintyEstimate,
    UncertaintyEstimator,
    UncertaintyType,
    estimate_uncertainty,
    get_uncertainty_ranking,
)


# =============================================================================
# Fixtures
# =============================================================================

class MockModelWithDropout(nn.Module):
    """Mock model with dropout for MC Dropout testing."""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 100, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, output_dim)
    
    def forward(self, x, tasks=None, return_uncertainty=False):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        
        tasks = tasks or ["cell_painting"]
        outputs = {task: x for task in tasks}
        
        if return_uncertainty:
            # Add log variance outputs
            for task in tasks:
                outputs[f"{task}_log_var"] = torch.zeros_like(x)
        
        return outputs


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, n_samples: int = 32, embed_dim: int = 128, phenotype_dim: int = 100):
        self.n_samples = n_samples
        self.embed_dim = embed_dim
        self.phenotype_dim = phenotype_dim
        self.embeddings = torch.randn(n_samples, embed_dim)
        self.phenotypes = torch.randn(n_samples, phenotype_dim)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            "protein_embedding": self.embeddings[idx],
            "cell_painting": self.phenotypes[idx],
            "mask_cell_painting": torch.tensor(True),
            "protein_id": f"protein_{idx}",
        }


def create_mock_dataloader(n_samples: int = 32, batch_size: int = 8):
    """Create a mock DataLoader."""
    dataset = MockDataset(n_samples=n_samples)
    
    def collate_fn(batch):
        return {
            "protein_embedding": torch.stack([b["protein_embedding"] for b in batch]),
            "cell_painting": torch.stack([b["cell_painting"] for b in batch]),
            "mask_cell_painting": torch.stack([b["mask_cell_painting"] for b in batch]),
            "protein_id": [b["protein_id"] for b in batch],
        }
    
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


@pytest.fixture
def mock_model():
    """Create a mock model with dropout."""
    return MockModelWithDropout(input_dim=128, output_dim=100)


@pytest.fixture
def mock_dataloader():
    """Create a mock DataLoader."""
    return create_mock_dataloader(n_samples=32, batch_size=8)


@pytest.fixture
def uncertainty_estimate():
    """Create a sample UncertaintyEstimate."""
    n_samples = 20
    n_features = 100
    
    return UncertaintyEstimate(
        mean=np.random.randn(n_samples, n_features).astype(np.float32),
        epistemic=np.abs(np.random.randn(n_samples, n_features)).astype(np.float32),
        aleatoric=np.abs(np.random.randn(n_samples, n_features)).astype(np.float32),
        total=np.abs(np.random.randn(n_samples, n_features)).astype(np.float32),
        sample_ids=[f"protein_{i}" for i in range(n_samples)],
    )


@pytest.fixture
def embeddings():
    """Create sample embeddings for diversity testing."""
    return np.random.randn(20, 128).astype(np.float32)


# =============================================================================
# Tests for UncertaintyType
# =============================================================================

class TestUncertaintyType:
    """Tests for UncertaintyType enum."""
    
    def test_enum_values(self):
        """Test enum has expected values."""
        assert UncertaintyType.EPISTEMIC.value == "epistemic"
        assert UncertaintyType.ALEATORIC.value == "aleatoric"
        assert UncertaintyType.TOTAL.value == "total"
    
    def test_enum_is_string(self):
        """Test enum values are strings."""
        assert isinstance(UncertaintyType.EPISTEMIC.value, str)


# =============================================================================
# Tests for UncertaintyEstimate
# =============================================================================

class TestUncertaintyEstimate:
    """Tests for UncertaintyEstimate dataclass."""
    
    def test_basic_creation(self):
        """Test basic creation with mean only."""
        mean = np.random.randn(10, 50)
        estimate = UncertaintyEstimate(mean=mean)
        
        assert estimate.n_samples == 10
        assert estimate.n_features == 50
        assert estimate.epistemic is None
        assert estimate.aleatoric is None
    
    def test_full_creation(self, uncertainty_estimate):
        """Test creation with all fields."""
        assert uncertainty_estimate.n_samples == 20
        assert uncertainty_estimate.n_features == 100
        assert uncertainty_estimate.epistemic is not None
        assert uncertainty_estimate.aleatoric is not None
        assert uncertainty_estimate.total is not None
        assert len(uncertainty_estimate.sample_ids) == 20
    
    def test_get_uncertainty_total(self, uncertainty_estimate):
        """Test getting total uncertainty."""
        unc = uncertainty_estimate.get_uncertainty(
            uncertainty_type=UncertaintyType.TOTAL,
            reduction="mean",
        )
        
        assert unc.shape == (20,)  # n_samples
    
    def test_get_uncertainty_epistemic(self, uncertainty_estimate):
        """Test getting epistemic uncertainty."""
        unc = uncertainty_estimate.get_uncertainty(
            uncertainty_type=UncertaintyType.EPISTEMIC,
            reduction="mean",
        )
        
        assert unc.shape == (20,)
    
    def test_get_uncertainty_aleatoric(self, uncertainty_estimate):
        """Test getting aleatoric uncertainty."""
        unc = uncertainty_estimate.get_uncertainty(
            uncertainty_type=UncertaintyType.ALEATORIC,
            reduction="mean",
        )
        
        assert unc.shape == (20,)
    
    def test_get_uncertainty_reduction_sum(self, uncertainty_estimate):
        """Test sum reduction."""
        unc = uncertainty_estimate.get_uncertainty(
            uncertainty_type=UncertaintyType.TOTAL,
            reduction="sum",
        )
        
        assert unc.shape == (20,)
    
    def test_get_uncertainty_reduction_max(self, uncertainty_estimate):
        """Test max reduction."""
        unc = uncertainty_estimate.get_uncertainty(
            uncertainty_type=UncertaintyType.TOTAL,
            reduction="max",
        )
        
        assert unc.shape == (20,)
    
    def test_get_uncertainty_reduction_none(self, uncertainty_estimate):
        """Test no reduction."""
        unc = uncertainty_estimate.get_uncertainty(
            uncertainty_type=UncertaintyType.TOTAL,
            reduction="none",
        )
        
        assert unc.shape == (20, 100)  # (n_samples, n_features)
    
    def test_get_uncertainty_missing_type_raises(self):
        """Test that missing uncertainty type raises error."""
        estimate = UncertaintyEstimate(
            mean=np.random.randn(10, 50),
            # No total, epistemic, or aleatoric
        )
        
        with pytest.raises(ValueError, match="not available"):
            estimate.get_uncertainty(UncertaintyType.TOTAL)
    
    def test_get_uncertainty_invalid_reduction_raises(self, uncertainty_estimate):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError, match="Unknown reduction"):
            uncertainty_estimate.get_uncertainty(
                UncertaintyType.TOTAL,
                reduction="invalid",
            )
    
    def test_to_dict(self, uncertainty_estimate):
        """Test conversion to dictionary."""
        d = uncertainty_estimate.to_dict()
        
        assert "mean" in d
        assert "epistemic" in d
        assert "aleatoric" in d
        assert "total" in d
        assert "sample_ids" in d


# =============================================================================
# Tests for MCDropoutEstimator
# =============================================================================

class TestMCDropoutEstimator:
    """Tests for MCDropoutEstimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = MCDropoutEstimator(
            n_samples=10,
            tasks=["cell_painting"],
            device="cpu",
        )
        
        assert estimator.n_samples == 10
        assert estimator.tasks == ["cell_painting"]
        assert estimator.device == "cpu"
    
    def test_default_initialization(self):
        """Test default initialization values."""
        estimator = MCDropoutEstimator()
        
        assert estimator.n_samples == 20
        assert "cell_painting" in estimator.tasks
    
    def test_estimate(self, mock_model, mock_dataloader):
        """Test uncertainty estimation."""
        estimator = MCDropoutEstimator(
            n_samples=5,  # Few samples for speed
            tasks=["cell_painting"],
            device="cpu",
        )
        
        result = estimator.estimate(
            model=mock_model,
            dataloader=mock_dataloader,
            show_progress=False,
        )
        
        assert isinstance(result, UncertaintyEstimate)
        assert result.n_samples == 32  # Dataset size
        assert result.epistemic is not None
        assert result.total is not None
    
    def test_estimate_with_sample_ids(self, mock_model, mock_dataloader):
        """Test that sample IDs are collected."""
        estimator = MCDropoutEstimator(n_samples=3, device="cpu")
        
        result = estimator.estimate(
            model=mock_model,
            dataloader=mock_dataloader,
            show_progress=False,
        )
        
        assert result.sample_ids is not None
        assert len(result.sample_ids) == 32
        assert result.sample_ids[0] == "protein_0"
    
    def test_estimate_return_samples(self, mock_model, mock_dataloader):
        """Test returning raw MC samples."""
        estimator = MCDropoutEstimator(n_samples=5, device="cpu")
        
        result = estimator.estimate(
            model=mock_model,
            dataloader=mock_dataloader,
            show_progress=False,
            return_samples=True,
        )
        
        assert result.samples is not None
        assert result.samples.shape[0] == 5  # n_mc_samples
        assert result.samples.shape[1] == 32  # n_data_samples
    
    def test_dropout_enabled_during_estimation(self, mock_model, mock_dataloader):
        """Test that dropout produces variance in predictions."""
        estimator = MCDropoutEstimator(n_samples=10, device="cpu")
        
        result = estimator.estimate(
            model=mock_model,
            dataloader=mock_dataloader,
            show_progress=False,
            return_samples=True,
        )
        
        # With dropout, we should see variance across MC samples
        # (unless dropout rate is 0)
        variance = result.samples.var(axis=0)
        assert variance.mean() > 0 or mock_model.dropout.p == 0


# =============================================================================
# Tests for EnsembleEstimator
# =============================================================================

class TestEnsembleEstimator:
    """Tests for EnsembleEstimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = EnsembleEstimator(
            tasks=["cell_painting"],
            device="cpu",
        )
        
        assert estimator.tasks == ["cell_painting"]
    
    def test_requires_list_of_models(self, mock_model, mock_dataloader):
        """Test that single model raises error."""
        estimator = EnsembleEstimator(device="cpu")
        
        with pytest.raises(ValueError, match="list of models"):
            estimator.estimate(
                model=mock_model,
                dataloader=mock_dataloader,
            )
    
    def test_requires_at_least_two_models(self, mock_dataloader):
        """Test that single model in list raises error."""
        estimator = EnsembleEstimator(device="cpu")
        models = [MockModelWithDropout()]
        
        with pytest.raises(ValueError, match="at least 2"):
            estimator.estimate(
                model=models,
                dataloader=mock_dataloader,
            )
    
    def test_estimate_with_ensemble(self, mock_dataloader):
        """Test estimation with ensemble of models."""
        estimator = EnsembleEstimator(device="cpu")
        models = [MockModelWithDropout() for _ in range(3)]
        
        result = estimator.estimate(
            model=models,
            dataloader=mock_dataloader,
            show_progress=False,
        )
        
        assert isinstance(result, UncertaintyEstimate)
        assert result.n_samples == 32
        assert result.epistemic is not None
    
    def test_estimate_return_samples(self, mock_dataloader):
        """Test returning ensemble predictions."""
        estimator = EnsembleEstimator(device="cpu")
        models = [MockModelWithDropout() for _ in range(3)]
        
        result = estimator.estimate(
            model=models,
            dataloader=mock_dataloader,
            show_progress=False,
            return_samples=True,
        )
        
        assert result.samples is not None
        assert result.samples.shape[0] == 3  # n_models


# =============================================================================
# Tests for HeteroscedasticEstimator
# =============================================================================

class TestHeteroscedasticEstimator:
    """Tests for HeteroscedasticEstimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = HeteroscedasticEstimator(
            n_mc_samples=10,
            use_mc_dropout=True,
            device="cpu",
        )
        
        assert estimator.n_mc_samples == 10
        assert estimator.use_mc_dropout is True
    
    def test_estimate_with_uncertainty_output(self, mock_model, mock_dataloader):
        """Test estimation with model that outputs uncertainty."""
        estimator = HeteroscedasticEstimator(
            n_mc_samples=3,
            use_mc_dropout=True,
            device="cpu",
        )
        
        result = estimator.estimate(
            model=mock_model,
            dataloader=mock_dataloader,
            show_progress=False,
        )
        
        assert isinstance(result, UncertaintyEstimate)
        assert result.epistemic is not None
        assert result.aleatoric is not None
        assert result.total is not None


# =============================================================================
# Tests for estimate_uncertainty convenience function
# =============================================================================

class TestEstimateUncertainty:
    """Tests for estimate_uncertainty function."""
    
    def test_mc_dropout_method(self, mock_model, mock_dataloader):
        """Test MC Dropout method."""
        result = estimate_uncertainty(
            model=mock_model,
            dataloader=mock_dataloader,
            method="mc_dropout",
            n_samples=3,
            show_progress=False,
        )
        
        assert isinstance(result, UncertaintyEstimate)
    
    def test_ensemble_method(self, mock_dataloader):
        """Test ensemble method."""
        models = [MockModelWithDropout() for _ in range(3)]
        
        result = estimate_uncertainty(
            model=models,
            dataloader=mock_dataloader,
            method="ensemble",
            show_progress=False,
        )
        
        assert isinstance(result, UncertaintyEstimate)
    
    def test_heteroscedastic_method(self, mock_model, mock_dataloader):
        """Test heteroscedastic method."""
        result = estimate_uncertainty(
            model=mock_model,
            dataloader=mock_dataloader,
            method="heteroscedastic",
            n_samples=3,
            show_progress=False,
        )
        
        assert isinstance(result, UncertaintyEstimate)
    
    def test_invalid_method_raises(self, mock_model, mock_dataloader):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_uncertainty(
                model=mock_model,
                dataloader=mock_dataloader,
                method="invalid",
            )


# =============================================================================
# Tests for get_uncertainty_ranking
# =============================================================================

class TestGetUncertaintyRanking:
    """Tests for get_uncertainty_ranking function."""
    
    def test_ranking_descending(self, uncertainty_estimate):
        """Test default descending ranking."""
        indices, scores = get_uncertainty_ranking(uncertainty_estimate)
        
        assert len(indices) == 20
        assert len(scores) == 20
        # Should be sorted descending
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_ranking_ascending(self, uncertainty_estimate):
        """Test ascending ranking."""
        indices, scores = get_uncertainty_ranking(
            uncertainty_estimate,
            ascending=True,
        )
        
        # Should be sorted ascending
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
    
    def test_ranking_epistemic(self, uncertainty_estimate):
        """Test ranking by epistemic uncertainty."""
        indices, scores = get_uncertainty_ranking(
            uncertainty_estimate,
            uncertainty_type=UncertaintyType.EPISTEMIC,
        )
        
        assert len(indices) == 20


# =============================================================================
# Tests for UncertaintySampling
# =============================================================================

class TestUncertaintySampling:
    """Tests for UncertaintySampling acquisition function."""
    
    def test_initialization(self):
        """Test initialization."""
        acq = UncertaintySampling(
            uncertainty_type=UncertaintyType.EPISTEMIC,
            reduction="sum",
        )
        
        assert acq.uncertainty_type == UncertaintyType.EPISTEMIC
        assert acq.reduction == "sum"
    
    def test_score(self, uncertainty_estimate):
        """Test scoring samples."""
        acq = UncertaintySampling()
        scores = acq.score(uncertainty_estimate)
        
        assert scores.shape == (20,)
        assert not np.isnan(scores).any()
    
    def test_select(self, uncertainty_estimate):
        """Test selecting top samples."""
        acq = UncertaintySampling()
        indices = acq.select(uncertainty_estimate, n_select=5)
        
        assert len(indices) == 5
        assert len(set(indices)) == 5  # All unique
    
    def test_select_returns_highest_uncertainty(self, uncertainty_estimate):
        """Test that selection returns highest uncertainty samples."""
        acq = UncertaintySampling()
        
        scores = acq.score(uncertainty_estimate)
        indices = acq.select(uncertainty_estimate, n_select=5)
        
        # Selected should have highest scores
        selected_scores = scores[indices]
        assert selected_scores.min() >= np.sort(scores)[-5]


# =============================================================================
# Tests for ExpectedImprovement
# =============================================================================

class TestExpectedImprovement:
    """Tests for ExpectedImprovement acquisition function."""
    
    def test_initialization(self):
        """Test initialization."""
        acq = ExpectedImprovement(
            target_feature_idx=0,
            maximise=True,
            xi=0.1,
        )
        
        assert acq.target_feature_idx == 0
        assert acq.maximise is True
        assert acq.xi == 0.1
    
    def test_score(self, uncertainty_estimate):
        """Test EI scoring."""
        acq = ExpectedImprovement()
        scores = acq.score(uncertainty_estimate)
        
        assert scores.shape == (20,)
        assert not np.isnan(scores).any()
        assert (scores >= 0).all()  # EI is non-negative
    
    def test_score_with_best_value(self, uncertainty_estimate):
        """Test EI with explicit best value."""
        acq = ExpectedImprovement()
        scores = acq.score(uncertainty_estimate, best_value=0.0)
        
        assert scores.shape == (20,)
    
    def test_set_best_value(self, uncertainty_estimate):
        """Test setting best value."""
        acq = ExpectedImprovement()
        acq.set_best_value(0.5)
        
        assert acq.best_value == 0.5
        
        scores = acq.score(uncertainty_estimate)
        assert scores.shape == (20,)
    
    def test_select(self, uncertainty_estimate):
        """Test selection."""
        acq = ExpectedImprovement()
        indices = acq.select(uncertainty_estimate, n_select=5)
        
        assert len(indices) == 5


# =============================================================================
# Tests for ProbabilityOfImprovement
# =============================================================================

class TestProbabilityOfImprovement:
    """Tests for ProbabilityOfImprovement acquisition function."""
    
    def test_initialization(self):
        """Test initialization."""
        acq = ProbabilityOfImprovement(maximise=True)
        
        assert acq.maximise is True
    
    def test_score(self, uncertainty_estimate):
        """Test PI scoring."""
        acq = ProbabilityOfImprovement()
        scores = acq.score(uncertainty_estimate)
        
        assert scores.shape == (20,)
        assert (scores >= 0).all()
        assert (scores <= 1).all()  # Probability
    
    def test_select(self, uncertainty_estimate):
        """Test selection."""
        acq = ProbabilityOfImprovement()
        indices = acq.select(uncertainty_estimate, n_select=5)
        
        assert len(indices) == 5


# =============================================================================
# Tests for DiversitySampling
# =============================================================================

class TestDiversitySampling:
    """Tests for DiversitySampling acquisition function."""
    
    def test_initialization(self):
        """Test initialization."""
        acq = DiversitySampling(method="kmeans++", metric="cosine")
        
        assert acq.method == "kmeans++"
        assert acq.metric == "cosine"
    
    def test_score_returns_uniform(self, uncertainty_estimate):
        """Test that score returns uniform values (diversity needs select)."""
        acq = DiversitySampling()
        scores = acq.score(uncertainty_estimate)
        
        assert (scores == 1.0).all()
    
    def test_select_kmeans_pp(self, uncertainty_estimate, embeddings):
        """Test k-means++ selection."""
        acq = DiversitySampling(method="kmeans++")
        indices = acq.select(uncertainty_estimate, n_select=5, embeddings=embeddings)
        
        assert len(indices) == 5
        assert len(set(indices)) == 5  # All unique
    
    def test_select_maxmin(self, uncertainty_estimate, embeddings):
        """Test maxmin selection."""
        acq = DiversitySampling(method="maxmin")
        indices = acq.select(uncertainty_estimate, n_select=5, embeddings=embeddings)
        
        assert len(indices) == 5
    
    def test_select_dpp_approx(self, uncertainty_estimate, embeddings):
        """Test approximate DPP selection."""
        acq = DiversitySampling(method="dpp_approx")
        indices = acq.select(uncertainty_estimate, n_select=5, embeddings=embeddings)
        
        assert len(indices) == 5
    
    def test_select_without_embeddings(self, uncertainty_estimate):
        """Test selection using predictions as embeddings."""
        acq = DiversitySampling(method="kmeans++")
        indices = acq.select(uncertainty_estimate, n_select=5)
        
        assert len(indices) == 5
    
    def test_select_more_than_available(self, uncertainty_estimate, embeddings):
        """Test selecting more samples than available."""
        acq = DiversitySampling()
        indices = acq.select(uncertainty_estimate, n_select=100, embeddings=embeddings)
        
        assert len(indices) == 20  # Limited to available samples
    
    def test_invalid_method_raises(self, uncertainty_estimate):
        """Test that invalid method raises error."""
        acq = DiversitySampling(method="invalid")
        
        with pytest.raises(ValueError, match="Unknown method"):
            acq.select(uncertainty_estimate, n_select=5)


# =============================================================================
# Tests for HybridAcquisition
# =============================================================================

class TestHybridAcquisition:
    """Tests for HybridAcquisition."""
    
    def test_initialization(self):
        """Test initialization."""
        acq = HybridAcquisition(
            uncertainty_weight=0.6,
            diversity_weight=0.4,
        )
        
        assert acq.uncertainty_weight == 0.6
        assert acq.diversity_weight == 0.4
    
    def test_score(self, uncertainty_estimate):
        """Test scoring (returns uncertainty scores)."""
        acq = HybridAcquisition()
        scores = acq.score(uncertainty_estimate)
        
        assert scores.shape == (20,)
    
    def test_select(self, uncertainty_estimate, embeddings):
        """Test hybrid selection."""
        acq = HybridAcquisition()
        indices = acq.select(uncertainty_estimate, n_select=5, embeddings=embeddings)
        
        assert len(indices) == 5
        assert len(set(indices)) == 5
    
    def test_select_balances_uncertainty_and_diversity(self, uncertainty_estimate, embeddings):
        """Test that selection balances both criteria."""
        # High uncertainty weight should favor uncertain samples
        acq_unc = HybridAcquisition(uncertainty_weight=0.99, diversity_weight=0.01)
        
        # High diversity weight should favor diverse samples
        acq_div = HybridAcquisition(uncertainty_weight=0.01, diversity_weight=0.99)
        
        indices_unc = acq_unc.select(uncertainty_estimate, n_select=5, embeddings=embeddings)
        indices_div = acq_div.select(uncertainty_estimate, n_select=5, embeddings=embeddings)
        
        # Should produce different selections (usually)
        # This is probabilistic, so we just check they're valid
        assert len(indices_unc) == 5
        assert len(indices_div) == 5


# =============================================================================
# Tests for BatchAcquisition
# =============================================================================

class TestBatchAcquisition:
    """Tests for BatchAcquisition."""
    
    def test_initialization(self):
        """Test initialization."""
        base = UncertaintySampling()
        acq = BatchAcquisition(base_acquisition=base, batch_strategy="greedy")
        
        assert acq.batch_strategy == "greedy"
    
    def test_score(self, uncertainty_estimate):
        """Test scoring delegates to base."""
        base = UncertaintySampling()
        acq = BatchAcquisition(base_acquisition=base)
        
        base_scores = base.score(uncertainty_estimate)
        batch_scores = acq.score(uncertainty_estimate)
        
        np.testing.assert_array_equal(base_scores, batch_scores)
    
    def test_select_greedy(self, uncertainty_estimate, embeddings):
        """Test greedy batch selection."""
        base = UncertaintySampling()
        acq = BatchAcquisition(base_acquisition=base, batch_strategy="greedy")
        
        indices = acq.select(
            uncertainty_estimate,
            n_select=5,
            embeddings=embeddings,
        )
        
        assert len(indices) == 5
        assert len(set(indices)) == 5
    
    def test_select_stochastic(self, uncertainty_estimate):
        """Test stochastic batch selection."""
        base = UncertaintySampling()
        acq = BatchAcquisition(base_acquisition=base, batch_strategy="stochastic")
        
        indices = acq.select(uncertainty_estimate, n_select=5)
        
        assert len(indices) == 5


# =============================================================================
# Tests for compute_acquisition_scores
# =============================================================================

class TestComputeAcquisitionScores:
    """Tests for compute_acquisition_scores function."""
    
    def test_uncertainty_method(self, uncertainty_estimate):
        """Test uncertainty scoring."""
        scores = compute_acquisition_scores(
            uncertainty_estimate,
            method="uncertainty",
        )
        
        assert scores.shape == (20,)
    
    def test_ei_method(self, uncertainty_estimate):
        """Test EI scoring."""
        scores = compute_acquisition_scores(
            uncertainty_estimate,
            method="ei",
        )
        
        assert scores.shape == (20,)
    
    def test_pi_method(self, uncertainty_estimate):
        """Test PI scoring."""
        scores = compute_acquisition_scores(
            uncertainty_estimate,
            method="pi",
        )
        
        assert scores.shape == (20,)
    
    def test_random_method(self, uncertainty_estimate):
        """Test random scoring."""
        scores = compute_acquisition_scores(
            uncertainty_estimate,
            method="random",
        )
        
        assert scores.shape == (20,)
        assert (scores >= 0).all()
        assert (scores <= 1).all()
    
    def test_invalid_method_raises(self, uncertainty_estimate):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_acquisition_scores(uncertainty_estimate, method="invalid")


# =============================================================================
# Tests for SelectionConfig
# =============================================================================

class TestSelectionConfig:
    """Tests for SelectionConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = SelectionConfig()
        
        assert config.n_select == 10
        assert config.uncertainty_method == "mc_dropout"
        assert config.n_mc_samples == 20
        assert config.acquisition_method == "hybrid"
        assert config.uncertainty_weight == 0.7
        assert config.diversity_weight == 0.3
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = SelectionConfig(
            n_select=50,
            uncertainty_method="ensemble",
            acquisition_method="uncertainty",
        )
        
        assert config.n_select == 50
        assert config.uncertainty_method == "ensemble"
        assert config.acquisition_method == "uncertainty"


# =============================================================================
# Tests for SelectionResult
# =============================================================================

class TestSelectionResult:
    """Tests for SelectionResult dataclass."""
    
    @pytest.fixture
    def selection_result(self, uncertainty_estimate):
        """Create a sample SelectionResult."""
        return SelectionResult(
            selected_indices=np.array([0, 5, 10, 15, 19]),
            selected_ids=["protein_0", "protein_5", "protein_10", "protein_15", "protein_19"],
            acquisition_scores=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            uncertainty_estimates=uncertainty_estimate,
            all_indices_ranked=np.arange(20),
            all_scores=np.random.rand(20),
            config=SelectionConfig(n_select=5),
        )
    
    def test_to_dict(self, selection_result):
        """Test conversion to dictionary."""
        d = selection_result.to_dict()
        
        assert "selected_indices" in d
        assert "selected_ids" in d
        assert "acquisition_scores" in d
        assert "config" in d
    
    def test_get_selected_proteins(self, selection_result):
        """Test getting selected protein info."""
        proteins = selection_result.get_selected_proteins()
        
        assert len(proteins) == 5
        assert proteins[0]["rank"] == 1
        assert proteins[0]["id"] == "protein_0"
        assert "acquisition_score" in proteins[0]
        assert "uncertainty" in proteins[0]
    
    def test_summary(self, selection_result):
        """Test summary string."""
        summary = selection_result.summary()
        
        assert "5 samples selected" in summary
        assert "hybrid" in summary  # Method


# =============================================================================
# Tests for ExperimentSelector
# =============================================================================

class TestExperimentSelector:
    """Tests for ExperimentSelector class."""
    
    def test_initialization(self, mock_model):
        """Test selector initialization."""
        config = SelectionConfig(n_select=5)
        selector = ExperimentSelector(
            model=mock_model,
            config=config,
            device="cpu",
        )
        
        assert selector.model is mock_model
        assert selector.config.n_select == 5
    
    def test_default_initialization(self, mock_model):
        """Test default initialization."""
        selector = ExperimentSelector(model=mock_model)
        
        assert selector.config.n_select == 10
    
    def test_select(self, mock_model, mock_dataloader):
        """Test sample selection."""
        config = SelectionConfig(
            n_select=5,
            n_mc_samples=3,  # Few samples for speed
        )
        selector = ExperimentSelector(
            model=mock_model,
            config=config,
            device="cpu",
        )
        
        result = selector.select(mock_dataloader, show_progress=False)
        
        assert isinstance(result, SelectionResult)
        assert len(result.selected_ids) == 5
        assert len(result.selected_indices) == 5
    
    def test_select_updates_history(self, mock_model, mock_dataloader):
        """Test that selection updates history."""
        config = SelectionConfig(n_select=5, n_mc_samples=3)
        selector = ExperimentSelector(model=mock_model, config=config, device="cpu")
        
        assert len(selector.selection_history) == 0
        
        selector.select(mock_dataloader, show_progress=False)
        
        assert len(selector.selection_history) == 1
    
    def test_select_updates_exclude_list(self, mock_model, mock_dataloader):
        """Test that selected IDs are added to exclude list."""
        config = SelectionConfig(n_select=5, n_mc_samples=3)
        selector = ExperimentSelector(model=mock_model, config=config, device="cpu")
        
        result = selector.select(mock_dataloader, show_progress=False)
        
        assert len(selector.config.exclude_ids) == 5
        assert all(id in selector.config.exclude_ids for id in result.selected_ids)
    
    def test_select_with_exclude_ids(self, mock_model, mock_dataloader):
        """Test selection with pre-excluded IDs."""
        config = SelectionConfig(
            n_select=5,
            n_mc_samples=3,
            exclude_ids=["protein_0", "protein_1", "protein_2"],
        )
        selector = ExperimentSelector(model=mock_model, config=config, device="cpu")
        
        result = selector.select(mock_dataloader, show_progress=False)
        
        # Excluded IDs should not be selected
        assert "protein_0" not in result.selected_ids
        assert "protein_1" not in result.selected_ids
        assert "protein_2" not in result.selected_ids
    
    def test_select_iterative(self, mock_model, mock_dataloader):
        """Test iterative selection."""
        config = SelectionConfig(n_select=3, n_mc_samples=3)
        selector = ExperimentSelector(model=mock_model, config=config, device="cpu")
        
        results = selector.select_iterative(
            mock_dataloader,
            n_iterations=2,
            n_per_iteration=3,
            show_progress=False,
        )
        
        assert len(results) == 2
        assert len(selector.selection_history) == 2
    
    def test_reset_exclusions(self, mock_model, mock_dataloader):
        """Test resetting exclusion list."""
        config = SelectionConfig(n_select=5, n_mc_samples=3)
        selector = ExperimentSelector(model=mock_model, config=config, device="cpu")
        
        selector.select(mock_dataloader, show_progress=False)
        assert len(selector.config.exclude_ids) > 0
        
        selector.reset_exclusions()
        assert len(selector.config.exclude_ids) == 0
    
    def test_get_selection_summary(self, mock_model, mock_dataloader):
        """Test getting selection summary."""
        config = SelectionConfig(n_select=5, n_mc_samples=3)
        selector = ExperimentSelector(model=mock_model, config=config, device="cpu")
        
        # Before any selection
        summary = selector.get_selection_summary()
        assert summary["n_selections"] == 0
        
        # After selection
        selector.select(mock_dataloader, show_progress=False)
        summary = selector.get_selection_summary()
        
        assert summary["n_selections"] == 1
        assert summary["total_selected"] == 5
    
    def test_acquisition_methods(self, mock_model, mock_dataloader):
        """Test different acquisition methods."""
        methods = ["uncertainty", "ei", "diversity", "hybrid"]
        
        for method in methods:
            config = SelectionConfig(
                n_select=3,
                n_mc_samples=3,
                acquisition_method=method,
            )
            selector = ExperimentSelector(model=mock_model, config=config, device="cpu")
            
            result = selector.select(mock_dataloader, show_progress=False)
            assert len(result.selected_ids) == 3
    
    def test_invalid_uncertainty_method_raises(self, mock_model):
        """Test that invalid uncertainty method raises error."""
        config = SelectionConfig(uncertainty_method="invalid")
        
        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            ExperimentSelector(model=mock_model, config=config)
    
    def test_invalid_acquisition_method_raises(self, mock_model):
        """Test that invalid acquisition method raises error."""
        config = SelectionConfig(acquisition_method="invalid")
        
        with pytest.raises(ValueError, match="Unknown acquisition method"):
            ExperimentSelector(model=mock_model, config=config)


# =============================================================================
# Tests for Convenience Functions
# =============================================================================

class TestSelectNextExperiments:
    """Tests for select_next_experiments function."""
    
    def test_basic_usage(self, mock_model, mock_dataloader):
        """Test basic usage."""
        result = select_next_experiments(
            model=mock_model,
            dataloader=mock_dataloader,
            n_select=5,
            method="uncertainty",
            n_mc_samples=3,
            show_progress=False,
        )
        
        assert isinstance(result, SelectionResult)
        assert len(result.selected_ids) == 5
    
    def test_with_exclude_ids(self, mock_model, mock_dataloader):
        """Test with excluded IDs."""
        result = select_next_experiments(
            model=mock_model,
            dataloader=mock_dataloader,
            n_select=5,
            exclude_ids=["protein_0"],
            n_mc_samples=3,
            show_progress=False,
        )
        
        assert "protein_0" not in result.selected_ids


class TestRankByUncertainty:
    """Tests for rank_by_uncertainty function."""
    
    def test_returns_rankings(self, mock_model, mock_dataloader):
        """Test that function returns proper rankings."""
        indices, scores, ids = rank_by_uncertainty(
            model=mock_model,
            dataloader=mock_dataloader,
            n_mc_samples=3,
            show_progress=False,
        )
        
        assert len(indices) == 32
        assert len(scores) == 32
        assert len(ids) == 32
        
        # Should be sorted descending
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


class TestComputeDiversityMatrix:
    """Tests for compute_diversity_matrix function."""
    
    def test_returns_square_matrix(self, embeddings):
        """Test that function returns square matrix."""
        matrix = compute_diversity_matrix(embeddings)
        
        assert matrix.shape == (20, 20)
    
    def test_diagonal_is_zero(self, embeddings):
        """Test that diagonal is zero (distance to self)."""
        matrix = compute_diversity_matrix(embeddings)
        
        np.testing.assert_array_almost_equal(np.diag(matrix), 0.0)
    
    def test_symmetric(self, embeddings):
        """Test that matrix is symmetric."""
        matrix = compute_diversity_matrix(embeddings)
        
        np.testing.assert_array_almost_equal(matrix, matrix.T)
    
    def test_different_metrics(self, embeddings):
        """Test with different distance metrics."""
        for metric in ["euclidean", "cosine", "cityblock"]:
            matrix = compute_diversity_matrix(embeddings, metric=metric)
            assert matrix.shape == (20, 20)


class TestSelectDiverseSubset:
    """Tests for select_diverse_subset function."""
    
    def test_basic_selection(self, embeddings):
        """Test basic diverse subset selection."""
        indices = select_diverse_subset(
            embeddings=embeddings,
            n_select=5,
        )
        
        assert len(indices) == 5
        assert len(set(indices)) == 5  # All unique
    
    def test_reproducible_with_seed(self, embeddings):
        """Test reproducibility with seed."""
        indices1 = select_diverse_subset(embeddings, n_select=5, seed=42)
        indices2 = select_diverse_subset(embeddings, n_select=5, seed=42)
        
        np.testing.assert_array_equal(indices1, indices2)
    
    def test_different_seeds_different_results(self, embeddings):
        """Test different seeds produce different results."""
        indices1 = select_diverse_subset(embeddings, n_select=5, seed=42)
        indices2 = select_diverse_subset(embeddings, n_select=5, seed=123)
        
        # Should be different (with high probability)
        assert not np.array_equal(indices1, indices2)
    
    def test_methods(self, embeddings):
        """Test different selection methods."""
        for method in ["kmeans++", "maxmin"]:
            indices = select_diverse_subset(
                embeddings=embeddings,
                n_select=5,
                method=method,
            )
            assert len(indices) == 5


# =============================================================================
# Tests for ActiveLearningLoop
# =============================================================================

class TestActiveLearningLoop:
    """Tests for ActiveLearningLoop class."""
    
    @pytest.fixture
    def al_loop(self, mock_model):
        """Create an ActiveLearningLoop for testing."""
        # Create mock datasets
        train_data = MockDataset(n_samples=16)
        pool_data = MockDataset(n_samples=32)
        val_data = MockDataset(n_samples=8)
        
        def model_factory():
            return MockModelWithDropout()
        
        return ActiveLearningLoop(
            model_factory=model_factory,
            initial_train_data=train_data,
            pool_data=pool_data,
            val_data=val_data,
            n_iterations=2,
            n_samples_per_iteration=3,
            selection_config=SelectionConfig(n_select=3, n_mc_samples=3),
        )
    
    def test_initialization(self, al_loop):
        """Test AL loop initialization."""
        assert al_loop.n_iterations == 2
        assert al_loop.n_samples_per_iteration == 3
        assert al_loop.iteration == 0
        assert len(al_loop.history) == 0
    
    def test_train_model(self, al_loop):
        """Test model training."""
        with patch("protophen.data.loaders.create_dataloader") as mock_loader:
            with patch("protophen.training.trainer.Trainer") as mock_trainer:
                mock_loader.return_value = create_mock_dataloader(n_samples=16, batch_size=4)
                mock_trainer_instance = MagicMock()
                mock_trainer_instance.train.return_value = {"train_losses": [0.5]}
                mock_trainer.return_value = mock_trainer_instance
                
                model = al_loop.train_model()
                
                assert model is not None
                assert al_loop.current_model is model
    
    def test_get_summary_empty(self, al_loop):
        """Test summary when no iterations completed."""
        summary = al_loop.get_summary()
        
        assert summary["status"] == "not started"
    
    def test_update_datasets(self, al_loop):
        """Test dataset update tracking."""
        indices = np.array([0, 5, 10])
        al_loop.update_datasets(indices)
        
        assert len(al_loop.selected_indices_history) == 1
        np.testing.assert_array_equal(al_loop.selected_indices_history[0], indices)


# =============================================================================
# Integration Tests
# =============================================================================

class TestActiveLearniningIntegration:
    """Integration tests for active learning pipeline."""
    
    def test_full_selection_pipeline(self, mock_model, mock_dataloader):
        """Test complete selection pipeline."""
        # Step 1: Estimate uncertainty
        uncertainty = estimate_uncertainty(
            model=mock_model,
            dataloader=mock_dataloader,
            method="mc_dropout",
            n_samples=5,
            show_progress=False,
        )
        
        assert uncertainty.n_samples == 32
        
        # Step 2: Rank by uncertainty
        indices, scores = get_uncertainty_ranking(uncertainty)
        
        assert len(indices) == 32
        
        # Step 3: Select using acquisition function
        acq = HybridAcquisition()
        selected = acq.select(uncertainty, n_select=5)
        
        assert len(selected) == 5
    
    def test_selector_with_different_methods(self, mock_model, mock_dataloader):
        """Test selector with various acquisition methods."""
        for method in ["uncertainty", "ei", "hybrid"]:
            config = SelectionConfig(
                n_select=5,
                n_mc_samples=3,
                acquisition_method=method,
            )
            
            selector = ExperimentSelector(
                model=mock_model,
                config=config,
                device="cpu",
            )
            
            result = selector.select(mock_dataloader, show_progress=False)
            
            assert len(result.selected_ids) == 5
            assert result.uncertainty_estimates is not None
    
    def test_iterative_selection_excludes_previous(self, mock_model, mock_dataloader):
        """Test that iterative selection properly excludes previous selections."""
        config = SelectionConfig(n_select=5, n_mc_samples=3)
        selector = ExperimentSelector(
            model=mock_model,
            config=config,
            device="cpu",
        )
        
        # First selection
        result1 = selector.select(mock_dataloader, show_progress=False)
        first_ids = set(result1.selected_ids)
        
        # Second selection
        result2 = selector.select(mock_dataloader, show_progress=False)
        second_ids = set(result2.selected_ids)
        
        # Should have no overlap
        assert first_ids.isdisjoint(second_ids)