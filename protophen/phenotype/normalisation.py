"""
Normalisation and batch correction for phenotypic data.

This module provides tools for normalising Cell Painting features
and correcting batch effects between plates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from protophen.data.phenotype import PhenotypeDataset
from protophen.utils.logging import logger


# =============================================================================
# Normalisation Functions
# =============================================================================

def robust_mad_normalise(
    x: np.ndarray,
    center: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust normalisation using median and MAD.
    
    Args:
        x: Data array of shape (n_samples, n_features)
        center: Pre-computed median (for applying to new data)
        scale: Pre-computed MAD (for applying to new data)
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Tuple of (normalised_data, median, mad)
    """
    if center is None:
        center = np.nanmedian(x, axis=0)
    
    if scale is None:
        # MAD = median(|x - median(x)|)
        scale = np.nanmedian(np.abs(x - center), axis=0)
        # Convert MAD to standard deviation equivalent
        scale = scale * 1.4826
    
    # Prevent division by zero
    scale = np.where(scale < epsilon, epsilon, scale)
    
    normalised = (x - center) / scale
    
    return normalised, center, scale


def zscore_normalise(
    x: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalisation.
    
    Args:
        x: Data array of shape (n_samples, n_features)
        mean: Pre-computed mean
        std: Pre-computed standard deviation
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Tuple of (normalised_data, mean, std)
    """
    if mean is None:
        mean = np.nanmean(x, axis=0)
    
    if std is None:
        std = np.nanstd(x, axis=0)
    
    std = np.where(std < epsilon, epsilon, std)
    
    normalised = (x - mean) / std
    
    return normalised, mean, std


def normalise_plate(
    features: np.ndarray,
    control_indices: Optional[np.ndarray] = None,
    method: str = "robust_mad",
) -> np.ndarray:
    """
    Normalise features within a plate.
    
    If control_indices provided, normalisation statistics are computed
    from controls only (recommended for Cell Painting).
    
    Args:
        features: Feature matrix (n_samples, n_features)
        control_indices: Indices of negative control wells
        method: Normalisation method ("robust_mad" or "zscore")
        
    Returns:
        Normalised feature matrix
    """
    if control_indices is not None and len(control_indices) > 0:
        reference_data = features[control_indices]
    else:
        reference_data = features
    
    if method == "robust_mad":
        normalised, _, _ = robust_mad_normalise(
            features,
            center=np.nanmedian(reference_data, axis=0),
            scale=np.nanmedian(np.abs(reference_data - np.nanmedian(reference_data, axis=0)), axis=0) * 1.4826
        )
    elif method == "zscore":
        normalised, _, _ = zscore_normalise(
            features,
            mean=np.nanmean(reference_data, axis=0),
            std=np.nanstd(reference_data, axis=0)
        )
    else:
        raise ValueError(f"Unknown normalisation method: {method}")
    
    return normalised


# =============================================================================
# Normaliser Class
# =============================================================================

@dataclass
class NormaliserConfig:
    """Configuration for feature normalisation."""
    
    method: Literal["robust_mad", "zscore", "minmax", "none"] = "robust_mad"
    clip_outliers: bool = True
    outlier_threshold: float = 5.0
    use_controls: bool = True
    control_column: str = "treatment"
    control_value: str = "DMSO"


class Normaliser:
    """
    Normalise phenotypic features.
    
    Attributes:
        config: Normalisation configuration
        is_fitted: Whether normaliser has been fitted
        
    Example:
        >>> normaliser = Normaliser(method="robust_mad")
        >>> 
        >>> # Fit on training data
        >>> normaliser.fit(train_features)
        >>> 
        >>> # Transform new data
        >>> normalised = normaliser.transform(test_features)
    """
    
    def __init__(
        self,
        method: str = "robust_mad",
        clip_outliers: bool = True,
        outlier_threshold: float = 5.0,
    ):
        """
        Initialise normaliser.
        
        Args:
            method: Normalisation method
            clip_outliers: Whether to clip outliers after normalisation
            outlier_threshold: Threshold for clipping (in std units)
        """
        self.config = NormaliserConfig(
            method=method,
            clip_outliers=clip_outliers,
            outlier_threshold=outlier_threshold,
        )
        
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> "Normaliser":
        """
        Fit normaliser to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            self
        """
        if self.config.method == "robust_mad":
            self.center_ = np.nanmedian(X, axis=0)
            self.scale_ = np.nanmedian(np.abs(X - self.center_), axis=0) * 1.4826
        elif self.config.method == "zscore":
            self.center_ = np.nanmean(X, axis=0)
            self.scale_ = np.nanstd(X, axis=0)
        elif self.config.method == "minmax":
            self.center_ = np.nanmin(X, axis=0)
            self.scale_ = np.nanmax(X, axis=0) - self.center_
        elif self.config.method == "none":
            self.center_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        # Prevent division by zero
        self.scale_ = np.where(self.scale_ < 1e-8, 1e-8, self.scale_)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Normalised feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Normaliser must be fitted before transform")
        
        normalised = (X - self.center_) / self.scale_
        
        # Clip outliers
        if self.config.clip_outliers:
            threshold = self.config.outlier_threshold
            normalised = np.clip(normalised, -threshold, threshold)
        
        return normalised
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform normalised data."""
        if not self.is_fitted:
            raise RuntimeError("Normaliser must be fitted before inverse_transform")
        
        return X * self.scale_ + self.center_


# =============================================================================
# Batch Correction
# =============================================================================

class BatchCorrector:
    """
    Correct batch effects between plates.
    
    Supports several batch correction methods:
    - center: Center each batch to zero mean
    - zscore: Z-score normalise within each batch
    - robust: Robust normalisation within each batch
    - reference: Align all batches to a reference batch
    
    Example:
        >>> corrector = BatchCorrector(method="robust")
        >>> 
        >>> # Correct batch effects
        >>> corrected = corrector.fit_transform(features, batch_labels)
    """
    
    def __init__(
        self,
        method: Literal["center", "zscore", "robust", "reference"] = "robust",
        reference_batch: Optional[str] = None,
    ):
        """
        Initialise batch corrector.
        
        Args:
            method: Batch correction method
            reference_batch: Reference batch to align others to (for 'reference' method)
        """
        self.method = method
        self.reference_batch = reference_batch
        
        self._batch_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self._global_stats: Dict[str, np.ndarray] = {}
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> "BatchCorrector":
        """
        Fit batch correction parameters.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            batch_labels: Batch label for each sample
            
        Returns:
            self
        """
        unique_batches = np.unique(batch_labels)
        
        # Compute per-batch statistics
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_data = X[batch_mask]
            
            if self.method in ["robust", "reference"]:
                center = np.nanmedian(batch_data, axis=0)
                scale = np.nanmedian(np.abs(batch_data - center), axis=0) * 1.4826
            else:  # zscore or center
                center = np.nanmean(batch_data, axis=0)
                scale = np.nanstd(batch_data, axis=0) if self.method == "zscore" else np.ones(X.shape[1])
            
            # Prevent division by zero
            scale = np.where(scale < 1e-8, 1e-8, scale)
            
            self._batch_stats[str(batch)] = {
                "center": center,
                "scale": scale,
                "n_samples": batch_mask.sum(),
            }
        
        # Compute global statistics
        if self.method in ["robust", "reference"]:
            self._global_stats["center"] = np.nanmedian(X, axis=0)
            self._global_stats["scale"] = np.nanmedian(
                np.abs(X - self._global_stats["center"]), axis=0
            ) * 1.4826
        else:
            self._global_stats["center"] = np.nanmean(X, axis=0)
            self._global_stats["scale"] = np.nanstd(X, axis=0)
        
        self._global_stats["scale"] = np.where(
            self._global_stats["scale"] < 1e-8, 1e-8, self._global_stats["scale"]
        )
        
        # For reference method, store reference batch stats
        if self.method == "reference" and self.reference_batch is not None:
            if str(self.reference_batch) not in self._batch_stats:
                raise ValueError(f"Reference batch '{self.reference_batch}' not found")
            self._reference_stats = self._batch_stats[str(self.reference_batch)]
        
        self.is_fitted = True
        logger.info(f"Fitted BatchCorrector on {len(unique_batches)} batches")
        
        return self
    
    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Apply batch correction.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            batch_labels: Batch label for each sample
            
        Returns:
            Batch-corrected feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("BatchCorrector must be fitted before transform")
        
        corrected = np.zeros_like(X)
        
        for batch in np.unique(batch_labels):
            batch_mask = batch_labels == batch
            batch_data = X[batch_mask]
            batch_str = str(batch)
            
            # Get batch stats (use global if batch not seen during fitting)
            if batch_str in self._batch_stats:
                batch_center = self._batch_stats[batch_str]["center"]
                batch_scale = self._batch_stats[batch_str]["scale"]
            else:
                logger.warning(f"Batch '{batch}' not seen during fitting, using global stats")
                batch_center = self._global_stats["center"]
                batch_scale = self._global_stats["scale"]
            
            if self.method == "center":
                # Center to global mean
                corrected[batch_mask] = batch_data - batch_center + self._global_stats["center"]
            
            elif self.method == "zscore":
                # Z-score within batch, then rescale to global
                z_scored = (batch_data - batch_center) / batch_scale
                corrected[batch_mask] = z_scored * self._global_stats["scale"] + self._global_stats["center"]
            
            elif self.method == "robust":
                # Robust normalise within batch, then rescale to global
                normalised = (batch_data - batch_center) / batch_scale
                corrected[batch_mask] = normalised * self._global_stats["scale"] + self._global_stats["center"]
            
            elif self.method == "reference":
                # Align to reference batch
                if hasattr(self, '_reference_stats'):
                    normalised = (batch_data - batch_center) / batch_scale
                    corrected[batch_mask] = (
                        normalised * self._reference_stats["scale"] + self._reference_stats["center"]
                    )
                else:
                    corrected[batch_mask] = batch_data
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        return corrected
    
    def fit_transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, batch_labels).transform(X, batch_labels)
    
    def get_batch_summary(self) -> pd.DataFrame:
        """Get summary of batch statistics."""
        if not self.is_fitted:
            return pd.DataFrame()
        
        records = []
        for batch, stats in self._batch_stats.items():
            records.append({
                "batch": batch,
                "n_samples": stats["n_samples"],
                "center_mean": float(np.mean(stats["center"])),
                "center_std": float(np.std(stats["center"])),
                "scale_mean": float(np.mean(stats["scale"])),
                "scale_std": float(np.std(stats["scale"])),
            })
        
        return pd.DataFrame(records)


# =============================================================================
# Feature Selection
# =============================================================================

class FeatureSelector:
    """
    Select informative features from phenotypic data.
    
    Supports multiple selection strategies:
    - variance: Remove low-variance features
    - correlation: Remove highly correlated features
    - nan: Remove features with too many missing values
    
    Example:
        >>> selector = FeatureSelector(
        ...     min_variance=1e-6,
        ...     max_correlation=0.95,
        ...     max_nan_fraction=0.05,
        ... )
        >>> selected = selector.fit_transform(features)
        >>> print(f"Selected {selector.n_selected_} of {features.shape[1]} features")
    """
    
    def __init__(
        self,
        min_variance: float = 1e-6,
        max_correlation: float = 0.95,
        max_nan_fraction: float = 0.05,
    ):
        """
        Initialise feature selector.
        
        Args:
            min_variance: Minimum variance threshold
            max_correlation: Maximum correlation threshold
            max_nan_fraction: Maximum fraction of NaN values allowed
        """
        self.min_variance = min_variance
        self.max_correlation = max_correlation
        self.max_nan_fraction = max_nan_fraction
        
        self.selected_indices_: Optional[np.ndarray] = None
        self.removed_indices_: Optional[np.ndarray] = None
        self.removal_reasons_: Dict[int, str] = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> "FeatureSelector":
        """
        Fit feature selector.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            self
        """
        n_features = X.shape[1]
        to_remove = set()
        
        # Step 1: Remove features with too many NaNs
        nan_fractions = np.isnan(X).mean(axis=0)
        nan_removed = np.where(nan_fractions > self.max_nan_fraction)[0]
        for idx in nan_removed:
            to_remove.add(idx)
            self.removal_reasons_[idx] = f"nan_fraction={nan_fractions[idx]:.3f}"
        
        logger.debug(f"Removed {len(nan_removed)} features due to high NaN fraction")
        
        # Step 2: Remove low-variance features
        # Compute variance ignoring NaNs
        variances = np.nanvar(X, axis=0)
        low_var_removed = np.where(variances < self.min_variance)[0]
        for idx in low_var_removed:
            if idx not in to_remove:
                to_remove.add(idx)
                self.removal_reasons_[idx] = f"low_variance={variances[idx]:.2e}"
        
        logger.debug(f"Removed {len(low_var_removed)} low-variance features")
        
        # Step 3: Remove highly correlated features
        remaining_indices = [i for i in range(n_features) if i not in to_remove]
        
        if len(remaining_indices) > 1 and self.max_correlation < 1.0:
            X_remaining = X[:, remaining_indices]
            
            # Compute correlation matrix (handling NaNs)
            # Use pandas for pairwise complete correlation
            df = pd.DataFrame(X_remaining)
            corr_matrix = df.corr().abs().values
            
            # Find highly correlated pairs
            corr_removed = set()
            for i in range(len(remaining_indices)):
                if i in corr_removed:
                    continue
                for j in range(i + 1, len(remaining_indices)):
                    if j in corr_removed:
                        continue
                    if corr_matrix[i, j] > self.max_correlation:
                        # Remove the one with lower variance
                        var_i = variances[remaining_indices[i]]
                        var_j = variances[remaining_indices[j]]
                        remove_idx = j if var_i >= var_j else i
                        corr_removed.add(remove_idx)
            
            for local_idx in corr_removed:
                global_idx = remaining_indices[local_idx]
                to_remove.add(global_idx)
                self.removal_reasons_[global_idx] = "high_correlation"
            
            logger.debug(f"Removed {len(corr_removed)} highly correlated features")
        
        # Store results
        self.selected_indices_ = np.array([i for i in range(n_features) if i not in to_remove])
        self.removed_indices_ = np.array(list(to_remove))
        
        self.is_fitted = True
        logger.info(f"Selected {len(self.selected_indices_)} of {n_features} features")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to selected features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Reduced feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform")
        
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    @property
    def n_selected_(self) -> int:
        """Number of selected features."""
        return len(self.selected_indices_) if self.selected_indices_ is not None else 0
    
    @property
    def n_removed_(self) -> int:
        """Number of removed features."""
        return len(self.removed_indices_) if self.removed_indices_ is not None else 0
    
    def get_removal_summary(self) -> pd.DataFrame:
        """Get summary of removed features."""
        if not self.removal_reasons_:
            return pd.DataFrame()
        
        records = [
            {"feature_index": idx, "reason": reason}
            for idx, reason in self.removal_reasons_.items()
        ]
        return pd.DataFrame(records)