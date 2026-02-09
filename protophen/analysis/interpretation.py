"""
Model interpretation for ProToPhen.

This module provides tools for understanding model predictions:
- SHAP (SHapley Additive exPlanations) values
- Gradient-based feature importance
- Integrated gradients for attribution
- Attention weight analysis
- Feature ablation studies

References:
    Lundberg, S.M. and Lee, S.I., 2017. A unified approach to interpreting 
    model predictions. NeurIPS. https://arxiv.org/abs/1705.07874
    
    Sundararajan, M., Taly, A. and Yan, Q., 2017. Axiomatic attribution for 
    deep networks. ICML. https://arxiv.org/abs/1703.01365
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from protophen.utils.logging import logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InterpretationConfig:
    """Configuration for model interpretation."""
    
    # SHAP settings
    shap_n_samples: int = 100  # Background samples for SHAP
    shap_n_evals: int = 500  # Number of evaluations for kernel SHAP
    
    # Integrated gradients settings
    ig_n_steps: int = 50  # Number of interpolation steps
    ig_baseline: Literal["zero", "random", "mean"] = "zero"
    
    # Gradient settings
    gradient_method: Literal["vanilla", "smooth", "guided"] = "vanilla"
    smooth_grad_n_samples: int = 50
    smooth_grad_noise_std: float = 0.1
    
    # Feature ablation
    ablation_method: Literal["zero", "mean", "noise"] = "zero"
    
    # Device
    device: str = "cuda"
    
    # Batch size for interpretations
    batch_size: int = 32


# =============================================================================
# Base Interpreter
# =============================================================================

class BaseInterpreter(ABC):
    """Abstract base class for model interpreters."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InterpretationConfig] = None,
    ):
        """
        Initialise interpreter.
        
        Args:
            model: Model to interpret
            config: Interpretation configuration
        """
        self.model = model
        self.config = config or InterpretationConfig()
        self.device = torch.device(self.config.device)
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @abstractmethod
    def explain(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate explanations for inputs.
        
        Args:
            inputs: Input tensor
            targets: Optional target indices/values
            **kwargs: Method-specific arguments
            
        Returns:
            Dictionary containing attribution scores
        """
        pass


# =============================================================================
# Gradient-Based Interpretation
# =============================================================================

class GradientInterpreter(BaseInterpreter):
    """
    Gradient-based feature attribution.
    
    Computes the gradient of model output with respect to inputs
    to understand which input features are most influential.
    
    Supports:
    - Vanilla gradients
    - SmoothGrad (averaged gradients with noise)
    - Guided backpropagation
    
    Example:
        >>> interpreter = GradientInterpreter(model)
        >>> attributions = interpreter.explain(embeddings, task="cell_painting")
        >>> print(attributions["gradients"].shape)  # (batch, input_dim)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InterpretationConfig] = None,
    ):
        super().__init__(model, config)
        
        # Store original ReLU forward functions for guided backprop
        self._relu_hooks = []
    
    def _register_guided_hooks(self) -> None:
        """Register hooks for guided backpropagation."""
        def guided_relu_hook(module, grad_input, grad_output):
            # Only propagate positive gradients
            return (torch.clamp(grad_input[0], min=0.0),)
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(guided_relu_hook)
                self._relu_hooks.append(hook)
    
    def _remove_guided_hooks(self) -> None:
        """Remove guided backpropagation hooks."""
        for hook in self._relu_hooks:
            hook.remove()
        self._relu_hooks = []
    
    def _compute_gradients(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute gradients of output with respect to inputs.
        
        Args:
            inputs: Input embeddings
            task: Which task output to compute gradients for
            target_idx: Specific output index (None = sum all outputs)
            
        Returns:
            Gradient tensor of same shape as inputs
        """
        inputs = inputs.clone().detach().requires_grad_(True)
        inputs = inputs.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs, tasks=[task])
        
        if task not in outputs:
            raise ValueError(f"Task '{task}' not in model outputs")
        
        output = outputs[task]
        
        # Select target for gradient computation
        if target_idx is not None:
            if output.dim() > 1:
                target = output[:, target_idx].sum()
            else:
                target = output.sum()
        else:
            target = output.sum()
        
        # Backward pass
        target.backward()
        
        gradients = inputs.grad.clone()
        
        return gradients
    
    def _smooth_gradients(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
        n_samples: Optional[int] = None,
        noise_std: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute SmoothGrad (averaged gradients with noise).
        
        Args:
            inputs: Input embeddings
            task: Which task output to compute gradients for
            target_idx: Specific output index
            n_samples: Number of noisy samples
            noise_std: Standard deviation of noise
            
        Returns:
            Averaged gradient tensor
        """
        n_samples = n_samples or self.config.smooth_grad_n_samples
        noise_std = noise_std or self.config.smooth_grad_noise_std
        
        accumulated_grads = torch.zeros_like(inputs)
        
        for _ in range(n_samples):
            # Add noise to inputs
            noise = torch.randn_like(inputs) * noise_std
            noisy_inputs = inputs + noise
            
            # Compute gradients
            grads = self._compute_gradients(noisy_inputs, task, target_idx)
            accumulated_grads += grads
        
        return accumulated_grads / n_samples
    
    def explain(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
        method: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate gradient-based explanations.
        
        Args:
            inputs: Input embeddings of shape (batch, input_dim)
            task: Which task to explain
            target_idx: Specific output feature to explain
            method: Override config gradient method
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - gradients: Raw gradient attributions
                - importance: Absolute gradient importance
                - saliency: Input × gradient
        """
        method = method or self.config.gradient_method
        inputs = inputs.to(self.device)
        
        if method == "guided":
            self._register_guided_hooks()
        
        try:
            if method == "smooth":
                gradients = self._smooth_gradients(
                    inputs, task, target_idx, **kwargs
                )
            else:
                gradients = self._compute_gradients(inputs, task, target_idx)
        finally:
            if method == "guided":
                self._remove_guided_hooks()
        
        # Compute derived attributions
        importance = gradients.abs()
        saliency = inputs * gradients  # Input × gradient
        
        return {
            "gradients": gradients.cpu(),
            "importance": importance.cpu(),
            "saliency": saliency.cpu(),
        }
    
    def feature_importance(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        aggregate: Literal["mean", "max", "l2"] = "mean",
    ) -> np.ndarray:
        """
        Compute aggregated feature importance scores.
        
        Args:
            inputs: Input embeddings of shape (batch, input_dim)
            task: Which task to analyse
            aggregate: Aggregation method across batch
            
        Returns:
            Feature importance array of shape (input_dim,)
        """
        explanations = self.explain(inputs, task=task)
        importance = explanations["importance"].numpy()
        
        if aggregate == "mean":
            return importance.mean(axis=0)
        elif aggregate == "max":
            return importance.max(axis=0)
        elif aggregate == "l2":
            return np.sqrt((importance ** 2).sum(axis=0))
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")


# =============================================================================
# Integrated Gradients
# =============================================================================

class IntegratedGradientsInterpreter(BaseInterpreter):
    """
    Integrated Gradients attribution method.
    
    Computes attributions by integrating gradients along a path
    from a baseline to the input. This method satisfies the
    axioms of sensitivity and implementation invariance.
    
    Reference:
        Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks.
    
    Example:
        >>> interpreter = IntegratedGradientsInterpreter(model)
        >>> attributions = interpreter.explain(
        ...     embeddings,
        ...     task="cell_painting",
        ...     target_idx=0,
        ... )
        >>> print(attributions["attributions"].shape)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InterpretationConfig] = None,
    ):
        super().__init__(model, config)
    
    def _get_baseline(
        self,
        inputs: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Get baseline for integrated gradients.
        
        Args:
            inputs: Input tensor (for shape reference)
            method: Baseline method (zero, random, mean)
            
        Returns:
            Baseline tensor of same shape as inputs
        """
        method = method or self.config.ig_baseline
        
        if method == "zero":
            return torch.zeros_like(inputs)
        elif method == "random":
            return torch.randn_like(inputs) * 0.1
        elif method == "mean":
            return inputs.mean(dim=0, keepdim=True).expand_as(inputs)
        else:
            raise ValueError(f"Unknown baseline method: {method}")
    
    def _interpolate(
        self,
        baseline: torch.Tensor,
        inputs: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """
        Create interpolated inputs between baseline and input.
        
        Args:
            baseline: Baseline tensor
            inputs: Input tensor
            n_steps: Number of interpolation steps
            
        Returns:
            Interpolated tensor of shape (n_steps, *inputs.shape)
        """
        alphas = torch.linspace(0, 1, n_steps, device=inputs.device)
        
        # Shape: (n_steps, batch, features)
        interpolated = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * (
            inputs.unsqueeze(0) - baseline.unsqueeze(0)
        )
        
        return interpolated
    
    def explain(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute integrated gradients attributions.
        
        Args:
            inputs: Input embeddings of shape (batch, input_dim)
            task: Which task to explain
            target_idx: Specific output feature to explain (None = sum all)
            baseline: Custom baseline (uses config default if None)
            n_steps: Number of integration steps
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - attributions: Integrated gradients attributions
                - convergence_delta: Approximation error
        """
        n_steps = n_steps or self.config.ig_n_steps
        inputs = inputs.to(self.device)
        
        # Get baseline
        if baseline is None:
            baseline = self._get_baseline(inputs)
        baseline = baseline.to(self.device)
        
        # Create interpolated inputs
        interpolated = self._interpolate(baseline, inputs, n_steps)
        
        # Compute gradients at each interpolation step
        accumulated_grads = torch.zeros_like(inputs)
        
        for step in range(n_steps):
            step_inputs = interpolated[step].clone().detach().requires_grad_(True)
            
            # Forward pass
            outputs = self.model(step_inputs, tasks=[task])
            output = outputs[task]
            
            # Select target
            if target_idx is not None:
                if output.dim() > 1:
                    target = output[:, target_idx].sum()
                else:
                    target = output.sum()
            else:
                target = output.sum()
            
            # Backward pass
            target.backward()
            
            accumulated_grads += step_inputs.grad
        
        # Average gradients and multiply by (input - baseline)
        avg_grads = accumulated_grads / n_steps
        attributions = (inputs - baseline) * avg_grads
        
        # Compute convergence delta (should be close to output difference)
        with torch.no_grad():
            output_input = self.model(inputs, tasks=[task])[task]
            output_baseline = self.model(baseline, tasks=[task])[task]
            
            if target_idx is not None:
                output_diff = output_input[:, target_idx] - output_baseline[:, target_idx]
            else:
                output_diff = output_input.sum(dim=-1) - output_baseline.sum(dim=-1)
            
            attribution_sum = attributions.sum(dim=-1)
            convergence_delta = (output_diff - attribution_sum).abs()
        
        return {
            "attributions": attributions.cpu(),
            "convergence_delta": convergence_delta.cpu(),
            "baseline": baseline.cpu(),
        }
    
    def explain_batch(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute integrated gradients for a batch with progress bar.
        
        Args:
            inputs: Input embeddings
            task: Which task to explain
            target_idx: Specific output feature
            show_progress: Show progress bar
            
        Returns:
            Dictionary of attributions
        """
        batch_size = self.config.batch_size
        n_samples = inputs.shape[0]
        
        all_attributions = []
        all_deltas = []
        
        iterator = range(0, n_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Integrated Gradients")
        
        for i in iterator:
            batch = inputs[i:i + batch_size]
            result = self.explain(batch, task=task, target_idx=target_idx)
            all_attributions.append(result["attributions"])
            all_deltas.append(result["convergence_delta"])
        
        return {
            "attributions": torch.cat(all_attributions, dim=0),
            "convergence_delta": torch.cat(all_deltas, dim=0),
        }


# =============================================================================
# Feature Ablation
# =============================================================================

class FeatureAblationInterpreter(BaseInterpreter):
    """
    Feature ablation for understanding feature importance.
    
    Measures feature importance by ablating (removing/replacing)
    individual features or groups of features and observing
    the change in model output.
    
    Example:
        >>> interpreter = FeatureAblationInterpreter(model)
        >>> importance = interpreter.compute_importance(
        ...     embeddings,
        ...     task="cell_painting",
        ...     feature_groups={"esm2": (0, 1280), "physicochemical": (1280, 1719)},
        ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InterpretationConfig] = None,
    ):
        super().__init__(model, config)
    
    def _ablate_features(
        self,
        inputs: torch.Tensor,
        feature_indices: Union[int, List[int], slice],
        reference_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Ablate specified features.
        
        Args:
            inputs: Input tensor
            feature_indices: Which features to ablate
            reference_value: Value to replace with (uses config method if None)
            
        Returns:
            Ablated input tensor
        """
        ablated = inputs.clone()
        
        if reference_value is not None:
            ablated[:, feature_indices] = reference_value
        elif self.config.ablation_method == "zero":
            ablated[:, feature_indices] = 0.0
        elif self.config.ablation_method == "mean":
            ablated[:, feature_indices] = inputs[:, feature_indices].mean(dim=0)
        elif self.config.ablation_method == "noise":
            noise = torch.randn_like(ablated[:, feature_indices])
            ablated[:, feature_indices] = noise
        
        return ablated
    
    @torch.no_grad()
    def explain(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature ablation importance.
        
        Args:
            inputs: Input embeddings
            task: Which task to analyse
            target_idx: Specific output feature
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing feature importance scores
        """
        inputs = inputs.to(self.device)
        n_features = inputs.shape[-1]
        
        # Get baseline prediction
        baseline_output = self.model(inputs, tasks=[task])[task]
        
        if target_idx is not None:
            baseline_value = baseline_output[:, target_idx]
        else:
            baseline_value = baseline_output.mean(dim=-1)
        
        # Ablate each feature and measure change
        importance = torch.zeros(n_features, device=self.device)
        
        for i in tqdm(range(n_features), desc="Feature ablation"):
            ablated = self._ablate_features(inputs, i)
            ablated_output = self.model(ablated, tasks=[task])[task]
            
            if target_idx is not None:
                ablated_value = ablated_output[:, target_idx]
            else:
                ablated_value = ablated_output.mean(dim=-1)
            
            # Importance = change in output when feature is ablated
            importance[i] = (baseline_value - ablated_value).abs().mean()
        
        return {
            "importance": importance.cpu(),
            "baseline_output": baseline_value.mean().cpu(),
        }
    
    @torch.no_grad()
    def compute_group_importance(
        self,
        inputs: torch.Tensor,
        feature_groups: Dict[str, Tuple[int, int]],
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute importance for groups of features.
        
        Useful for understanding contribution of different
        embedding types (ESM-2 vs physicochemical).
        
        Args:
            inputs: Input embeddings
            task: Which task to analyse
            feature_groups: Dict mapping group names to (start, end) indices
            target_idx: Specific output feature
            
        Returns:
            Dictionary mapping group names to importance scores
        """
        inputs = inputs.to(self.device)
        
        # Get baseline prediction
        baseline_output = self.model(inputs, tasks=[task])[task]
        
        if target_idx is not None:
            baseline_value = baseline_output[:, target_idx].mean()
        else:
            baseline_value = baseline_output.mean()
        
        group_importance = {}
        
        for group_name, (start, end) in feature_groups.items():
            ablated = self._ablate_features(inputs, slice(start, end))
            ablated_output = self.model(ablated, tasks=[task])[task]
            
            if target_idx is not None:
                ablated_value = ablated_output[:, target_idx].mean()
            else:
                ablated_value = ablated_output.mean()
            
            group_importance[group_name] = (baseline_value - ablated_value).abs().item()
        
        return group_importance


# =============================================================================
# SHAP Interpretation
# =============================================================================

class SHAPInterpreter(BaseInterpreter):
    """
    SHAP (SHapley Additive exPlanations) interpreter.
    
    Computes SHAP values for model explanations using the KernelSHAP
    algorithm. SHAP values have nice theoretical properties including
    local accuracy, missingness, and consistency.
    
    Note: Requires the 'shap' package for full functionality.
    For simple use cases, a built-in approximation is provided.
    
    Example:
        >>> interpreter = SHAPInterpreter(model, background_data=train_embeddings)
        >>> shap_values = interpreter.explain(test_embeddings, task="cell_painting")
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        config: Optional[InterpretationConfig] = None,
    ):
        """
        Initialise SHAP interpreter.
        
        Args:
            model: Model to interpret
            background_data: Background dataset for SHAP (subset of training data)
            config: Interpretation configuration
        """
        super().__init__(model, config)
        
        self.background_data = background_data
        self._shap_explainer = None
    
    def _setup_shap_explainer(
        self,
        task: str,
        target_idx: Optional[int] = None,
    ) -> None:
        """Setup SHAP KernelExplainer."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP package required. Install with: pip install shap"
            )
        
        if self.background_data is None:
            raise ValueError("Background data required for SHAP explainer")
        
        # Subsample background data if needed
        n_background = min(self.config.shap_n_samples, len(self.background_data))
        indices = torch.randperm(len(self.background_data))[:n_background]
        background = self.background_data[indices].numpy()
        
        # Create prediction function for SHAP
        def predict_fn(x):
            x_tensor = torch.from_numpy(x).float().to(self.device)
            with torch.no_grad():
                outputs = self.model(x_tensor, tasks=[task])
            
            output = outputs[task].cpu().numpy()
            
            if target_idx is not None:
                return output[:, target_idx]
            return output
        
        self._shap_explainer = shap.KernelExplainer(predict_fn, background)
    
    def explain(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
        n_evals: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Compute SHAP values for inputs.
        
        Args:
            inputs: Input embeddings
            task: Which task to explain
            target_idx: Specific output feature to explain
            n_evals: Number of model evaluations for SHAP
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing SHAP values and expected value
        """
        n_evals = n_evals or self.config.shap_n_evals
        
        # Setup explainer if needed
        if self._shap_explainer is None:
            self._setup_shap_explainer(task, target_idx)
        
        # Convert to numpy
        inputs_np = inputs.cpu().numpy()
        
        # Compute SHAP values
        shap_values = self._shap_explainer.shap_values(
            inputs_np,
            nsamples=n_evals,
        )
        
        return {
            "shap_values": shap_values,
            "expected_value": self._shap_explainer.expected_value,
            "inputs": inputs_np,
        }
    
    def explain_approximate(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
        n_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute approximate SHAP values without the shap package.
        
        Uses sampling-based approximation of Shapley values.
        Less accurate but doesn't require additional dependencies.
        
        Args:
            inputs: Input embeddings
            task: Which task to explain
            target_idx: Specific output feature
            n_samples: Number of permutation samples
            
        Returns:
            Dictionary containing approximate SHAP values
        """
        inputs = inputs.to(self.device)
        batch_size, n_features = inputs.shape
        
        # Get baseline (mean of background or zeros)
        if self.background_data is not None:
            baseline = self.background_data.mean(dim=0).to(self.device)
        else:
            baseline = torch.zeros(n_features, device=self.device)
        
        # Initialise SHAP values
        shap_values = torch.zeros_like(inputs)
        
        with torch.no_grad():
            for sample_idx in range(batch_size):
                x = inputs[sample_idx]
                
                for _ in range(n_samples):
                    # Random permutation of features
                    perm = torch.randperm(n_features)
                    
                    # Compute marginal contributions
                    prev_output = None
                    z = baseline.clone()
                    
                    for i, feat_idx in enumerate(perm):
                        # Add feature to coalition
                        z[feat_idx] = x[feat_idx]
                        
                        # Compute model output
                        output = self.model(z.unsqueeze(0), tasks=[task])[task]
                        
                        if target_idx is not None:
                            current_output = output[0, target_idx]
                        else:
                            current_output = output[0].mean()
                        
                        # Marginal contribution
                        if prev_output is not None:
                            shap_values[sample_idx, feat_idx] += (
                                current_output - prev_output
                            )
                        
                        prev_output = current_output
                
                shap_values[sample_idx] /= n_samples
        
        return {
            "shap_values": shap_values.cpu(),
            "baseline": baseline.cpu(),
        }


# =============================================================================
# Attention Weight Analysis
# =============================================================================

class AttentionAnalyser:
    """
    Analyse attention weights from attention-based fusion models.
    
    Useful when using attention-based embedding fusion to understand
    how the model weighs different embedding types.
    
    Example:
        >>> analyser = AttentionAnalyser(model)
        >>> weights = analyser.get_fusion_weights(embeddings)
        >>> print(weights)  # {"esm2": 0.6, "physicochemical": 0.4}
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialise attention analyser.
        
        Args:
            model: Model with attention-based fusion
        """
        self.model = model
        self._attention_weights: Dict[str, torch.Tensor] = {}
        self._hooks = []
    
    def _register_hooks(self) -> None:
        """Register forward hooks to capture attention weights."""
        
        def attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    # MultiheadAttention returns (output, attention_weights)
                    self._attention_weights[name] = output[1].detach()
            return hook
        
        # Find attention modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(attention_hook(name))
                self._hooks.append(hook)
    
    def _remove_hooks(self) -> None:
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_weights = {}
    
    @torch.no_grad()
    def get_attention_weights(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from forward pass.
        
        Args:
            inputs: Input embeddings
            task: Task for forward pass
            
        Returns:
            Dictionary of attention weights by layer name
        """
        self._register_hooks()
        
        try:
            self.model.eval()
            _ = self.model(inputs, tasks=[task])
            weights = self._attention_weights.copy()
        finally:
            self._remove_hooks()
        
        return weights
    
    def get_fusion_weights(
        self,
        inputs: torch.Tensor,
        embedding_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get average fusion weights for embedding types.
        
        Args:
            inputs: Input embeddings
            embedding_names: Names of embedding types
            
        Returns:
            Dictionary mapping embedding names to average weights
        """
        attention_weights = self.get_attention_weights(inputs)
        
        if not attention_weights:
            logger.warning("No attention weights found in model")
            return {}
        
        # Average across all attention layers and samples
        all_weights = []
        for layer_weights in attention_weights.values():
            # Average over batch and heads
            avg_weights = layer_weights.mean(dim=(0, 1))
            all_weights.append(avg_weights)
        
        combined = torch.stack(all_weights).mean(dim=0)
        
        # Map to embedding names
        if embedding_names is None:
            embedding_names = [f"embedding_{i}" for i in range(len(combined))]
        
        return {
            name: combined[i].item()
            for i, name in enumerate(embedding_names)
            if i < len(combined)
        }


# =============================================================================
# Comprehensive Model Interpreter
# =============================================================================

class ModelInterpreter:
    """
    Unified interface for model interpretation.
    
    Combines multiple interpretation methods for comprehensive
    model understanding.
    
    Example:
        >>> interpreter = ModelInterpreter(model, train_embeddings)
        >>> 
        >>> # Get gradient-based importance
        >>> grad_results = interpreter.gradient_importance(test_embeddings)
        >>> 
        >>> # Get integrated gradients
        >>> ig_results = interpreter.integrated_gradients(test_embeddings)
        >>> 
        >>> # Get feature ablation importance
        >>> ablation_results = interpreter.feature_ablation(
        ...     test_embeddings,
        ...     feature_groups={"esm2": (0, 1280), "physchem": (1280, 1719)},
        ... )
        >>> 
        >>> # Generate comprehensive report
        >>> report = interpreter.generate_report(test_embeddings)
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        config: Optional[InterpretationConfig] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialise model interpreter.
        
        Args:
            model: Model to interpret
            background_data: Background data for SHAP
            config: Interpretation configuration
            feature_names: Names for input features
        """
        self.model = model
        self.config = config or InterpretationConfig()
        self.feature_names = feature_names
        self.background_data = background_data
        
        # Initialise sub-interpreters
        self._gradient_interpreter = GradientInterpreter(model, config)
        self._ig_interpreter = IntegratedGradientsInterpreter(model, config)
        self._ablation_interpreter = FeatureAblationInterpreter(model, config)
        
        if background_data is not None:
            self._shap_interpreter = SHAPInterpreter(model, background_data, config)
        else:
            self._shap_interpreter = None
        
        self._attention_analyser = AttentionAnalyser(model)
    
    def gradient_importance(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        method: str = "vanilla",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute gradient-based feature importance.
        
        Args:
            inputs: Input embeddings
            task: Which task to analyse
            method: Gradient method (vanilla, smooth, guided)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with gradient attributions and importance scores
        """
        results = self._gradient_interpreter.explain(
            inputs, task=task, method=method, **kwargs
        )
        
        # Compute aggregated importance
        results["feature_importance"] = results["importance"].mean(dim=0).numpy()
        
        return results
    
    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        target_idx: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute integrated gradients attributions.
        
        Args:
            inputs: Input embeddings
            task: Which task to analyse
            target_idx: Specific output feature
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with IG attributions
        """
        return self._ig_interpreter.explain(
            inputs, task=task, target_idx=target_idx, **kwargs
        )
    
    def feature_ablation(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        feature_groups: Optional[Dict[str, Tuple[int, int]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute feature ablation importance.
        
        Args:
            inputs: Input embeddings
            task: Which task to analyse
            feature_groups: Optional groups of features
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with ablation importance scores
        """
        if feature_groups is not None:
            return self._ablation_interpreter.compute_group_importance(
                inputs, task=task, feature_groups=feature_groups, **kwargs
            )
        else:
            return self._ablation_interpreter.explain(
                inputs, task=task, **kwargs
            )
    
    def shap_values(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        approximate: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values.
        
        Args:
            inputs: Input embeddings
            task: Which task to analyse
            approximate: Use built-in approximation instead of SHAP package
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with SHAP values
        """
        if self._shap_interpreter is None:
            if approximate:
                return SHAPInterpreter(
                    self.model, self.background_data, self.config
                ).explain_approximate(inputs, task=task, **kwargs)
            else:
                raise ValueError(
                    "Background data required for SHAP. "
                    "Use approximate=True for built-in approximation."
                )
        
        if approximate:
            return self._shap_interpreter.explain_approximate(
                inputs, task=task, **kwargs
            )
        else:
            return self._shap_interpreter.explain(inputs, task=task, **kwargs)
    
    def attention_weights(
        self,
        inputs: torch.Tensor,
        embedding_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get attention weights from model.
        
        Args:
            inputs: Input embeddings
            embedding_names: Names of embedding types
            
        Returns:
            Dictionary with attention weights
        """
        raw_weights = self._attention_analyser.get_attention_weights(inputs)
        fusion_weights = self._attention_analyser.get_fusion_weights(
            inputs, embedding_names
        )
        
        return {
            "raw_weights": raw_weights,
            "fusion_weights": fusion_weights,
        }
    
    def generate_report(
        self,
        inputs: torch.Tensor,
        task: str = "cell_painting",
        feature_groups: Optional[Dict[str, Tuple[int, int]]] = None,
        include_shap: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive interpretation report.
        
        Args:
            inputs: Input embeddings
            task: Which task to analyse
            feature_groups: Feature groups for ablation
            include_shap: Whether to compute SHAP values
            
        Returns:
            Comprehensive interpretation report
        """
        logger.info("Generating interpretation report...")
        
        report = {
            "task": task,
            "n_samples": inputs.shape[0],
            "n_features": inputs.shape[1],
        }
        
        # Gradient importance
        logger.info("Computing gradient importance...")
        report["gradient"] = self.gradient_importance(inputs, task=task)
        
        # Integrated gradients
        logger.info("Computing integrated gradients...")
        report["integrated_gradients"] = self.integrated_gradients(
            inputs, task=task
        )
        
        # Feature ablation
        logger.info("Computing feature ablation importance...")
        if feature_groups:
            report["ablation_groups"] = self.feature_ablation(
                inputs, task=task, feature_groups=feature_groups
            )
        report["ablation"] = self._ablation_interpreter.explain(inputs, task=task)
        
        # SHAP (optional, can be slow)
        if include_shap:
            logger.info("Computing SHAP values...")
            try:
                report["shap"] = self.shap_values(inputs, task=task)
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
                report["shap"] = None
        
        # Attention weights
        try:
            report["attention"] = self.attention_weights(inputs)
        except Exception:
            report["attention"] = None
        
        # Summary statistics
        report["summary"] = self._compute_summary(report)
        
        logger.info("Report generation complete")
        return report
    
    def _compute_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics from report."""
        summary = {}
        
        # Top features by gradient importance
        if "gradient" in report and "feature_importance" in report["gradient"]:
            importance = report["gradient"]["feature_importance"]
            top_indices = np.argsort(importance)[-10:][::-1]
            
            if self.feature_names:
                top_features = [
                    (self.feature_names[i], importance[i])
                    for i in top_indices
                ]
            else:
                top_features = [
                    (f"feature_{i}", importance[i])
                    for i in top_indices
                ]
            
            summary["top_features_gradient"] = top_features
        
        # Convergence check for integrated gradients
        if "integrated_gradients" in report:
            delta = report["integrated_gradients"]["convergence_delta"]
            summary["ig_convergence_mean"] = delta.mean().item()
            summary["ig_convergence_max"] = delta.max().item()
        
        # Ablation group importance
        if "ablation_groups" in report:
            summary["group_importance"] = report["ablation_groups"]
        
        return summary


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_feature_importance(
    model: nn.Module,
    inputs: torch.Tensor,
    task: str = "cell_painting",
    method: Literal["gradient", "ig", "ablation"] = "gradient",
    **kwargs,
) -> np.ndarray:
    """
    Compute feature importance using specified method.
    
    Args:
        model: Model to interpret
        inputs: Input embeddings
        task: Which task to analyse
        method: Interpretation method
        **kwargs: Additional arguments
        
    Returns:
        Feature importance array of shape (n_features,)
    """
    if method == "gradient":
        interpreter = GradientInterpreter(model)
        return interpreter.feature_importance(inputs, task=task, **kwargs)
    
    elif method == "ig":
        interpreter = IntegratedGradientsInterpreter(model)
        result = interpreter.explain(inputs, task=task, **kwargs)
        return result["attributions"].abs().mean(dim=0).numpy()
    
    elif method == "ablation":
        interpreter = FeatureAblationInterpreter(model)
        result = interpreter.explain(inputs, task=task, **kwargs)
        return result["importance"].numpy()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def explain_prediction(
    model: nn.Module,
    inputs: torch.Tensor,
    task: str = "cell_painting",
    target_idx: Optional[int] = None,
    background_data: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Generate explanation for a prediction.
    
    Convenience function combining multiple interpretation methods.
    
    Args:
        model: Model to interpret
        inputs: Input embeddings (single sample or batch)
        task: Which task to explain
        target_idx: Specific output feature
        background_data: Background data for SHAP
        
    Returns:
        Dictionary with multiple explanations
    """
    interpreter = ModelInterpreter(model, background_data=background_data)
    
    return {
        "gradient": interpreter.gradient_importance(inputs, task=task),
        "integrated_gradients": interpreter.integrated_gradients(
            inputs, task=task, target_idx=target_idx
        ),
    }


def get_embedding_contribution(
    model: nn.Module,
    inputs: torch.Tensor,
    embedding_ranges: Dict[str, Tuple[int, int]],
    task: str = "cell_painting",
) -> Dict[str, float]:
    """
    Get contribution of each embedding type to predictions.
    
    Args:
        model: Model to interpret
        inputs: Input embeddings
        embedding_ranges: Dict mapping embedding names to (start, end) indices
        task: Which task to analyse
        
    Returns:
        Dictionary mapping embedding names to contribution scores
    """
    interpreter = FeatureAblationInterpreter(model)
    return interpreter.compute_group_importance(
        inputs, task=task, feature_groups=embedding_ranges
    )