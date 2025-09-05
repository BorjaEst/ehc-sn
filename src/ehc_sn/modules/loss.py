import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class GramianOrthogonalityLoss(nn.Module):
    """Gramian orthogonality loss for decorrelated sparse representations.

    This loss function encourages activations to be orthogonal by minimizing the
    difference between the normalized correlation matrix and the identity matrix.
    It promotes decorrelated representations where different neurons encode
    independent features, which is crucial for pattern separation in the
    entorhinal-hippocampal circuit.

    The loss is computed as:
        L_gramian = ||C - I||²_F
    where C is the normalized correlation matrix of activations and I is the
    identity matrix. The Frobenius norm ensures all pairwise correlations are
    minimized equally.

    Attributes:
        center (bool): Whether to center activations before computing correlations.
            Centering removes the mean, making the correlation matrix more stable.
        eps (float): Small value for numerical stability when normalizing by
            standard deviation. Prevents division by zero for constant activations.

    Example:
        >>> loss_fn = GramianOrthogonalityLoss(center=True, eps=1e-6)
        >>> activations = torch.randn(32, 64)  # (batch_size, features)
        >>> loss = loss_fn(activations)
        >>> print(f"Orthogonality loss: {loss.item():.4f}")
    """

    def __init__(self, center: bool = True, eps: float = 1e-6):
        """Initialize Gramian orthogonality loss.

        Args:
            center: Whether to center activations before normalization. Centering
                removes the mean activation, making correlations more stable and
                interpretable. Default: True.
            eps: Small epsilon value for numerical stability when computing standard
                deviation. Prevents division by zero for constant activations.
                Default: 1e-6.
        """
        super().__init__()
        self.center = center
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Gramian orthogonality loss.

        Computes the Frobenius norm of the difference between the normalized
        correlation matrix and the identity matrix. This encourages activations
        to be decorrelated and orthogonal.

        Args:
            z: Activations tensor of shape (B, D) where B is batch size and
                D is the number of features/neurons.

        Returns:
            Scalar tensor containing the Gramian orthogonality loss.

        Raises:
            AssertionError: If input tensor is not 2-dimensional.

        Note:
            The loss is scale-invariant due to normalization by standard deviation,
            making it robust to different activation magnitudes.
        """
        assert z.dim() == 2, "z must be (B, D)"
        B, D = z.shape

        # Center activations if requested
        if self.center:
            zc = z - z.mean(dim=0, keepdim=True)
        else:
            zc = z

        # Normalize by standard deviation
        std = zc.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        zn = zc / std

        # Compute correlation matrix and compare to identity
        C = (zn.T @ zn) / float(B)
        I = torch.eye(D, device=z.device, dtype=z.dtype)
        return (C - I).pow(2).sum()


class HomeostaticActivityLoss(nn.Module):
    """Homeostatic activity loss for maintaining target firing rates.

    This loss function combines mean firing rate regulation with minimum activity
    constraints to ensure healthy population dynamics in neural representations.
    It mimics biological homeostatic mechanisms that maintain stable neural
    activity levels in the hippocampus and related brain regions.

    The loss consists of two components:
    1. Firing rate loss: (mean_rate - target_rate)² to maintain target activity
    2. Minimum activity loss: Penalty when fewer than min_active neurons fire per sample

    This dual constraint ensures representations are neither too sparse (which could
    lose information) nor too dense (which reduces pattern separation capacity).

    Attributes:
        target_rate (float): Target mean firing rate for the neural population.
            Typical values range from 0.05 to 0.20 based on biological observations.
        min_active (int): Minimum number of neurons that should be active per sample.
            Ensures each input activates enough neurons for robust representation.
        eps (float): Small value for numerical stability in computations.

    Example:
        >>> loss_fn = HomeostaticActivityLoss(target_rate=0.10, min_active=8)
        >>> activations = torch.rand(32, 64)  # (batch_size, features)
        >>> loss = loss_fn(activations)
        >>> components = loss_fn.get_components(activations)
        >>> print(f"Total: {loss.item():.4f}, Rate: {components['firing_rate'].item():.4f}")
    """

    def __init__(self, target_rate: float = 0.10, min_active: int = 8, eps: float = 1e-6):
        """Initialize homeostatic activity loss.

        Args:
            target_rate: Target mean firing rate for the neural population.
                Should be between 0.0 and 1.0, with typical biological values
                around 0.05-0.20. Default: 0.10.
            min_active: Minimum number of neurons that should be active (> 1e-6)
                per sample. Ensures robust distributed representations. Default: 8.
            eps: Small epsilon value for numerical stability. Default: 1e-6.
        """
        super().__init__()
        self.target_rate = target_rate
        self.min_active = min_active
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute homeostatic activity loss.

        Combines firing rate regulation with minimum activity constraints to
        maintain healthy population dynamics. The total loss is:
        L_homeo = L_rate + 0.1 * L_min_activity

        Args:
            z: Activations tensor of shape (B, D) where B is batch size and
                D is the number of features/neurons.

        Returns:
            Scalar tensor containing the combined homeostatic loss.

        Raises:
            AssertionError: If input tensor is not 2-dimensional.

        Note:
            The 0.1 weighting factor for minimum activity loss preserves the
            original balance from the composite loss function.
        """
        assert z.dim() == 2, "z must be (B, D)"

        # Mean firing rate loss
        mean_rate = z.mean(dim=0)
        l_rate = (mean_rate - self.target_rate).pow(2).mean()

        # Minimum activity constraint
        active_per_sample = (z > 1e-6).float().sum(dim=1)
        min_activity_loss = torch.clamp(self.min_active - active_per_sample, min=0).pow(2).mean()

        # Combined homeostatic term (preserve original weighting)
        return l_rate + 0.1 * min_activity_loss


class TargetL1SparsityLoss(nn.Module):
    """Target-based L1 sparsity loss with ReLU thresholding.

    This loss function promotes sparsity by penalizing activations that exceed a target
    threshold using ReLU activation. It computes max(0, mean(|z|) - target_rate),
    which allows the network to maintain some baseline activity while discouraging
    excessive activation.

    The ReLU-based approach is biologically motivated - it models homeostatic
    mechanisms where neurons only incur metabolic costs when they exceed baseline
    firing rates. This encourages sparse coding while permitting necessary activity
    for information encoding.

    Attributes:
        target_rate (float): Target mean absolute activation level. Only activations
            exceeding this threshold are penalized. When set to 0.0, all activations
            are penalized (pure L1 penalty).

    Example:
        >>> # Pure L1 sparsity (target_rate=0.0)
        >>> loss_fn = TargetL1SparsityLoss(target_rate=0.0)
        >>> activations = torch.rand(32, 64)  # (batch_size, features)
        >>> loss = loss_fn(activations)
        >>> print(f"Pure L1 loss: {loss.item():.4f}")

        >>> # Target-based sparsity (only penalize excess activity)
        >>> loss_fn = TargetL1SparsityLoss(target_rate=0.05)
        >>> loss = loss_fn(activations)
        >>> print(f"Target-based L1 loss: {loss.item():.4f}")
    """

    def __init__(self, target_rate: float = 0.0):
        """Initialize target-based L1 sparsity loss.

        Args:
            target_rate: Target mean absolute activation level. Only penalizes
                activations that exceed this threshold. When set to 0.0, applies
                pure L1 penalty to all activations. Typical values for target-based
                loss are 0.01-0.1. Default: 0.0.
        """
        super().__init__()
        self.target_rate = target_rate

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute target-based L1 sparsity loss.

        Applies ReLU-thresholded L1 penalty: max(0, mean(|z|) - target_rate).
        This allows baseline activity up to target_rate while penalizing excess
        activation, promoting sparse representations with controlled activity levels.

        Args:
            z: Activations tensor of shape (B, D) where B is batch size and
                D is the number of features/neurons.

        Returns:
            Scalar tensor containing the target-based L1 sparsity loss.

        Raises:
            AssertionError: If input tensor is not 2-dimensional.

        Note:
            The ReLU thresholding ensures the loss is zero when mean activation
            is below target_rate, allowing the network to maintain necessary
            baseline activity for information encoding.
        """
        assert z.dim() == 2, "z must be (B, D)"

        # Compute mean absolute activation and apply ReLU threshold
        mean_abs_activation = z.abs().mean()
        return torch.relu(mean_abs_activation - self.target_rate)
