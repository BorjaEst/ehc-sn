"""
Loss functions for entorhinal-hippocampal circuit modeling.

This module provides specialized loss functions designed for training neural network
models of the entorhinal-hippocampal circuit. The losses implement biologically-inspired
constraints that encourage sparse, decorrelated representations similar to those
observed in hippocampal place cells and grid cells.

Key Features:
    - Gramian orthogonality loss for decorrelated representations
    - Homeostatic activity loss for stable firing rate regulation
    - Target-based L1 sparsity with ReLU thresholding
    - Scale-invariant formulations for stable training across dimensions

Classes:
    GramianOrthogonalityLoss: Promotes orthogonal latent representations
    HomeostaticActivityLoss: Maintains target firing rates with minimum activity
    TargetL1SparsityLoss: Encourages sparse coding with baseline activity tolerance

Examples:
    >>> # Create combined loss for autoencoder training
    >>> gramian_loss = GramianOrthogonalityLoss(center=True)
    >>> homeo_loss = HomeostaticActivityLoss(target_rate=0.1, min_active=8)
    >>>
    >>> # Apply to encoder activations
    >>> activations = encoder(inputs)  # Shape: (batch, latent_dim)
    >>> total_loss = gramian_loss(activations) + homeo_loss(activations)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GramianOrthogonalityLoss(nn.Module):
    """Gramian orthogonality loss for decorrelated sparse representations.

    This loss function encourages activations to be orthogonal by minimizing the
    difference between the normalized correlation matrix and the identity matrix.
    It promotes decorrelated representations where different neurons encode
    independent features, which is crucial for pattern separation in the
    entorhinal-hippocampal circuit.

    The loss implements the Frobenius norm of the correlation matrix deviation:
        L_gramian = mean((C - I)²) where C = (Z^T @ Z) / B

    This formulation ensures all pairwise correlations are minimized equally
    while preventing quadratic scaling with feature dimension that would
    otherwise dominate composite loss functions.

    Biological Motivation:
        Orthogonal representations maximize information capacity and enable
        effective pattern separation, similar to place cell firing patterns
        in the hippocampus where different cells encode distinct spatial locations.

    Attributes:
        center: Whether to center activations before computing correlations.
            Centering removes the mean, making the correlation matrix more stable
            and interpretable. Recommended for most applications.
        eps: Small value for numerical stability when normalizing by standard
            deviation. Prevents division by zero for constant activations.

    Examples:
        >>> # Standard orthogonality loss with centering
        >>> loss_fn = GramianOrthogonalityLoss(center=True)
        >>> activations = torch.randn(32, 64)  # (batch_size, features)
        >>> loss = loss_fn(activations)
        >>> print(f"Orthogonality loss: {loss.item():.4f}")

        >>> # Without centering for pre-normalized inputs
        >>> loss_fn = GramianOrthogonalityLoss(center=False)
        >>> loss = loss_fn(normalized_activations)
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
        """Compute Gramian orthogonality loss for decorrelated representations.

        This method calculates the mean squared deviation between the normalized
        correlation matrix of input activations and the identity matrix. The loss
        encourages orthogonal representations by penalizing correlations between
        different neurons/features.

        The computation involves:
        1. Optional mean centering of activations for stable correlations
        2. Standard deviation normalization for scale invariance
        3. Correlation matrix computation via normalized cross-products
        4. Mean squared error with identity matrix (prevents D^2 scaling)

        Mathematical formulation:
            L_gramian = mean((C - I)²) where C = (Z^T @ Z) / B
            Z is normalized activations, I is identity, B is batch size

        Args:
            z: Input activations tensor of shape (batch_size, num_features).
                Each row represents one sample's feature activations.

        Returns:
            Scalar loss tensor encouraging orthogonal feature representations.
            Typical values range from 0.0 (perfect orthogonality) to ~1.0.

        Raises:
            AssertionError: If input tensor is not 2-dimensional (batch, features).

        Note:
            Using mean instead of sum prevents the loss from scaling quadratically
            with feature dimension, ensuring stable training across different
            latent sizes and proper balance with other loss components.
        """
        assert z.dim() == 2, "z must be (B, D)"
        B, D = z.shape

        # Optional centering
        if self.center:
            zc = z - z.mean(dim=0, keepdim=True)
        else:
            zc = z

        # Standard deviation normalization (scale invariance)
        std = zc.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        zn = zc / std

        # Correlation-like matrix
        C = (zn.T @ zn) / float(B)
        I = torch.eye(D, device=z.device, dtype=z.dtype)
        return (C - I).pow(2).mean()


class HomeostaticActivityLoss(nn.Module):
    """Homeostatic activity loss for maintaining target firing rates.

    This loss function combines mean firing rate regulation with minimum activity
    constraints to ensure healthy population dynamics in neural representations.
    It mimics biological homeostatic mechanisms that maintain stable neural
    activity levels in the hippocampus and related brain regions.

    The loss implements a dual-constraint system observed in biological networks:
    1. Firing rate homeostasis: Maintains population activity near target levels
    2. Minimum activity constraint: Ensures sufficient neurons participate per sample

    Mathematical formulation:
        L_homeo = L_rate + 0.1 * L_min_activity
        L_rate = mean((mean_rate - target_rate)²)
        L_min_activity = mean(max(0, min_active - active_count)²)

    This dual constraint ensures representations are neither too sparse (losing
    information) nor too dense (reducing pattern separation capacity), mimicking
    the balanced activity observed in hippocampal place cell populations.

    Biological Motivation:
        Homeostatic mechanisms in the brain maintain stable firing rates despite
        varying inputs, ensuring robust neural computation. This loss encourages
        similar stability in artificial neural representations.

    Attributes:
        target_rate: Target mean firing rate for the neural population.
            Typical biological values range from 0.05 to 0.20 based on
            hippocampal recordings. Lower values promote sparser representations.
        min_active: Minimum number of neurons that should be active per sample.
            Ensures each input activates enough neurons for robust distributed
            representation. Should be much smaller than total feature count.
        eps: Small value for numerical stability in computations.

    Examples:
        >>> # Standard homeostatic loss for sparse representations
        >>> loss_fn = HomeostaticActivityLoss(target_rate=0.10, min_active=8)
        >>> activations = torch.rand(32, 64)  # (batch_size, features)
        >>> loss = loss_fn(activations)
        >>> print(f"Homeostatic loss: {loss.item():.4f}")

        >>> # Denser representations with higher target rate
        >>> loss_fn = HomeostaticActivityLoss(target_rate=0.20, min_active=16)
        >>> loss = loss_fn(activations)
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
        """Compute homeostatic activity loss for balanced neural populations.

        This method implements a dual-constraint loss function that maintains
        healthy neural population dynamics by regulating both mean firing rates
        and minimum activity levels. It combines two biologically-motivated
        constraints observed in hippocampal circuits.

        The loss consists of two components:
        1. Firing rate regulation: Penalizes deviation from target mean activity
        2. Minimum activity constraint: Ensures sufficient neurons are active

        Mathematical formulation:
            L_homeo = L_rate + 0.1 * L_min_activity
            L_rate = mean((mean_rate - target_rate)²)
            L_min_activity = mean(max(0, min_active - active_count)²)

        Args:
            z: Input activations tensor of shape (batch_size, num_features).
                Values typically in [0, 1] range representing neural firing rates.

        Returns:
            Scalar loss tensor encouraging target firing rates and minimum activity.
            Typical values range from 0.0 (perfect homeostasis) to ~0.1.

        Raises:
            AssertionError: If input tensor is not 2-dimensional (batch, features).

        Note:
            The 0.1 weighting factor for minimum activity maintains the original
            balance between rate regulation and activity constraints. Neurons are
            considered "active" if their activation exceeds 1e-6 threshold.
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
    excessive activation above the threshold.

    The ReLU-based approach is biologically motivated, modeling homeostatic
    mechanisms where neurons only incur metabolic costs when they exceed baseline
    firing rates. This encourages sparse coding while permitting necessary activity
    for information encoding, similar to energy-efficient coding in biological
    neural networks.

    Mathematical formulation:
        L_sparsity = max(0, mean(|z|) - target_rate)

    When target_rate = 0.0, this reduces to pure L1 sparsity penalty.
    When target_rate > 0.0, it allows controlled baseline activity levels.

    Biological Motivation:
        Sparse coding in biological systems conserves metabolic energy while
        maintaining representational capacity. The threshold mechanism reflects
        the fact that neurons have baseline firing rates below which no
        additional cost is incurred.

    Attributes:
        target_rate: Target mean absolute activation level. Only activations
            exceeding this threshold are penalized. When set to 0.0, all
            activations are penalized (pure L1). Typical values 0.01-0.1.

    Examples:
        >>> # Pure L1 sparsity (no baseline tolerance)
        >>> loss_fn = TargetL1SparsityLoss(target_rate=0.0)
        >>> activations = torch.rand(32, 64)  # (batch_size, features)
        >>> loss = loss_fn(activations)
        >>> print(f"Pure L1 loss: {loss.item():.4f}")

        >>> # Target-based sparsity (allows baseline activity)
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
        """Compute target-based L1 sparsity loss with ReLU thresholding.

        This method applies a biologically-motivated sparsity penalty that only
        penalizes activations exceeding a baseline threshold. The ReLU-based
        approach models metabolic costs where neurons incur penalties only when
        firing above baseline levels, allowing necessary activity for encoding.

        The loss computation:
        1. Calculate mean absolute activation across all features
        2. Apply ReLU threshold to allow baseline activity up to target_rate
        3. Return penalty only for excess activation above threshold

        Mathematical formulation:
            L_sparsity = max(0, mean(|z|) - target_rate)

        Args:
            z: Input activations tensor of shape (batch_size, num_features).
                Absolute values are computed internally for sparsity calculation.

        Returns:
            Scalar loss tensor promoting sparse representations above baseline.
            Returns 0.0 when mean activity is below target_rate, positive otherwise.

        Raises:
            AssertionError: If input tensor is not 2-dimensional (batch, features).

        Note:
            When target_rate=0.0, this becomes pure L1 sparsity penalty.
            For target_rate > 0.0, allows controlled baseline activity while
            still encouraging overall sparsity in the representation.
        """
        assert z.dim() == 2, "z must be (B, D)"

        # Compute mean absolute activation and apply ReLU threshold
        mean_abs_activation = z.abs().mean()
        return torch.relu(mean_abs_activation - self.target_rate)
