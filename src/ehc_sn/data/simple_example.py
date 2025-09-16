import collections.abc
import math
import random
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Type, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ehc_sn.modules import dfa, srtp
from ehc_sn.trainers.feed_forward import FeedbackTainer


# -------------------------------------------------------------------------------------------
class DataParams(BaseModel):
    """Parameters for synthetic dataset generation."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Data settings
    input_dim: int = Field(default=256, ge=8, le=512, description="Dimensionality of input features")
    latent_dim: int = Field(default=32, ge=8, le=512, description="Dimensionality of latent features")
    sparsity: float = Field(default=0.25, ge=0.1, le=0.9, description="Fraction of active latent dimensions per sample")
    noise_level: float = Field(default=0.05, ge=0.0, le=0.2, description="Noise level added to generated data")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducible generation")


# -------------------------------------------------------------------------------------------
class SyntheticDataset(Dataset):
    """Map-style Dataset that generates synthetic autoencoder data on-the-fly.

    This implementation follows PyTorch Lightning best practices:
    - Generates single samples in __getitem__ (not batches)
    - Uses deterministic seeding per sample for reproducibility
    - Scales well with large datasets (no precomputation)
    """

    def __init__(self, n_samples: int, params: DataParams):
        """Initialize synthetic dataset.

        Args:
            n_samples: Number of samples in the dataset
            params: Data generation parameters
        """
        self.n_samples = n_samples
        self.params = params

        # Precompute feature group mappings for efficiency
        self.features_per_latent = self.params.input_dim // self.params.latent_dim
        self.n_active_per_sample = max(1, int(self.params.latent_dim * self.params.sparsity))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Generate a single sample deterministically based on index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input, target) tensors for autoencoder training
        """
        # Create deterministic generator for this sample
        g = torch.Generator().manual_seed(self.params.seed + idx)

        # Generate sparse latent representation
        latent = torch.zeros(self.params.latent_dim)
        active_dims = torch.randperm(self.params.latent_dim, generator=g)[: self.n_active_per_sample]
        latent[active_dims] = torch.randn(self.n_active_per_sample, generator=g).abs()

        # Generate input features based on latent factors
        x = torch.zeros(self.params.input_dim)

        for lat_idx in range(self.params.latent_dim):
            # Define which input features this latent dimension controls
            start_feat = lat_idx * self.features_per_latent
            end_feat = min(start_feat + self.features_per_latent, self.params.input_dim)

            # Generate patterns for this feature group
            for feat_idx in range(start_feat, end_feat):
                pattern_weight = 0.8 + 0.4 * math.sin(feat_idx * math.pi / self.features_per_latent)
                x[feat_idx] = latent[lat_idx] * pattern_weight

        # Add noise and clamp to [0, 1]
        noise = self.params.noise_level * torch.randn(self.params.input_dim, generator=g)
        x = torch.clamp(x + noise, 0.0, 1.0)

        # For autoencoder, target is the same as input
        return x, x


# -------------------------------------------------------------------------------------------
class DataGenerator:
    """Factory for creating synthetic datasets following PyTorch Lightning conventions.

    This generator creates Dataset objects (not batches) that work seamlessly with
    Lightning's DataModule and DataLoader systems.
    """

    def __init__(self, params: DataParams):
        """Initialize data generator with parameters.

        Args:
            params: Data generation parameters
        """
        self.params = params

    def __call__(self, n_samples: int) -> Dataset:
        """Create a dataset with the specified number of samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Dataset object that yields individual samples
        """
        return SyntheticDataset(n_samples, self.params)


# -------------------------------------------------------------------------------------------
def create_dataset(n: int, input_dim: int, latent_dim: int, seed: int = 42) -> Tuple[Tensor, Tensor]:
    """Create batch dataset (legacy function, prefer SyntheticDataset for new code).

    Args:
        n: Number of samples
        input_dim: Input dimensionality
        latent_dim: Latent dimensionality
        seed: Random seed for reproducibility

    Returns:
        Tuple of (input, target) tensors
    """
    # Create sparse latent factors
    g = torch.Generator().manual_seed(seed)
    latent = torch.zeros(n, latent_dim)

    # Each sample activates only a subset of latent dimensions (sparsity)
    n_active_per_sample = max(1, latent_dim // 4)  # 25% sparsity in latent space

    for i in range(n):
        # Randomly select which latent dimensions to activate
        active_dims = torch.randperm(latent_dim, generator=g)[:n_active_per_sample]
        latent[i, active_dims] = torch.randn(n_active_per_sample, generator=g).abs()  # Positive activations

    # Create feature groups - each latent dimension controls a group of input features
    features_per_latent = input_dim // latent_dim
    x = torch.zeros(n, input_dim)

    for lat_idx in range(latent_dim):
        # Define which input features this latent dimension controls
        start_feat = lat_idx * features_per_latent
        end_feat = min(start_feat + features_per_latent, input_dim)

        # Generate patterns for this feature group
        for feat_idx in range(start_feat, end_feat):
            # Create distinct patterns with some correlation within groups
            pattern_weight = 0.8 + 0.4 * torch.sin(torch.tensor(feat_idx * math.pi / features_per_latent))
            x[:, feat_idx] = latent[:, lat_idx] * pattern_weight

    # Add small amount of noise and ensure values are in [0, 1]
    noise = 0.05 * torch.randn(x.shape, generator=g)
    x = torch.clamp(x + noise, 0.0, 1.0)

    return x, x


# -------------------------------------------------------------------------------------------
def make_figure(original: Tensor, reconstruction: Tensor, n_samples: int = 4) -> None:
    """Plot original batch data vs reconstructions side by side."""
    # Convert to numpy for plotting
    x = original.cpu().numpy()
    y = reconstruction.cpu().numpy()

    # Limit number of samples to plot
    n_samples = min(n_samples, original.shape[0])

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_samples):
        # Plot original data
        axes[0, i].plot(x[i], "b-", linewidth=1, alpha=0.7, label="Original")
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].set_ylim(0, 1)
        axes[0, i].grid(True, alpha=0.3)
        if i == 0:
            axes[0, i].set_ylabel("Original")

        # Plot reconstruction
        axes[1, i].plot(y[i], "r-", linewidth=1, alpha=0.7, label="Reconstruction")
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_xlabel("Feature Index")
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed")

    # Calculate and display MSE in title
    plt.suptitle(f"Data vs Reconstruction")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test both the new SyntheticDataset and legacy create_dataset function
    print("=== Testing SyntheticDataset (Recommended) ===")

    # Create parameters and generator
    params = DataParams(input_dim=64, latent_dim=16, sparsity=0.25, noise_level=0.05, seed=42)
    generator = DataGenerator(params)

    # Create dataset
    dataset = generator(n_samples=1000)
    print(f"Dataset length: {len(dataset)}")

    # Test individual sample generation
    x1, y1 = dataset[0]
    x2, y2 = dataset[0]  # Should be identical (deterministic)
    print(f"Sample shapes: x={x1.shape}, y={y1.shape}")
    print(f"Deterministic: {torch.allclose(x1, x2)}")

    # Create DataLoader and test batching
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    print(f"Number of batches: {len(dataloader)}")

    # Get first batch and visualize
    batch_x, batch_y = next(iter(dataloader))
    print(f"Batch shapes: x={batch_x.shape}, y={batch_y.shape}")

    # Visualize some samples
    make_figure(batch_x, batch_y, n_samples=4)

    print("\n=== Testing Legacy create_dataset Function ===")

    # Test legacy function for comparison
    x_legacy, y_legacy = create_dataset(100, 64, 16, seed=42)
    print(f"Legacy shapes: x={x_legacy.shape}, y={y_legacy.shape}")

    # Create TensorDataset for comparison
    legacy_dataset = TensorDataset(x_legacy, y_legacy)
    legacy_dataloader = DataLoader(legacy_dataset, batch_size=32, shuffle=False)

    # Compare first samples (should be very similar with same seed)
    legacy_batch_x, legacy_batch_y = next(iter(legacy_dataloader))
    print(f"Legacy batch shapes: x={legacy_batch_x.shape}, y={legacy_batch_y.shape}")

    # Check if first samples are similar (they should be with same seed)
    x_first_new, _ = dataset[0]
    x_first_legacy = legacy_batch_x[0]
    print(f"First samples similar: {torch.allclose(x_first_new, x_first_legacy, atol=1e-4)}")
