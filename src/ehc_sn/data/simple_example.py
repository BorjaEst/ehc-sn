import math
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


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
if __name__ == "__main__":
    # Test SyntheticDataset
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
