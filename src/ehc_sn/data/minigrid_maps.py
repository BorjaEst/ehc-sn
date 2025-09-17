"""MiniGrid map generation following standard dataset patterns.

Provides functional map generation compatible with existing datamodule architecture.
Uses deterministic seeding per sample for reproducible training data.
Environment ID determines output dimensions naturally.
"""

from typing import Tuple

import gymnasium as gym
import minigrid.envs  # noqa: F401
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.utils.data import Dataset


# -------------------------------------------------------------------------------------------
class MiniGridParams(BaseModel):
    """Parameters for MiniGrid map generation."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Generation settings
    env_id: str = Field(default="MiniGrid-MemoryS17Random-v0", description="MiniGrid environment ID")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducible generation")


# -------------------------------------------------------------------------------------------
class MiniGridDataset(Dataset):
    """Map-style Dataset that generates MiniGrid maps on-the-fly.

    This implementation follows PyTorch Lightning best practices:
    - Generates single samples in __getitem__ (not batches)
    - Uses deterministic seeding per sample for reproducibility
    - Scales well with large datasets (no precomputation)
    - Output size determined by environment specification
    """

    def __init__(self, n_samples: int, params: MiniGridParams):
        """Initialize MiniGrid dataset.

        Args:
            n_samples: Number of samples in the dataset
            params: Map generation parameters
        """
        self.n_samples = n_samples
        self.params = params

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Generate a single MiniGrid map sample deterministically.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input, target) map tensors for autoencoder training
        """
        # Deterministic per-sample seed (does not mutate global RNG state)
        sample_seed = self.params.seed + idx

        # Generate map using the functional interface
        map_tensor = generate(
            seed=sample_seed,
            env_id=self.params.env_id,
        )

        # For autoencoder, target is the same as input
        return map_tensor, map_tensor


# -------------------------------------------------------------------------------------------
class MiniGridGenerator:
    """Factory for creating MiniGrid datasets following PyTorch Lightning conventions.

    This generator creates Dataset objects (not batches) that work seamlessly with
    Lightning's DataModule and DataLoader systems.
    """

    def __init__(self, params: MiniGridParams):
        """Initialize MiniGrid generator with parameters.

        Args:
            params: Map generation parameters
        """
        self.params = params

    def __call__(self, n_samples: int) -> Dataset:
        """Create a dataset with the specified number of samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Dataset object that yields individual MiniGrid map samples
        """
        return MiniGridDataset(n_samples, self.params)


# -------------------------------------------------------------------------------------------
def generate(seed: int, env_id: str = "MiniGrid-MultiRoom-N4-v0") -> Tensor:
    """Generate a MiniGrid map tensor deterministically.

    Output dimensions are determined by the environment specification naturally.
    No artificial resizing is performed to respect the environment's intended scale.
    Always includes goal channel for consistent output format.

    Args:
        seed: Random seed for deterministic generation
        env_id: MiniGrid environment ID (determines output size)

    Returns:
        Float32 tensor of shape (2,H,W) where:
        - Channel 0: walls (1=wall, 0=free)
        - Channel 1: goal (1=goal, 0=other)

    Raises:
        ImportError: If minigrid package not available
        RuntimeError: If environment creation fails
    """
    # Create environment and reset with seed
    env = gym.make(env_id, render_mode=None)
    try:
        env.reset(seed=seed)

        # Extract grid information at native resolution
        layers = _extract_grid(env)

        return layers.contiguous()

    finally:
        env.close()


# -------------------------------------------------------------------------------------------
def _extract_grid(env) -> Tensor:
    """Extract wall and goal masks from MiniGrid environment using vectorized operations.

    Args:
        env: MiniGrid environment instance

    Returns:
        Tensor of shape (2, H, W) with [walls, goals] channels

    Raises:
        RuntimeError: If environment lacks grid attribute
    """
    grid = getattr(env.unwrapped, "grid", None)
    if grid is None:
        raise RuntimeError("MiniGrid environment missing grid attribute")

    # Get encoded grid representation
    encoded = grid.encode()  # Shape: (H, W, 3) - (object_type, color, state)

    # Convert to tensor for vectorized operations
    encoded_tensor = torch.from_numpy(encoded).long()  # Shape: (H, W, 3)

    # Extract object types (first channel)
    obj_types = encoded_tensor[:, :, 0]  # Shape: (H, W)

    # Create channel masks using vectorized operations
    walls = (obj_types == 2).float()  # Wall object type
    goals = (obj_types == 8).float()  # Goal object type

    # Stack channels: [walls, goals]
    layers = torch.stack([walls, goals], dim=0)  # Shape: (2, H, W)

    return layers


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test different environment sizes
    print("=== Testing Environment-First MiniGrid Generation ===")

    environments = [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-MemoryS17Random-v0",
        "MiniGrid-MultiRoom-N6-v0",
    ]

    for env_id in environments:
        print(f"\n--- Testing {env_id} ---")

        # Test generation function directly
        map_tensor = generate(seed=42, env_id=env_id)
        print(f"Native size: {map_tensor.shape}")

        # Check for deterministic generation
        map_tensor2 = generate(seed=42, env_id=env_id)
        print(f"Deterministic: {torch.equal(map_tensor, map_tensor2)}")

        # Show wall density
        from ehc_sn.utils.maps import wall_density

        density = wall_density(map_tensor[0:1])  # walls channel only
        print(f"Wall density: {density:.3f}")

    # Test dataset interface
    print(f"\n--- Testing Dataset Interface ---")
    params = MiniGridParams(env_id="MiniGrid-MemoryS13Random-v0")
    generator = MiniGridGenerator(params)
    dataset = generator(n_samples=3)

    print(f"Dataset length: {len(dataset)}")
    for i in range(2):
        map_tensor, target = dataset[i]
        print(f"Sample {i}: {map_tensor.shape}")
