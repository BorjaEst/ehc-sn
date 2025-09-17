"""2D obstacle map generation using MiniGrid environments.

This module generates binary obstacle maps by extracting walls from MiniGrid environments.
Maps where 1 represents obstacles/walls and 0 represents free space, providing realistic
maze-like structures for spatial navigation testing.
"""

from typing import Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.utils.data import Dataset

from ehc_sn.constants.types import ObstacleMap
from ehc_sn.data.minigrid_maps import generate as minigrid_generate


# -------------------------------------------------------------------------------------------
class DataParams(BaseModel):
    """Parameters for obstacle map generation using MiniGrid environments."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Environment settings
    env_id: str = Field(default="MiniGrid-MultiRoom-N6-v0", description="MiniGrid environment ID to use")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducible generation")

    # Optional preprocessing
    invert_walls: bool = Field(default=False, description="Invert walls (1=free, 0=wall)")


# -------------------------------------------------------------------------------------------
class ObstacleMapDataset(Dataset):
    """Map-style Dataset that generates obstacle maps using MiniGrid environments.

    Extracts wall/obstacle patterns from MiniGrid environments, providing realistic
    maze-like structures. Uses deterministic seeding per sample for reproducibility.
    """

    def __init__(self, n_samples: int, params: DataParams):
        """Initialize obstacle map dataset.

        Args:
            n_samples: Number of samples in the dataset
            params: Generation parameters including environment ID
        """
        self.n_samples = n_samples
        self.params = params

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[ObstacleMap, ObstacleMap]:
        """Generate a single obstacle map sample deterministically.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input, target) obstacle maps for autoencoder training
        """
        # Deterministic per-sample seed (does not mutate global RNG state)
        sample_seed = self.params.seed + idx

        # Generate MiniGrid map and extract walls channel
        minigrid_tensor = minigrid_generate(seed=sample_seed, env_id=self.params.env_id)

        # Extract walls channel (channel 0)
        walls = minigrid_tensor[0]  # Shape: (H, W)

        # Apply inversion if requested
        if self.params.invert_walls:
            obstacle_map = 1.0 - walls  # Invert: 1=free, 0=wall
        else:
            obstacle_map = walls  # Keep: 1=wall, 0=free

        # For autoencoder, target is the same as input
        return obstacle_map, obstacle_map


# -------------------------------------------------------------------------------------------
class DataGenerator:

    def __init__(self, params: DataParams):
        self.params = params

    def __call__(self, n_samples: int) -> Dataset:
        return ObstacleMapDataset(n_samples, self.params)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test obstacle map generation using MiniGrid
    print("=== Testing ObstacleMapDataset with MiniGrid ===")

    # Test with different MiniGrid environments
    environments = [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-MultiRoom-N6-v0",
        "MiniGrid-MemoryS17Random-v0",
    ]

    for env_id in environments:
        print(f"\n--- Testing {env_id} ---")

        # Create parameters and generator
        params = DataParams(env_id=env_id, seed=42, invert_walls=False)
        generator = DataGenerator(params)

        # Create dataset
        dataset = generator(n_samples=10)
        print(f"Dataset length: {len(dataset)}")

        # Test individual sample generation
        map1, target1 = dataset[0]
        map2, target2 = dataset[0]  # Should be identical (deterministic)
        print(f"Sample shapes: map={map1.shape}, target={target1.shape}")
        print(f"Deterministic: {torch.allclose(map1, map2)}")
        print(f"Obstacle density: {torch.mean(map1).item():.3f}")

        # Show a sample obstacle map
        sample_map = map1.numpy()
        print(f"Sample obstacle map (1=obstacle, 0=free):")
        for row in sample_map[:8]:  # Show first 8 rows
            print("".join(["█" if cell > 0.5 else "·" for cell in row[:16]]))  # Show first 16 cols

        if len(sample_map) > 8:
            print("... (truncated)")

    # Test inversion feature
    print(f"\n--- Testing Wall Inversion ---")
    params_normal = DataParams(env_id="MiniGrid-Empty-8x8-v0", seed=42, invert_walls=False)
    params_inverted = DataParams(env_id="MiniGrid-Empty-8x8-v0", seed=42, invert_walls=True)

    generator_normal = DataGenerator(params_normal)
    generator_inverted = DataGenerator(params_inverted)

    dataset_normal = generator_normal(n_samples=1)
    dataset_inverted = generator_inverted(n_samples=1)

    map_normal, _ = dataset_normal[0]
    map_inverted, _ = dataset_inverted[0]

    print(f"Normal density: {torch.mean(map_normal).item():.3f}")
    print(f"Inverted density: {torch.mean(map_inverted).item():.3f}")
    print(f"Sum should be 1.0: {torch.mean(map_normal + map_inverted).item():.3f}")

    print("\n✓ MiniGrid-based obstacle map generation test completed!")
