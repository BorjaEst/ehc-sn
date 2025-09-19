"""Example usage of the obstacle maps module with MiniGrid environments."""

import torch
from torch.utils.data import DataLoader

from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.binary_map import BinaryMapFigure, BinaryMapParams


def main():
    """Demonstrate 2D obstacle cognitive map generation and visualization using MiniGrid."""

    # -----------------------------------------------------------------------------------
    # Setup parameters
    # -----------------------------------------------------------------------------------
    # Use MiniGrid environment for realistic maze generation
    obstacle_params = DataParams(env_id="MiniGrid-MultiRoom-N6-v0", seed=42)

    # Configure binary map visualization
    figure_params = BinaryMapParams(
        fig_width=12.0,
        fig_height=8.0,
        fig_dpi=100,
        title="MiniGrid Obstacle Maps - Realistic Maze Structures",
        grid=True,
        frame=True,
    )

    # -----------------------------------------------------------------------------------
    # Generate obstacle maps
    # -----------------------------------------------------------------------------------
    print("Generating obstacle maps using MiniGrid environments...")

    # Create generator and dataset
    generator = DataGenerator(obstacle_params)
    dataset = generator(n_samples=200)

    # Create DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Get a batch of obstacle maps
    batch = next(iter(dataloader))
    inputs, targets = batch

    density = torch.mean(inputs, dim=(1, 2)).cpu().numpy()
    print(f"Generated batch: {inputs.shape}")
    print(f"Obstacle density range: {density.min():.3f} - {density.max():.3f}")

    # -----------------------------------------------------------------------------------
    # Visualize with BinaryMapFigure
    # -----------------------------------------------------------------------------------
    print("Creating visualization...")

    # Create figure for individual map visualization
    figure = BinaryMapFigure(figure_params)

    # Show first few samples from the batch
    import matplotlib.pyplot as plt

    # Create subplot figure to show multiple maps
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("MiniGrid Obstacle Maps - Sample Variations", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < len(inputs):
            # Extract single map (H, W) from batch
            single_map = inputs[i]

            # Use BinaryMapFigure to plot on the provided axes
            figure.plot(single_map, ax=ax)
            ax.set_title(f"Sample {i+1}")
        else:
            ax.axis("off")

    plt.tight_layout()

    # Save the figure
    fig.savefig("/home/borja/temp/ehc-sn/obstacle_maps_demo.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to obstacle_maps_demo.png")

    # -----------------------------------------------------------------------------------
    # Show simplified API features
    # -----------------------------------------------------------------------------------
    print("\nMiniGrid-based obstacle maps module features:")
    print("✓ DataParams: Configurable MiniGrid environment ID and seed")
    print("✓ ObstacleMapDataset: Extracts walls from realistic MiniGrid environments")
    print("✓ DataGenerator: Simple factory pattern for dataset creation")
    print("✓ Integration with BinaryMapFigure for 2D visualization")
    print("✓ Deterministic generation with per-sample seeding")
    print("✓ Multiple environment types (Empty, MultiRoom, Memory, etc.)")
    print("✓ Optional wall inversion for different conventions")


if __name__ == "__main__":
    main()
