
import numpy as np

from ehc_sn.spatial_scaffold import MemorySpatialScaffold
from tools.visualize_scaffold import visualize_scaffold


def main():
    """Main demonstration function."""
    # Create a memory spatial scaffold
    scaffold = MemorySpatialScaffold(dimensions=(100, 100), resolution=1.0)
    
    # Generate a simple trajectory
    trajectory = []
    for t in range(50):
        x = 50 + 30 * np.cos(t * 0.1)
        y = 50 + 30 * np.sin(t * 0.1)
        trajectory.append((x, y))
    
    # Simulate movement along the trajectory
    for position in trajectory:
        scaffold.update_position(position)
        scaffold.decay_memory(0.9)
    
    # Visualize the final state
    visualize_scaffold(scaffold)
    
    print("Memory Spatial Scaffold Demonstration")
    print(f"Current position: {scaffold.current_position}")
    print(f"Grid activations: {scaffold.grid_activations}")
    print(f"Number of active place cells: {np.sum(scaffold.place_activations > 0.3)}")

if __name__ == "__main__":
    main()