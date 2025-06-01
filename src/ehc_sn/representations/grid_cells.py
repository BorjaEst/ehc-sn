# /home/borja/projects/spatial-navigation/ehc-sn/grid_cells.py

import numpy as np

class GridCellModule:
    """
    Implementation of grid cell-like firing patterns.
    
    Grid cells fire in a hexagonal lattice pattern when an animal traverses 
    an environment, providing a metric for spatial navigation.
    """
    
    def __init__(self, scale=10.0, orientation=0.0, dimensions=(100, 100), resolution=1.0):
        """
        Initialize a grid cell module.
        
        Parameters:
        -----------
        scale : float
            The spacing between grid firing fields
        orientation : float
            The orientation of the grid pattern in radians
        """
        self.scale = scale
        self.orientation = orientation
        self.dimensions = dimensions
        self.resolution = resolution
        
        # Calculate the number of cells in each dimension
        self.grid_size = (
            int(dimensions[0] / resolution),
            int(dimensions[1] / resolution)
        )
        
        # Create the grid cell firing pattern
        self.pattern = self._create_grid_pattern()
        
    def _create_grid_pattern(self):
        """Create the hexagonal grid pattern."""
        x = np.arange(0, self.dimensions[0], self.resolution)
        y = np.arange(0, self.dimensions[1], self.resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Create three sets of parallel waves (60 degrees apart)
        k = 2 * np.pi / self.scale
        
        # First wave vector
        kx1 = k * np.cos(self.orientation)
        ky1 = k * np.sin(self.orientation)
        
        # Second wave vector (rotated 60 degrees)
        kx2 = k * np.cos(self.orientation + np.pi/3)
        ky2 = k * np.sin(self.orientation + np.pi/3)
        
        # Third wave vector (rotated 120 degrees)
        kx3 = k * np.cos(self.orientation + 2*np.pi/3)
        ky3 = k * np.sin(self.orientation + 2*np.pi/3)
        
        # Calculate grid pattern as the sum of three sinusoidal gratings
        wave1 = np.cos(xx * kx1 + yy * ky1)
        wave2 = np.cos(xx * kx2 + yy * ky2)
        wave3 = np.cos(xx * kx3 + yy * ky3)
        
        grid = (wave1 + wave2 + wave3) / 3.0
        
        # Normalize to 0-1 range
        grid = (grid + 1) / 2
        
        return grid.T  # Transpose to match the grid_size
    
    def get_activation(self, position):
        """Get the grid cell activation at a specific position."""
        cell_x = int(position[0] / self.resolution)
        cell_y = int(position[1] / self.resolution)
        
        # Ensure position is within bounds
        cell_x = max(0, min(cell_x, self.grid_size[0] - 1))
        cell_y = max(0, min(cell_y, self.grid_size[1] - 1))
        
        return self.pattern[cell_x, cell_y]