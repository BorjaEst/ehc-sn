# /home/borja/projects/spatial-navigation/ehc-sn/place_cells.py

import numpy as np
from scipy.spatial.distance import cdist

class PlaceCellEnsemble:
    """
    Implementation of hippocampal place cell-like firing patterns.
    
    Place cells fire when an animal visits a specific location in an environment.
    """
    
    def __init__(self, num_cells=100, dimensions=(100, 100), field_width=10.0):
        """
        Initialize a place cell ensemble.
        
        Parameters:
        -----------
        num_cells : int
            Number of place cells in the ensemble
        dimensions : tuple
            The dimensions of the space (width, height)
        field_width : float
            The width of each place field (sigma of Gaussian)
        """
        self.num_cells = num_cells
        self.dimensions = dimensions
        self.field_width = field_width
        
        # Randomly assign preferred locations for each place cell
        self.centers = np.random.rand(num_cells, 2)
        self.centers[:, 0] *= dimensions[0]
        self.centers[:, 1] *= dimensions[1]
        
    def get_activations(self, position):
        """Get activation values for all place cells given a position."""
        position = np.array(position).reshape(1, 2)
        
        # Calculate distance from position to each cell's center
        distances = cdist(position, self.centers)[0]
        
        # Calculate activation using Gaussian function
        activations = np.exp(-0.5 * (distances / self.field_width)**2)
        
        return activations
    
    def get_population_vector(self, position):
        """Get the normalized population vector of place cell activations."""
        activations = self.get_activations(position)
        
        # Normalize the population vector
        norm = np.linalg.norm(activations)
        if norm > 0:
            activations = activations / norm
            
        return activations