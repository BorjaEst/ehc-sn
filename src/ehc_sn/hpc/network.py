import numpy as np
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt

from ehc_sn.hpc.place_cells import PlaceCellsLayer


class HPC:
    """Implementation of Hippocampus (HPC) network."""

    def __init__(
        self,
        place_cells: PlaceCellsLayer,
        noise_level: float = 0.1,
        memory_decay: float = 0.05,
    ):
        """Initialize the hippocampus network."""
        self.place_cells = place_cells
        self.noise_level = noise_level
        self.memory_decay = memory_decay
        self.current_position = None
        self.memory_trace = []  # Store activity patterns as episodic memory
        
    def set_position(self, position: Tuple[float, float]) -> np.ndarray:
        """
        Update the activity of all place cells based on the current position.
        
        Parameters
        ----------
        position : tuple
            (x, y) coordinates of current position
            
        Returns
        -------
        ndarray
            Updated place cell activities
        """
        self.current_position = np.asarray(position)
        
        # Get baseline activity
        activity = self.place_cells.set_position(position)
        
        # Add noise to simulate biological variability
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, size=activity.shape)
            activity = np.clip(activity + noise, 0, 1)  # Ensure values stay within [0,1]
            self.place_cells.activity = activity
            
        return activity
    
    def update_from_grid_cells(self, grid_activity: np.ndarray) -> np.ndarray:
        """
        Update place cell activity based on input from grid cells.
        
        Parameters
        ----------
        grid_activity : ndarray
            Activity pattern from grid cells
            
        Returns
        -------
        ndarray
            Updated place cell activities
        """
        # In a full implementation, this would use the learned weights between
        # grid cells and place cells to update place cell activity.
        # For simplicity, we just pass through the current position.
        if self.current_position is not None:
            return self.set_position(self.current_position)
        return self.place_cells.activity
    
    def store_memory(self) -> None:
        """
        Store current place cell activity pattern in episodic memory.
        """
        if self.current_position is not None:
            # Make a copy of the current activity pattern
            memory = {
                'position': self.current_position.copy(),
                'activity': self.place_cells.activity.copy()
            }
            self.memory_trace.append(memory)
    
    def recall_memory(self, pattern: np.ndarray) -> Optional[dict]:
        """
        Retrieve stored memory pattern most similar to the provided pattern.
        
        Parameters
        ----------
        pattern : ndarray
            Activity pattern to compare with stored memories
            
        Returns
        -------
        dict or None
            Retrieved memory with highest overlap, or None if no memories
        """
        if not self.memory_trace:
            return None
            
        max_overlap = -1
        best_match = None
        
        for memory in self.memory_trace:
            overlap = self.calculate_overlap(pattern, memory['activity'])
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = memory
                
        return best_match
    
    def calculate_overlap(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Calculate the overlap (cosine similarity) between two activity patterns.
        
        Parameters
        ----------
        pattern1 : ndarray
            First activity pattern
        pattern2 : ndarray
            Second activity pattern
            
        Returns
        -------
        float
            Overlap score between 0 and 1
        """
        # Cosine similarity: dot product divided by the product of magnitudes
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return np.dot(pattern1, pattern2) / (norm1 * norm2)
    
    def show(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Visualize place cell activity across the environment.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.place_cells.show(ax=ax, title="Hippocampal Place Cell Activity")
        
        # If we have a current position, mark it
        if self.current_position is not None:
            ax.plot(
                self.current_position[0],
                self.current_position[1],
                'o',
                color='white',
                markersize=10,
                markeredgecolor='black',
                label='Current Position'
            )
            ax.legend()
            
        plt.tight_layout()
        plt.show()
