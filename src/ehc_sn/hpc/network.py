from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ehc_sn.hpc.place_cells import PlaceCellsLayer


class HPC:
    # Implementation of Hippocampus (HPC) network
    # This class integrates place cell activity with memory functions
    # and interactions with grid cell input from MEC
    # See __init__.py for the public API documentation

    def __init__(
        self,
        place_cells: PlaceCellsLayer,
        noise_level: float = 0.1,
        memory_decay: float = 0.05,
    ):
        # Initialize the hippocampus network
        # Parameters:
        # - place_cells: Layer of place cells to be managed
        # - noise_level: Amount of noise added to place cell activities
        # - memory_decay: Rate at which stored memories decay over time

        self.place_cells = place_cells
        self.noise_level = noise_level
        self.memory_decay = memory_decay
        self.current_position = None
        self.memory_trace = []  # Store activity patterns as episodic memory

    def set_position(self, position: Tuple[float, float]) -> np.ndarray:
        # Update the activity of all place cells based on the current position
        # Place cell activity follows a Gaussian tuning curve around preferred locations
        # The biological basis is described in O'Keefe & Dostrovsky (1971)

        self.current_position = np.asarray(position)

        # Get baseline activity
        activity = self.place_cells.set_position(position)

        # Add noise to simulate biological variability
        # This models the stochastic nature of neural firing
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, size=activity.shape)
            activity = np.clip(activity + noise, 0, 1)  # Ensure values stay within [0,1]
            self.place_cells.activity = activity

        return activity

    def update_from_grid_cells(self, grid_activity: np.ndarray) -> np.ndarray:
        # Update place cell activity based on input from grid cells
        # In a biological system, this represents the feedforward connections
        # from grid cells in the MEC to place cells in the hippocampus
        # Reference: Solstad et al. (2006), From grid cells to place cells: A mathematical model

        # In a full implementation, this would use the learned weights between
        # grid cells and place cells to update place cell activity.
        # For simplicity, we just pass through the current position.
        if self.current_position is not None:
            return self.set_position(self.current_position)
        return self.place_cells.activity

    def store_memory(self) -> None:
        # Store current place cell activity pattern in episodic memory
        # This models the hippocampus's role in episodic memory formation
        # Reference: Buzsáki & Moser (2013), Memory, navigation and theta rhythm
        # in the hippocampal-entorhinal system

        if self.current_position is not None:
            # Make a copy of the current activity pattern
            memory = {
                "position": self.current_position.copy(),
                "activity": self.place_cells.activity.copy(),
            }
            self.memory_trace.append(memory)

    def recall_memory(self, pattern: np.ndarray) -> Optional[dict]:
        # Retrieve stored memory pattern most similar to the provided pattern
        # This implements a basic pattern completion mechanism, a key function
        # of the hippocampal CA3 region
        # Reference: Rolls (2013), The mechanisms for pattern completion and pattern
        # separation in the hippocampus

        if not self.memory_trace:
            return None

        max_overlap = -1
        best_match = None

        for memory in self.memory_trace:
            overlap = self.calculate_overlap(pattern, memory["activity"])
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = memory

        return best_match

    def calculate_overlap(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        # Calculate the overlap (cosine similarity) between two activity patterns
        # This is a key operation for pattern completion and memory retrieval
        # Neurobiological basis: Cell assemblies with overlapping activation patterns
        # represent similar memories or experiences

        # Cosine similarity: dot product divided by the product of magnitudes
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return np.dot(pattern1, pattern2) / (norm1 * norm2)

    def show(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        # Visualize place cell activity across the environment
        # Creates a spatial activity map showing the firing rate of place cells
        # at each location in the environment

        fig, ax = plt.subplots(figsize=figsize)
        self.place_cells.show(ax=ax, title="Hippocampal Place Cell Activity")

        # If we have a current position, mark it
        if self.current_position is not None:
            ax.plot(
                self.current_position[0],
                self.current_position[1],
                "o",
                color="white",
                markersize=10,
                markeredgecolor="black",
                label="Current Position",
            )
            ax.legend()

        plt.tight_layout()
        plt.show()
