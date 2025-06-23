"""Custom type definitions for the entorhinal-hippocampal circuit library."""

from typing import Dict, List, Optional, Tuple, TypedDict

from torch import Tensor

# Types for spatial positions and coordinates
Position = Tuple[int, int]  # (row, column) or (y, x) position in grid
GridSize = Tuple[int, int]  # (height, width) of a grid
Direction = Tuple[int, int]  # (dy, dx) movement direction

# Types for grid map representation
ObstacleMap = Tensor  # 2D binary tensor with 1s for obstacles
GoalMap = Tensor  # 2D binary tensor with 1 at goal location

# Types for cognitive map representation
ValueMap = Tensor  # 2D tensor with values (e.g., distance to goal)
DistanceMap = ValueMap  # 2D tensor with distance metrics
CognitiveMap = Tensor  # 3D multichannel of ValueMaps, e.g., for different features

# EHC model component types
NeuralActivity = Tensor  # Neural firing patterns in a region
RegionalState = Tensor  # Hidden state of a region
Embedding = Tensor  # Feature representation in latent space
Weights = Tensor  # Connection weights between regions

# Type for circuit states across all regions
CircuitStates = Dict[str, Tensor]
