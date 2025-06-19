"""Custom type definitions for the entorhinal-hippocampal circuit library."""

from typing import Dict, List, Optional, Tuple, TypedDict

import torch

# Types for spatial positions and coordinates
Position = Tuple[int, int]  # (row, column) or (y, x) position in grid
GridSize = Tuple[int, int]  # (height, width) of a grid
Direction = Tuple[int, int]  # (dy, dx) movement direction

# Types for grid map representation
ObstacleMap = torch.Tensor  # 2D binary tensor with 1s for obstacles
GoalMap = torch.Tensor  # 2D binary tensor with 1 at goal location

# Types for cognitive map representation
ValueMap = torch.Tensor  # 2D tensor with values (e.g., distance to goal)
CognitiveMap = ValueMap  # 2D tensor representing spatial knowledge
DistanceMap = ValueMap  # 2D tensor with distance metrics

# EHC model component types
NeuralActivity = torch.Tensor  # Neural firing patterns in a region
RegionalState = torch.Tensor  # Hidden state of a region
Embedding = torch.Tensor  # Feature representation in latent space
Weights = torch.Tensor  # Connection weights between regions

# Type for circuit states across all regions
CircuitStates = Dict[str, torch.Tensor]
