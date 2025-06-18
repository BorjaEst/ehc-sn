"""Custom type definitions for the entorhinal-hippocampal circuit library."""

from typing import Dict, List, Optional, Tuple, TypedDict, Union

import torch

# Type for cognitive map representation
CognitiveMap = torch.Tensor


# Configuration type for region parameters
class RegionConfig(TypedDict):
    input_size: int
    hidden_size: int
    output_size: int
    activation: str
    dropout: float


# Configuration type for connection parameters
class ConnectionConfig(TypedDict):
    source: str
    target: str
    weight_type: str  # "trainable" or "fixed"
    initial_weight: Optional[torch.Tensor]
