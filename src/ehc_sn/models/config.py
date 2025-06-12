import os
from pathlib import Path
from typing import Dict, List, Optional

import tomli
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class Layer(BaseModel):
    """The layer settings."""

    model_config = ConfigDict(extra="forbid")
    description: Optional[str] = Field(None, description="Optional description of the layer")
    neuron: str = Field(..., description="Neuron type from available neurons")
    size: PositiveInt = Field(..., description="Number of neurons in the layer")


class Connection(BaseModel):
    """Parameters for synaptic connections."""

    model_config = ConfigDict(extra="forbid")
    target: str = Field(..., description="Target layer name")
    source: str = Field(..., description="Source layer name or input name")
    synapse: str = Field(..., description="Synapse type from available synapses")


class Network(BaseModel):
    """The network settings."""

    model_config = ConfigDict(extra="forbid")
    layers: Dict[str, Layer] = Field(default_factory=dict, description="The layers of the network")
    connections: List[Connection] = Field(default_factory=list, description="Synaptic connections between layers")


def load_model(file_path: Path) -> Network:
    # Open and read the TOML file
    with open(file_path, "rb") as f:
        config_data = tomli.load(f)

    # Validate and return the Network model using Pydantic v2
    return Network.model_validate(config_data)
