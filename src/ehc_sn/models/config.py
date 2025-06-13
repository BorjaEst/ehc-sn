import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import tomli
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class Layer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: Optional[str] = Field(None, description="Optional description of the layer's purpose")
    neuron: str = Field(..., description="Neuron type from available neurons (e.g., 'place_cell', 'grid_cell')")
    size: PositiveInt = Field(..., description="Number of neurons in the layer (must be positive)")


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = Field(..., description="Target layer name receiving the connection")
    source: str = Field(..., description="Source layer name or input name sending signals")
    synapse: str = Field(..., description="Synapse type from available synapses (e.g., 'excitatory', 'inhibitory')")


class Network(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layers: Dict[str, Layer] = Field(default_factory=dict, description="The layers of the network, keyed by name")
    connections: List[Connection] = Field(default_factory=list, description="Synaptic connections between layers")


def load_model(file_path: Union[str, Path]) -> Network:
    # Convert string path to Path object if needed
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    # Open and read the TOML file
    try:
        with open(file_path, "rb") as f:
            config_data = tomli.load(f)
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"Error parsing TOML file {file_path}: {e}")

    # Validate and return the Network model using Pydantic v2
    try:
        return Network.model_validate(config_data)
    except Exception as e:
        raise ValueError(f"Invalid network configuration in {file_path}: {e}")
