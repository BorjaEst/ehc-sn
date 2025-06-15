"""Parameter definitions for EHC spatial navigation models."""

from typing import Dict, Optional

import tomli
from pydantic import BaseModel, ConfigDict, Field

from ehc_sn.settings import config


class Synapse(BaseModel):
    """Parameters for a synapse type."""

    model_config = ConfigDict(extra="forbid")
    description: str = Field(..., description="Description of the synapse")
    weight_max: Optional[float] = Field(None, description="Maximum weight value for plasticity")
    weight_min: Optional[float] = Field(None, description="Minimum weight value for plasticity")
    normalize: bool = Field(False, description="Whether to normalize weights")
    learning_rate: float = Field(..., description="Learning rate for plasticity")


class Neuron(BaseModel):
    """Parameters for a neuron population."""

    model_config = ConfigDict(extra="forbid")
    description: str = Field(..., description="Description of the neuron")
    activation_lim: Optional[float] = Field(None, description="Maximum activation value")
    activation_function: str = Field(..., description="Activation function used by the neuron")


class Parameters(BaseModel):
    """Parameters for the EHC spatial navigation model."""

    model_config = ConfigDict(extra="forbid")
    synapses: Dict[str, Synapse] = Field(..., description="Synapse configurations by type")
    neurons: Dict[str, Neuron] = Field(..., description="Neuron configurations by type")


# Open and read the TOML file
with open(config.parameters_file, "rb") as f:
    config_data = tomli.load(f)

# Use Pydantic v2's model_validate method
_parameters = Parameters.model_validate(config_data)
neurons = _parameters.neurons
synapses = _parameters.synapses


__all__ = ["Synapse", "Neuron", "synapses", "neurons"]
