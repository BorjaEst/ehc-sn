"""Parameter definitions for EHC spatial navigation models."""

from typing import Dict

import tomli
from pydantic import BaseModel, ConfigDict, Field

from ehc_sn.settings import config


class Synapse(BaseModel):
    """Parameters for a synapse type."""

    model_config = ConfigDict(extra="forbid")
    description: str = Field(..., description="Description of the synapse")
    w_init: float = Field(..., description="Initial weight value")
    w_max: float = Field(..., description="Maximum weight value for plasticity")
    w_min: float = Field(..., description="Minimum weight value for plasticity")
    learning_rate: float = Field(..., description="Learning rate for plasticity")


class Neuron(BaseModel):
    """Parameters for a neuron population."""

    model_config = ConfigDict(extra="forbid")
    description: str = Field(..., description="Description of the neuron")
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
parameters = Parameters.model_validate(config_data)


__all__ = ["Synapse", "Synapses", "Neuron", "Neurons", "Parameters", "parameters"]
