from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt
from pydantic_settings import BaseSettings


class Neurons(BaseModel):
    """Parameters for a neuron population."""

    model_config = ConfigDict(extra="forbid")
    size: PositiveInt = Field(..., description="Number of neurons in the population")


class Synapses(BaseModel):
    """Parameters for synaptic connections."""

    model_config = ConfigDict(extra="forbid")
    size: PositiveInt = Field(..., description="Number of target neurons")
    w_init: float = Field(0.1, description="Initial weight value")


class Layer(BaseModel):
    """The layer settings."""

    model_config = ConfigDict(extra="forbid")
    neurons: Neurons = Field(..., description="Neuron population parameters")
    synapses: dict[str, Synapses] = Field({}, description="Synaptic connection parameters")


class Network(BaseModel):
    """The network settings."""

    model_config = ConfigDict(extra="forbid")
    layers: dict[str, Layer] = Field({}, description="The layers of the network")


class Model(BaseSettings):
    """The model settings."""

    model_config = ConfigDict(extra="forbid")
    hpc: Network = Field(..., description="Hippocampal network parameters")
    mec: Network = Field(..., description="Entorhinal-hippocampal complex network parameters")
