"""Module for the model class."""

from typing import Any

import norse.torch as snn
import torch
from ehc_sn import config
from norse.torch.functional.stdp import STDPParameters, STDPState
from pydantic import ConfigDict  # fmt: skip
from pydantic import field_serializer  # fmt: skip
from pydantic import model_validator  # fmt: skip
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name
# pylint: disable=too-many-function-args


class CellParameters(BaseModel):
    """The LIF refractory settings."""

    model_config = ConfigDict(extra="forbid")
    rho_reset: NonNegativeInt = Field(default=5, description="(steps) Refractory period")
    tau_syn_inv: NonNegativeFloat = Field(default=10.0, description="Inverse synaptic time constant")
    tau_mem_inv: NonNegativeFloat = Field(default=4.0, description="Inverse membrane time constant")
    v_leak: float = Field(default=-65, description="Leak potential")
    v_th: float = Field(default=-50, description="Threshold potential")
    v_reset: float = Field(default=-65, description="Reset potential")
    alpha: NonNegativeFloat = Field(default=0.5, description="Surrogate gradient computation parameter")

    @field_serializer("*")
    @classmethod
    def to_tensor(cls, value):
        """Convert the LIF parameters to a tensor."""
        return torch.tensor(value).to(config.device)

    def parameters(self) -> snn.LIFRefracParameters:
        """Return the LIFRefrac parameters as a norse object."""
        lif = snn.LIFParameters(**self.model_dump(exclude={"rho_reset"}))
        rho_reset = torch.tensor(self.rho_reset).to(config.device)
        return snn.LIFRefracParameters(lif, rho_reset)


class Plasticity(BaseModel):
    """The plasticity settings."""

    model_config = ConfigDict(extra="forbid")
    a_pre: NonNegativeFloat = Field(default=1.0, description="Contribution of presynaptic spikes to trace")
    a_post: NonNegativeFloat = Field(default=1.0, description="Contribution of postsynaptic spikes to trace")
    tau_pre_inv: NonNegativeFloat = Field(default=20, description="(1/s) Inverse decay of presynaptic spike trace")
    tau_post_inv: NonNegativeFloat = Field(default=20, description="(1/s) Inverse decay of postsynaptic spike trace")
    w_min: NonNegativeFloat = Field(default=0.0, description="Lower bound on synaptic weights (should be < w_max)")
    w_max: NonNegativeFloat = Field(default=1.0, description="Upper bound on synaptic weight (should be > w_min)")
    eta_plus: NonNegativeFloat = Field(default=1e-3, lt=1, description="Learning rate for potentiation (<<1)")
    eta_minus: NonNegativeFloat = Field(default=1e-3, lt=1, description="Learning rate for depression (<<1)")
    mu: NonNegativeFloat = Field(default=0.5, le=1, description="Exponent for multiplicative STDP (<= 1)")

    @model_validator(mode="after")
    @classmethod
    def validate_weight_limits(cls, data: Any) -> Any:
        """Validate the synaptic weight limits."""
        if data.w_min >= data.w_max:
            raise ValueError("The minimum weight must be less than the maximum weight.")
        return data

    @field_serializer("*")
    @classmethod
    def to_tensor(cls, value):
        """Convert the LIF parameters to a tensor."""
        return torch.tensor(value).to(config.device)

    def parameters(self) -> STDPParameters:
        """Return the LIFRefrac parameters as a norse object."""
        return STDPParameters(**self.model_dump())


class Synapses(BaseModel):
    """The synapse settings."""

    model_config = ConfigDict(extra="forbid")
    epsilon: NonNegativeFloat = Field(default=0.02, description="Probability of any connection")
    init_value: NonNegativeFloat = Field(default=1.0, description="Initial weight of the connections")
    input_size: PositiveInt = Field(..., description="Size of the input")
    stdp: Plasticity = Plasticity()  # STDP parameters

    def make_mask(self, layer_size: int) -> torch.Tensor:
        """Return the mask for the synapses."""
        mask = torch.rand(layer_size, self.input_size) < self.epsilon
        return mask.to(config.device)

    def state(self, layer_size: int) -> STDPState:
        """Return the STDP state for the synapses."""
        return STDPState(
            t_pre=torch.zeros(1, self.input_size).to(config.device),
            t_post=torch.zeros(1, layer_size).to(config.device),
        )


class Layer(BaseModel):
    """The layer settings."""

    model_config = ConfigDict(extra="forbid")
    population: PositiveInt = Field(..., description="Size of the population")
    cells: CellParameters = CellParameters()  # LIFRefrac parameters
    synapses: dict[str, Synapses] = Field({}, description="The synapses of the layer")


class Network(BaseModel):
    """The network settings."""

    model_config = ConfigDict(extra="forbid")
    layers: dict[str, Layer] = Field({}, description="The layers of the network")
