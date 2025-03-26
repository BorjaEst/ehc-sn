"""Module for the model class."""

from typing import Any

import norse.torch as snn
import torch
from ehc_sn import config
from norse.torch.functional import stdp
from pydantic import ConfigDict  # fmt: skip
from pydantic import field_serializer  # fmt: skip
from pydantic import model_validator  # fmt: skip
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name


class LIFRefracCell(BaseModel):
    """The LIF refractory settings of the model."""

    model_config = ConfigDict(extra="forbid")
    rho_reset: NonNegativeInt = Field(default=0, description="(steps) Refractory period")
    tau_syn_inv: NonNegativeFloat = Field(default=2.0, description="(1/ms) Inverse synaptic time constant")
    tau_mem_inv: NonNegativeFloat = Field(default=0.05, description="(1/ms) Inverse membrane time constant")
    v_leak: float = Field(default=-65, description="(mV) Leak potential")
    v_th: float = Field(default=-50, description="(mV) Threshold potential")
    v_reset: float = Field(default=-65, description="(mV) Reset potential")
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


class Layer(BaseModel):
    """The layer settings of the EI model."""

    model_config = ConfigDict(extra="forbid")
    population: PositiveInt = Field(..., description="Size of the population")
    input_size: PositiveInt = Field(..., description="Size of the input")
    epsilon: NonNegativeFloat = Field(default=0.2, description="Probability of any connection")
    init_weight: NonNegativeFloat = Field(default=20.0, description="Initial weight of the connections")
    cell: LIFRefracCell = LIFRefracCell()

    def cell_parameters(self) -> snn.LIFRefracParameters:
        """Return the LIFRefrac parameters as a norse object."""
        return self.cell.parameters()

    def spawn_connections(self) -> torch.Tensor:
        """Return the mask for the layer connections."""
        return torch.rand(self.population, self.input_size) < self.epsilon

    def spawn_weights(self) -> torch.Tensor:
        """Return the weights for the layer connections."""
        return self.init_weight * self.spawn_connections().to(config.device)


class Plasticity(BaseModel):
    """The plasticity settings of the EI model."""

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

    def parameters(self) -> stdp.STDPParameters:
        """Return the LIFRefrac parameters as a norse object."""
        return stdp.STDPParameters(**self.model_dump())


class Network(BaseModel):
    """The network settings of the EI model."""

    model_config = ConfigDict(extra="forbid")
    layers: dict[str, Layer] = Field({}, description="The layers of the network")
    plasticity: Plasticity = Plasticity()  # STDP parameters

    def stdp_state(self, l1: str, l2: str) -> stdp.STDPState:
        """Return the STDP state for the layer."""
        l1_pop, l2_pop = self.layers[l1].population, self.layers[l2].population
        return stdp.STDPState(
            t_pre=torch.zeros(1, l1_pop).to(config.device),
            t_post=torch.zeros(1, l2_pop).to(config.device),
        )
