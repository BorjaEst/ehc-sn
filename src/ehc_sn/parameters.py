"""Module for the model class."""

from typing import Any

import norse.torch as snn
import torch
from ehc_sn import config
from norse.torch.functional import stdp
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt, field_serializer, model_validator
from pydantic import ConfigDict


# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name


class LIFCell(BaseModel):
    """The LIF neuron settings of the EI model."""

    model_config = ConfigDict(extra="forbid")
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

    def parameters(self) -> snn.LIFParameters:
        """Return the LIF parameters as a norse object."""
        return snn.LIFParameters(**self.model_dump())

    def cell(self) -> snn.LIFCell:
        """Return the LIF cell as a norse object."""
        return snn.LIFCell(p=self.parameters()).to(config.device)


class LIFRefracCell(LIFCell):
    """The LIF refractory settings of the EI model."""

    model_config = ConfigDict(extra="forbid")
    rho_reset: NonNegativeFloat = Field(default=5, description="(ms) Refractory period")

    def parameters(self) -> snn.LIFRefracParameters:
        """Return the LIFRefrac parameters as a norse object."""
        lif = snn.LIFParameters(**self.model_dump(exclude={"rho_reset"}))
        rho_reset = torch.tensor(self.rho_reset).to(config.device)
        return snn.LIFRefracParameters(lif, rho_reset)

    def cell(self) -> snn.LIFRefracCell:
        """Return the LIFRefrac cell as a norse object."""
        return snn.LIFRefracCell(p=self.parameters()).to(config.device)


class Layer(BaseModel):
    """The layer settings of the EI model."""

    model_config = ConfigDict(extra="forbid")
    population: PositiveInt = Field(default=80, description="Size of the population")
    cell: LIFRefracCell = LIFRefracCell()


class Synapses(BaseModel):
    """The connectivity settings of the EI model."""

    model_config = ConfigDict(extra="forbid")
    epsilon: NonNegativeFloat = Field(default=0.02, description="Probability of any connection")
    w: dict[str, NonNegativeFloat] = Field({}, description="Initial synaptic weights to layer (nS)")


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
    synapses: dict[str, Synapses] = Field({}, description="The synaptic connections between layers")
    plasticity: Plasticity = Plasticity()  # STDP parameters

    def mask(self, l1: str, l2: str) -> torch.Tensor:
        """Return the mask for the layer."""
        l1_pop, l2_pop = self.layers[l1].population, self.layers[l2].population
        return (torch.rand(l1_pop, l2_pop) < self.synapses[l1].epsilon).T.to(config.device)

    def weights(self, l1: str, l2: str) -> torch.Tensor:
        """Return the weights for the layer."""
        l1_pop, l2_pop = self.layers[l1].population, self.layers[l2].population
        return self.synapses[l1].w[l2] * torch.ones(l1_pop, l2_pop).T.to(config.device)

    def stdp_state(self, l1: str, l2: str) -> stdp.STDPState:
        """Return the STDP state for the layer."""
        l1_pop, l2_pop = self.layers[l1].population, self.layers[l2].population
        return stdp.STDPState(
            t_pre=torch.zeros(1, l1_pop).to(config.device),
            t_post=torch.zeros(1, l2_pop).to(config.device),
        )
