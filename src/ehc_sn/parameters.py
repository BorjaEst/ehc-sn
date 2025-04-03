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
        return torch.tensor(value, device=config.device)

    def parameters(self) -> snn.LIFRefracParameters:
        """Return the LIFRefrac parameters as a norse object."""
        lif = snn.LIFParameters(**self.model_dump(exclude={"rho_reset"}))
        rho_reset = torch.tensor(self.rho_reset, device=config.device)
        return snn.LIFRefracParameters(lif, rho_reset)


class Plasticity(BaseModel):
    """The plasticity settings."""

    model_config = ConfigDict(extra="forbid")
    gain: NonNegativeFloat = Field(default=1.00, description="Contribution of spikes to trace")
    tau: NonNegativeFloat = Field(default=0.01, gt=0, description="Inverse decay of spike trace")

    @field_serializer("*")
    @classmethod
    def to_tensor(cls, value):
        """Convert the LIF parameters to a tensor."""
        return torch.tensor(value, device=config.device)


class Synapses(BaseModel):
    """The synapse settings."""

    model_config = ConfigDict(extra="forbid")
    input_size: PositiveInt = Field(..., description="Size for the number of inputs")
    epsilon: NonNegativeFloat = Field(default=0.02, description="Probability of any connection")
    w_init: NonNegativeFloat = Field(default=0.1, description="Initial weight of the connections")
    w_min: float = Field(default=0.0, description="Lower bound on synaptic weights (should be < w_max)")
    w_max: float = Field(default=10.0, description="Upper bound on synaptic weight (should be > w_min)")
    learning_rate: NonNegativeFloat = Field(default=1e-3, lt=1, description="Learning rate (<<1)")
    ltp: Plasticity = Plasticity(tau=0.01)  # STDP parameters for Long-Term Potentiation
    ltd: Plasticity = Plasticity(tau=0.02)  # STDP parameters for Long-Term Depression

    @model_validator(mode="after")
    @classmethod
    def validate_weight_limits(cls, data: Any) -> Any:
        """Validate the synaptic weight limits."""
        if data.w_min >= data.w_max:
            raise ValueError("The minimum weight must be less than the maximum weight.")
        return data

    def make_mask(self, layer_size: int) -> torch.Tensor:
        """Return the mask for the synapses."""
        n, m = layer_size, self.input_size
        return torch.rand(n, m, device=config.device) < self.epsilon

    def state(self, layer_size: int) -> STDPState:
        """Return the STDP state for the synapses."""
        return STDPState(
            t_pre=torch.zeros(1, self.input_size, device=config.device),
            t_post=torch.zeros(1, layer_size, device=config.device),
        )

    def stdp_parameters(self) -> STDPParameters:
        """Return the STDP parameters for the synapses."""
        return STDPParameters(
            stdp_algorithm="additive",  # Fastest; all algorithms can overlap 
            a_post=torch.tensor(self.ltd.gain, device=config.device),
            a_pre=torch.tensor(self.ltp.gain, device=config.device),
            tau_post_inv=torch.tensor(1/self.ltd.tau, device=config.device),
            tau_pre_inv=torch.tensor(1/self.ltp.tau, device=config.device),
            eta_minus=self.learning_rate, eta_plus=self.learning_rate,
            w_min=self.w_min, w_max=self.w_max,
        ) # fmt: skip


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
