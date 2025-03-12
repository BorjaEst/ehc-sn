"""Module for the model class."""

from abc import ABC, abstractmethod
from typing import Optional

import norse.torch as snn
import torch
from ehc_sn import config
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt, field_serializer
from pydantic_settings import BaseSettings
from torch import nn

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name


class PopulationsModel(BaseModel):
    """The populations settings of the EI model."""

    n_exc: PositiveInt = Field(default=80, description="Size of excitatory population E")
    n_inh: PositiveInt = Field(default=20, description="Size of inhibitory population I")


class LIFModel(BaseModel):
    """The LIF neuron settings of the EI model."""

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


class LIFRefracModel(BaseModel):
    """The LIF refractory settings of the EI model."""

    lif: LIFModel = LIFModel()
    rho_reset: NonNegativeFloat = Field(default=5, description="(ms) Refractory period")

    @field_serializer("rho_reset")
    @classmethod
    def to_tensor(cls, value):
        """Convert the refractory period to a tensor."""
        return torch.tensor(value).to(config.device)


class ConnectivityModel(BaseModel):
    """The connectivity settings of the EI model."""

    epsilon: NonNegativeFloat = Field(default=0.02, description="Probability of any connection (EE,EI,IE,II)")
    g_ee: NonNegativeFloat = Field(default=3, description="Weight of excitatory to excitatory synapses")
    g_ei: NonNegativeFloat = Field(default=3, description="Weight of excitatory to inhibitory synapses")
    g_ii: NonNegativeFloat = Field(default=30, description="Weight of inhibitory to inhibitory synapses")
    g_ie: NonNegativeFloat = Field(default=30, description="Weight of inhibitory to excitatory synapses")
    chi: NonNegativeFloat = Field(default=5, description="Potentiation factor of excitatory weights")


class PlasticityModel(BaseModel):
    """The plasticity settings of the EI model."""

    tau_stdp: NonNegativeFloat = Field(default=20, description="(ms) Decay of (pre and post) synaptic trace")
    eta: NonNegativeFloat = Field(default=1e-4, description="Learning rate")
    alpha: NonNegativeFloat = Field(default=0.12, description="Presynaptic offset")
    w_min: NonNegativeFloat = Field(default=0, description="Minimum inhibitory synaptic weight")
    w_max: NonNegativeFloat = Field(default=300, description="Maximum inhibitory synaptic weight")


class EIParameters(BaseSettings):
    """The settings of a Excitatory Inhibitory network model."""

    populations: PopulationsModel = PopulationsModel()
    excitatory: LIFRefracModel = LIFRefracModel()
    inhibitory: LIFRefracModel = LIFRefracModel()
    connectivity: ConnectivityModel = ConnectivityModel()
    plasticity: PlasticityModel = PlasticityModel()


class BaseEIModel(nn.Module, ABC):
    """Base model class for connectivity models."""

    def __init__(self, p: EIParameters) -> None:
        super().__init__()
        self.excitatory = BaseEIModel.layer_exc(p.excitatory)
        self.inihibitory = BaseEIModel.layer_inh(p.inhibitory)
        self.masks = BaseEIModel.create_masks(p.populations, p.connectivity)
        self.populations = p.populations
        self.chi = p.connectivity.chi  # TODO: how to use this?
        self.plasticity = p.plasticity
        self.e, self.i = None, None

    @staticmethod
    def layer_exc(p: LIFRefracModel) -> snn.LIFRefracCell:
        """Create an excitatory layer."""
        lif = snn.LIFParameters(**p.lif.model_dump())
        rho_reset = torch.tensor(p.rho_reset).to(config.device)
        parameters = snn.LIFRefracParameters(lif, rho_reset)
        return snn.LIFRefracCell(p=parameters).to(config.device)

    @staticmethod
    def layer_inh(p: LIFRefracModel) -> snn.LIFRefracCell:
        """Create an inhibitory layer."""
        lif = snn.LIFParameters(**p.lif.model_dump())
        rho_reset = torch.tensor(p.rho_reset).to(config.device)
        parameters = snn.LIFRefracParameters(lif, rho_reset)
        return snn.LIFRefracCell(p=parameters).to(config.device)

    @staticmethod
    def create_masks(pp: PopulationsModel, conn: ConnectivityModel) -> dict:
        """Create the masks for the model."""
        return {
            "ee": (conn.g_ee * torch.rand(pp.n_exc, pp.n_exc) < conn.epsilon).to(config.device),
            "ei": (conn.g_ei * torch.rand(pp.n_exc, pp.n_inh) < conn.epsilon).to(config.device),
            "ii": (conn.g_ii * torch.rand(pp.n_inh, pp.n_inh) < conn.epsilon).to(config.device),
            "ie": (conn.g_ie * torch.rand(pp.n_inh, pp.n_exc) < conn.epsilon).to(config.device),
        }

    def reset(self):
        """Reset the state of the model."""
        zeros = torch.zeros(self.populations.n_exc).to(config.device)
        self.e = self.excitatory(zeros, None)
        zeros = torch.zeros(self.populations.n_inh).to(config.device)
        self.i = self.inihibitory(zeros, None)

    @abstractmethod
    def step(self, x):
        """Execute a single step of the model with input x."""

    def forward(self, exc_current):
        """Forward pass."""
        return torch.stack([self.step(x) for x in exc_current])


class EIModel(BaseEIModel):
    """Excitatory Inhibitory Network Model."""

    def __init__(self, p: Optional[EIParameters] = None) -> None:
        super().__init__(p or EIParameters())
        self.w_ee = self.masks["ee"] * torch.ones(self.masks["ee"].shape).to(config.device)
        self.w_ei = self.masks["ei"] * torch.ones(self.masks["ei"].shape).to(config.device)
        self.w_ii = self.masks["ii"] * torch.ones(self.masks["ii"].shape).to(config.device)
        self.w_ie = self.masks["ie"] * torch.ones(self.masks["ie"].shape).to(config.device)
        self.reset()
        self.w_ii = nn.Parameter(self.w_ii)

    def step(self, x):  # TODO: Check if masks are required or STDP err does not prop
        xe = x - self.i[0] @ (self.masks["ie"] * self.w_ie)
        xe = xe + self.e[0] @ (self.masks["ee"] * self.w_ee)
        self.e = self.excitatory(xe, self.e[1])
        xi = self.e[0] @ (self.masks["ei"] * self.w_ei)
        xi = xi - self.i[0] @ (self.masks["ii"] * self.w_ii)
        self.i = self.inihibitory(xi, self.i[1])
        return self.e[0]
