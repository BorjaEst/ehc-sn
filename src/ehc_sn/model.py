"""Module for the model class."""

from typing import Optional

import norse.torch as snn
import torch
from pydantic import Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings
from torch import nn


class EIParameters(BaseSettings):
    """Parameters for the EI model."""

    exc_size: PositiveInt = Field(
        100, le=1000, description="Number of excitatory neurons"
    )
    inh_size: PositiveInt = Field(
        100, le=1000, description="Number of inhibitory neurons"
    )
    rr_conn: PositiveFloat = Field(
        0.1, le=1.0, description="Recurrent connection probability"
    )
    ei_prop: PositiveFloat = Field(
        0.1, description="Weight proportion of excitatory to inhibitory"
    )
    ie_prop: PositiveFloat = Field(
        0.1, description="Weight proportion of inhibitory to excitatory"
    )


class EIModel(nn.Module):
    """Excitatory Inhibitory Network Model."""

    def __init__(self, p: Optional[EIParameters] = None) -> None:
        self.p = p = p or EIParameters(**{})
        super().__init__()
        self.excitatory = EIModel.excitatory_layer(p)
        self.inihibitory = snn.LIFCell()
        self.e = self.i = None, None
        self.reset()
        self.w_ei = p.ei_prop * torch.rand(p.exc_size, p.inh_size)
        self.w_ie = p.ie_prop * torch.rand(p.inh_size, p.exc_size)

    @staticmethod
    def excitatory_layer(p: EIParameters):
        """Create an excitatory layer."""
        rr_matrix = torch.rand(p.exc_size, p.exc_size) < p.rr_conn
        return snn.LIFRecurrentCell(
            input_size=p.exc_size,
            hidden_size=p.exc_size,
            input_weights=torch.eye(p.exc_size).type(torch.float32),
            recurrent_weights=rr_matrix.type(torch.float32),
        )

    def reset(self):
        """Reset the state of the model."""
        self.e = self.excitatory(torch.zeros(self.p.exc_size), None)
        self.i = self.inihibitory(torch.zeros(self.p.inh_size), None)

    @property
    def e_state(self):
        """State of excitatory neurons."""
        return self.e[1]  # State is the second element of the tuple

    @property
    def i_state(self):
        """State of inhibitory neurons."""
        return self.i[1]  # State is the second element of the tuple

    def step(self, x):
        """Step the model forward."""
        xe = x - self.i[0] @ self.w_ie  # feedforward inhibition
        self.e = self.excitatory(xe, self.e_state)
        xi = self.e[0] @ self.w_ei  # feedback excitation
        self.i = self.inihibitory(xi, self.i_state)
        return self.e[0]

    def forward(self, input_currents):
        """Forward pass."""
        return torch.stack([self.step(x) for x in input_currents])
