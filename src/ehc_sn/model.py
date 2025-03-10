"""Module for the model class."""

import norse.torch as snn
from torch import nn
import torch


class EIModel(nn.Module):
    def __init__(self, e_size=1000, i_size=100):
        super().__init__()
        self.excitatory = EIModel.excitatory_layer(e_size)
        self.inihibitory = snn.LIFCell()
        self.shape = (e_size, i_size)
        self.reset()
        self.w_ei = 0.1 * torch.rand(e_size, i_size)
        self.w_ie = 0.1 * torch.rand(i_size, e_size)

    @staticmethod
    def excitatory_layer(e_size, rr_conn=0.1):
        rr_matrix = torch.rand(e_size, e_size) < rr_conn
        return snn.LIFRecurrentCell(
            input_size=e_size,
            hidden_size=e_size,
            input_weights=torch.eye(e_size).type(torch.float32),
            recurrent_weights=rr_matrix.type(torch.float32),
        )

    def reset(self):
        self.e = self.excitatory(torch.zeros(self.shape[0]), None)
        self.i = self.inihibitory(torch.zeros(self.shape[1]), None)

    @property
    def e_state(self):
        return self.e[1]  # state of the last neuron

    @property
    def i_state(self):
        return self.i[1]  # state of the last neuron

    def step(self, x):
        xe = x - self.i[0] @ self.w_ie  # feedforward inhibition
        self.e = self.excitatory(xe, self.e_state)
        xi = self.e[0] @ self.w_ei  # feedback excitation
        self.i = self.inihibitory(xi, self.i_state)
        return self.e[0]

    def forward(self, input_currents):
        return torch.stack([self.step(x) for x in input_currents])


if __name__ == "__main__":
    model = EIModel(e_size=10)
    input_currents = torch.rand(100, 10)  # 100 steps, 10 neurons
    output = model(input_currents)
