from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, jit

from ehc_sn import nn, parameters


class HPC(nn.Network):
    def forward(self, x: Tensor, mec: Tensor) -> Tensor:
        x = torch.cat([x, mec], dim=1)  # Concatenate inputs
        x = self.layers["place_cells"](x)
        return x

    @property
    def activation(self) -> Tensor:
        return torch.cat(
            [self.layers["place_cells"].neurons.activation],
            dim=1,
        )


class MEC(nn.Network):
    def forward(self, x: Tensor, hpc: Tensor) -> Tensor:
        x = torch.cat([x, hpc], dim=1)  # Concatenate inputs
        for task in [
            jit.fork(self.layers["grid_1"], x),
            jit.fork(self.layers["grid_1"], x),
            jit.fork(self.layers["grid_1"], x),
        ]:
            jit.wait(task)
        return self.activation

    @property
    def activation(self) -> Tensor:
        return torch.cat(
            [
                self.layers["grid_1"].neurons.activation,
                self.layers["grid_2"].neurons.activation,
                self.layers["grid_3"].neurons.activation,
            ],
            dim=1,
        )


class EHC_SN(nn.Module):
    def __init__(self, p: parameters.Model, **kwargs):
        super().__init__(**kwargs)
        self.hpc = HPC(p.hpc, **kwargs)
        self.mec = MEC(p.mec, **kwargs)

    def forward(self, hpc_input: Tensor, mec_input: Tensor) -> Tensor:
        for task in [
            jit.fork(self.hpc, hpc_input, self.mec.activation),
            jit.fork(self.mec, mec_input, self.hpc.activation),
        ]:
            jit.wait(task)
            return self.activation

    @property
    def activation(self) -> Tensor:
        return torch.cat(
            [self.hpc.activation, self.mec.activation],
            dim=1,
        )
