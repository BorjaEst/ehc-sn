from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ehc_sn import nn, parameters


class HPC(nn.Network):
    def forward(self, x: Tensor) -> Tensor:
        x = self.layers["place_cells"](x)
        return x


class MEC(nn.Network):
    def forward(self, x: Tensor) -> Tensor:
        x = self.layers["grid_1"](x)
        x = self.layers["grid_2"](x)
        x = self.layers["grid_3"](x)
        return x


class EHC_SN(nn.Module):
    def __init__(self, p: parameters.Model, **kwargs):
        super().__init__(**kwargs)
        self.hpc = HPC(p.hpc, **kwargs)
        self.mec = MEC(p.mec, **kwargs)

    def forward(self, hpc_input: Tensor) -> Tensor:
        hpc_output = self.hpc(hpc_input)
        ehc_output = self.mec(hpc_output)
        return ehc_output
