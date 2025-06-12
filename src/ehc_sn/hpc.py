from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ehc_sn import nn


class Network(nn.Network):
    def forward(self, x: Tensor) -> Tensor:
        x = self.layers["layer_1"](x)
        x = self.layers["layer_2"](x)
        return x
