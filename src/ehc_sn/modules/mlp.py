import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.types import Device, Dtype
from ehc_sn.utils import find_multiple, trunc_normal_init_


# =================================================================================================
class MLPConfig(BaseModel, extra="forbid"):
    """Configuration for the MLP (feed-forward) block used in HRM transformer layers."""

    hidden_size: int = Field(
        ...,
        ge=32,
        frozen=True,
        description="Hidden size of the MLP block.",
    )
    expansion: float = Field(
        default=4.0,
        gt=1.0,
        description="Expansion factor for the MLP layers in the transformer blocks.",
    )


# =================================================================================================
class SwiGLU(nn.Module):
    """SwiGLU feed-forward (MLP) block with a gated activation.

    This implements the common SwiGLU pattern:

    - Project from ``hidden_size`` to an intermediate width and split into
      ``gate`` and ``up`` parts.
    - Apply ``silu`` to the gate and multiply elementwise with the up branch.
    - Project back to ``hidden_size``.

    The intermediate width is computed from ``expansion`` and rounded/aligned
    to a multiple of 256 for efficiency.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: MLPConfig, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize the SwiGLU block.

        Args:
            hidden_size: Input and output hidden dimension.
            expansion: Expansion multiplier used to compute the intermediate width.
                The internal width is derived as ``round(expansion * hidden_size * 2/3)``
                and then aligned to a multiple of 256.
        """
        super().__init__()
        self._config = config

        inter = find_multiple(round(config.expansion * config.hidden_size * 2 / 3), 256)
        self.gate_up_proj = nn.Linear(config.hidden_size, inter * 2, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:  # -------------------------------------------------------
        """Initialize projection weights with truncated normal matching legacy ``CastedLinear``.

        Std formulas (fan_in = input dimension of each projection):
            - ``gate_up_proj``: ``std = 1 / sqrt(hidden_size)``
            - ``down_proj``:    ``std = 1 / sqrt(inter)``  (recovered via ``in_features``)
        """
        trunc_normal_init_(self.gate_up_proj.weight, std=1.0 / math.sqrt(self._config.hidden_size))
        trunc_normal_init_(self.down_proj.weight, std=1.0 / math.sqrt(self.down_proj.in_features))

    @property
    def config(self) -> MLPConfig:
        """Configuration of the SwiGLU block."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor,
    ) -> Tensor:  # fmt: skip
        """Apply the SwiGLU transformation.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Tensor with the same shape and dtype as the input.

        Notes:
            - This module is shape-preserving in the last dimension.
            - The underlying projections are performed by
              :class:`~ehc_sn.modules.projections.CastedLinear`.
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# =================================================================================================
class MLP(torch.nn.Module):
    """Simple 2-layer MLP (legacy utility).

    This class supports either:
        - a single module (scalar ``in_dim``/``out_dim``), or
        - a list of independent modules (list ``in_dim``/``out_dim``)

    Notes:
        This is a legacy helper used by older components (e.g. autoencoder). New
        code should generally prefer :class:`SwiGLU` or a purpose-built module.
    """

    def __init__(
        self, in_dim, out_dim, activation=(torch.nn.functional.elu, None), hidden_dim=None, bias=(True, True)
    ):
        """Create the MLP.

        Args:
            in_dim: Input dimension or list of input dimensions.
            out_dim: Output dimension or list of output dimensions.
            activation: Tuple ``(hidden_activation, output_activation)``.
            hidden_dim: Hidden dimension(s). If None, uses mean of in/out.
            bias: Tuple indicating whether each layer uses bias.
        """
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(MLP, self).__init__()
        # Check if this network consists of module: are input and output dimensions lists? If not, make them (but remember it wasn't)
        if type(in_dim) is list:
            self.is_list = True
        else:
            in_dim = [in_dim]
            out_dim = [out_dim]
            self.is_list = False
        # Find number of modules
        self.N = len(in_dim)
        # Create weights (input->hidden, hidden->output) for each module
        self.w = torch.nn.ModuleList([])
        for n in range(self.N):
            # If number of hidden dimensions is not specified: mean of input and output
            if hidden_dim is None:
                hidden = int(np.mean([in_dim[n], out_dim[n]]))
            else:
                hidden = hidden_dim[n] if self.is_list else hidden_dim
            # Each module has two sets of weights: input->hidden and hidden->output
            self.w.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.Linear(in_dim[n], hidden, bias=bias[0]),
                        torch.nn.Linear(hidden, out_dim[n], bias=bias[1]),
                    ]
                )
            )
        # Copy activation function for hidden layer and output layer
        self.activation = activation
        # Initialise all weights
        with torch.no_grad():
            for from_layer in range(2):
                for n in range(self.N):
                    # Set weights to xavier initalisation
                    torch.nn.init.xavier_normal_(self.w[n][from_layer].weight)
                    # Set biases to 0
                    if bias[from_layer]:
                        self.w[n][from_layer].bias.fill_(0.0)

    def set_weights(self, from_layer, value):
        """Set the weights of one layer for all modules.

        Args:
            from_layer: Layer index (0=input->hidden, 1=hidden->output).
            value: Tensor or scalar, or list thereof when using modular mode.
        """
        # If single value is provided: copy it for each module
        if type(value) is not list:
            input_value = [value for n in range(self.N)]
        else:
            input_value = value
        # Run through all modules and set weights starting from requested layer to the specified value
        with torch.no_grad():
            # MLP is setup as follows: w[module][layer] is Linear object, w[module][layer].weight is Parameter object for linear weights, w[module][layer].weight.data is tensor of weight values
            for n in range(self.N):
                # If a tensor is provided: copy the tensor to the weights
                if type(input_value[n]) is torch.Tensor:
                    self.w[n][from_layer].weight.copy_(input_value[n])
                # If only a single value is provided: set that value everywhere
                else:
                    self.w[n][from_layer].weight.fill_(input_value[n])

    def forward(self, data):
        """Apply the MLP to the provided input(s)."""
        # Make input data into list, if this network doesn't consist of modules
        if self.is_list:
            input_data = data
        else:
            input_data = [data]
        # Run input through network for each module
        output = []
        for n in range(self.N):
            # Pass through first weights from input to hidden layer
            module_output = self.w[n][0](input_data[n])
            # Apply hidden layer activation
            if self.activation[0] is not None:
                module_output = self.activation[0](module_output)
            # Pass through second weights from hidden to output layer
            module_output = self.w[n][1](module_output)
            # Apply output layer activation
            if self.activation[1] is not None:
                module_output = self.activation[1](module_output)
            # Transpose output again to go back to column vectors instead of row vectors
            output.append(module_output)
        # If this network doesn't consist of modules: select output from first module to return
        if not self.is_list:
            output = output[0]
        # And return output
        return output
        return output
