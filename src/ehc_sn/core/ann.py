from typing import Any, Dict, Iterable, List, Optional, Tuple

from torch import Tensor, nn


class Layer(nn.Module):
    def __init__(self, synapses: nn.Module, activation: Optional[nn.Module] = None):
        super().__init__()
        self.register_buffer("currents", None)  # Starts without current values
        self.register_buffer("neurons", None)  # Starts without activation values
        self.synapses = synapses
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        self.currents = self.synapses(input)
        self.neurons = self.activation(self.currents)
        return self.neurons


if __name__ == "__main__":
    # Usage example
    pass  # TODO
