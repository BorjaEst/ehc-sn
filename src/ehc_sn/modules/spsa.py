import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.utils.noise import rademacher_like


# -------------------------------------------------------------------------------------------
class Linear(nn.Linear):
    def __init__(self, *args, epsilon: float = 1e-2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = float(epsilon)
        self._deltas: Optional[List[Tensor]] = None

    def forward(self, input: Tensor) -> Tensor:
        # Detach to enforce locality (no upstream gradient)
        return super().forward(input.detach())

    def feedback(self, coefficient: Tensor) -> None:
        self.weight.grad = coefficient * self._deltas[0]
        if self.bias is not None:
            self.bias.grad = coefficient * self._deltas[1]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"epsilon={self.epsilon}"  # fmt: skip
        )

    def prepare_perturbation(self) -> None:
        self._deltas = [rademacher_like(self.weight)]
        if self.bias is not None:
            self._deltas.append(rademacher_like(self.bias))

    @torch.no_grad()
    def apply_perturbation(self, scale: float) -> None:
        self.weight.data.add_(scale * self.epsilon * self._deltas[0])
        if self.bias is not None:
            self.bias.data.add_(scale * self.epsilon * self._deltas[1])


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal sanity example using an optimizer
    torch.manual_seed(0)
    layer = Linear(4, 3, epsilon=1e-2)
    opt = torch.optim.SGD(layer.parameters(), lr=1e-2)

    x = torch.randn(8, 4)
    y = torch.randn(8, 3)

    # SPSA ± cycle
    opt.zero_grad()
    layer.prepare_perturbation()

    layer.apply_perturbation(+1.0)
    yhat_p = layer(x)
    Lp = nn.MSELoss(reduction="mean")(yhat_p, y)

    layer.apply_perturbation(-2.0)
    yhat_m = layer(x)
    Lm = nn.MSELoss(reduction="mean")(yhat_m, y)

    layer.apply_perturbation(+1.0)  # restore
    coefficient = (Lp - Lm) / (2.0 * layer.epsilon)
    layer.feedback(coefficient)  # fills .grad

    opt.step()
    print("SPSA step done via optimizer.")
