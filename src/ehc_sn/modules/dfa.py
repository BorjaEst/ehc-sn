from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn


# -------------------------------------------------------------------------------------------
class Linear(nn.Linear):
    def __init__(self, target_features: int, *args, device=None, dtype=None):
        super().__init__(*args, device=device, dtype=dtype)
        self.target_features = target_features
        fb_weights = torch.zeros(target_features, self.out_features, device=device, dtype=dtype)
        self.register_buffer("fb_weight", fb_weights)
        self.reset_weights()  # Initialize weights properly

    # -----------------------------------------------------------------------------------
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.detach())

    # -----------------------------------------------------------------------------------
    def feedback(self, error: Tensor, context: Tensor) -> None:
        delta = torch.matmul(error, self.fb_weight)
        torch.autograd.backward(context, delta)

    # -----------------------------------------------------------------------------------
    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fb_weight)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage example
    pass  # TODO
