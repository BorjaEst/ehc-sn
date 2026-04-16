"""MazeHard task-side decoder surfaces."""

from __future__ import annotations

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.tasks.maze_hard import MazeHardTaskOutput


# =============================================================================
class MazeHardTokenDecoder(nn.Module):
    """ """

    def __init__(  # ----------------------------------------------------------
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """ """
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(  # -----------------------------------------------------------
        self,
        outputs,  # TODO: specify type
        *,
        device: Device | None = None,
    ) -> MazeHardTaskOutput:
        """ """
        logits = self.lm_head(z_H[:, 1:])  # Strip CLS → (B, S, vocab_size)
        return


# =============================================================================
__all__ = ["MazeHardTokenDecoder"]
