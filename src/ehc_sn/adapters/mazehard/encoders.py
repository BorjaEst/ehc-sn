"""MazeHard task-side decoder surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.tasks.maze_hard.contracts import MazeHardTaskInput


# =============================================================================
class MazeHardEncoder(ABC):
    """Abstract base class for MazeHard token encoders."""

    @abstractmethod
    def __init__(  # ----------------------------------------------------------
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Standard constructor signature for MazeHard encoders."""

    @abstractmethod
    def forward(  # -----------------------------------------------------------
        self,
        batch: MazeHardTaskInput,
    ) -> Any:
        """Encode a MazeHard task batch into token embeddings suitable for HRM input."""


# =============================================================================
__all__ = ["MazeHardEncoder"]
