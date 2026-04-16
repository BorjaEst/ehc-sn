"""MazeHard task-side decoder surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype

from ehc_sn.tasks.maze_hard.contracts import MazeHardTaskOutput


# =============================================================================
class MazeHardDecoder(ABC):
    """Abstract base class for MazeHard token decoders."""

    @abstractmethod
    def __init__(  # ----------------------------------------------------------
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Standard constructor signature for MazeHard decoders."""

    @abstractmethod
    def forward(  # -----------------------------------------------------------
        self,
        logits: Any,
    ) -> MazeHardTaskOutput:
        """Decode model output into a MazeHard task output."""


# =============================================================================
__all__ = ["MazeHardDecoder"]
