""" """

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


# =============================================================================
@dataclass(frozen=True)
class MazeHardTaskOutput:
    """Task-owned MazeHard prediction payload."""

    task_logits: Tensor
