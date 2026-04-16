"""Navigation task-side decoder surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.tasks.navigation.contracts import NavigationTaskOutput


# =============================================================================
class NavigationDecoder(ABC):
    """Model-facing latent features consumed by navigation task decoders."""

    @abstractmethod
    def __init__(  # ---------------------------------------------------------
        self,
        observation_dim: int,
        latent_dim: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Initialize the navigation decoder heads over model-facing features."""

    @abstractmethod
    def forward(  # -----------------------------------------------------------
        self,
        outputs: Any,
    ) -> NavigationTaskOutput:
        """Decode model-facing navigation features into a navigation task output."""


# =============================================================================
__all__ = ["NavigationDecoder"]
