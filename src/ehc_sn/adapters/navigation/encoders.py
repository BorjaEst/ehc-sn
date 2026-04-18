"""Navigation task-side encoder surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.autoencoder import AutoencoderSettings
from ehc_sn.tasks.navigation.contracts import NavigationTaskInput


# =============================================================================
class NavigationEncoder(ABC):
    """Model-facing latent features produced by navigation task encoders."""

    @abstractmethod
    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        feature_dim: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Initialize the navigation decoder heads over model-facing features."""

    @abstractmethod
    def forward(  # -----------------------------------------------------------
        self,
        batch: NavigationTaskInput,
    ) -> Any:
        """Encode a batch of navigation task inputs into model-facing features."""


# =============================================================================
__all__ = ["NavigationEncoder"]
