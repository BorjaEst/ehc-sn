"""Arena task-side encoder surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import device as Device
from torch import dtype as Dtype

from ehc_sn.tasks.arena.contracts import ArenaTaskInput


# =============================================================================
class ArenaEncoder(ABC):
    """Model-facing latent features produced by arena task encoders."""

    @abstractmethod
    def __init__(
        self,
        observation_dim: int,
        feature_dim: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Initialize the arena encoder over task inputs."""

    @abstractmethod
    def forward(
        self,
        batch: ArenaTaskInput,
    ) -> Any:
        """Encode a batch of arena task inputs into model-facing features."""


# =============================================================================
__all__ = ["ArenaEncoder"]
