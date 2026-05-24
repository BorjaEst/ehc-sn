"""Arena task-side decoder surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import device as Device
from torch import dtype as Dtype

from ehc_sn.tasks.arena.contracts import ArenaTaskOutput


# =============================================================================
class ArenaDecoder(ABC):
    """Model-facing latent features consumed by arena task decoders."""

    @abstractmethod
    def __init__(  # ----------------------------------------------------------
        self,
        observation_dim: int,
        latent_dim: int,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Initialize the arena decoder heads over model-facing features."""

    @abstractmethod
    def forward(
        self,
        outputs: Any,
    ) -> ArenaTaskOutput:
        """Decode model-facing features into an arena task output."""


# =============================================================================
__all__ = ["ArenaDecoder"]
