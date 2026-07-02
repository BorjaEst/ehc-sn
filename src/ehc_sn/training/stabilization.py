"""Training-stabilization helpers.

This module owns reusable training-machinery configs and wrappers that are not
specific to any one model family.  At present it exposes:

- :class:`TargetNetworkConfig`: EMA-lagged target-network config.
- :class:`TargetAdapterModule`: ``nn.Module`` wrapper that ensures the frozen
  target backbone is registered in Lightning's ``state_dict`` so checkpoints
  capture its weights.
"""

from __future__ import annotations

import copy

import torch
from pydantic import BaseModel, Field
from torch import nn


# =============================================================================
class TargetNetworkConfig(BaseModel, extra="forbid"):
    """Optional EMA-lagged target network for TD bootstrap stabilization.

    When enabled, the learner creates a frozen copy of the online backbone and
    updates it via Polyak averaging after every optimizer step:
    ``target ← (1 - τ) * target + τ * online``.

    This config is model-agnostic.  Any Lightning module that uses an
    ``nn.Module``-based backbone can import and reuse it.
    """

    enabled: bool = Field(
        default=False,
        description="Whether to use an EMA-lagged target backbone for "
        "value / Q bootstrap targets.",
    )
    tau: float = Field(
        default=0.005,
        ge=0.001,
        le=1.0,
        description="Polyak (EMA) update coefficient: "
        "target ← (1 - tau) * target + tau * online.",
    )


# =============================================================================
class TargetAdapterModule(nn.Module):
    """``nn.Module`` wrapper that owns a frozen EMA-lagged target backbone.

    Without this wrapper the raw deep copy is a plain Python attribute that
    Lightning ignores in ``state_dict`` / ``load_state_dict``.  Wrapping it in
    an ``nn.Module`` child guarantees checkpoint persistence.

    .. code-block:: python

        self._target_adapter = TargetAdapterModule(self.adapter)

    Attributes:
        target: The frozen backbone (``evaluation()``, ``requires_grad_(False)``).
    """

    def __init__(self, source: nn.Module) -> None:
        """Create a frozen EMA copy of *source*.

        Args:
            source: The online backbone whose architecture and current weights
                will be deep-copied for the target.
        """
        super().__init__()
        target = copy.deepcopy(source)
        target.requires_grad_(False)
        target.evaluation()
        self.target = target

    @property
    def config(self) -> object:
        """Delegate to the wrapped backbone's config if it exists."""
        return getattr(self.target, "config", None)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Forward through the frozen target backbone."""
        return self.target(*args, **kwargs)

    def ema_update(self, online: nn.Module, tau: float) -> None:
        """One-step Polyak update: ``target ← (1 - τ) * target + τ * online``.

        Args:
            online: The online backbone whose parameters are mixed into the
                target.
            tau: EMA coefficient (smaller → slower tracking).
        """
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), online.parameters()):
                tp.lerp_(op, tau)


# =============================================================================
__all__ = [
    "TargetNetworkConfig",
    "TargetAdapterModule",
]
