"""Shared head base abstractions.

This module contains only universal rollout-head wiring that is valid across
multiple head families. Family-specific logic lives in sibling internal
modules such as ``_token.py`` and ``_variational.py``.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel
from torch import nn

from ehc_sn.types import Batch


# =================================================================================================
class ControllerWithInitialState(Protocol):
    """Controller protocol used by :class:`BaseLossHead`."""

    def initial_state(self, batch_sample: Batch) -> Any:
        """Build the initial rollout carry for a batch sample."""


# =================================================================================================
class BaseLossHead[ControllerT: ControllerWithInitialState, ConfigT: BaseModel](
    nn.Module
): # fmt: skip
    """Minimal wiring base for all rollout-based loss heads.

    Owns exactly:

    * **Controller storage** — wraps the controller and exposes it via ``controller``.
    * **Config storage** — wraps the config and exposes it via ``config``.
    * **Carry initialization** — delegates to ``controller.initial_state`` via
      ``initial_carry``.

    Does **not** know about labels, loss functions, logits layout, or metrics.
    Those concerns belong to family-specific layers or concrete subclasses.

    Output contract:
        Concrete ``forward`` methods must return
        ``(step_output, new_carry, all_halted)``.
    """

    def __init__(self, controller: ControllerT, config: ConfigT) -> None:
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> ControllerT:
        """Return the wrapped controller."""
        return self._controller

    @property
    def config(self) -> ConfigT:
        """Return the head configuration."""
        return self._config

    def initial_carry(self, batch_sample: Batch) -> Any:
        """Initialize rollout carry/state from an example batch."""
        return self.controller.initial_state(batch_sample)


# =================================================================================================
__all__ = ["BaseLossHead", "ControllerWithInitialState"]
