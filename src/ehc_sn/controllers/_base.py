"""Shared controller bases and rollout-state helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from pydantic import BaseModel
from torch import Tensor

from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class RolloutBackbone[ModelState, ModelOutput](Protocol):
    """Protocol for the backbone model wrapped by a rollout controller."""

    def init_state(  # ---------------------------------------------------------
        self,
        batch_size: int,
    ) -> ModelState: ...

    def reset_state(  # --------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: ModelState,
    ) -> ModelState: ...

    def __call__(  # -----------------------------------------------------------
        self,
        batch: Batch,
        state: ModelState | None = None,
    ) -> tuple[ModelOutput, ModelState]: ...


# =============================================================================
@dataclass
class RolloutState[ModelState](DetachMixin):
    """Per-slot rollout state carried across controller steps.

    Attributes:
        model_state: Backbone recurrent state; shape and type are backbone-specific.
        steps: Per-slot step counters of shape ``(B,)``; reset to 0 on halt.
        halted: Per-slot done/reset flag of shape ``(B,)``; ``True`` means the slot
            halted on the *previous* step and will be refreshed at the start of the
            *next* step.
        data: Per-slot input/label buffers (for example ``"observation"``,
            ``"input_ids"``, ``"labels"``); each
            value has shape ``(B, ...)``.  Halted slots receive fresh rows from the
            incoming batch before the backbone forward pass.
    """

    model_state: ModelState
    steps: Tensor
    halted: Tensor
    data: dict[str, Tensor]


# =============================================================================
class BaseController[ModelState, ConfigT: BaseModel]:
    """Shared infrastructure for slot-based rollout controllers.

    A rollout controller drives a backbone across a variable number of steps,
    maintaining per-slot state that persists until a slot halts (partial-reset
    semantics).  This class owns exactly the following:

    * **Slot lifecycle** — ``initial_slots`` allocates per-slot buffers;
      ``refresh_slot_data`` copies fresh batch rows into halted slots;
      ``advance_steps`` increments active-slot counters and resets halted ones.
    * **Backbone access** — the backbone is stored and exposed via ``backbone``;
      the controller is the *only* party that should call ``backbone.reset_state``
      and ``backbone.__call__``.

    What this base does **not** own:

    * **Halting semantics** — deciding which slots are done and when is fully
      delegated to each subclass ``step`` implementation.
    * **Action selection** — any policy (greedy argmax, categorical sampling,
      exploration gating) lives in the subclass private hook
      ``_select_action_and_done``.
    * **Token loss computation** — no loss is computed here.

    Subclasses must implement ``step(state, batch, **options)`` following this
    sequence:

    1. Call ``self.refresh_slot_data(batch, state)`` to overwrite halted slots.
    2. Call ``self.backbone.reset_state(state.halted, state.model_state)``.
    3. Call ``self.backbone(data, model_state)`` for the forward pass; it returns
       ``(output, model_state)`` — output first, state second.
    4. Call ``self.advance_steps(state)`` to update per-slot counters.
    5. Call ``self._select_action_and_done(...)`` — the single algorithm hook;
       it must return at minimum ``(action: Tensor, done: Tensor[bool])``;
       ``done`` becomes the ``halted`` field of the new carry.

    .. note::
        ``initial_slots`` requires at least one tensor-valued batch entry to
        infer batch size and device.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        backbone: RolloutBackbone[ModelState],
        config: ConfigT,
    ) -> None:
        self._backbone = backbone
        self._config = config

    @property
    def backbone(self) -> RolloutBackbone[ModelState]:
        """Return the wrapped backbone."""
        return self._backbone

    @property
    def config(self) -> ConfigT:
        """Return the controller configuration."""
        return self._config

    def initial_slots(  # -----------------------------------------------------
        self,
        batch_sample: Batch,
    ) -> RolloutState[ModelState]:
        """Allocate the initial per-slot rollout buffers from a batch sample.

        All slots start as halted (``halted=True``, ``steps=0``), so the first
        ``refresh_slot_data`` call will fill every slot with incoming batch data.

        .. warning::
            Requires at least one tensor-valued batch entry with shape
            ``(B, ...)``. All keys are allocated as empty buffers with matching
            dtype and shape.
        """
        anchor = batch_anchor_tensor(batch_sample)
        batch_size = int(anchor.shape[0])
        device = anchor.device
        return RolloutState(
            model_state=self.backbone.init_state(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            data=self.make_empty_slot_data(batch_sample),
        )

    @staticmethod
    def make_empty_slot_data(  # ----------------------------------------------
        batch_sample: Batch,
    ) -> dict[str, Tensor]:
        """Allocate per-slot buffers matching an example batch."""
        return {
            key: torch.empty_like(value) for key, value in batch_sample.items()
        }

    def refresh_slot_data(  # -------------------------------------------------
        self,
        batch: Batch,
        state: RolloutState[ModelState],
    ) -> dict[str, Tensor]:
        """Refresh slot buffers for halted rows."""
        batch, halted, data = batch, state.halted, state.data
        return {
            key: torch.where(
                halted.view((-1,) + (1,) * (value.ndim - 1)), value, data[key]
            )
            for key, value in batch.items()
        }

    def advance_steps(  # -----------------------------------------------------
        self,
        state: RolloutState[ModelState],
    ) -> Tensor:
        """Advance per-slot step counters using halted rows as reset points."""
        steps, halted = state.steps, state.halted
        return torch.where(halted, torch.zeros_like(steps), steps) + 1


# =============================================================================
def batch_anchor_tensor(  # ---------------------------------------------------
    batch: Batch,
) -> Tensor:
    """Return one representative batch tensor used to infer batch size and device.

    The generic controller layer must not assume a semantic key such as
    ``"observation"`` or ``"input_ids"``. It only requires a non-empty mapping
    of tensor-valued batch entries that share the leading batch dimension.
    """
    for value in batch.values():
        if isinstance(value, Tensor):
            return value
    raise ValueError("Batch must contain at least one tensor-valued entry.")


# =============================================================================
__all__ = [
    "BaseController",
    "RolloutBackbone",
    "RolloutState",
    "batch_anchor_tensor",
]
