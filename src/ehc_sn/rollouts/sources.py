"""Passive rollout sources used by rollout runners."""

from __future__ import annotations

import torch
from torch import Tensor

from ehc_sn.rollouts.partial_reset import PartialResetBatchAssembler
from ehc_sn.rollouts.runtime import HaltedCarry
from ehc_sn.types import Batch


# =============================================================================
class RepeatSource:
    """Yield the same batch repeatedly, optionally with a fixed horizon."""

    def __init__(  # -----------------------------------------------------------
        self,
        batch: Batch,
        *,
        max_rollout_steps: int | None = None,
        stop_on_halt: bool = False,
        freeze_halted: bool = False,
    ) -> None:
        """Initialize the source with a batch to repeat and an optional rollout horizon."""
        self._batch = batch
        self._max_rollout_steps = max_rollout_steps
        self._steps = 0
        self._halted: Tensor | None = None
        self._stop_on_halt = stop_on_halt
        self._freeze_halted = freeze_halted

    def __iter__(self) -> RepeatSource:
        """Return self as an iterator."""
        return self

    def __next__(self) -> Batch:
        """Yield the batch, stopping if the maximum rollout steps have been reached."""
        if self._stop_on_halt and self._halted is not None:
            if bool(self._halted.all()):
                raise StopIteration
        if (
            self._max_rollout_steps is not None
            and self._steps >= self._max_rollout_steps
        ):
            raise StopIteration
        self._steps += 1
        return self._batch

    def update(  # ------------------------------------------------------------
        self,
        *,
        carry: HaltedCarry,
    ) -> None:
        """Update halt state and optionally freeze halted rows to their last data."""
        self._halted = carry.halted
        if not self._freeze_halted:
            return
        carry_data = getattr(carry, "data", None)
        if not isinstance(carry_data, dict) or not carry_data:
            return
        halted = carry.halted
        updated: dict[str, Tensor] = {}
        for key, value in self._batch.items():
            carry_value = carry_data.get(key)
            if (
                isinstance(value, Tensor)
                and isinstance(carry_value, Tensor)
                and value.shape == carry_value.shape
            ):
                updated[key] = torch.where(
                    halted.view((-1,) + (1,) * (value.ndim - 1)),
                    carry_value,
                    value,
                )
            else:
                updated[key] = value
        self._batch = updated


# =============================================================================
class PartialResetSource:
    """Passive refill source for partial-reset recurrent training."""

    def __init__(  # ----------------------------------------------------------
        self,
        *,
        incoming: Batch,
        assembler: PartialResetBatchAssembler,
        carry0: HaltedCarry,
    ) -> None:
        """Initialize the source with the first batch, assembler, and initial carry."""
        self._incoming = incoming
        self._template = incoming
        self._assembler = assembler
        self._reset_mask = carry0.halted
        self._started = False

    def __iter__(self) -> "PartialResetSource":
        """Return self as an iterator."""
        return self

    def __next__(self) -> Batch:
        """Yield the next batch, handling the initial batch and subsequent."""
        if not self._started:
            self._started = True
            return self._assembler.ingest_and_make_step_batch(
                incoming=self._incoming,
                reset_mask=self._reset_mask,
            )

        if not self._reset_mask.any():
            return self._template

        step_batch = self._assembler.refill_only(
            template=self._template, reset_mask=self._reset_mask
        )
        if step_batch is None:
            raise StopIteration
        return step_batch

    def update(  # ------------------------------------------------------------
        self,
        *,
        carry: HaltedCarry,
    ) -> None:
        """Update the refill mask from the latest batch-aligned halt mask."""
        self._reset_mask = carry.halted


# =============================================================================
__all__ = ["PartialResetSource", "RepeatSource"]
