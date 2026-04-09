"""Passive rollout sources used by rollout runners."""

from __future__ import annotations

from ehc_sn.rollouts.runtime import HaltedCarry
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.types import Batch


# =================================================================================================
class RepeatSource:
    """Yield the same batch repeatedly, optionally with a fixed horizon."""

    def __init__(self, batch: Batch, *, max_steps: int | None = None) -> None:
        self._batch = batch
        self._max_steps = max_steps
        self._steps = 0

    def __iter__(self) -> "RepeatSource":
        return self

    def __next__(self) -> Batch:
        if self._max_steps is not None and self._steps >= self._max_steps:
            raise StopIteration
        self._steps += 1
        return self._batch

    def update(self, *, carry: HaltedCarry) -> None:
        """Ignore feedback because the source is purely repetitive."""
        _ = carry


# =================================================================================================
class PartialResetSource:
    """Passive refill source for partial-reset recurrent training."""

    def __init__(self, *, incoming: Batch, assembler: PartialResetBatchAssembler, carry0: HaltedCarry) -> None:
        self._incoming = incoming
        self._template = incoming
        self._assembler = assembler
        self._reset_mask = carry0.halted
        self._started = False

    def __iter__(self) -> "PartialResetSource":
        return self

    def __next__(self) -> Batch:
        if not self._started:
            self._started = True
            return self._assembler.ingest_and_make_step_batch(
                incoming=self._incoming,
                reset_mask=self._reset_mask,
            )

        if not self._reset_mask.any():
            return self._template

        step_batch = self._assembler.refill_only(template=self._template, reset_mask=self._reset_mask)
        if step_batch is None:
            raise StopIteration
        return step_batch

    def update(self, *, carry: HaltedCarry) -> None:
        """Update the refill mask from the latest batch-aligned halt mask."""
        self._reset_mask = carry.halted


# =================================================================================================
__all__ = ["PartialResetSource", "RepeatSource"]
