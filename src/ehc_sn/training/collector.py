""" """

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Dict

from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.types import Batch


# =================================================================================================
class PartialResetCollector(Iterator[Batch]):
    """
    Iterator that yields per-step batches and stops when refill is impossible.

    This mirrors RL collector semantics: done slots must be reset/refilled; if
    a refill cannot be satisfied, iteration stops.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, *, incoming: Batch, assembler: PartialResetBatchAssembler, carry0: Any,
    ) -> None:  # fmt: skip
        """ """
        self._incoming = incoming
        self._template = incoming
        self._assembler = assembler
        self._reset_mask = carry0.halted
        self._started = False

    def __iter__(  # ------------------------------------------------------------------------------
        self,
    ) -> "PartialResetCollector":  # fmt: skip
        return self

    def __next__(  # ------------------------------------------------------------------------------
        self,
    ) -> Batch:  # fmt: skip
        if not self._started:
            self._started = True
            return self._assembler.ingest_and_make_step_batch(
                incoming=self._incoming, reset_mask=self._reset_mask
            )

        if not self._reset_mask.any():
            return self._template

        step_batch = self._assembler.refill_only(template=self._template, reset_mask=self._reset_mask)
        if step_batch is None:
            raise StopIteration
        return step_batch

    def update(  # --------------------------------------------------------------------------------
        self, *, carry: Any,
    ) -> None:  # fmt: skip
        self._reset_mask = carry.halted


# =================================================================================================
__all__ = ["PartialResetCollector"]
