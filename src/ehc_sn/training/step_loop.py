""" """

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterator, Mapping, Optional, Protocol

from torch import Tensor

from ehc_sn.types import Batch, StepBatchSource


# =============================================================================
@dataclass(frozen=True)
class StepContext:
    """Trace context for a single compute step."""

    batch: Mapping[str, Tensor]
    carry: Any
    outputs: Any
    step_module: Any


# =============================================================================
class StepModule(Protocol):
    """Protocol for step modules used by `StepLoop`."""

    def initial_carry(  # -----------------------------------------------------
        self,
        batch_sample: Batch,
    ) -> Any:
        """Return the initial carry for a given batch sample."""

    def __call__(  # ----------------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        **options,
    ) -> tuple[Any, Any, bool]:
        """Compute the step outputs and next carry for a given batch and carry."""


# =============================================================================
class StepLoop(Iterator[tuple[int, StepContext]]):
    """Iterate a step module over a batch source for diagnostic or training
    traces.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        step_module: StepModule,
        batch: StepBatchSource,
        carry0: Any,
        *,
        max_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initialize the step loop with the step module, batch source,
        initial carry, optional maximum steps, and optional step options.
        """
        self.step_module = step_module
        self.options = options or {}
        self.batch_iter = enumerate(islice(batch, max_steps), start=0)
        self.carry = carry0

    def __iter__(  # ----------------------------------------------------------
        self,
    ) -> StepLoop:
        """Return self as an iterator."""
        return self

    def __next__(  # ----------------------------------------------------------
        self,
    ) -> tuple[int, StepContext]:
        """Advance the step loop by one step, returning the current step index
        and context.
        """
        t_trace, batch = next(self.batch_iter)
        outputs, carry, done = self.step_module(
            batch, self.carry, **self.options
        )
        self.carry = carry
        self.batch_iter = iter([]) if done else self.batch_iter

        context = StepContext(
            batch=batch,
            carry=carry,
            outputs=outputs,
            step_module=self.step_module,
        )
        return t_trace, context


# =============================================================================
__all__ = ["Batch", "StepLoop", "StepBatchSource", "StepContext", "StepModule"]
