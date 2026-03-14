from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol, Tuple

from torch import Tensor

# TODO: We need to move this to typing module
Batch = Dict[str, Tensor]
StepBatchSource = Iterator[Batch]


# =================================================================================================
@dataclass(frozen=True)
class StepContext:
    """Trace context for a single compute step."""

    batch: Mapping[str, Tensor]
    carry: Any
    outputs: Any
    step_module: Any


# =================================================================================================
class StepModule(Protocol):
    """Protocol for step modules used by `StepLoop`."""

    def initial_carry(  # ------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> Any:  # fmt: skip
        ...  # fmt: skip

    def __call__(  # -----------------------------------------------------------------------------
        self, batch: Batch, carry: Any, **options
    ) -> tuple[Any, Any, bool]:  # fmt: skip
        ...  # fmt: skip


# =================================================================================================
class StepLoop(Iterator[tuple[int, StepContext]]):
    """Iterate a step module over a batch source for diagnostic or training traces."""

    def __init__(  # -----------------------------------------------------------------------------
        self, step_module: StepModule, batch: StepBatchSource, carry0: Any,
        *,
        max_steps: Optional[int] = None, options: Optional[Mapping[str, object]] = None,
    ) -> None:  # fmt: skip
        self.step_module = step_module
        self.options = options or {}
        self.batch_iter = enumerate(islice(batch, max_steps), start=0)
        self.carry = carry0

    def __iter__(  # -----------------------------------------------------------------------------
        self,
    ) -> StepLoop:  # fmt: skip
        return self

    def __next__(  # -----------------------------------------------------------------------------
        self,
    ) -> tuple[int, StepContext]:  # fmt: skip
        t_trace, batch = next(self.batch_iter)
        outputs, carry, done = self.step_module(batch, self.carry, **self.options)
        self.carry = carry
        self.batch_iter = iter([]) if done else self.batch_iter

        context = StepContext(batch=batch, carry=carry, outputs=outputs, step_module=self.step_module)
        return t_trace, context


__all__ = ["Batch", "StepLoop", "StepBatchSource", "StepContext", "StepModule"]
