"""Executed rollout runtime primitives.

This module defines the canonical executed-trajectory data structures and
runtime protocols used by learners and observers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

from torch import Tensor

from ehc_sn.types import Batch


# =================================================================================================
@dataclass(frozen=True)
class StepRecord:
    """Executed controller step.

    This record is intentionally minimal and stores only the executed batch,
    the post-step carry, the controller outputs, and the step index.
    """

    index: int
    batch: Batch
    carry: Any
    outputs: Any
    all_halted: bool


# =================================================================================================
@dataclass(frozen=True)
class RolloutChunk:
    """Finite executed rollout fragment produced by a runner."""

    records: tuple[StepRecord, ...]
    final_carry: Any
    source_exhausted: bool = False

    @property
    def length(self) -> int:
        """Return number of executed steps in the chunk."""
        return len(self.records)

    @property
    def last_record(self) -> StepRecord:
        """Return the final executed step in the chunk."""
        if not self.records:
            raise ValueError("RolloutChunk has no step records.")
        return self.records[-1]


# =================================================================================================
@dataclass(frozen=True)
class ObservedStep:
    """Objective-scored step context consumed by metrics and trace observers."""

    index: int
    batch: Batch
    carry: Any
    outputs: Any


# =================================================================================================
@dataclass(frozen=True)
class EvaluatedChunk:
    """Objective-scored rollout fragment returned by a pure objective."""

    steps: tuple[ObservedStep, ...]
    loss: Tensor
    final_carry: Any
    source_exhausted: bool = False

    @property
    def last_step(self) -> ObservedStep:
        """Return the final scored step in the chunk."""
        if not self.steps:
            raise ValueError("EvaluatedChunk has no observed steps.")
        return self.steps[-1]


# =================================================================================================
class HaltedCarry(Protocol):
    """Minimal carry surface required by rollout sources.

    Sources receive the latest controller carry after each executed step. Any
    source that performs partial reset depends on ``halted`` being a batch-
    aligned boolean tensor whose rows match the batch most recently emitted by
    the source.
    """

    halted: Tensor


# =================================================================================================
class Source(Protocol):
    """Passive rollout batch supplier used by runners.

    Runners call ``update(carry=...)`` after each controller step so sources can
    react to the latest batch-aligned halt state when deciding how to emit the
    next batch.
    """

    def __iter__(self) -> "Source": ...

    def __next__(self) -> Batch: ...

    def update(self, *, carry: HaltedCarry) -> None:
        """Receive the latest controller carry for source-side state updates."""
        ...


# =================================================================================================
class StepController(Protocol):
    """Controller protocol required by rollout runners."""

    def initial_state(self, batch_sample: Batch) -> Any: ...

    def step(self, state: Any, batch: Batch, **options: Any) -> tuple[Any, Any]: ...


# =================================================================================================
class Runner(Protocol):
    """Executed rollout driver."""

    def run(
        self,
        *,
        source: Source,
        controller: StepController,
        carry: Any,
        max_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
    ) -> RolloutChunk: ...


# =================================================================================================
class SingleStepRunner:
    """Execute exactly one controller step from a passive source."""

    def run(
        self,
        *,
        source: Source,
        controller: StepController,
        carry: Any,
        max_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
    ) -> RolloutChunk:
        """Return a one-step rollout chunk."""
        _ = max_steps
        options_dict = dict(options or {})
        try:
            batch = next(source)
        except StopIteration as exc:
            raise ValueError("SingleStepRunner source produced no batch.") from exc

        carry, outputs = controller.step(carry, batch, **options_dict)
        source.update(carry=carry)
        record = StepRecord(
            index=0,
            batch=batch,
            carry=carry,
            outputs=outputs,
            all_halted=bool(carry.halted.all()),
        )
        return RolloutChunk(records=(record,), final_carry=carry, source_exhausted=False)


# =================================================================================================
class RecurrentRunner:
    """Execute a finite recurrent rollout chunk from a passive source."""

    def run(
        self,
        *,
        source: Source,
        controller: StepController,
        carry: Any,
        max_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
    ) -> RolloutChunk:
        """Return a rollout chunk terminated by halt, source exhaustion, or max steps."""
        options_dict = dict(options or {})
        records: list[StepRecord] = []
        source_exhausted = False
        step_idx = 0

        while max_steps is None or step_idx < max_steps:
            try:
                batch = next(source)
            except StopIteration:
                source_exhausted = True
                break

            carry, outputs = controller.step(carry, batch, **options_dict)
            source.update(carry=carry)
            all_halted = bool(carry.halted.all())
            records.append(
                StepRecord(
                    index=step_idx,
                    batch=batch,
                    carry=carry,
                    outputs=outputs,
                    all_halted=all_halted,
                )
            )
            step_idx += 1

            if all_halted:
                break

        if not records:
            raise ValueError("RecurrentRunner did not execute any steps.")
        return RolloutChunk(records=tuple(records), final_carry=carry, source_exhausted=source_exhausted)


# =================================================================================================
__all__ = [
    "EvaluatedChunk",
    "HaltedCarry",
    "ObservedStep",
    "RecurrentRunner",
    "RolloutChunk",
    "Runner",
    "SingleStepRunner",
    "Source",
    "StepController",
    "StepRecord",
]
