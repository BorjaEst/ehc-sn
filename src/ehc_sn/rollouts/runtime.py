"""Executed rollout runtime primitives.

This module defines the canonical executed-trajectory data structures and
runtime protocols used by learners and observers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Mapping, Optional, Protocol, TypeVar

from torch import Tensor

from ehc_sn.types import Batch


# =================================================================================================
class StopReason(str, Enum):
    """Canonical reasons a runner can stop normal execution."""

    ALL_HALTED = "all_halted"
    SINGLE_STEP_COMPLETED = "single_step_completed"
    SOURCE_EXHAUSTED = "source_exhausted"
    STEP_LIMIT_REACHED = "step_limit_reached"


# =================================================================================================
class ExecutionHaltError(RuntimeError):
    """Raised when a runner hits a hard execution limit before halting cleanly."""

    def __init__(self, *, hard_max_steps: int, executed_steps: int) -> None:
        super().__init__(f"Runner exceeded hard step limit {hard_max_steps} after {executed_steps} executed steps.")
        self.hard_max_steps = hard_max_steps
        self.executed_steps = executed_steps


# =================================================================================================
class HaltedCarry(Protocol):
    """Minimal carry surface required by rollout sources.

    Sources receive the latest controller carry after each executed step. Any
    source that performs partial reset depends on ``halted`` being a batch-
    aligned boolean tensor whose rows match the batch most recently emitted by
    the source.
    """

    halted: Tensor


CarryT = TypeVar("CarryT", bound=HaltedCarry)
ControllerOutputT = TypeVar("ControllerOutputT")


# =================================================================================================
class ObjectiveStepOutput(Protocol):
    """Minimal scored-step output exposed outside objective modules."""

    @property
    def loss(self) -> Tensor: ...

    metrics: object
    signals: Mapping[str, object]


ScoredOutputT = TypeVar("ScoredOutputT", bound=ObjectiveStepOutput)


# =================================================================================================
@dataclass(frozen=True)
class StepRecord(Generic[CarryT, ControllerOutputT]):
    """Executed controller step.

    This record is intentionally minimal and stores only the executed batch,
    the post-step carry, the controller outputs, and the step index.
    """

    index: int
    batch: Batch
    carry: CarryT
    outputs: ControllerOutputT
    all_halted: bool


# =================================================================================================
@dataclass(frozen=True)
class RolloutExecution(Generic[CarryT]):
    """Recordless execution summary returned when a runner skips capture."""

    final_carry: CarryT
    executed_steps: int
    source_exhausted: bool = False
    stop_reason: StopReason = StopReason.SOURCE_EXHAUSTED

    @property
    def length(self) -> int:
        """Return number of executed steps in the execution summary."""
        return self.executed_steps


# =================================================================================================
@dataclass(frozen=True)
class RolloutChunk(Generic[CarryT, ControllerOutputT]):
    """Record-bearing executed rollout fragment produced by a runner."""

    records: tuple[StepRecord[CarryT, ControllerOutputT], ...]
    final_carry: CarryT
    executed_steps: int
    source_exhausted: bool = False
    stop_reason: StopReason = StopReason.SOURCE_EXHAUSTED

    @property
    def length(self) -> int:
        """Return number of executed steps in the chunk."""
        return self.executed_steps

    @property
    def last_record(self) -> StepRecord[CarryT, ControllerOutputT]:
        """Return the final executed step in the chunk."""
        if not self.records:
            raise ValueError("RolloutChunk has no step records.")
        return self.records[-1]


# =================================================================================================
@dataclass(frozen=True)
class ObservedStep(Generic[CarryT, ScoredOutputT]):
    """Objective-scored step context consumed by metrics and trace observers."""

    index: int
    batch: Batch
    carry: CarryT
    outputs: ScoredOutputT


# =================================================================================================
@dataclass(frozen=True)
class EvaluatedChunk(Generic[CarryT, ScoredOutputT]):
    """Objective-scored rollout fragment returned by a pure objective."""

    steps: tuple[ObservedStep[CarryT, ScoredOutputT], ...]
    loss: Tensor
    final_carry: CarryT
    source_exhausted: bool = False

    @property
    def last_step(self) -> ObservedStep[CarryT, ScoredOutputT]:
        """Return the final scored step in the chunk."""
        if not self.steps:
            raise ValueError("EvaluatedChunk has no observed steps.")
        return self.steps[-1]


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
class StepController(Protocol[CarryT, ControllerOutputT]):
    """Controller protocol required by rollout runners."""

    def initial_state(self, batch_sample: Batch) -> CarryT: ...

    def step(self, state: CarryT, batch: Batch, **options: object) -> tuple[CarryT, ControllerOutputT]: ...


# =================================================================================================
class RolloutRecordObserver(Protocol[CarryT, ControllerOutputT]):
    """Passive per-step consumer used during runner execution."""

    def __call__(self, record: StepRecord[CarryT, ControllerOutputT]) -> None: ...


# =================================================================================================
class Runner(Protocol[CarryT, ControllerOutputT]):
    """Executed rollout driver."""

    def run(
        self,
        *,
        source: Source,
        controller: StepController[CarryT, ControllerOutputT],
        carry: CarryT,
        max_steps: Optional[int] = None,
        hard_max_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
        record_observer: RolloutRecordObserver[CarryT, ControllerOutputT] | None = None,
        capture_records: bool = True,
    ) -> RolloutChunk[CarryT, ControllerOutputT] | RolloutExecution[CarryT]: ...


# =================================================================================================
def _validate_limit(name: str, limit: int | None) -> None:
    """Validate that an optional runner limit is strictly positive."""
    if limit is not None and limit <= 0:
        raise ValueError(f"{name} must be positive when provided, got {limit}.")


# =================================================================================================
def _validate_single_step_limit(name: str, limit: int | None) -> None:
    """Validate that single-step runners only accept unit pacing limits."""
    _validate_limit(name, limit)
    if limit is not None and limit != 1:
        raise ValueError(f"{name} must be 1 or None for SingleStepRunner, got {limit}.")


# =================================================================================================
class SingleStepRunner:
    """Execute exactly one controller step from a passive source."""

    def run(
        self,
        *,
        source: Source,
        controller: StepController[CarryT, ControllerOutputT],
        carry: CarryT,
        max_steps: Optional[int] = None,
        hard_max_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
        record_observer: RolloutRecordObserver[CarryT, ControllerOutputT] | None = None,
        capture_records: bool = True,
    ) -> RolloutChunk[CarryT, ControllerOutputT] | RolloutExecution[CarryT]:
        """Return a one-step rollout chunk."""
        _validate_single_step_limit("max_steps", max_steps)
        _validate_single_step_limit("hard_max_steps", hard_max_steps)
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
        if record_observer is not None:
            record_observer(record)
        stop_reason = StopReason.ALL_HALTED if record.all_halted else StopReason.SINGLE_STEP_COMPLETED
        if not capture_records:
            return RolloutExecution(
                final_carry=carry,
                executed_steps=1,
                source_exhausted=False,
                stop_reason=stop_reason,
            )
        return RolloutChunk(
            records=(record,),
            final_carry=carry,
            executed_steps=1,
            source_exhausted=False,
            stop_reason=stop_reason,
        )


# =================================================================================================
class RecurrentRunner:
    """Execute a finite recurrent rollout chunk from a passive source."""

    def run(
        self,
        *,
        source: Source,
        controller: StepController[CarryT, ControllerOutputT],
        carry: CarryT,
        max_steps: Optional[int] = None,
        hard_max_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
        record_observer: RolloutRecordObserver[CarryT, ControllerOutputT] | None = None,
        capture_records: bool = True,
    ) -> RolloutChunk[CarryT, ControllerOutputT] | RolloutExecution[CarryT]:
        """Return a rollout chunk terminated by halt, source exhaustion, or step limit."""
        _validate_limit("max_steps", max_steps)
        _validate_limit("hard_max_steps", hard_max_steps)
        options_dict = dict(options or {})
        records: list[StepRecord[CarryT, ControllerOutputT]] = []
        source_exhausted = False
        stop_reason = StopReason.STEP_LIMIT_REACHED
        step_idx = 0

        while True:
            if hard_max_steps is not None and step_idx >= hard_max_steps:
                raise ExecutionHaltError(hard_max_steps=hard_max_steps, executed_steps=step_idx)
            if max_steps is not None and step_idx >= max_steps:
                stop_reason = StopReason.STEP_LIMIT_REACHED
                break

            try:
                batch = next(source)
            except StopIteration:
                source_exhausted = True
                stop_reason = StopReason.SOURCE_EXHAUSTED
                break

            carry, outputs = controller.step(carry, batch, **options_dict)
            source.update(carry=carry)
            record = StepRecord(
                index=step_idx,
                batch=batch,
                carry=carry,
                outputs=outputs,
                all_halted=bool(carry.halted.all()),
            )
            if record_observer is not None:
                record_observer(record)
            if capture_records:
                records.append(record)
            step_idx += 1

            if record.all_halted:
                stop_reason = StopReason.ALL_HALTED
                break

        if step_idx == 0:
            raise ValueError("RecurrentRunner did not execute any steps.")
        if not capture_records:
            return RolloutExecution(
                final_carry=carry,
                executed_steps=step_idx,
                source_exhausted=source_exhausted,
                stop_reason=stop_reason,
            )
        return RolloutChunk(
            records=tuple(records),
            final_carry=carry,
            executed_steps=step_idx,
            source_exhausted=source_exhausted,
            stop_reason=stop_reason,
        )


# =================================================================================================
__all__ = [
    "EvaluatedChunk",
    "ExecutionHaltError",
    "HaltedCarry",
    "ObjectiveStepOutput",
    "ObservedStep",
    "RecurrentRunner",
    "RolloutExecution",
    "RolloutChunk",
    "RolloutRecordObserver",
    "Runner",
    "SingleStepRunner",
    "Source",
    "StepController",
    "StepRecord",
    "StopReason",
]
