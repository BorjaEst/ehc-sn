"""Executed rollout runtime primitives.

This module defines the canonical executed-trajectory data structures and
runtime protocols used by learners and observers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from types import SimpleNamespace
from typing import Any, Generic, Mapping, Optional, Protocol, TypeVar

from torch import Tensor

from ehc_sn.types import Batch


# =============================================================================
class StopReason(str, Enum):
    """Canonical reasons a runner can stop normal execution."""

    ALL_HALTED = "all_halted"
    SINGLE_STEP_COMPLETED = "single_step_completed"
    SOURCE_EXHAUSTED = "source_exhausted"
    STEP_LIMIT_REACHED = "step_limit_reached"


# =============================================================================
class ExecutionHaltError(RuntimeError):
    """Raised when a runner hits a hard execution limit before halting cleanly."""

    def __init__(  # ----------------------------------------------------------
        self,
        *,
        hard_max_rollout_steps: int,
        executed_steps: int,
    ) -> None:
        """Initialize the error with the hard limit and executed step count."""
        super().__init__(
            f"Runner exceeded hard rollout step limit {hard_max_rollout_steps} "
            f"after {executed_steps} executed steps."
        )
        self.hard_max_rollout_steps = hard_max_rollout_steps
        self.executed_steps = executed_steps


# =============================================================================
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


# =============================================================================
class ObjectiveStepOutput(Protocol):
    """Minimal scored-step output exposed outside objective modules."""

    @property
    def loss(self) -> Tensor:
        """Return the scalar loss tensor for this step."""

    metrics: object
    signals: Mapping[str, object]


ScoredOutputT = TypeVar("ScoredOutputT", bound=ObjectiveStepOutput)


# =============================================================================
@dataclass(frozen=True)
class CarrySnapshot:
    """Frozen post-step projection of controller carry.

    Snapshots are lean, frozen views of continuity state after the step finishes.
    They are not the authoritative current-step payload; use
    ``StepRecord.executed_frame`` (or the backward-compatible ``record.batch``)
    when step-truth tensors are required. Snapshot fields are limited to
    lightweight continuity facts and current-step extracts and must exclude
    resident per-trajectory payloads.
    """

    halted: Tensor
    steps: Tensor | None = None
    data: dict[str, Any] | None = None
    static_data: dict[str, Any] | None = None
    model_state: Any = None
    env_td: Any = None


# =============================================================================
@dataclass(frozen=True)
class StepRecord(Generic[ControllerOutputT]):
    """Executed controller step.

    This record stores the executed batch, a frozen post-step carry snapshot,
    the controller outputs, and the step index.

    Fields
    ------
    batch:
        Backward-compatible alias; always contains the *executed_frame* content
        (the exact tensors consumed by the model, objective, and diagnostics).
        Do not confuse with the raw source batch — use ``sampled_input`` for that.
    sampled_input:
        What the source proposed: the raw batch returned by the source before the
        controller processed it.  May contain large trajectory arrays not consumed
        by active slots.  ``None`` when the runner does not populate this field.
    executed_frame:
        The exact tensors actually consumed by the model, objective, and
        diagnostics for this step.  For replay controllers this is the
        carry-owned step slice, independent of the source batch.
        ``None`` when the runner does not populate this field (legacy path).
    """

    index: int
    batch: Batch
    snapshot: CarrySnapshot
    outputs: ControllerOutputT
    all_halted: bool
    sampled_input: Batch | None = None
    executed_frame: Batch | None = None

    @property
    def carry(self) -> CarrySnapshot:
        """Backward-compatible alias for the frozen post-step snapshot."""
        return self.snapshot


# =============================================================================
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


# =============================================================================
@dataclass(frozen=True)
class RolloutChunk(Generic[CarryT, ControllerOutputT]):
    """Record-bearing executed rollout fragment produced by a runner."""

    records: tuple[StepRecord[ControllerOutputT], ...]
    final_carry: CarryT
    executed_steps: int
    source_exhausted: bool = False
    stop_reason: StopReason = StopReason.SOURCE_EXHAUSTED

    @property
    def length(self) -> int:
        """Return number of executed steps in the chunk."""
        return self.executed_steps

    @property
    def last_record(self) -> StepRecord[ControllerOutputT]:
        """Return the final executed step in the chunk."""
        if not self.records:
            raise ValueError("RolloutChunk has no step records.")
        return self.records[-1]


# =============================================================================
@dataclass(frozen=True)
class ObservedStep(Generic[ScoredOutputT]):
    """Objective-scored step context consumed by metrics and trace observers.

    Fields
    ------
    batch:
        Backward-compatible alias; always contains the *executed_frame* content.
    sampled_input:
        Raw source-batch proposal; ``None`` when not populated by the runner.
    executed_frame:
        Exact tensors consumed by the model and objective on this step.
        ``None`` when not populated (legacy path).
    """

    index: int
    batch: Batch
    snapshot: CarrySnapshot
    outputs: ScoredOutputT
    sampled_input: Batch | None = None
    executed_frame: Batch | None = None

    @property
    def carry(self) -> CarrySnapshot:
        """Backward-compatible alias for the frozen post-step snapshot."""
        return self.snapshot


# =============================================================================
@dataclass(frozen=True)
class EvaluatedChunk(Generic[CarryT, ScoredOutputT]):
    """Objective-scored rollout fragment returned by a pure objective."""

    steps: tuple[ObservedStep[ScoredOutputT], ...]
    loss: Tensor
    final_carry: CarryT
    source_exhausted: bool = False

    @property
    def last_step(self) -> ObservedStep[ScoredOutputT]:
        """Return the final scored step in the chunk."""
        if not self.steps:
            raise ValueError("EvaluatedChunk has no observed steps.")
        return self.steps[-1]


# =============================================================================
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


# =============================================================================
class StepController(Protocol[CarryT, ControllerOutputT]):
    """Controller protocol required by rollout runners."""

    def initial_state(  # -----------------------------------------------------
        self,
        batch_sample: Batch,
    ) -> CarryT:
        """Return the initial carry state for the given source batch sample."""

    def step(  # --------------------------------------------------------------
        self,
        state: CarryT,
        batch: Batch,
        **options: object,
    ) -> tuple[CarryT, ControllerOutputT]:
        """Execute one step of the controller and return new carry and outputs."""


# =============================================================================
class RolloutRecordObserver(Protocol[ControllerOutputT]):
    """Passive per-step consumer used during runner execution."""

    def __call__(  # ----------------------------------------------------------
        self,
        record: StepRecord[ControllerOutputT],
    ) -> None:
        """Observe a single executed step record during rollout execution."""


# =============================================================================
class Runner(Protocol[CarryT, ControllerOutputT]):
    """Executed rollout driver."""

    def run(  # ---------------------------------------------------------------
        self,
        *,
        source: Source,
        controller: StepController[CarryT, ControllerOutputT],
        carry: CarryT,
        max_rollout_steps: Optional[int] = None,
        hard_max_rollout_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
        record_observer: RolloutRecordObserver[ControllerOutputT] | None = None,
        capture_records: bool = True,
    ) -> RolloutChunk[CarryT, ControllerOutputT] | RolloutExecution[CarryT]:
        """Execute a rollout and return a chunk of step records or an execution summary."""


# =============================================================================
def _snapshot_value(  # -------------------------------------------------------
    value: Any,
    *,
    path: str,
) -> Any:
    """Return a structural clone suitable for frozen rollout records.

    The snapshot preserves tensor/device semantics while avoiding generic
    ``copy.deepcopy`` over autograd-bearing state. Mutable values not covered by
    this helper should expose an explicit ``clone()`` method or be modeled as
    dataclasses/containers of supported values. Unsupported mutable values fail
    fast so new carry fields cannot silently bypass snapshot freezing.
    """
    if isinstance(value, Tensor):
        return value.clone()
    if value is None or isinstance(value, (bool, int, float, str, bytes, Enum)):
        return value

    clone = getattr(value, "clone", None)
    if callable(clone):
        try:
            return clone()
        except TypeError as exc:
            raise TypeError(
                f"Unsupported snapshot value at {path}: {type(value).__name__} "
                " exposes clone() but it "
                "cannot be called without arguments."
            ) from exc

    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(
            **{
                name: _snapshot_value(item, path=f"{path}.{name}")
                for name, item in vars(value).items()
            }
        )
    if isinstance(value, dict):
        return {
            name: _snapshot_value(item, path=f"{path}[{name!r}]")
            for name, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _snapshot_value(item, path=f"{path}[{idx}]")
            for idx, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _snapshot_value(item, path=f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ]
    if isinstance(value, set):
        return {
            _snapshot_value(item, path=f"{path}[{idx}]")
            for idx, item in enumerate(value)
        }
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _snapshot_value(
                    getattr(value, field.name), path=f"{path}.{field.name}"
                )
                for field in fields(value)
            },
        )
    raise TypeError(
        f"Unsupported snapshot value at {path}: {type(value).__name__}. "
        "Snapshot-compatible values must be tensors, scalars, dataclasses, "
        "SimpleNamespace, standard containers, or expose clone()."
    )


# =============================================================================
def _snapshot_carry(  # -------------------------------------------------------
    carry: HaltedCarry,
) -> CarrySnapshot:
    """Return the closed post-step carry snapshot stored in rollout records."""
    return CarrySnapshot(
        halted=_snapshot_value(carry.halted, path="carry.halted"),
        steps=_snapshot_value(
            getattr(carry, "steps", None), path="carry.steps"
        ),
        data=_snapshot_value(getattr(carry, "data", None), path="carry.data"),
        static_data=_snapshot_value(
            getattr(carry, "static_data", None), path="carry.static_data"
        ),
        model_state=_snapshot_value(
            getattr(carry, "model_state", None), path="carry.model_state"
        ),
        env_td=_snapshot_value(
            getattr(carry, "env_td", None), path="carry.env_td"
        ),
    )


# =============================================================================
def _validate_limit(  # -------------------------------------------------------
    name: str,
    limit: int | None,
) -> None:
    """Validate that an optional runner limit is strictly positive."""
    if limit is not None and limit <= 0:
        raise ValueError(f"{name} must be positive when provided, got {limit}.")


# =============================================================================
def _validate_single_step_limit(  # -------------------------------------------
    name: str,
    limit: int | None,
) -> None:
    """Validate that single-step runners only accept unit pacing limits."""
    _validate_limit(name, limit)
    if limit is not None and limit != 1:
        raise ValueError(
            f"{name} must be 1 or None for SingleStepRunner, got {limit}."
        )


# =============================================================================
class SingleStepRunner:
    """Execute exactly one controller step from a passive source."""

    def run(  # ---------------------------------------------------------------
        self,
        *,
        source: Source,
        controller: StepController[CarryT, ControllerOutputT],
        carry: CarryT,
        max_rollout_steps: Optional[int] = None,
        hard_max_rollout_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
        record_observer: RolloutRecordObserver[ControllerOutputT] | None = None,
        capture_records: bool = True,
    ) -> RolloutChunk[CarryT, ControllerOutputT] | RolloutExecution[CarryT]:
        """Return a one-step rollout chunk."""
        _validate_single_step_limit("max_rollout_steps", max_rollout_steps)
        _validate_single_step_limit(
            "hard_max_rollout_steps", hard_max_rollout_steps
        )
        options_dict = dict(options or {})
        try:
            batch = next(source)
        except StopIteration as exc:
            raise ValueError(
                "SingleStepRunner source produced no batch."
            ) from exc

        carry, outputs = controller.step(carry, batch, **options_dict)
        snapshot = _snapshot_carry(carry)
        source.update(carry=carry)
        # executed_frame: the actual step tensors consumed by the model — taken from
        # carry.data which the controller sets from resident_payload for replay controllers.
        executed = dict(getattr(carry, "data", None) or {})
        record = StepRecord(
            index=0,
            batch=executed,
            sampled_input=batch,
            executed_frame=executed,
            snapshot=snapshot,
            outputs=outputs,
            all_halted=bool(snapshot.halted.all()),
        )
        if record_observer is not None:
            record_observer(record)
        stop_reason = (
            StopReason.ALL_HALTED
            if record.all_halted
            else StopReason.SINGLE_STEP_COMPLETED
        )
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


# =============================================================================
class RecurrentRunner:
    """Execute a finite recurrent rollout chunk from a passive source."""

    def run(  # ---------------------------------------------------------------
        self,
        *,
        source: Source,
        controller: StepController[CarryT, ControllerOutputT],
        carry: CarryT,
        max_rollout_steps: Optional[int] = None,
        hard_max_rollout_steps: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
        record_observer: RolloutRecordObserver[ControllerOutputT] | None = None,
        capture_records: bool = True,
    ) -> RolloutChunk[CarryT, ControllerOutputT] | RolloutExecution[CarryT]:
        """Return a rollout chunk terminated by halt, source exhaustion, or step limit."""
        _validate_limit("max_rollout_steps", max_rollout_steps)
        _validate_limit("hard_max_rollout_steps", hard_max_rollout_steps)
        options_dict = dict(options or {})
        records: list[StepRecord[ControllerOutputT]] = []
        source_exhausted = False
        stop_reason = StopReason.STEP_LIMIT_REACHED
        step_idx = 0

        while True:
            if (
                hard_max_rollout_steps is not None
                and step_idx >= hard_max_rollout_steps
            ):
                raise ExecutionHaltError(
                    hard_max_rollout_steps=hard_max_rollout_steps,
                    executed_steps=step_idx,
                )
            if max_rollout_steps is not None and step_idx >= max_rollout_steps:
                stop_reason = StopReason.STEP_LIMIT_REACHED
                break

            try:
                batch = next(source)
            except StopIteration:
                source_exhausted = True
                stop_reason = StopReason.SOURCE_EXHAUSTED
                break

            carry, outputs = controller.step(carry, batch, **options_dict)
            snapshot = _snapshot_carry(carry)
            source.update(carry=carry)
            # executed_frame: carry-owned step tensors, independent of source batch.
            executed = dict(getattr(carry, "data", None) or {})
            record = StepRecord(
                index=step_idx,
                batch=executed,
                sampled_input=batch,
                executed_frame=executed,
                snapshot=snapshot,
                outputs=outputs,
                all_halted=bool(snapshot.halted.all()),
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


# =============================================================================
__all__ = [
    "CarrySnapshot",
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
