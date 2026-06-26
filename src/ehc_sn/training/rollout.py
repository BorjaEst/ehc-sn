"""Training-owned rollout orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from ehc_sn.rollouts.materialization import (
    EvaluatedChunk,
    ObservedStep,
)
from ehc_sn.rollouts.runtime import (
    RolloutChunk,
    RolloutExecution,
    Runner,
    Source,
    StepController,
    StepRecord,
)
from ehc_sn.rollouts.scoring import (
    RolloutScorer,
    ScoredRecord,
    score_rollout_chunk,
    score_rollout_record,
)
from ehc_sn.traces.observer import TraceObserver, TraceSpec
from ehc_sn.traces.sink import InMemoryTraceSink, TraceSink
from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
@dataclass(frozen=True)
class CapturedRolloutResult:
    """Pair an executed rollout chunk with its objective-scored result."""

    chunk: RolloutChunk
    evaluated: EvaluatedChunk


# =============================================================================
@dataclass(frozen=True)
class StreamingRolloutResult:
    """Streaming rollout evaluation without full-step materialization."""

    execution: RolloutExecution
    loss: Tensor
    last_step: ObservedStep


# =============================================================================
@dataclass(frozen=True)
class StreamingRolloutResultWithTrace:
    """Streaming rollout evaluation with a materialized trace tree.

    The trace is produced by a :class:`TraceSink` and returned as a
    finalized :class:`~ehc_sn.traces.trace_tree.TraceTree`.
    """

    execution: RolloutExecution
    loss: Tensor
    last_step: ObservedStep
    trace_tree: TraceTree


# =============================================================================
def run_captured_rollout(  # -------------------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    snapshot_model_state: bool = False,
) -> RolloutChunk:
    """Execute a rollout and return the captured chunk."""
    executed = runner.run(
        source=source,
        controller=controller,
        carry=carry,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        options=dict(runner_options or {}),
        snapshot_model_state=snapshot_model_state,
    )
    if not isinstance(executed, RolloutChunk):
        raise TypeError(
            "Captured rollout execution requires step records, but the "
            "runner returned a recordless execution summary."
        )
    return executed


# =============================================================================
def score_captured_rollout(  # -----------------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    objective: RolloutScorer,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    scoring_input_builder: Callable[[StepRecord], object] | None = None,
    snapshot_model_state: bool = False,
) -> CapturedRolloutResult:
    """Execute a rollout chunk and score it with a pure objective."""
    executed = run_captured_rollout(
        runner=runner,
        source=source,
        controller=controller,
        carry=carry,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        runner_options=runner_options,
        snapshot_model_state=snapshot_model_state,
    )
    evaluated = score_rollout_chunk(
        executed, objective, scoring_input_builder=scoring_input_builder
    )
    return CapturedRolloutResult(chunk=executed, evaluated=evaluated)


# =============================================================================
def score_rollout_streaming(  # ----------------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    objective: RolloutScorer,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    scoring_input_builder: Callable[[StepRecord], object],
    observed_step_observer: Callable[[ObservedStep], None] | None = None,
    scored_record_observer: Callable[["ScoredRecord"], None] | None = None,
    snapshot_model_state: bool = False,
) -> StreamingRolloutResult:
    """Execute a rollout and score records on the fly without storing the full chunk.

    Parameters
    ----------
    observed_step_observer:
        Optional callback invoked on each scored ``ObservedStep`` (detached
        presentation evidence, no graph connection).  Safe for metric updates.
    scored_record_observer:
        Optional callback invoked on each ``ScoredRecord`` *after* the step
        is scored but *before* the record is discarded.  The ``ScoredRecord``
        contains the live loss tensor with its autograd graph intact.  This
        is the appropriate hook for per-step backward.

        .. warning::

            The ``scored_record_observer`` receives a graph-bearing loss.
            Call ``backward()`` or ``detach()`` before returning, or the
            graph will leak.
    """
    total_loss: Tensor | None = None
    last_step: ObservedStep | None = None

    def observe_record(record: StepRecord) -> None:
        nonlocal total_loss, last_step
        scored = score_rollout_record(
            record, objective, input_builder=scoring_input_builder
        )
        if scored_record_observer is not None:
            scored_record_observer(scored)
        if observed_step_observer is not None:
            observed_step_observer(scored.observed_step)
        last_step = scored.observed_step
        total_loss = (
            scored.loss if total_loss is None else total_loss + scored.loss
        )

    executed = runner.run(
        source=source,
        controller=controller,
        carry=carry,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        options=dict(runner_options or {}),
        record_observer=observe_record,
        capture_records=False,
        snapshot_model_state=snapshot_model_state,
    )
    if isinstance(executed, RolloutChunk):
        raise TypeError(
            "Streaming rollout evaluation expects a recordless execution "
            "summary, but the runner returned a captured chunk."
        )
    if total_loss is None or last_step is None:
        raise ValueError("Streaming rollout evaluation received no steps.")
    return StreamingRolloutResult(
        execution=executed, loss=total_loss, last_step=last_step
    )


# =============================================================================
def score_rollout_streaming_with_trace(  # ------------------------------------
    *,
    runner: Runner,
    source: Source,
    controller: StepController,
    carry: Any,
    objective: RolloutScorer,
    trace_spec: TraceSpec,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    scoring_input_builder: Callable[[StepRecord], object],
    trace_sink: TraceSink | None = None,
    observed_step_observer: Callable[[ObservedStep], None] | None = None,
    scored_record_observer: Callable[["ScoredRecord"], None] | None = None,
    snapshot_model_state: bool = False,
) -> StreamingRolloutResultWithTrace:
    """Execute a rollout, score records, and extract a trace in one streaming pass.

    Unlike :func:`score_captured_rollout` which accumulates all
    ``StepRecord`` objects, this function calls ``runner.run`` with
    ``capture_records=False`` and processes each record ephemerally
    through a callback.  The trace is written to a caller-chosen
    :class:`TraceSink` as fields are extracted.

    Parameters
    ----------
    runner, source, controller, carry, objective:
        Same as :func:`score_rollout_streaming`.
    trace_spec:
        Trace specification defining which fields to extract.
    max_rollout_steps, hard_max_rollout_steps, runner_options:
        Same as :func:`score_rollout_streaming`.
    scoring_input_builder:
        Callable that builds scoring input from each ``StepRecord``.
    trace_sink:
        Optional sink that receives extracted trace payloads.  When
        ``None``, an ``InMemoryTraceSink`` is created automatically.
    observed_step_observer, scored_record_observer:
        Same as :func:`score_rollout_streaming`.
    snapshot_model_state:
        Whether to snapshot ``model_state`` in the runner's carry snapshot.
        Automatically set to ``trace_spec.requires_model_state()`` when not
        explicitly provided.

    Returns
    -------
    StreamingRolloutResultWithTrace
        Streaming result with a finalized ``TraceArtifact``.
    """
    total_loss: Tensor | None = None
    last_step: ObservedStep | None = None
    trace_observer = TraceObserver(trace_spec)
    if trace_sink is None:
        trace_sink = InMemoryTraceSink(trace_spec)
    if not snapshot_model_state:
        snapshot_model_state = trace_spec.requires_model_state()

    def observe_record(record: StepRecord) -> None:
        nonlocal total_loss, last_step
        scored = score_rollout_record(
            record, objective, input_builder=scoring_input_builder
        )
        if scored_record_observer is not None:
            scored_record_observer(scored)
        if observed_step_observer is not None:
            observed_step_observer(scored.observed_step)
        last_step = scored.observed_step
        total_loss = (
            scored.loss if total_loss is None else total_loss + scored.loss
        )
        # Extract trace fields from the ephemeral record into the sink.
        trace_observer.observe(record, step_index=record.index, sink=trace_sink)

    executed = runner.run(
        source=source,
        controller=controller,
        carry=carry,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        options=dict(runner_options or {}),
        record_observer=observe_record,
        capture_records=False,
        snapshot_model_state=snapshot_model_state,
    )
    if isinstance(executed, RolloutChunk):
        raise TypeError(
            "Streaming rollout evaluation expects a recordless execution "
            "summary, but the runner returned a captured chunk."
        )
    if total_loss is None or last_step is None:
        raise ValueError("Streaming rollout evaluation received no steps.")
    return StreamingRolloutResultWithTrace(
        execution=executed,
        loss=total_loss,
        last_step=last_step,
        trace_tree=trace_sink.finalize(),
    )


# =============================================================================
__all__ = [
    "CapturedRolloutResult",
    "StreamingRolloutResult",
    "StreamingRolloutResultWithTrace",
    "run_captured_rollout",
    "score_captured_rollout",
    "score_rollout_streaming",
    "score_rollout_streaming_with_trace",
]
