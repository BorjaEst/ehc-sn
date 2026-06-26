"""Reusable top-level replay evaluation execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from torch import Tensor

from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationExecutor,
    EvaluationSourceProvider,
    EvaluationTraceRequest,
)
from ehc_sn.rollouts.materialization import EvaluatedChunk, ObservedStep
from ehc_sn.rollouts.runtime import (
    RolloutExecution,
    Runner,
    StepController,
    StepRecord,
)
from ehc_sn.rollouts.scoring import RolloutScorer, score_rollout_record
from ehc_sn.rollouts.sources import RepeatSource
from ehc_sn.traces.observer import TraceObserver
from ehc_sn.traces.sink import InMemoryTraceSink, TraceSink
from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
def execute_replay_evaluation_batch(
    *,
    case: EvaluationCaseBatch,
    runner: Runner,
    controller: StepController,
    carry: Any,
    objective: RolloutScorer,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    scoring_input_builder: Callable[[StepRecord], object] | None = None,
    trace_request: EvaluationTraceRequest | None = None,
) -> EvaluationCaseResult:
    """Execute and score one replay case with optional trace materialization.

    Uses a streaming execution path: ``StepRecord`` objects are ephemeral
    and discarded after scoring and trace extraction.  Observed steps are
    retained for metric aggregation; the full carry snapshots are not.
    """
    trace_sink: TraceSink | None = None
    trace_observer: TraceObserver | None = None
    if trace_request is not None:
        trace_sink = InMemoryTraceSink(trace_request.trace_spec)
        trace_observer = TraceObserver(trace_request.trace_spec)

    total_loss: Tensor | None = None
    observed_steps: list[ObservedStep] = []

    def observe_record(record: StepRecord) -> None:
        nonlocal total_loss
        scored = score_rollout_record(
            record, objective, input_builder=scoring_input_builder
        )
        observed_steps.append(scored.observed_step)
        total_loss = (
            scored.loss if total_loss is None else total_loss + scored.loss
        )
        if trace_observer is not None and trace_sink is not None:
            trace_observer.observe(
                record, step_index=record.index, sink=trace_sink
            )

    snapshot_model_state = (
        _trace_request_needs_model_state(trace_request)
        if trace_request is not None
        else False
    )
    executed = runner.run(
        source=RepeatSource(case.batch),
        controller=controller,
        carry=carry,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        options=dict(runner_options or {}),
        record_observer=observe_record,
        capture_records=False,
        snapshot_model_state=snapshot_model_state,
    )

    if total_loss is None or not observed_steps:
        raise ValueError("Evaluation received no steps.")

    trace_tree: TraceTree | None = None
    if trace_sink is not None and trace_request is not None:
        trace_tree = trace_sink.finalize()
        if trace_request.trace_meta is not None:
            trace_tree.attach_meta(trace_request.trace_meta, overwrite=True)

    evaluation = EvaluatedChunk(
        steps=tuple(observed_steps),
        loss=total_loss,
        final_carry=(
            executed.final_carry
            if isinstance(executed, RolloutExecution)
            else carry
        ),
        source_exhausted=(
            executed.source_exhausted
            if isinstance(executed, RolloutExecution)
            else False
        ),
    )

    return EvaluationCaseResult(
        case_id=case.case_id,
        evaluated=evaluation,
        source_context=case.source_context,
        trace=trace_tree,
    )


# =============================================================================
def iter_evaluation_regime(
    provider: EvaluationSourceProvider,
    executor: EvaluationExecutor,
    *,
    max_batches: int = 0,
    max_samples: int | None = None,
    trace_request: EvaluationTraceRequest | None = None,
    prepare_case_batch: (
        Callable[[EvaluationCaseBatch], EvaluationCaseBatch] | None
    ) = None,
) -> Iterator[EvaluationCaseResult]:
    """Yield executor results for all provider batches in one regime.

    ``prepare_case_batch`` exists for out-of-loop orchestration such as named
    regime execution, where provider batches must be normalized or moved to the
    active runtime device before entering the family evaluation seam.
    """
    for case in provider.provide_cases(
        max_batches=max_batches, max_samples=max_samples
    ):
        if prepare_case_batch is not None:
            case = prepare_case_batch(case)
        yield executor.execute_evaluation_batch(
            case,
            trace_request=trace_request,
        )


# =============================================================================
def _trace_request_needs_model_state(
    trace_request: EvaluationTraceRequest | None,
) -> bool:
    """Return whether requested trace fields require carry.model_state snapshots."""
    if trace_request is None:
        return False
    return trace_request.trace_spec.requires_model_state()


# =============================================================================
__all__ = ["execute_replay_evaluation_batch", "iter_evaluation_regime"]
