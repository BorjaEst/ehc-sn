"""Reusable top-level replay evaluation execution helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationExecutor,
    EvaluationSourceProvider,
    EvaluationTraceRequest,
)
from ehc_sn.objectives.rollout import RolloutScorer
from ehc_sn.rollouts.runtime import Runner, StepController
from ehc_sn.rollouts.sources import RepeatSource
from ehc_sn.traces.rollout import observe_rollout_chunk
from ehc_sn.training.rollout import score_captured_rollout


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
    objective_options: Mapping[str, object] | None = None,
    trace_request: EvaluationTraceRequest | None = None,
) -> EvaluationCaseResult:
    """Execute and score one replay case with optional trace materialization."""
    evaluation = score_captured_rollout(
        runner=runner,
        source=RepeatSource(case.batch),
        controller=controller,
        carry=carry,
        objective=objective,
        max_rollout_steps=max_rollout_steps,
        hard_max_rollout_steps=hard_max_rollout_steps,
        runner_options=runner_options,
        objective_options=objective_options,
    )

    trace = None
    if trace_request is not None:
        trace = observe_rollout_chunk(
            evaluation.chunk,
            trace_request.trace_spec,
            trace_meta=trace_request.trace_meta,
        )

    return EvaluationCaseResult(
        case_id=case.case_id,
        evaluated=evaluation.evaluated,
        source_context=case.source_context,
        trace=trace,
    )


# =============================================================================
def iter_evaluation_regime(
    provider: EvaluationSourceProvider,
    executor: EvaluationExecutor,
    *,
    max_batches: int = 0,
    trace_request: EvaluationTraceRequest | None = None,
) -> Iterator[EvaluationCaseResult]:
    """Yield executor results for all provider batches in one regime."""
    for case in provider.provide_cases(max_batches=max_batches):
        yield executor.execute_evaluation_batch(
            case,
            trace_request=trace_request,
        )


# =============================================================================
__all__ = ["execute_replay_evaluation_batch", "iter_evaluation_regime"]
