"""Reusable top-level replay evaluation execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ehc_sn.contracts.dependencies import Dependency, DependencyKind
from ehc_sn.evaluation.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseContext,
    EvaluationCaseResult,
    EvaluationConsumer,
    EvaluationExecutor,
    EvaluationSourceProvider,
    EvaluationTraceRequest,
)
from ehc_sn.rollouts.materialization import ObservedStep
from ehc_sn.rollouts.runtime import (
    RolloutExecution,
    Runner,
    StepController,
    StepRecord,
)
from ehc_sn.rollouts.scoring import (
    RolloutAccumulator,
    RolloutScorer,
    score_rollout_record,
)
from ehc_sn.rollouts.sources import RepeatSource
from ehc_sn.traces.observer import StepContext, TraceObserver
from ehc_sn.traces.sink import InMemoryTraceSink
from ehc_sn.traces.trace_tree import TraceTree

if TYPE_CHECKING:
    from ehc_sn.types import Batch


# =============================================================================
def execute_replay_evaluation_batch(
    *,
    case: EvaluationCaseBatch,
    runner: Runner,
    controller: StepController,
    carry: Any,
    objective: RolloutScorer,
    model: nn.Module | None = None,
    max_rollout_steps: int | None = None,
    hard_max_rollout_steps: int | None = None,
    runner_options: Mapping[str, object] | None = None,
    scoring_input_builder: Callable[[StepRecord], object] | None = None,
    consumers: Sequence[EvaluationConsumer] = (),
    trace_request: EvaluationTraceRequest | None = None,
    metric_observer: Callable[[ObservedStep], None] | None = None,
    case_meta_fn: (
        Callable[[Batch, dict[str, object]], dict[str, object]] | None
    ) = None,
) -> EvaluationCaseResult:
    """Execute and score one replay case with consumer per-step observers.

    Uses a streaming execution path: ``StepRecord`` objects are ephemeral
    and discarded after scoring and consumer observation.  Under
    ``torch.inference_mode()`` the autograd graph does not accumulate
    across recurrent steps, and a ``RolloutAccumulator`` retains only
    sufficient statistics (loss sum + final step) rather than a full
    list of ``ObservedStep`` per step.

    When ``trace_request`` is provided, builds a ``TraceObserver`` and
    ``InMemoryTraceSink`` directly (bypassing ``TraceConsumer`` for
    backward compatibility).  The observer writes per-step payloads to
    the sink during the rollout, and the finalized ``TraceTree`` is
    returned via ``EvaluationCaseResult.trace``.

    Consumers receive ``update(ctx)`` at every step, ``finalize()`` after
    the runner loop, and ``close()`` in a ``finally`` block (reverse
    order).  Duplicate consumer names raise ``ValueError`` before any
    execution.

    ``case_meta_fn`` is called with ``batch`` after the rollout and should
    return a JSON-serializable dict of per-case metadata (task-level static
    data like cell_type, start_flag).  The result is returned via
    ``consumer_results["case_meta"]`` for the orchestrator to route to the
    ``TraceConsumer``.

    The complete function body executes under ``torch.inference_mode()``
    so that no autograd graph is constructed during evaluation — every
    recurrent step, scoring call, and trace observation is covered.
    """
    # ---- Validate consumer names -------------------------------------------
    _validate_unique_consumer_names(consumers)

    # ---- Build trace capture plumbing --------------------------------------
    trace_observer: TraceObserver | None = None
    trace_sink: InMemoryTraceSink | None = None
    if trace_request is not None:
        trace_observer = TraceObserver(trace_request.trace_spec)
        trace_sink = InMemoryTraceSink(trace_request.trace_spec, max_steps=128)

    # ---- Union consumer + trace dependencies --------------------------------
    all_deps: set[Dependency] = set()
    for consumer in consumers:
        all_deps.update(consumer.dependencies)
    if trace_request is not None:
        for dep in trace_request.trace_spec.resolved_dependencies():
            if isinstance(dep, Dependency):
                all_deps.add(dep)
            elif isinstance(dep, str):
                # Some trace fields still use plain strings for dependencies
                # (legacy).  Treat as model-view dependencies by name.
                all_deps.add(Dependency(DependencyKind.MODEL_VIEW, dep))

    required_views = frozenset(
        dep.name for dep in all_deps if dep.kind is DependencyKind.MODEL_VIEW
    )

    # ---- Per-case consumer lifecycle (begin_case) ---------------------------
    case_ctx = EvaluationCaseContext(
        case_id=case.case_id,
        batch_size=case.n_samples,
    )
    for consumer in consumers:
        consumer.begin_case(case_ctx)

    try:
        with torch.inference_mode():
            accumulator: RolloutAccumulator = RolloutAccumulator()

            def observe_record(record: StepRecord) -> None:
                scored = score_rollout_record(
                    record, objective, input_builder=scoring_input_builder
                )
                accumulator.observe(scored)
                # Fold per-step metrics into the optional metric observer
                # (accumulates RatioStat increments across all steps).
                if metric_observer is not None:
                    metric_observer(scored.observed_step)
                # Extract semantic views from the combined dependency set.
                views: dict[str, Tensor] = {}
                if required_views and model is not None:
                    ms = record.snapshot.model_state
                    views = model.trace_views(required_views, state=ms)
                ctx = StepContext(
                    index=record.index,
                    record=record,
                    views=views,
                )
                for consumer in consumers:
                    consumer.update(ctx)
                if trace_observer is not None:
                    assert trace_sink is not None
                    trace_observer.observe(
                        ctx, step_index=record.index, sink=trace_sink
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
            )

            final_carry = (
                executed.final_carry
                if isinstance(executed, RolloutExecution)
                else carry
            )
            source_exhausted = (
                executed.source_exhausted
                if isinstance(executed, RolloutExecution)
                else False
            )

            evaluation = accumulator.finalize(
                final_carry=final_carry,
                source_exhausted=source_exhausted,
            )

        # inference_mode ends here — the EvaluatedChunk is now safe to
        # pass to post-rollout lifecycle hooks.

        # Extract per-case metadata if case_meta_fn is provided.
        case_meta: dict[str, object] = {}
        if case_meta_fn is not None:
            try:
                case_meta = case_meta_fn(case.batch)
            except Exception:
                pass

        # End per-case lifecycle for all consumers (reverse order).
        for consumer in reversed(consumers):
            consumer.end_case(case_ctx)

        # Finalize trace sink (legacy path) and attach metadata.
        trace: TraceTree | None = None
        if trace_sink is not None:
            trace = trace_sink.finalize()
            if trace_request is not None and trace_request.trace_meta:
                trace.attach_meta(trace_request.trace_meta)

        return EvaluationCaseResult(
            case_id=case.case_id,
            evaluated=evaluation,
            source_context=case.source_context,
            trace=trace,
            consumer_results={"case_meta": case_meta} if case_meta else {},
        )
    except BaseException:
        # On failure, end_case is skipped for this case.
        # The runner-level close() handles cleanup.
        raise


# =============================================================================
def iter_evaluation_regime(
    provider: EvaluationSourceProvider,
    executor: EvaluationExecutor,
    *,
    consumers: Sequence[EvaluationConsumer] = (),
    max_batches: int = 0,
    max_samples: int | None = None,
    trace_request: EvaluationTraceRequest | None = None,
    prepare_case_batch: (
        Callable[[EvaluationCaseBatch], EvaluationCaseBatch] | None
    ) = None,
) -> Iterator[EvaluationCaseResult]:
    """Yield executor results for all provider batches in one regime.

    ``consumers`` are forwarded to the executor for per-case lifecycle
    (``begin_case``, ``update``, ``end_case``).  The caller owns the
    run-level lifecycle (``begin_run``, ``finalize``, ``close``).

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
            consumers=consumers,
            trace_request=trace_request,
        )


# =============================================================================
# =============================================================================
# =============================================================================
def _validate_unique_consumer_names(
    consumers: Sequence[EvaluationConsumer],
) -> None:
    """Raise ``ValueError`` if any two consumers share the same ``name``."""
    names = [consumer.name for consumer in consumers]
    if len(names) != len(set(names)):
        from collections import Counter

        dupes = {name for name, count in Counter(names).items() if count > 1}
        raise ValueError(
            "Evaluation consumer names must be unique; "
            f"duplicates={sorted(dupes)!r}"
        )


# =============================================================================
__all__ = ["execute_replay_evaluation_batch", "iter_evaluation_regime"]
