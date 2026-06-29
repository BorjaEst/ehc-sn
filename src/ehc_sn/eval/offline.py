"""Offline evaluation runner — checkpoint + config → eval artifact.

Produces a persisted eval-artifact bundle from a model-family checkpoint
without a Lightning Trainer or training callback.  Uses only public APIs
from ``ehc_sn.eval``.

The runner receives a pre-resolved :class:`EvaluationExperiment` — all
provider, regime, capture, and identity resolution has already been
performed by the experiment-specific builder.

Constructs run-scoped consumers from ``plan.aggregate_specs`` and manages
their lifecycle (``begin_run``, ``begin_case`` / ``end_case`` per case,
``finalize``, ``close``).

Usage::

    from ehc_sn.analysis import CompiledFigurePlan
    from ehc_sn.eval.invocation import (
        EvaluationExecutionRequest,
        ResolvedModelArtifact,
    )
    from ehc_sn.eval.contracts import EvaluationExperiment

    artifact_dir = run_offline_eval(
        experiment=experiment,
        execution=EvaluationExecutionRequest(
            experiment=experiment,
            model_artifact=ResolvedModelArtifact(
                requested_uri="best.ckpt",
                resolved_source="/path/to/best.ckpt",
            ),
            output_dir=Path("/tmp/smoke"),
            device="cpu",
        ),
        plan=compiled_plan,
    )

Boundary rules:
    - Must not import from ``ehc_sn.reporting`` or ``ehc_sn.report_metrics``.
    - Must not call private ``_build_*`` or ``_hydrate_*`` from
      ``ehc_sn.eval.artifacts``.
    - Must not import from ``lightning/``, ``models/``, ``tasks/``,
      ``adapters/``, or ``callbacks/``.
    - Must not import from ``experiments/<task>/<family>/`` (receives
      pre-resolved objects).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ehc_sn.eval.artifacts import (
    _hydrate_executor_from_source,
    _initialize_eval_runtime,
    collect_regime_artifact_bundle,
    resolve_provider,
)
from ehc_sn.eval.consumers import TraceConsumer
from ehc_sn.eval.contracts import (
    AggregateBuildContext,
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationConsumer,
    EvaluationRunContext,
    ProducedArtifact,
)
from ehc_sn.eval.executor import iter_evaluation_regime
from ehc_sn.eval.invocation import (
    EvaluationExecutionRequest,
    EvaluationPrecision,
)
from ehc_sn.eval.reuse import (
    compute_checkpoint_sha256,
    compute_plan_digest_from_plan,
    find_compatible_artifact,
)
from ehc_sn.traces import TraceObserver
from ehc_sn.traces.sink import ZarrTraceSink

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ehc_sn.analysis.compiler import CompiledFigurePlan
    from ehc_sn.eval.contracts import EvaluationExperiment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device_batch(
    case: EvaluationCaseBatch,
    device: torch.device,
) -> EvaluationCaseBatch:
    """Move all tensor values in a case batch to *device*."""
    moved = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in case.batch.items()
    }
    return replace(case, batch=moved)


def _detach_case_result(
    result: EvaluationCaseResult,
) -> EvaluationCaseResult:
    """Detach all tensor values in a case result to CPU for safe persistence."""
    evaluated = result.evaluated
    safe_evaluated = replace(
        evaluated,
        loss=evaluated.loss.detach().to(device="cpu"),
    )
    return replace(result, evaluated=safe_evaluated, trace=None)


def _execute_analysis_specs(
    *,
    plan: CompiledFigurePlan,
    produced_artifacts: tuple[ProducedArtifact, ...],
    workspace: Path,
) -> list[ProducedArtifact]:
    """Run post-evaluation analysis specs and return their produced artifacts."""
    # Build an inventory of produced artifacts keyed by manifest_key.
    artifact_inventory: dict[str, Path] = {}
    for pa in produced_artifacts:
        artifact_inventory[pa.key.manifest_key()] = pa.path

    analysis_results: list[ProducedArtifact] = []
    for analysis_spec in plan.analysis_specs:
        # Check that all input requirements are satisfied.
        input_paths: dict[str, Path] = {}
        for req in analysis_spec.inputs:
            key = req.key.manifest_key()
            if key in artifact_inventory:
                input_paths[key] = workspace / artifact_inventory[key]
            else:
                # Try looking up in the aggregate directory.
                agg_path = workspace / "aggregates" / f"{req.key.name}.zarr"
                if agg_path.exists():
                    input_paths[key] = agg_path

        if len(input_paths) != len(analysis_spec.inputs):
            # Skip analysis if inputs not available — caller
            # can rerun with full evaluation.
            continue

        # Call the analysis runner.
        try:
            agg_input = next(iter(input_paths.values()))
            analysis_output = analysis_spec.runner(
                aggregate_artifact_path=agg_input,
                output_dir=workspace
                / "analyses"
                / f"{analysis_spec.name}.zarr",
            )
            analysis_results.extend(analysis_output)
        except Exception:
            logger.exception(
                "Analysis spec %r failed; skipping.",
                analysis_spec.name,
            )
    return analysis_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_offline_eval(
    *,
    experiment: EvaluationExperiment,
    execution: EvaluationExecutionRequest,
    plan: CompiledFigurePlan,
) -> Path:
    """Run one evaluation regime offline and persist the artifact bundle.

    Constructs run-scoped consumers from ``plan.aggregate_specs`` and manages
    their lifecycle (``begin_run``, ``begin_case`` / ``end_case`` per case,
    ``finalize``, ``close``).  A trace consumer is added when the experiment
    defines a trace spec.

    Consumers write output into a private staging workspace.  The artifact
    collector copies staged files into the final output directory during
    atomic commit.  This ensures partial outputs are never visible and
    ``_SUCCESS`` marks a complete artifact bundle.

    Parameters
    ----------
    experiment:
        Pre-resolved evaluation experiment returned by an experiment-specific
        builder.
    execution:
        Combined resolved experiment and model artifact for execution,
        including checkpoint path, output directory, device, and
        bounded-execution overrides.
    plan:
        Compiled figure plan.  Pass a baseline plan (empty requested_figures)
        for a trace-only evaluation without aggregates or analyses.

    Returns
    -------
    Path
        Resolved absolute path to the written artifact directory.

    Raises
    ------
    RuntimeError
        If no cases are produced by the provider.
    """
    device_str = execution.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # ---- Reuse check --------------------------------------------------------
    checkpoint_sha256: str = ""
    plan_digest: str = ""

    if not execution.no_reuse:
        model_source = Path(execution.model_artifact.resolved_source)
        # For artifact directories, hash the weights file for stable identity.
        if model_source.is_dir() and (model_source / "weights.pt").is_file():
            hash_target = model_source / "weights.pt"
        else:
            hash_target = model_source
        checkpoint_sha256 = compute_checkpoint_sha256(hash_target)
        plan_digest = compute_plan_digest_from_plan(
            plan,
            capture_profile_name=experiment.capture_profile or "",
            capture_profile_version=1,
        )

        existing = find_compatible_artifact(
            execution.output_dir.parent,
            plan_digest,
            alias=execution.output_dir.name,
            checkpoint_sha256=checkpoint_sha256,
        )
        if existing is not None:
            print(f"Reusing existing artifact at {existing}")
            return existing.resolve()

    # ---- Hydrate executor (CPU first, then move to device) ------------------
    # Load weights on CPU, then transfer the executor to the target device.
    # This avoids a period where both the initialized GPU parameters and the
    # CPU checkpoint tensors are live simultaneously.
    executor = experiment.executor
    model_source = Path(execution.model_artifact.resolved_source)
    _hydrate_executor_from_source(executor, str(model_source))
    executor.to(device)

    # Compute SHA-256 if not already done.
    if not checkpoint_sha256:
        checkpoint_sha256 = compute_checkpoint_sha256(model_source)

    # ---- Initialize evaluation runtime --------------------------------------
    _initialize_eval_runtime(executor)

    # ---- Put executor in evaluation mode ------------------------------------
    executor.eval()

    # ---- Resolve provider ---------------------------------------------------
    provider = resolve_provider(
        experiment.provider_spec.ref,
        experiment.provider_spec.settings,
        batch_size=experiment.provider_spec.batch_size,
    )

    identity = experiment.identity
    task_name = identity.task if identity is not None else "unknown"

    # ---- Build consumers and run context ------------------------------------
    run_scoped_consumers: list[EvaluationConsumer] = []

    run_ctx = EvaluationRunContext(
        run_id=experiment.regime_id,
        checkpoint_sha256="",
        dataset_identity=None,
    )

    for agg_spec in plan.aggregate_specs:
        build_ctx = AggregateBuildContext(
            experiment=experiment,
            run=run_ctx,
            output_root=execution.output_dir,
            device=device,
        )
        consumer = agg_spec.factory(build_ctx)
        run_scoped_consumers.append(consumer)

    # Add a trace consumer if the experiment has a trace spec.
    # Consumers write to a private staging workspace, not the final output dir.
    output_dir = execution.output_dir
    workspace = output_dir.parent / f".{output_dir.name}.work"
    workspace.mkdir(parents=True, exist_ok=True)

    if experiment.trace_spec is not None:
        trace_sink = ZarrTraceSink(
            path=workspace / "traces",
            spec=experiment.trace_spec,
            chunk_size=256,
        )
        trace_observer = TraceObserver(experiment.trace_spec)
        run_scoped_consumers.append(
            TraceConsumer(trace_observer, trace_sink, name="trace")
        )

    # Create autocast context if needed.
    precision = execution.precision
    autocast_ctx = nullcontext()
    if device.type == "cuda":
        _autocast_dtype: torch.dtype | None = None
        if precision == EvaluationPrecision.BF16_MIXED:
            _autocast_dtype = torch.bfloat16
        elif precision == EvaluationPrecision.FLOAT16_MIXED:
            _autocast_dtype = torch.float16
        if _autocast_dtype is not None:
            autocast_ctx = torch.autocast(
                device_type="cuda", dtype=_autocast_dtype
            )

    # Track consumers whose begin_run completed for ordered cleanup.
    started_consumers: list[EvaluationConsumer] = []

    try:
        # ---- begin_run -------------------------------------------------------
        for consumer in run_scoped_consumers:
            consumer.begin_run(run_ctx)
            started_consumers.append(consumer)

        # ---- Batch preparation ------------------------------------------------
        def _prepare_batch(case: EvaluationCaseBatch) -> EvaluationCaseBatch:
            return _to_device_batch(case, device)

        # ---- Execute cases (generator consumed INSIDE autocast) --------------
        safe_results: list[EvaluationCaseResult] = []

        with autocast_ctx:
            for result in iter_evaluation_regime(
                provider,
                executor,
                consumers=tuple(started_consumers),
                max_batches=execution.max_batches,
                max_samples=None,
                trace_request=None,
                prepare_case_batch=_prepare_batch,
            ):
                safe_results.append(_detach_case_result(result))

        # ---- Attach per-case metadata to the TraceConsumer --------------------
        # Extract case_meta from consumer_results and route it to the trace
        # consumer so the trace index carries static task-level metadata
        # (cell_type, start_flag, etc.) needed by figure rendering.
        for consumer in started_consumers:
            if hasattr(consumer, "attach_case_meta"):
                consumer: Any = consumer  # noqa: PLW2901
        for result in safe_results:
            cr = result.consumer_results or {}
            case_meta = cr.get("case_meta")
            if isinstance(case_meta, dict) and case_meta:
                for consumer in started_consumers:
                    if hasattr(consumer, "attach_case_meta"):
                        try:
                            consumer.attach_case_meta(result.case_id, case_meta)
                        except Exception:
                            pass

        # ---- Finalize consumers BEFORE artifact collection --------------------
        produced_artifacts: list[ProducedArtifact] = []
        for consumer in started_consumers:
            produced_artifacts.extend(consumer.finalize())

        # ---- Run post-evaluation analysis specs ------------------------------
        if plan.analysis_specs:
            analysis_artifacts = _execute_analysis_specs(
                plan=plan,
                produced_artifacts=tuple(produced_artifacts),
                workspace=workspace,
            )
            produced_artifacts.extend(analysis_artifacts)

        # ---- Collect and commit the artifact bundle --------------------------
        collect_regime_artifact_bundle(
            case_results=iter(safe_results),
            executor=executor,
            provider=provider,
            regime_id=experiment.regime_id,
            regime_kind=experiment.regime_kind,
            run_dir=output_dir,
            trace_spec=experiment.trace_spec,
            max_batches=execution.max_batches,
            trigger_kind="offline",
            epoch=0,
            step=0,
            produced_artifacts=tuple(produced_artifacts),
            provenance_overrides={
                "alias": execution.output_dir.name,
                "model_family": identity.model_family if identity else None,
                "trace_paradigm": identity.trace_paradigm if identity else None,
                "task": task_name,
                "capture_profile": experiment.capture_profile,
                "capture_profile_version": 1,
                "capture_include": [],
                "capture_exclude": [],
                "request": {
                    "max_batches": execution.max_batches,
                    "max_samples": None,
                    "batch_size": experiment.provider_spec.batch_size,
                    "max_cases": experiment.capture_max_cases,
                },
                "checkpoint_sha256": checkpoint_sha256,
                "plan_digest": plan_digest,
            },
        )

    finally:
        # ---- Close consumers (resource cleanup, outermost finally) -----------
        for consumer in reversed(started_consumers):
            try:
                consumer.close()
            except Exception:
                logger.exception(
                    "Failed to close evaluation consumer %r",
                    consumer.name,
                )
        # ---- Clean up staging workspace --------------------------------------
        try:
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            logger.exception(
                "Failed to clean up staging workspace %r", workspace
            )

    return output_dir.resolve()


# ---------------------------------------------------------------------------
__all__ = ["run_offline_eval"]
