"""Offline evaluation runner — checkpoint + config → eval artifact v3.

Produces a persisted eval-artifact bundle from a model-family checkpoint
without a Lightning Trainer or training callback.  Uses only public APIs
from ``ehc_sn.eval``.

The runner receives a pre-resolved :class:`EvaluationExperiment` — all
provider, regime, capture, and identity resolution has already been
performed by the experiment-specific builder.

Usage::

    from ehc_sn.experiments._infra import (
        EvaluationExperiment,
        EvaluationRunRequest,
    )

    artifact_dir = run_offline_eval(
        experiment=experiment,
        request=EvaluationRunRequest(
            checkpoint_path=Path("best.ckpt"),
            output_dir=Path("/tmp/smoke"),
            device="cpu",
        ),
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

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ehc_sn.eval.artifacts import (
    _hydrate_executor_from_checkpoint,
    collect_regime_artifact_bundle,
    resolve_provider,
)
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import iter_evaluation_regime
from ehc_sn.experiments._infra import (
    EvaluationExperiment,
    EvaluationRunRequest,
)

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_offline_eval(
    *,
    experiment: EvaluationExperiment,
    request: EvaluationRunRequest,
) -> Path:
    """Run one evaluation regime offline and persist the artifact bundle.

    Parameters
    ----------
    experiment:
        Pre-resolved evaluation experiment returned by an experiment-specific
        builder.  Contains the executor, provider specification, regime
        identity, trace specification, and experiment identity.
    request:
        Run-instance parameters — checkpoint, output directory, device,
        and bounded-execution overrides.

    Returns
    -------
    Path
        Resolved absolute path to the written artifact directory.

    Raises
    ------
    RuntimeError
        If no cases are produced by the provider.
    FileNotFoundError
        If ``checkpoint_path`` does not exist (delegated to checkpoint loading).
    ValueError
        If the capture profile cannot be resolved for the experiment's trace
        paradigm (delegated to ``resolve_capture_profile``).
    KeyError
        If a required summary field from the aggregation hook collides with
        reserved keys.
    """
    device = torch.device(request.device)

    # ---- Hydrate executor ---------------------------------------------------
    executor = experiment.executor
    executor.to(device)

    # ---- Load checkpoint ----------------------------------------------------
    _hydrate_executor_from_checkpoint(executor, request.checkpoint_path)

    # ---- Resolve provider ---------------------------------------------------
    provider = resolve_provider(
        experiment.provider_spec.ref, experiment.provider_spec.settings
    )

    # ---- Build trace request ------------------------------------------------
    trace_request: EvaluationTraceRequest | None = None
    if experiment.trace_spec is not None:
        trace_request = EvaluationTraceRequest(trace_spec=experiment.trace_spec)

    identity = experiment.identity
    task_name = identity.task if identity is not None else "unknown"

    # ---- Build case result generator ----------------------------------------
    # Cases are streamed to collect_regime_artifact_bundle one at a time.
    # No list of all case results is retained — only lightweight summary rows
    # accumulate inside the artifact writer for the final manifest.
    case_results = iter_evaluation_regime(
        provider,
        executor,
        max_batches=request.max_batches,
        trace_request=trace_request,
        prepare_case_batch=lambda case: _to_device_batch(case, device),
    )

    # ---- Persist (streaming) ------------------------------------------------
    # collect_regime_artifact_bundle writes each case to .npz as it arrives
    # and returns a lightweight regime result with summary metrics.
    regime_result = collect_regime_artifact_bundle(
        case_results=case_results,
        executor=executor,
        provider=provider,
        regime_id=experiment.regime_id,
        regime_kind=experiment.regime_kind,
        run_dir=request.output_dir,
        trace_spec=experiment.trace_spec,
        max_batches=request.max_batches,
        trigger_kind="offline",
        epoch=0,
        step=0,
        provenance_overrides={
            "model_family": identity.model_family if identity else None,
            "trace_paradigm": identity.trace_paradigm if identity else None,
            "task": task_name,
            "capture_profile": experiment.capture_profile,
            "capture_profile_version": 1,
            "capture_include": list(experiment.capture_include),
            "capture_exclude": list(experiment.capture_exclude),
        },
    )

    return request.output_dir.resolve()


# ---------------------------------------------------------------------------
__all__ = ["run_offline_eval"]
