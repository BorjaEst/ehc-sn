"""Offline evaluation runner — checkpoint + config → eval artifact v3.

Produces a persisted eval-artifact bundle from a model-family checkpoint
without a Lightning Trainer or training callback.  Uses only public APIs
from ``ehc_sn.eval``.

Usage::

    artifact_dir = run_offline_eval(
        model_family="hrm_v2",
        executor_config_path=Path("config.toml"),
        checkpoint_path=Path("best.ckpt"),
        task="mazehard",
        provider_ref="ehc_sn.tasks.mazehard.providers.MazeHardReplayProvider",
        provider_settings={"dataset_path": str(data_root), "n_cases": 2},
        regime_id="mazehard_smoke",
        regime_kind="diagnostic",
        output_dir=Path("/tmp/mazehard_smoke"),
        device="cpu",
    )

Boundary rules:
    - Must not import from ``ehc_sn.reporting`` or ``ehc_sn.report_metrics``.
    - Must not call private ``_build_*`` or ``_hydrate_*`` from
      ``ehc_sn.eval.artifacts``.
    - Must not import from ``lightning/``, ``models/``, ``tasks/``,
      ``adapters/``, or ``callbacks/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import torch

from ehc_sn.eval.artifacts import (
    EvalArtifactExecutorRef,
    load_executor_from_artifact,
    persist_regime_artifact_bundle,
    resolve_provider,
)
from ehc_sn.eval.contracts import (
    EvaluationCaseBatch,
    EvaluationCaseResult,
    EvaluationRegimeResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import iter_evaluation_regime
from ehc_sn.metrics.values import validate_metric_value
from ehc_sn.traces import TraceField
from ehc_sn.traces.specs import build_trace_spec

# ---------------------------------------------------------------------------
# Reserved keys that model aggregation hooks must not return.
# Mirrored from callbacks/evaluation.py to avoid cross-module coupling.
# ---------------------------------------------------------------------------

_RESERVED_SUMMARY_KEYS: frozenset[str] = frozenset({"n_cases", "loss"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_loss_scalar(evaluated: Any) -> float | None:
    """Extract a scalar loss from an evaluated chunk, or None."""
    loss = getattr(evaluated, "loss", None)
    if loss is None:
        return None
    if torch.is_tensor(loss):
        if loss.numel() != 1:
            return None
        return float(loss.detach().item())
    if isinstance(loss, Real):
        return float(loss)
    return None


def _validate_hook_metrics(
    task_summary: dict[str, object],
) -> dict[str, float | int]:
    """Validate and type-narrow a family-owned hook return value.

    Rules match ``callbacks/evaluation._validate_hook_metrics``:
        - Keys in ``_RESERVED_SUMMARY_KEYS`` are rejected.
        - Per-value validation delegates to
          :func:`ehc_sn.metrics.values.validate_metric_value`.
    """
    collided = _RESERVED_SUMMARY_KEYS & task_summary.keys()
    if collided:
        raise ValueError(
            "Hook returned reserved summary keys: " f"{sorted(collided)}."
        )

    validated: dict[str, float | int] = {}
    for key, value in task_summary.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Metric key must be str, got {type(key).__name__} ({key!r})."
            )
        if not key:
            raise ValueError("Metric key must be a non-empty string.")
        validated[key] = validate_metric_value(key=key, value=value)
    return validated


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
    model_family: str,
    executor_config_path: Path,
    checkpoint_path: Path,
    task: str,
    provider_ref: str,
    provider_settings: dict[str, Any],
    regime_id: str,
    regime_kind: Literal["diagnostic", "benchmark"],
    output_dir: Path,
    device: str | torch.device = "cpu",
    max_batches: int = 0,
    trace_keys: list[str] | None = None,
    extra_trace_fields: Sequence[TraceField] | None = None,
) -> Path:
    """Run one evaluation regime offline and persist the artifact bundle.

    Parameters
    ----------
    model_family:
        Canonical model-family identifier (e.g. ``"hrm_v2"``).  Validated
        by ``EvalArtifactExecutorRef``.
    executor_config_path:
        Path to a TOML config file for the model family's executor.
    checkpoint_path:
        Path to a weights-only checkpoint file.
    task:
        Canonical task identifier (e.g. ``"mazehard"``).
    provider_ref:
        Dotted import path to an ``EvaluationSourceProvider`` class.
    provider_settings:
        Keyword arguments forwarded to the provider constructor.
    regime_id:
        Unique regime identifier written into the artifact manifest.
    regime_kind:
        Regime classification — ``"diagnostic"`` or ``"benchmark"``.
    output_dir:
        Target directory for the persisted eval artifact.
    device:
        PyTorch device for executor and batch placement.
    max_batches:
        Maximum provider batches to run.  ``0`` means all.
    trace_keys:
        Optional list of semantic trace key strings (e.g.
        ``["act/halted", "pred/solution_overlay"]``).  When provided,
        an ``EvaluationTraceRequest`` is built from the executor's
        ``_trace_paradigm`` and passed to the evaluation loop so that
        trace material is persisted in the artifact bundle.  ``None``
        or empty list means no trace materialization (default).
    extra_trace_fields:
        Optional sequence of :class:`~ehc_sn.traces.TraceField` objects
        appended to the trace spec as *extra_fields* (e.g.
        ``HRM_HIDDEN_STATE_FIELDS``).  ``None`` or empty sequence
        means no extra fields.

    Returns
    -------
    Path
        Resolved absolute path to the written artifact directory.

    Raises
    ------
    RuntimeError
        If no cases are produced by the provider.
    FileNotFoundError
        If ``executor_config_path`` or ``checkpoint_path`` does not exist
        (delegated to ``load_executor_from_artifact``).
    ValueError
        If ``model_family`` is unsupported (delegated to
        ``EvalArtifactExecutorRef`` validation), or if ``trace_keys``
        includes unknown keys (delegated to ``build_trace_spec``).
    KeyError
        If a required summary field from the aggregation hook collides with
        reserved keys.
    """
    device = torch.device(device)

    # ---- Construct executor -------------------------------------------------
    ref = EvalArtifactExecutorRef(
        model_family=model_family,
        executor_config_path=executor_config_path,
        checkpoint_path=checkpoint_path,
    )
    executor = load_executor_from_artifact(ref)
    executor.to(device)

    # ---- Resolve provider and run cases -------------------------------------
    provider = resolve_provider(provider_ref, provider_settings)

    # Build optional trace request.
    trace_request: EvaluationTraceRequest | None = None
    if trace_keys:
        paradigm = getattr(executor, "_trace_paradigm", None)
        if paradigm is None:
            raise ValueError(
                "Executor does not have a _trace_paradigm attribute. "
                "Cannot build trace request without knowing the trace paradigm."
            )

        # Discover adapter-specific trace fields from the executor.
        extra = extra_trace_fields
        if extra is None:
            extra = getattr(executor, "_extra_trace_fields", None)

        trace_spec = build_trace_spec(
            paradigm,
            include_keys=trace_keys,  # type: ignore[arg-type]
            extra_fields=extra,
        )
        trace_request = EvaluationTraceRequest(trace_spec=trace_spec)

    case_results: list[EvaluationCaseResult] = []
    for result in iter_evaluation_regime(
        provider,
        executor,
        max_batches=max_batches,
        trace_request=trace_request,
        prepare_case_batch=lambda case: _to_device_batch(case, device),
    ):
        case_results.append(result)

    if not case_results:
        raise RuntimeError(
            "No evaluation cases were produced. "
            f"Provider: {provider_ref!r}. "
            f"Regime: {regime_id!r}."
        )

    # ---- Build summary ------------------------------------------------------
    losses = [
        value
        for value in (_extract_loss_scalar(r.evaluated) for r in case_results)
        if value is not None
    ]
    summary: dict[str, object] = {"n_cases": len(case_results)}
    if losses:
        summary["loss"] = sum(losses) / len(losses)

    # Optional model-family aggregation hook.
    aggregate = getattr(executor, "aggregate_evaluation_case_metrics", None)
    if aggregate is not None:
        task_summary = aggregate(
            task=task,
            regime_id=regime_id,
            regime_kind=regime_kind,
            case_results=case_results,
        )
        validated = _validate_hook_metrics(task_summary)
        summary.update(validated)

    # ---- Persist ------------------------------------------------------------
    regime_result = EvaluationRegimeResult(
        regime_id=regime_id,
        case_results=tuple(case_results),
        summary=summary,
    )
    persist_regime_artifact_bundle(
        run_dir=output_dir,
        task=task,
        regime_kind=regime_kind,
        regime_id=regime_id,
        trigger_kind="offline",
        epoch=0,
        step=0,
        regime_result=regime_result,
    )
    return output_dir.resolve()


# ---------------------------------------------------------------------------
__all__ = ["run_offline_eval"]
