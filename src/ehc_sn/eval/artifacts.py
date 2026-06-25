"""Eval-owned artifact collection and persistence helpers.

This module owns the long-term on-disk contract for persisted evaluation
artifacts (traces, metadata, manifests).
"""

from __future__ import annotations

import importlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

from ehc_sn.eval.contracts import (
    EvaluationCaseResult,
    EvaluationRegimeResult,
    EvaluationTraceRequest,
)
from ehc_sn.eval.executor import iter_evaluation_regime
from ehc_sn.figures import REGISTRY, list_figures
from ehc_sn.traces import build_trace_spec
from ehc_sn.traces.observer import TraceSpec
from ehc_sn.traces.trace_tree import TraceTree

_ARTIFACT_SCHEMA = "ehc_sn.eval.artifact.v3"
_MANIFEST_FILENAME = "manifest.json"
_SUCCESS_FILENAME = "_SUCCESS"


# =============================================================================
class EvalArtifactExecutorRef(BaseModel, extra="forbid"):
    """Typed artifact metadata used to reconstruct an evaluation executor."""

    schema_version: str = "1"
    artifact_type: Literal["evaluation_executor"] = "evaluation_executor"
    experiment_id: str = Field(..., min_length=1)
    executor_config_path: Path
    checkpoint_path: Path
    checkpoint_format: Literal["weights_only"] = "weights_only"

    @model_validator(mode="after")
    def _validate_experiment_id(self) -> "EvalArtifactExecutorRef":
        from ehc_sn.eval.registry import (
            get_evaluation_experiment_registration,
            list_experiment_ids,
        )

        self.experiment_id = self.experiment_id.strip().lower()
        try:
            get_evaluation_experiment_registration(self.experiment_id)
        except KeyError:
            raise ValueError(
                f"Unsupported experiment_id {self.experiment_id!r}. "
                f"Available: {list_experiment_ids()}."
            ) from None
        return self

    @classmethod
    def from_legacy_family(
        cls, *, model_family: str, task: str, **kwargs
    ) -> "EvalArtifactExecutorRef":
        """Construct from legacy ``model_family`` + ``task`` identity.

        Raises ``ValueError`` if the combination does not resolve to a
        known experiment_id.
        """
        from ehc_sn.eval.registry import (
            get_evaluation_experiment_registration,
            list_experiment_ids,
        )

        experiment_id = f"{model_family.strip().lower()}-{task.strip().lower()}"
        try:
            get_evaluation_experiment_registration(experiment_id)
        except KeyError:
            raise ValueError(
                f"Cannot resolve legacy model_family={model_family!r} "
                f"+ task={task!r} to a known experiment_id. "
                f"Available: {list_experiment_ids()}."
            ) from None
        return cls(experiment_id=experiment_id, **kwargs)


# =============================================================================
@dataclass(frozen=True)
class LoadedArtifactCase:
    """One persisted case trace loaded from an eval run directory."""

    case_id: str
    source_context: object | None
    trace: TraceTree
    temporal_semantics: dict[str, object] | None = None


# =============================================================================
def _extract_model_family(executor: Any) -> str | None:
    """Extract a model-family string from a loaded executor, if possible."""
    # Direct attribute on executor objects
    family = getattr(executor, "_model_family", None)
    if family is not None:
        return str(family)

    # Fallback: try _trace_paradigm as a proxy for model family
    paradigm = getattr(executor, "_trace_paradigm", None)
    if paradigm is not None:
        return str(paradigm)
    return None


# =============================================================================
def _resolve_temporal_semantics(executor: Any) -> dict[str, object]:
    """Extract temporal-semantics metadata from an evaluation executor.

    Defaults to ``"unknown"`` / ``None`` when the executor provides no
    information.  This is a best-effort extraction — fields are populated
    only when the executor exposes known attributes.
    """
    semantics: dict[str, object] = {
        "rollout_mode": "unknown",
        "carry_policy": "unknown",
        "bptt_chunk_size": None,
        "teacher_forcing": None,
    }

    # Try known executor attributes.
    rollout_mode = getattr(executor, "_evaluation_rollout_mode", None)
    if rollout_mode is not None:
        semantics["rollout_mode"] = rollout_mode

    carry_policy = getattr(executor, "_carry_policy", None)
    if carry_policy is not None:
        semantics["carry_policy"] = carry_policy

    config = getattr(executor, "config", None)
    if config is not None:
        bptt = getattr(config, "bptt_chunk_size", None)
        if bptt is not None:
            semantics["bptt_chunk_size"] = int(bptt)
        tf = getattr(config, "teacher_forcing", None)
        if tf is not None:
            semantics["teacher_forcing"] = bool(tf)

    return semantics


# =============================================================================
def _atomic_write_directory(
    tmp_dir: Path,
    final_dir: Path,
) -> None:
    """Atomically commit a temporary write directory to its final path.

    Writes ``_SUCCESS`` sentinel into *tmp_dir*, then renames the whole
    directory over *final_dir* (overwriting any previous artifact at that
    path).  Raises ``FileNotFoundError`` if *tmp_dir* does not exist.
    """
    if not tmp_dir.exists():
        raise FileNotFoundError(
            f"Cannot commit atomic write: temp dir does not exist: {tmp_dir}"
        )
    (tmp_dir / _SUCCESS_FILENAME).write_text("", encoding="utf-8")
    if final_dir.exists():
        import shutil

        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)


# =============================================================================
def collect_regime_artifact_bundle(
    *,
    executor: Any,
    provider: Any,
    regime_id: str,
    regime_kind: Literal["diagnostic", "benchmark"] = "diagnostic",
    run_dir: Path,
    trace_spec: TraceSpec | None = None,
    max_batches: int = 0,
    trigger_kind: str = "manual",
    epoch: int = 0,
    step: int = 0,
    provenance_overrides: dict[str, Any] | None = None,
) -> EvaluationRegimeResult:
    """Collect one evaluation regime and persist it as an artifact bundle.

    The provider and trace specification must already be resolved by the
    caller (typically an experiment-specific builder).  The old signature
    (accepting ``task``, ``provider_ref``, ``provider_settings``,
    ``trace_keys``, ``figure_names``) is removed — use
    :func:`collect_regime_artifact_bundle_from_ref` for artifact
    reconstruction or switch to the new ``EvaluationExperiment`` flow.
    """
    run_dir = Path(run_dir)
    model_family = _extract_model_family(executor)
    trace_paradigm = getattr(executor, "_trace_paradigm", None)
    temporal_semantics = _resolve_temporal_semantics(executor)

    # Resolve provenance overrides — caller may supply task/identity info.
    prov = provenance_overrides or {}
    resolved_task: str = prov.get(
        "task", _extract_model_family(executor) or "unknown"
    )

    # Build trace request from the pre-resolved trace spec.
    trace_request: EvaluationTraceRequest | None = None
    if trace_spec is not None:
        trace_request = EvaluationTraceRequest(trace_spec=trace_spec)

    # Write to temporary directory for atomic commit.
    tmp_dir = run_dir.with_suffix(".tmp")
    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    cases_dir = tmp_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    lightweight_case_results: list[EvaluationCaseResult] = []
    loss_sum = 0.0
    loss_count = 0

    for idx, result in enumerate(
        iter_evaluation_regime(
            provider,
            executor,
            max_batches=max_batches,
            trace_request=trace_request,
        )
    ):
        row = _persist_regime_case(
            run_dir=tmp_dir,
            result=result,
            index=idx,
        )
        summary_rows.append(
            {
                "case_id": row["case_id"],
                "source_context": row["source_context"],
                "loss": row["loss"],
                "has_trace": row["has_trace"],
            }
        )
        manifest_rows.append(row)

        loss = row["loss"]
        if isinstance(loss, Real):
            loss_sum += float(loss)
            loss_count += 1

        lightweight_case_results.append(
            EvaluationCaseResult(
                case_id=result.case_id,
                evaluated=SimpleNamespace(loss=loss),
                source_context=result.source_context,
                trace=None,
            )
        )

    summary: dict[str, object] = {"n_cases": len(summary_rows)}
    if loss_count > 0:
        summary["loss"] = loss_sum / float(loss_count)

    _write_regime_bundle_manifest(
        run_dir=tmp_dir,
        task=resolved_task,
        regime_summary=summary,
        summary_rows=summary_rows,
        manifest_rows=manifest_rows,
        regime_kind=regime_kind,
        regime_id=regime_id,
        trigger_kind=trigger_kind,
        epoch=epoch,
        step=step,
        model_family=model_family,
        trace_paradigm=trace_paradigm,
        temporal_semantics=temporal_semantics,
        capture_info=(
            {
                "profile": prov.get("capture_profile", "metrics_only"),
                "profile_version": prov.get("capture_profile_version", 1),
                "paradigm": prov.get("trace_paradigm", trace_paradigm),
                "requested": {
                    "include": prov.get("capture_include", []),
                    "exclude": prov.get("capture_exclude", []),
                },
                "resolved_fields": (
                    sorted(trace_spec.keys()) if trace_spec else []
                ),
            }
            if trace_spec is not None
            else None
        ),
    )
    _atomic_write_directory(tmp_dir, run_dir)
    return EvaluationRegimeResult(
        regime_id=regime_id,
        case_results=tuple(lightweight_case_results),
        summary=summary,
    )


# =============================================================================
def collect_regime_artifact_bundle_from_ref(
    *,
    artifact: EvalArtifactExecutorRef,
    task: str,
    provider_ref: str,
    provider_settings: dict[str, Any],
    run_dir: Path,
    regime_id: str,
    regime_kind: Literal["diagnostic", "benchmark"] = "diagnostic",
    max_batches: int = 0,
    trace_keys: list[str] | None = None,
    figure_names: list[str] | None = None,
    trigger_kind: str = "manual",
    epoch: int = 0,
    step: int = 0,
) -> EvaluationRegimeResult:
    """Collect and persist one artifact bundle from a typed executor ref.

    .. deprecated::
        Use the new ``EvaluationExperiment`` flow instead.  This function
        delegates to the new ``collect_regime_artifact_bundle`` with a
        resolved provider and trace spec.
    """
    executor = load_executor_from_artifact(artifact)
    trace_paradigm = getattr(executor, "_trace_paradigm", None)

    # Build trace request from keys (backward-compat path).
    trace_spec: TraceSpec | None = None
    all_keys: list[str] = list(trace_keys or [])
    if figure_names:
        from ehc_sn.figures import REGISTRY, list_figures

        list_figures()
        for name in figure_names:
            spec = REGISTRY.get(name)
            all_keys.extend(spec.trace_keys)
            all_keys.extend(spec.meta_keys)
    if all_keys and trace_paradigm is not None:
        from ehc_sn.traces.specs import build_trace_spec

        trace_spec = build_trace_spec(
            trace_paradigm, include_keys=set(all_keys)
        )

    provider = resolve_provider(provider_ref, provider_settings)

    return collect_regime_artifact_bundle(
        executor=executor,
        provider=provider,
        regime_id=regime_id,
        regime_kind=regime_kind,
        run_dir=run_dir,
        trace_spec=trace_spec,
        max_batches=max_batches,
        trigger_kind=trigger_kind,
        epoch=epoch,
        step=step,
        provenance_overrides={
            "task": task,
            "model_family": artifact.model_family,
            "trace_paradigm": trace_paradigm,
        },
    )


# =============================================================================
def load_executor_from_artifact(artifact: EvalArtifactExecutorRef) -> Any:
    """Construct and hydrate one family-specific evaluation executor."""
    config_path = Path(artifact.executor_config_path)
    checkpoint_path = Path(artifact.checkpoint_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Executor config path does not exist: {config_path}"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {checkpoint_path}"
        )

    config_map = tomllib.loads(config_path.read_text(encoding="utf-8"))
    executor = _build_executor_from_family_artifact(
        experiment_id=artifact.experiment_id,
        config_map=config_map,
    )
    _hydrate_executor_from_checkpoint(executor, checkpoint_path)
    _initialize_eval_runtime(executor)
    return executor


# =============================================================================
# =============================================================================
# Family registry access — dispatch lives in eval.registry
# =============================================================================


def _build_executor_from_family_artifact(
    *,
    experiment_id: str,
    config_map: dict[str, Any],
) -> Any:
    """Instantiate one evaluation executor from a resolved experiment_id.

    Dispatches through the experiment registry in :mod:`ehc_sn.eval.registry`.
    The config class validates the full TOML dict with ``model_validate``,
    eliminating manual key-by-key extraction.

    The executor's ``_trace_paradigm`` is read from the module itself,
    which every ``LightningModule`` subclass sets in ``__init__``.
    """
    from ehc_sn.eval.registry import get_evaluation_experiment_registration

    config_cls, build_fn = get_evaluation_experiment_registration(experiment_id)

    eval_config = config_cls.model_validate(config_map)
    built = build_fn(eval_config)

    # The builder now returns EvaluationExperiment; extract executor.
    from ehc_sn.experiments._infra import EvaluationExperiment as _EvalExp

    if isinstance(built, _EvalExp):
        executor = built.executor
    else:
        executor = built

    # Tag the executor so _build_trace_request can construct a trace spec
    # without calling set_eval_trace_keys.  Every LightningModule subclass
    # already sets _trace_paradigm in __init__.
    executor._trace_paradigm = getattr(executor, "_trace_paradigm", experiment_id.split("-")[0])  # type: ignore[attr-defined]
    return executor


# =============================================================================
# =============================================================================
# Generic case task-evidence helpers (task-agnostic duck-typed hook)
# =============================================================================


def _extract_case_task_evidence(
    result: EvaluationCaseResult,
) -> dict[str, np.ndarray] | None:
    """Extract task-evidence arrays from the case result, if attached.

    Checks for ``task_evidence_arrays`` on the ``source_context`` via
    duck-typing (``hasattr`` / ``isinstance(dict)``).  Returns ``None``
    when no arrays are present — this is the normal case for tasks that
    do not produce task-evidence sidecars.

    This function is intentionally generic: it does not import from
    ``tasks/arena`` or any task- or model-specific module.
    """
    ctx = result.source_context
    if hasattr(ctx, "task_evidence_arrays"):
        raw = ctx.task_evidence_arrays
        if isinstance(raw, dict) and raw:
            return raw
    if isinstance(ctx, dict):
        raw = ctx.get("task_evidence_arrays")
        if isinstance(raw, dict) and raw:
            return raw
    return None


def _extract_case_metadata(
    result: EvaluationCaseResult,
) -> dict[str, object] | None:
    """Extract case-level scalar metadata from the result, if attached.

    Same duck-typing pattern as :func:`_extract_case_task_evidence`.
    """
    ctx = result.source_context
    if hasattr(ctx, "case_metadata"):
        raw = ctx.case_metadata
        if isinstance(raw, dict):
            return raw
    if isinstance(ctx, dict):
        raw = ctx.get("case_metadata")
        if isinstance(raw, dict):
            return raw
    return None


# =============================================================================
def _hydrate_executor_from_checkpoint(
    executor: Any, checkpoint_path: Path
) -> None:
    """Load executor weights from a weights-only checkpoint artifact."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if {
            "optimizer_states",
            "lr_schedulers",
            "loops",
        }.intersection(raw):
            raise ValueError(
                "Unsupported checkpoint artifact: full trainer-resume checkpoints "
                "are not allowed for artifact collection. Provide a weights-only "
                "executor checkpoint instead."
            )
        state_dict = raw.get("state_dict", raw)
    else:
        state_dict = raw
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a state_dict mapping."
        )

    executor_state_keys = set(executor.state_dict().keys())
    matched = [key for key in state_dict if key in executor_state_keys]
    if not matched:
        raise ValueError(
            "Checkpoint state has no keys compatible with the constructed "
            f"executor for checkpoint {checkpoint_path!r}."
        )

    executor.load_state_dict(state_dict, strict=False)
    if hasattr(executor, "eval") and callable(executor.eval):
        executor.eval()


# =============================================================================
def _initialize_eval_runtime(executor: Any) -> None:
    """Initialize eval/test runtime when executor exposes setup()."""
    setup = getattr(executor, "setup", None)
    if callable(setup):
        setup("validate")


# =============================================================================
def persist_regime_artifact_bundle(
    *,
    run_dir: Path,
    task: str,
    regime_kind: Literal["diagnostic", "benchmark"],
    regime_id: str,
    trigger_kind: str,
    epoch: int,
    step: int,
    regime_result: EvaluationRegimeResult,
) -> None:
    """Persist regime artifacts to canonical manifest+portable format."""
    run_dir = Path(run_dir)
    tmp_dir = run_dir.with_suffix(".tmp")
    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    cases_dir = tmp_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for idx, result in enumerate(regime_result.case_results):
        source_context = _to_jsonable(result.source_context)
        loss = _extract_loss_scalar(result.evaluated)
        has_trace = result.trace is not None

        summary_rows.append(
            {
                "case_id": result.case_id,
                "source_context": source_context,
                "loss": loss,
                "has_trace": has_trace,
            }
        )

        row: dict[str, Any] = {
            "case_id": result.case_id,
            "source_context": source_context,
            "loss": loss,
            "has_trace": has_trace,
        }
        stem = f"{idx:04d}-{_sanitize_filename_component(result.case_id)}"
        if has_trace:
            dense_rel = Path("cases") / f"{stem}.dense.npz"
            meta_rel = Path("cases") / f"{stem}.meta.json"
            _write_dense_npz(tmp_dir / dense_rel, result.trace.export())
            (tmp_dir / meta_rel).write_text(
                json.dumps(
                    _to_jsonable(result.trace.get_meta()),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            row["dense_artifact"] = str(dense_rel)
            row["meta_artifact"] = str(meta_rel)

        # Optional per-case task-evidence sidecar.
        case_task_arrays = _extract_case_task_evidence(result)
        if case_task_arrays:
            task_rel = Path("cases") / f"{stem}.task.npz"
            _write_dense_npz(tmp_dir / task_rel, case_task_arrays)
            row["task_arrays_path"] = str(task_rel)
            case_meta = _extract_case_metadata(result)
            if case_meta:
                row["case_metadata"] = case_meta

        manifest_rows.append(row)

    _write_regime_bundle_manifest(
        run_dir=tmp_dir,
        task=task,
        regime_summary=dict(regime_result.summary),
        summary_rows=summary_rows,
        manifest_rows=manifest_rows,
        regime_kind=regime_kind,
        regime_id=regime_id,
        trigger_kind=trigger_kind,
        epoch=epoch,
        step=step,
    )
    _atomic_write_directory(tmp_dir, run_dir)


# =============================================================================
def _persist_regime_case(
    *,
    run_dir: Path,
    result: EvaluationCaseResult,
    index: int,
) -> dict[str, Any]:
    """Persist one case payload and return one manifest-ready case row."""
    source_context = _to_jsonable(result.source_context)
    loss = _extract_loss_scalar(result.evaluated)
    has_trace = result.trace is not None
    row: dict[str, Any] = {
        "case_id": result.case_id,
        "source_context": source_context,
        "loss": loss,
        "has_trace": has_trace,
    }
    if not has_trace:
        return row

    stem = f"{index:04d}-{_sanitize_filename_component(result.case_id)}"
    dense_rel = Path("cases") / f"{stem}.dense.npz"
    meta_rel = Path("cases") / f"{stem}.meta.json"
    dense = result.trace.export()
    meta = result.trace.get_meta()
    _write_dense_npz(run_dir / dense_rel, dense)
    (run_dir / meta_rel).write_text(
        json.dumps(_to_jsonable(meta), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    row["dense_artifact"] = str(dense_rel)
    row["meta_artifact"] = str(meta_rel)
    return row


# =============================================================================
def _write_regime_bundle_manifest(
    *,
    run_dir: Path,
    task: str,
    regime_summary: dict[str, object],
    summary_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    regime_kind: Literal["diagnostic", "benchmark"],
    regime_id: str,
    trigger_kind: str,
    epoch: int,
    step: int,
    model_family: str | None = None,
    trace_paradigm: str | None = None,
    temporal_semantics: dict[str, object] | None = None,
    capture_info: dict[str, object] | None = None,
) -> None:
    """Write canonical manifest payloads."""
    provenance: dict[str, object] = {}
    if model_family is not None:
        provenance["model_family"] = model_family
    if trace_paradigm is not None:
        provenance["trace_paradigm"] = trace_paradigm
    provenance["global_step"] = step
    provenance["epoch"] = epoch

    manifest: dict[str, object] = {
        "schema": _ARTIFACT_SCHEMA,
        "status": "complete",
        "task": task,
        "regime_id": regime_id,
        "regime_kind": regime_kind,
        "phase_kind": "diag" if regime_kind == "diagnostic" else "bench",
        "trigger_kind": trigger_kind,
        "epoch": epoch,
        "step": step,
        "provenance": provenance,
        "temporal_semantics": temporal_semantics
        or {
            "rollout_mode": "unknown",
            "carry_policy": "unknown",
            "bptt_chunk_size": None,
            "teacher_forcing": None,
        },
        "summary": _to_jsonable(dict(regime_summary)),
        "cases": manifest_rows,
    }
    if capture_info is not None:
        manifest["capture"] = capture_info
    (run_dir / _MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =============================================================================
def load_artifact_run_cases(run_dir: Path) -> list[LoadedArtifactCase]:
    """Load trace-bearing cases from a persisted artifact run directory."""
    run_dir = Path(run_dir)
    success_path = run_dir / _SUCCESS_FILENAME
    manifest_path = run_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json under {run_dir}.")
    if not success_path.exists():
        raise RuntimeError(
            f"Artifact at {run_dir} is missing _SUCCESS sentinel."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != _ARTIFACT_SCHEMA:
        raise ValueError(
            f"Unsupported artifact schema: {manifest.get('schema')!r}. "
            f"Expected {_ARTIFACT_SCHEMA!r}."
        )
    if manifest.get("status", "complete") != "complete":
        raise RuntimeError(
            f"Artifact at {run_dir} has status!=complete in manifest."
        )
    return _load_manifest_cases(run_dir, manifest)


# =============================================================================
def resolve_dotted_symbol(symbol_ref: str) -> Any:
    """Resolve one dotted symbol path to a Python object."""
    module_name, symbol_name = symbol_ref.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


# =============================================================================
def resolve_provider(
    provider_ref: str, provider_settings: dict[str, Any]
) -> Any:
    """Resolve and instantiate a provider class from dotted import path."""
    provider_cls = resolve_dotted_symbol(provider_ref)
    return provider_cls(**provider_settings)


# =============================================================================
def _load_manifest_cases(
    run_dir: Path,
    manifest: dict[str, Any],
) -> list[LoadedArtifactCase]:
    """Load trace-bearing cases from canonical manifest bundle payloads."""
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a list under key 'cases'.")

    loaded_cases: list[LoadedArtifactCase] = []
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("manifest.json case rows must be dictionaries.")
        case_id = row.get("case_id")
        has_trace = bool(row.get("has_trace", False))
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                "manifest.json case rows require a non-empty case_id."
            )
        if not has_trace:
            continue

        dense_rel = row.get("dense_artifact")
        meta_rel = row.get("meta_artifact")
        if not isinstance(dense_rel, str) or not dense_rel:
            raise ValueError(
                f"Case {case_id!r} missing dense_artifact in manifest."
            )
        if not isinstance(meta_rel, str) or not meta_rel:
            raise ValueError(
                f"Case {case_id!r} missing meta_artifact in manifest."
            )

        dense = _read_dense_npz(run_dir / dense_rel)
        meta_raw = json.loads((run_dir / meta_rel).read_text(encoding="utf-8"))
        if meta_raw is None:
            meta_raw = {}
        if not isinstance(meta_raw, dict):
            raise ValueError(
                f"Case {case_id!r} meta artifact must decode to a dict."
            )

        # Merge optional task-evidence sidecar into the dense dict so that
        # arena/* keys are available via TraceTree.get() after rehydration.
        task_arrays_rel = row.get("task_arrays_path")
        if isinstance(task_arrays_rel, str):
            task_dense = _read_dense_npz(run_dir / task_arrays_rel)
            # Model trace keys take precedence on collision.
            dense = {**task_dense, **dense}

        temporal_semantics = manifest.get("temporal_semantics")

        loaded_cases.append(
            LoadedArtifactCase(
                case_id=case_id,
                source_context=row.get("source_context"),
                trace=_rehydrate_trace_tree(dense=dense, meta=meta_raw),
                temporal_semantics=temporal_semantics,
            )
        )
    return loaded_cases


# =============================================================================
def _build_trace_request(
    *,
    executor: Any,
    trace_keys: list[str],
    figure_names: list[str],
) -> EvaluationTraceRequest:
    """Build trace request for collection using figure trace vocabulary.

    Trace keys are accumulated from explicit keys and figure requirements,
    then used to build a trace spec directly.  Does **not** call
    ``set_eval_trace_keys()`` on the executor — the executor's trace
    production is configured through its own ``diagnostic_trace_spec`` or
    through the model-family-aware ``build_trace_spec`` helper.
    """
    requested_trace_keys = set(trace_keys)
    if figure_names:
        # Force figures-package bootstrap so built-in names resolve in fresh processes.
        list_figures()
    for name in figure_names:
        spec = REGISTRY.get(name)
        requested_trace_keys.update(spec.trace_keys)
        requested_trace_keys.update(spec.meta_keys)

    paradigm = getattr(executor, "_trace_paradigm", None)
    if paradigm is not None and requested_trace_keys:
        return EvaluationTraceRequest(
            trace_spec=build_trace_spec(
                paradigm, include_keys=requested_trace_keys
            ),
            trace_meta=None,
        )

    trace_spec = getattr(executor, "trace_spec", None)
    if trace_spec is not None:
        return EvaluationTraceRequest(
            trace_spec=trace_spec,
            trace_meta=None,
        )
    raise ValueError(
        "Cannot build trace request: executor has no _trace_paradigm or trace_spec. "
        "Provide explicit trace_keys or figure_names."
    )


# =============================================================================
# Internal persistence helpers
# =============================================================================


def _sanitize_filename_component(name: str) -> str:
    """Sanitize a string for safe use as a filename component."""
    safe = re.sub(r"[^\w\-.]", "_", name)
    return safe.strip("._") or "unnamed"


def _to_jsonable(value: object) -> object:
    """Recursively convert a value to a JSON-serializable form."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().numpy())
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(v) for v in value)
    return str(value)


def _extract_loss_scalar(evaluated: Any) -> float | None:
    """Extract a scalar loss value from an evaluated chunk payload when available."""
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


def _write_dense_npz(path: Path, data: dict[str, Any]) -> None:
    """Write a dense payload dict to an NPZ file.

    Lists of arrays with equal shapes are stacked.  Lists with unequal shapes
    (e.g. per-frequency diagnostic tensors) are written as separate
    ``{key}/0``, ``{key}/1``, … keys so the rehydrated TraceTree can
    reconstruct the original tree-path structure.
    """
    arrays: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, torch.Tensor):
            arrays[key] = value.cpu().numpy()
        elif isinstance(value, (int, float)):
            arrays[key] = np.array(value)
        elif isinstance(value, list):
            try:
                arrays[key] = np.array(value)
            except ValueError:
                # Jagged list — write each element as its own indexed key.
                for i, sub in enumerate(value):
                    sub_key = f"{key}/{i}"
                    if isinstance(sub, torch.Tensor):
                        arrays[sub_key] = sub.cpu().numpy()
                    elif isinstance(sub, np.ndarray):
                        arrays[sub_key] = sub
                    else:
                        arrays[sub_key] = np.array(sub)
        else:
            arrays[key] = np.array(value)
    np.savez_compressed(path, **arrays)


def _read_dense_npz(path: Path) -> dict[str, np.ndarray]:
    """Read a dense payload dict from an NPZ file."""
    return dict(np.load(path, allow_pickle=True))


def _nest_slash_paths(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert flat slash-delimited keys into nested dicts.

    ``{"lec/filter/alpha_sigmoid": [0.99]}`` becomes
    ``{"lec": {"filter": {"alpha_sigmoid": [0.99]}}}`` so that
    ``_lookup_meta_path`` (which traverses slash-separated segments as
    nested dict levels) can resolve them.
    """
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        target = nested
        *segments, leaf = path.split("/")
        for seg in segments:
            target = target.setdefault(seg, {})
        target[leaf] = value
    return nested


def _rehydrate_trace_tree(
    dense: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> TraceTree:
    """Rehydrate a TraceTree from persisted dense + meta payloads."""
    trace = TraceTree()
    for key, leaf_strings in _split_nested_keys(dense):
        for leaf in leaf_strings:
            path = key
            if leaf:
                path = f"{path}/{leaf}"
            if path not in trace.path_to_index:
                trace.path_to_index[path] = len(trace.paths)
                trace.paths.append(tuple(path.split("/")))
                trace.path_strs.append(path)
                trace.leaf_is_numeric.append(True)
                # Infer batch_size from the first numeric leaf
                if trace.batch_size is None:
                    arr = dense.get(path)
                    if arr is not None and arr.ndim >= 1:
                        trace.batch_size = int(arr.shape[0])

    # Build dense_leaves array directly from rehydrated data so that
    # TraceTree.get() and figure rendering work on reloaded artifacts.
    n = len(trace.path_strs)
    dense_leaves: list[np.ndarray | None] = [None] * n
    for path, idx in trace.path_to_index.items():
        arr = dense.get(path)
        if arr is not None:
            dense_leaves[idx] = arr
    trace.dense_leaves = dense_leaves

    if meta:
        trace.attached_meta.update(_nest_slash_paths(meta))

    trace._rehydrated_dense = dense  # type: ignore[attr-defined]
    return trace


def _split_nested_keys(
    data: dict[str, np.ndarray],
) -> list[tuple[str, list[str]]]:
    """Split slash-delimited keys into top-level group and sub-leaf parts."""
    result: dict[str, set[str]] = {}
    for key in data:
        parts = key.split("/")
        if len(parts) == 1:
            result.setdefault(parts[0], set()).add("")
        else:
            group = parts[0]
            leaf = "/".join(parts[1:])
            result.setdefault(group, set()).add(leaf)
    return [(k, sorted(v)) for k, v in result.items()]


__all__ = [
    "EvalArtifactExecutorRef",
    "LoadedArtifactCase",
    "collect_regime_artifact_bundle",
    "collect_regime_artifact_bundle_from_ref",
    "load_artifact_run_cases",
    "load_executor_from_artifact",
    "persist_regime_artifact_bundle",
    "resolve_provider",
]
