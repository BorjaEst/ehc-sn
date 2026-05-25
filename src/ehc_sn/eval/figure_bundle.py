"""Eval-owned figure bundle collection and persistence helpers.

This module owns the long-term on-disk contract for persisted figure-ready
evaluation traces while keeping compatibility with legacy summary+pt payloads.
"""

from __future__ import annotations

import importlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Real
from pathlib import Path
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
from ehc_sn.traces.trace_tree import TraceTree

_BUNDLE_SCHEMA = "ehc_sn.eval.figure_bundle.v1"
_MANIFEST_FILENAME = "manifest.json"
_SUPPORTED_EXECUTOR_FAMILIES: frozenset[str] = frozenset(
    {"ehc-v1", "hrm-v1", "hrm-v2", "tem-v1", "tem-v2"}
)


# =============================================================================
class FigureBundleExecutorArtifact(BaseModel, extra="forbid"):
    """Typed artifact metadata used to reconstruct an evaluation executor."""

    schema_version: str = "1"
    artifact_type: Literal["evaluation_executor"] = "evaluation_executor"
    model_family: str = Field(..., min_length=1)
    executor_config_path: Path
    checkpoint_path: Path
    checkpoint_format: Literal["weights_only"] = "weights_only"

    @model_validator(mode="after")
    def _normalize_and_validate_family(self) -> "FigureBundleExecutorArtifact":
        self.model_family = self.model_family.strip().lower()
        if self.model_family not in _SUPPORTED_EXECUTOR_FAMILIES:
            raise ValueError(
                "Unsupported executor artifact model_family "
                f"{self.model_family!r}. Supported families: "
                f"{sorted(_SUPPORTED_EXECUTOR_FAMILIES)!r}."
            )
        return self


# =============================================================================
@dataclass(frozen=True)
class PersistedTraceCase:
    """One persisted case trace loaded from an eval run directory."""

    case_id: str
    source_context: object | None
    trace: TraceTree


# =============================================================================
def collect_regime_figure_bundle(
    *,
    executor: Any,
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
    write_legacy_compat: bool = True,
) -> EvaluationRegimeResult:
    """Collect one figure-ready evaluation regime and persist it as a bundle."""
    provider = resolve_provider(provider_ref, provider_settings)
    trace_request = _build_trace_request(
        executor=executor,
        trace_keys=trace_keys or [],
        figure_names=figure_names or [],
    )
    case_results = tuple(
        iter_evaluation_regime(
            provider,
            executor,
            max_batches=max_batches,
            trace_request=trace_request,
        )
    )

    losses = [
        value
        for value in (
            _extract_loss_scalar(result.evaluated) for result in case_results
        )
        if value is not None
    ]
    summary: dict[str, object] = {"n_cases": len(case_results)}
    if losses:
        summary["loss"] = sum(losses) / len(losses)

    regime_result = EvaluationRegimeResult(
        regime_id=regime_id,
        case_results=case_results,
        summary=summary,
    )
    persist_regime_figure_bundle(
        run_dir=run_dir,
        regime_kind=regime_kind,
        regime_id=regime_id,
        phase_kind="diag" if regime_kind == "diagnostic" else "bench",
        trigger_kind=trigger_kind,
        epoch=epoch,
        step=step,
        regime_result=regime_result,
        write_legacy_compat=write_legacy_compat,
    )
    return regime_result


# =============================================================================
def collect_regime_figure_bundle_from_artifact(
    *,
    artifact: FigureBundleExecutorArtifact,
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
    write_legacy_compat: bool = True,
) -> EvaluationRegimeResult:
    """Collect and persist one figure bundle from a typed artifact."""
    executor = load_executor_from_artifact(artifact)

    return collect_regime_figure_bundle(
        executor=executor,
        provider_ref=provider_ref,
        provider_settings=provider_settings,
        run_dir=run_dir,
        regime_id=regime_id,
        regime_kind=regime_kind,
        max_batches=max_batches,
        trace_keys=trace_keys,
        figure_names=figure_names,
        trigger_kind=trigger_kind,
        epoch=epoch,
        step=step,
        write_legacy_compat=write_legacy_compat,
    )


# =============================================================================
def load_executor_from_artifact(artifact: FigureBundleExecutorArtifact) -> Any:
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
        model_family=artifact.model_family,
        config_map=config_map,
    )
    _hydrate_executor_from_checkpoint(executor, checkpoint_path)
    _initialize_eval_runtime(executor)
    return executor


# =============================================================================
def _build_executor_from_family_artifact(
    *,
    model_family: str,
    config_map: dict[str, Any],
) -> Any:
    """Instantiate one training/evaluation executor for a supported family."""
    if model_family == "ehc-v1":
        from ehc_sn.lightning.ehc.ehc_v1 import (
            EHCV1TrainingModel,
            parse_ehc_v1_config,
        )

        return EHCV1TrainingModel(parse_ehc_v1_config(config_map))

    if model_family == "hrm-v1":
        from ehc_sn.lightning.hrm.hrm_v1 import (
            HRMV1ModelConfig,
            HRMV1TrainingModel,
        )

        return HRMV1TrainingModel(
            _build_model_config(config_map, HRMV1ModelConfig)
        )

    if model_family == "hrm-v2":
        from ehc_sn.lightning.hrm.hrm_v2 import (
            HRMV2ModelConfig,
            HRMV2TrainingModel,
        )

        return HRMV2TrainingModel(
            _build_model_config(config_map, HRMV2ModelConfig)
        )

    if model_family == "tem-v1":
        from ehc_sn.lightning.tem.tem_v1 import (
            TEMV1ModelConfig,
            TEMV1TrainingModel,
        )

        return TEMV1TrainingModel(
            _build_model_config(config_map, TEMV1ModelConfig)
        )

    if model_family == "tem-v2":
        from ehc_sn.lightning.tem.tem_v2 import (
            TEMV2ModelConfig,
            TEMV2TrainingModel,
        )

        return TEMV2TrainingModel(
            _build_model_config(config_map, TEMV2ModelConfig)
        )

    raise ValueError(
        "Unsupported executor artifact model_family "
        f"{model_family!r}. Supported families: "
        f"{sorted(_SUPPORTED_EXECUTOR_FAMILIES)!r}."
    )


# =============================================================================
def _build_model_config(config_map: dict[str, Any], config_cls: Any) -> Any:
    """Validate one training-model config from a larger TOML mapping."""
    field_names = set(config_cls.model_fields)
    filtered = {k: v for k, v in config_map.items() if k in field_names}
    return config_cls.model_validate(filtered)


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
                "are not allowed for figure collection. Provide a weights-only "
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
def persist_regime_figure_bundle(
    *,
    run_dir: Path,
    regime_kind: Literal["diagnostic", "benchmark"],
    regime_id: str,
    phase_kind: Literal["diag", "bench"],
    trigger_kind: str,
    epoch: int,
    step: int,
    regime_result: EvaluationRegimeResult,
    write_legacy_compat: bool = True,
) -> None:
    """Persist regime artifacts to canonical manifest+portable format."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = run_dir / "cases"
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
        if has_trace:
            stem = f"{idx:04d}-{_sanitize_filename_component(result.case_id)}"
            dense_rel = Path("cases") / f"{stem}.dense.npz"
            meta_rel = Path("cases") / f"{stem}.meta.json"
            _write_dense_npz(run_dir / dense_rel, result.trace.export())
            (run_dir / meta_rel).write_text(
                json.dumps(
                    _to_jsonable(result.trace.get_meta()),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            row["dense_artifact"] = str(dense_rel)
            row["meta_artifact"] = str(meta_rel)

            if write_legacy_compat:
                trace_payload = {
                    "case_id": result.case_id,
                    "source_context": result.source_context,
                    "dense": result.trace.export(),
                    "meta": result.trace.get_meta(),
                }
                trace_path = run_dir / f"{stem}.pt"
                torch.save(trace_payload, trace_path)

        manifest_rows.append(row)

    summary = {
        "regime_id": regime_result.regime_id,
        "regime_kind": regime_kind,
        "phase_kind": phase_kind,
        "trigger_kind": trigger_kind,
        "epoch": epoch,
        "step": step,
        "summary": _to_jsonable(dict(regime_result.summary)),
        "cases": summary_rows,
    }
    if write_legacy_compat:
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    manifest = {
        "schema": _BUNDLE_SCHEMA,
        "regime_id": regime_id,
        "regime_kind": regime_kind,
        "phase_kind": phase_kind,
        "trigger_kind": trigger_kind,
        "epoch": epoch,
        "step": step,
        "summary": _to_jsonable(dict(regime_result.summary)),
        "cases": manifest_rows,
    }
    (run_dir / _MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =============================================================================
def load_persisted_regime_run_cases(run_dir: Path) -> list[PersistedTraceCase]:
    """Load trace-bearing cases from a canonical or legacy run directory."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / _MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") == _BUNDLE_SCHEMA:
            return _load_manifest_cases(run_dir, manifest)
    return _load_legacy_cases(run_dir)


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
) -> list[PersistedTraceCase]:
    """Load trace-bearing cases from canonical manifest bundle payloads."""
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a list under key 'cases'.")

    loaded_cases: list[PersistedTraceCase] = []
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

        loaded_cases.append(
            PersistedTraceCase(
                case_id=case_id,
                source_context=row.get("source_context"),
                trace=_rehydrate_trace_tree(dense=dense, meta=meta_raw),
            )
        )
    return loaded_cases


# =============================================================================
def _load_legacy_cases(run_dir: Path) -> list[PersistedTraceCase]:
    """Load trace-bearing cases from legacy summary.json + per-case .pt files."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing canonical manifest.json and legacy summary.json under {run_dir}."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_cases = summary.get("cases")
    if not isinstance(summary_cases, list):
        raise ValueError("summary.json must contain a list under key 'cases'.")

    payload_by_case_id: dict[str, dict[str, Any]] = {}
    for payload_path in sorted(run_dir.glob("*.pt")):
        payload = torch.load(
            payload_path, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Trace payload at {payload_path} must be a dict.")
        case_id = payload.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                f"Trace payload at {payload_path} is missing a non-empty case_id."
            )
        payload_by_case_id[case_id] = payload

    loaded_cases: list[PersistedTraceCase] = []
    for row in summary_cases:
        if not isinstance(row, dict):
            raise ValueError("summary.json case rows must be dictionaries.")
        case_id = row.get("case_id")
        has_trace = bool(row.get("has_trace", False))
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                "summary.json case rows require a non-empty case_id."
            )
        if not has_trace:
            continue

        payload = payload_by_case_id.get(case_id)
        if payload is None:
            raise ValueError(
                "summary.json marks case as trace-bearing but payload is missing "
                f"for case_id={case_id!r}."
            )

        dense = payload.get("dense")
        meta = payload.get("meta")
        if dense is None:
            raise ValueError(
                f"Trace payload for case {case_id!r} is missing 'dense'."
            )
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise ValueError(
                f"Trace payload meta for case {case_id!r} must be a dict."
            )

        loaded_cases.append(
            PersistedTraceCase(
                case_id=case_id,
                source_context=payload.get("source_context"),
                trace=_rehydrate_trace_tree(dense=dense, meta=meta),
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
    """Build trace request for collection using figure trace vocabulary."""
    requested_trace_keys = set(trace_keys)
    if figure_names:
        # Force figures-package bootstrap so built-in names resolve in fresh processes.
        list_figures()
    for name in figure_names:
        requested_trace_keys.update(REGISTRY.get(name).trace_keys)

    if requested_trace_keys:
        set_trace_keys = getattr(executor, "set_eval_trace_keys", None)
        if callable(set_trace_keys):
            set_trace_keys(requested_trace_keys)

    trace_spec = getattr(executor, "trace_spec", None)
    if trace_spec is None:
        trace_spec = getattr(executor, "trace_specs", None)
    if trace_spec is None:
        raise RuntimeError(
            "Figure bundle collection requires trace_spec (or legacy trace_specs) "
            "on the executor."
        )
    return EvaluationTraceRequest(trace_spec=trace_spec)


# =============================================================================
def _write_dense_npz(path: Path, dense: Any) -> None:
    """Write exported trace dense payload in portable compressed format."""
    flat = _flatten_dense_tree(dense)
    if not flat:
        raise ValueError("Persisted trace dense payload is empty.")

    arrays: dict[str, np.ndarray] = {}
    paths: list[str] = []
    for idx, (dense_path, value) in enumerate(sorted(flat.items())):
        key = f"arr_{idx:05d}"
        arrays[key] = np.asarray(value)
        paths.append(dense_path)

    np.savez_compressed(
        path,
        __paths=np.asarray(paths, dtype=object),
        **arrays,
    )


# =============================================================================
def _read_dense_npz(path: Path) -> dict[str, Any]:
    """Read portable dense payload and return a flat path->array mapping."""
    with np.load(path, allow_pickle=True) as data:
        if "__paths" not in data:
            raise ValueError(f"Dense artifact is missing __paths: {path}")
        paths = data["__paths"].tolist()
        if not isinstance(paths, list):
            raise ValueError(
                f"Dense artifact __paths must decode to a list: {path}"
            )

        out: dict[str, Any] = {}
        for idx, dense_path in enumerate(paths):
            if not isinstance(dense_path, str) or not dense_path:
                raise ValueError(
                    f"Dense artifact has invalid path entry at {path}"
                )
            key = f"arr_{idx:05d}"
            if key not in data:
                raise ValueError(
                    f"Dense artifact {path} is missing payload array {key}."
                )
            out[dense_path] = np.asarray(data[key])
        return out


# =============================================================================
def _extract_loss_scalar(evaluated: Any) -> float | None:
    """Extract a scalar loss value from an evaluated payload."""
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


# =============================================================================
def _to_jsonable(value: Any) -> Any:
    """Convert nested objects into JSON-serializable primitives recursively."""
    if value is None:
        return None
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


# =============================================================================
def _sanitize_filename_component(value: str) -> str:
    """Normalize a string into a safe filename component."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "case"


# =============================================================================
def _rehydrate_trace_tree(*, dense: Any, meta: dict[str, Any]) -> TraceTree:
    """Rehydrate a TraceTree from persisted dense/meta artifact payloads."""
    trace = TraceTree()
    trace.finalize()

    flattened = (
        dense
        if isinstance(dense, dict)
        and all(isinstance(k, str) and "/" in k for k in dense)
        else _flatten_dense_tree(dense)
    )
    if not flattened:
        raise ValueError("Persisted trace dense payload is empty.")

    first_shape: tuple[int, ...] | None = None
    for path, value in flattened.items():
        array = np.asarray(value)
        trace.attach_dense(path, array, overwrite=True)
        if first_shape is None and array.ndim > 0:
            first_shape = tuple(array.shape)

    if first_shape is not None:
        trace.length = int(first_shape[0])
        if len(first_shape) > 1:
            trace.batch_size = int(first_shape[1])

    trace.attach_meta(meta, overwrite=True)
    environments = meta.get("environments")
    if isinstance(environments, list):
        trace.batch_size = len(environments)
    return trace


# =============================================================================
def _flatten_dense_tree(
    value: Any, prefix: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Flatten nested dense payload structures into slash-delimited paths."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            out.update(_flatten_dense_tree(child, prefix + (str(key),)))
        return out
    if isinstance(value, (list, tuple)):
        out: dict[str, Any] = {}
        for idx, child in enumerate(value):
            out.update(_flatten_dense_tree(child, prefix + (str(idx),)))
        return out
    if value is None:
        return {}
    if not prefix:
        raise ValueError(
            "Persisted dense payload must be nested under named paths."
        )
    return {"/".join(prefix): value}


__all__ = [
    "FigureBundleExecutorArtifact",
    "PersistedTraceCase",
    "collect_regime_figure_bundle",
    "collect_regime_figure_bundle_from_artifact",
    "load_persisted_regime_run_cases",
    "load_executor_from_artifact",
    "persist_regime_figure_bundle",
    "resolve_dotted_symbol",
    "resolve_provider",
]
