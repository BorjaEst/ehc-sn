"""evaluation-owned artifact collection and persistence helpers.

This module owns the long-term on-disk contract for persisted evaluation
artifacts (traces, metadata, manifests).
"""

from __future__ import annotations

import importlib
import json
import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

from ehc_sn.evaluation.contracts import (
    EvaluationCaseResult,
    EvaluationRegimeResult,
    ProducedArtifact,
)
from ehc_sn.figures import REGISTRY, list_figures
from ehc_sn.traces.observer import TraceSpec
from ehc_sn.traces.trace_tree import TraceTree

_ARTIFACT_SCHEMA = "ehp_sn.evaluation.artifact.v1"
_MANIFEST_FILENAME = "manifest.json"
_METRICS_FILENAME = "metrics.json"
_SUCCESS_FILENAME = "_SUCCESS"


# =============================================================================
class UnsupportedEvaluationArtifactSchema(ValueError):
    """Raised when an evaluation artifact manifest has a missing or mismatched schema."""

    def __init__(self, expected: str, actual: object, path: Path) -> None:
        self.expected = expected
        self.actual = actual
        self.path = path
        super().__init__(
            f"Unsupported evaluation artifact schema: {actual!r}. "
            f"Expected {expected!r} at {path}."
        )


# =============================================================================
class ZarrArtifactWriter:
    """Persist compact NumPy arrays as chunked Zarr groups.

    Writes each key in a ``data`` dict as a separate Zarr array under
    ``root/{name}/{key}``.  Skips fields whose values are not NumPy arrays
    (supports scalar metadata via ``.zattrs``).

    Requires ``zarr>=2.17``.  Raises ``ImportError`` at call time if ``zarr``
    is not installed.

    Usage::

        writer = ZarrArtifactWriter()
        writer.write("mec_grid_analysis", {"rate_maps": arr}, root)
    """

    @staticmethod
    def write(
        name: str,
        data: dict[str, np.ndarray],
        root: Path,
        *,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        """Write each array in *data* as a Zarr group under ``root/{name}/``.

        Args:
            name: Artifact name, used as the Zarr group directory name.
            data: Dict of array-name → NumPy array.  Non-array values are
                skipped silently.
            root: Parent directory for the artifact group.  Created if needed.
            chunks: Per-array chunk sizes.  When ``None``, defaults to spatial
                dims together and the unit axis chunked to min(64, last dim).

        Raises:
            ImportError: If ``zarr`` is not installed.
        """
        try:
            import numcodecs
            import zarr
        except ImportError as exc:
            raise ImportError(
                "zarr is required for ZarrArtifactWriter. "
                "Install with: pip install 'ehp-sn[evaluation]'"
            ) from exc

        group_dir = root / name
        group_dir.mkdir(parents=True, exist_ok=True)

        compressor = numcodecs.Blosc(cname="zstd", clevel=3)

        for key, arr in data.items():
            if not isinstance(arr, np.ndarray):
                continue

            arr = np.ascontiguousarray(arr)
            ndim = arr.ndim
            shape = arr.shape

            # Default chunking: spatial dims together, unit axis chunked.
            if chunks is not None:
                use_chunks = chunks
            elif ndim <= 1:
                use_chunks = shape
            elif ndim == 2:
                use_chunks = (shape[0], min(shape[1], 64))
            elif ndim == 3:
                use_chunks = (shape[0], shape[1], min(shape[2], 64))
            else:
                use_chunks = tuple(
                    min(s, 64) if i == ndim - 1 else s
                    for i, s in enumerate(shape)
                )

            arr_path = group_dir / key
            arr_path.mkdir(parents=True, exist_ok=True)

            z = zarr.open_array(
                str(arr_path),
                mode="w",
                shape=shape,
                chunks=use_chunks,
                dtype=arr.dtype,
                compressor=compressor,
                zarr_format=2,
                fill_value=None,
            )
            z[:] = arr

        # Write _SUCCESS sentinel.
        (group_dir / _SUCCESS_FILENAME).write_text("", encoding="utf-8")


# =============================================================================
@dataclass(frozen=True)
class LoadedArtifactCase:
    """One persisted case trace loaded from an evaluation run directory."""

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
# Streaming per-step metric accumulator (EvaluationConsumer)
# =============================================================================


class ArtifactMetricAccumulator:
    """Streaming per-step metric accumulator for artifact manifest production.

    Accumulates ``episode``, ``episode_tokens``, and ``extras`` RatioStat
    values from per-step ``ObservedStep`` objects during evaluation and
    returns compact CPU scalar dicts on ``finalize()``.

    This replaces the former post-hoc ``_extract_step_metric_increments``
    which iterated fully materialized ``EvaluatedChunk.steps``.
    """

    def __init__(self) -> None:
        self._completed_count: float = 0.0
        self._eligible_count: float = 0.0
        self._accuracy_sum: float = 0.0
        self._exact_sum: float = 0.0
        self._token_correct_sum: float = 0.0
        self._token_count_sum: float = 0.0
        self._has_data: bool = False

    def observe_step(self, step: object) -> None:
        """Accumulate metric increments from one observed step.

        Args:
            step: An ``ObservedStep`` whose ``outputs.metrics`` field
                contains ``episode``, ``episode_tokens``, and ``extras``
                with per-step increment semantics.  Silently skips when
                ``outputs.metrics`` is absent or ``None``.
        """
        outputs = getattr(step, "outputs", None)
        if outputs is None:
            return
        metrics = getattr(outputs, "metrics", None)
        if metrics is None:
            return

        ep = getattr(metrics, "episode", None)
        if ep is not None:
            self._completed_count += float(getattr(ep, "completed_count", 0.0))
            self._eligible_count += float(getattr(ep, "eligible_count", 0.0))
            self._accuracy_sum += float(getattr(ep, "accuracy_sum", 0.0))
            self._exact_sum += float(getattr(ep, "exact_sum", 0.0))
            self._has_data = True

        et = getattr(metrics, "episode_tokens", None)
        if et is not None:
            self._token_correct_sum += float(
                getattr(et, "token_correct_sum", 0.0)
            )
            self._token_count_sum += float(getattr(et, "token_count_sum", 0.0))
            self._has_data = True

    def finalize(self) -> dict[str, float | int]:
        """Return accumulated metric totals and reset internal state.

        Returns an empty dict when no metric data was observed (early
        exit or objective without episode/token metrics).
        """
        result: dict[str, float | int] = {}
        if not self._has_data:
            return result
        if self._completed_count > 0 or self._eligible_count > 0:
            result["completed_count"] = int(self._completed_count)
            result["eligible_count"] = int(self._eligible_count)
            result["accuracy_sum"] = self._accuracy_sum
            result["exact_sum"] = self._exact_sum
        if self._token_correct_sum > 0 or self._token_count_sum > 0:
            result["token_correct_sum"] = int(self._token_correct_sum)
            result["token_count_sum"] = int(self._token_count_sum)
        return result

    def reset(self) -> None:
        """Reset all accumulated state (for reuse across cases)."""
        self._completed_count = 0.0
        self._eligible_count = 0.0
        self._accuracy_sum = 0.0
        self._exact_sum = 0.0
        self._token_correct_sum = 0.0
        self._token_count_sum = 0.0
        self._has_data = False


# =============================================================================
# Streamed per-step metric extraction (backward-compatible helper)
# =============================================================================


def _accumulate_step_metric_increments(
    steps: tuple[Any, ...],
) -> dict[str, float | int]:
    """Accumulate per-step metric increments from an iterable of steps.

    Duck-types ``step.outputs.metrics`` for ``episode`` and
    ``episode_tokens`` fields.  Matching the same field names as
    the former ``_extract_step_metric_increments``.

    Args:
        steps: Iterable of objects with ``outputs.metrics`` containing
            ``episode`` and ``episode_tokens`` aggregations.

    Returns:
        Dict of summed metric increments.
    """
    inc: dict[str, float | int] = {}
    for step in steps:
        outputs = getattr(step, "outputs", None)
        if outputs is None:
            continue
        metrics = getattr(outputs, "metrics", None)
        if metrics is None:
            continue
        ep = getattr(metrics, "episode", None)
        if ep is not None:
            inc["completed_count"] = inc.get("completed_count", 0) + _to_int(
                getattr(ep, "completed_count", 0)
            )
            inc["eligible_count"] = inc.get("eligible_count", 0) + _to_int(
                getattr(ep, "eligible_count", 0)
            )
            inc["accuracy_sum"] = inc.get("accuracy_sum", 0.0) + _to_float(
                getattr(ep, "accuracy_sum", 0.0)
            )
            inc["exact_sum"] = inc.get("exact_sum", 0.0) + _to_float(
                getattr(ep, "exact_sum", 0.0)
            )
        ep_tok = getattr(metrics, "episode_tokens", None)
        if ep_tok is not None:
            inc["token_correct_sum"] = inc.get(
                "token_correct_sum", 0
            ) + _to_int(getattr(ep_tok, "token_correct_sum", 0))
            inc["token_count_sum"] = inc.get("token_count_sum", 0) + _to_int(
                getattr(ep_tok, "token_count_sum", 0)
            )
    return inc


# =============================================================================
# Hook metrics validation — mirrors logic from evaluation.offline
# =============================================================================

_RESERVED_SUMMARY_KEYS: frozenset[str] = frozenset({"n_cases", "loss"})


def _validate_hook_metrics(
    task_summary: dict[str, object],
) -> dict[str, float | int]:
    """Validate and type-narrow a family-owned hook return value."""
    from ehc_sn.metrics.values import validate_metric_value

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


def _to_int(value: object) -> int:
    """Coerce a numeric value to int (handles tensors and scalars)."""
    import torch

    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _to_float(value: object) -> float:
    """Coerce a numeric value to float (handles tensors and scalars)."""
    import torch

    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


# =============================================================================
def collect_regime_artifact_bundle(
    *,
    case_results: Iterable[EvaluationCaseResult] | None = None,
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
    analysis_artifact_descriptors: list[dict[str, object]] | None = None,
    produced_artifacts: Sequence[ProducedArtifact] = (),
) -> EvaluationRegimeResult:
    """Collect one evaluation regime and persist it as an artifact bundle.

    ``case_results`` is required — the caller owns device placement and
    batch preparation.  ``produced_artifacts`` are typed artifacts from
    consumer ``finalize()`` recorded in the manifest's ``"artifacts"``
    dict.

    ``analysis_artifact_descriptors`` are passed through to the manifest as
    ``"analysis_artifacts"`` entries.
    """
    if case_results is None:
        raise ValueError("case_results is required.")

    run_dir = Path(run_dir)
    model_family = _extract_model_family(executor)
    trace_paradigm = getattr(executor, "_trace_paradigm", None)
    temporal_semantics = _resolve_temporal_semantics(executor)

    # Resolve provenance overrides — caller may supply task/identity info.
    prov = provenance_overrides or {}
    resolved_task: str = prov.get(
        "task", _extract_model_family(executor) or "unknown"
    )

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

    # Per-step metric accumulators — populated via duck-typed extraction
    # from ``step.outputs.metrics`` (no imports from lightning/).
    total_completed_count: int = 0
    total_eligible_count: int = 0
    total_accuracy_sum: float = 0.0
    total_exact_sum: float = 0.0
    total_token_correct: int = 0
    total_token_count: int = 0
    total_samples: int = 0

    for idx, result in enumerate(case_results):
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

        # Extract per-step metric increments from each observed step.
        # The result may contain 0 or 1 steps (streaming path); iterate
        # whatever is available.
        evaluated = result.evaluated
        steps = getattr(evaluated, "steps", None)
        if steps:
            inc = _accumulate_step_metric_increments(steps)
            total_completed_count += int(inc.get("completed_count", 0))
            total_eligible_count += int(inc.get("eligible_count", 0))
            total_accuracy_sum += float(inc.get("accuracy_sum", 0.0))
            total_exact_sum += float(inc.get("exact_sum", 0.0))
            total_token_correct += int(inc.get("token_correct_sum", 0))
            total_token_count += int(inc.get("token_count_sum", 0))
        total_samples += result.n_samples

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
    if total_samples > 0:
        summary["n_samples"] = total_samples

    # Merge aggregated per-step metrics into the summary.
    if total_completed_count > 0 or total_eligible_count > 0:
        summary["n_sequence_completed"] = total_completed_count
        summary["n_sequence_eligible"] = total_eligible_count
    if total_token_correct > 0 or total_token_count > 0:
        summary["n_token_correct"] = total_token_correct
        summary["n_token_total"] = total_token_count
    if total_completed_count > 0:
        summary["sequence_accuracy"] = (
            total_accuracy_sum / total_completed_count
        )
        summary["sequence_exact"] = total_exact_sum / total_completed_count
    if total_token_count > 0:
        summary["token_accuracy"] = total_token_correct / total_token_count

    # ── Produced artifact persistence ───────────────────────────────────
    for artifact in produced_artifacts:
        artifact_path = tmp_dir / artifact.path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy staged consumer output into the temp directory for atomic commit.
        if artifact.source_path is not None and artifact.source_path.exists():
            if artifact.source_path.is_dir():
                import shutil

                shutil.copytree(
                    str(artifact.source_path),
                    str(artifact_path),
                    dirs_exist_ok=True,
                )
            else:
                import shutil

                shutil.copy2(str(artifact.source_path), str(artifact_path))
        elif not artifact_path.exists():
            # No source path and no existing copy — producer may have written
            # directly.  That is deprecated; warn once.
            pass

    # Optional model-family aggregation hook.
    aggregate = getattr(executor, "aggregate_evaluation_case_metrics", None)
    if aggregate is not None:
        try:
            task_summary = aggregate(
                task=resolved_task,
                regime_id=regime_id,
                regime_kind=regime_kind,
                case_results=tuple(lightweight_case_results),
            )
        except Exception:
            pass
        else:
            validated = _validate_hook_metrics(task_summary)
            summary.update(validated)

    # ── Patch manifest rows with trace_ref from Zarr index (if available) ──
    trace_index_path = tmp_dir / "traces" / "trace_index.json"
    if trace_index_path.exists():
        try:
            trace_index = json.loads(
                trace_index_path.read_text(encoding="utf-8")
            )
            index_cases = trace_index.get("cases", [])
            # Build case_id -> entry lookup.
            case_to_idx: dict[str, dict[str, object]] = {
                entry["case_id"]: entry for entry in index_cases
            }
            for row in manifest_rows:
                cid = row.get("case_id")
                if cid in case_to_idx:
                    row["trace_ref"] = {
                        "archive": "traces/behavioral.zarr",
                        "case_id": cid,
                    }
                    row["has_trace"] = True
        except Exception:
            pass

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
        analysis_artifacts=analysis_artifact_descriptors,
        produced_artifacts=produced_artifacts,
        checkpoint_sha256=str(prov.get("checkpoint_sha256", "")),
        plan_digest=str(prov.get("plan_digest", "")),
    )
    _atomic_write_directory(tmp_dir, run_dir)
    return EvaluationRegimeResult(
        regime_id=regime_id,
        case_results=tuple(lightweight_case_results),
        summary=summary,
    )


# =============================================================================


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
    """Load executor weights from a weights-only checkpoint artifact.

    Uses ``strict=True`` — any key mismatch raises ``RuntimeError``
    to prevent silent architecture/checkpoint incompatibility.
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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

    missing, unexpected = executor.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Strict state-dict loading failed for checkpoint "
            f"{checkpoint_path}:\n"
            f"  Missing keys: {sorted(missing)[:10]}\n"
            f"  Unexpected keys: {sorted(unexpected)[:10]}"
        )
    if hasattr(executor, "evaluation") and callable(executor.evaluation):
        executor.evaluation()


def _hydrate_executor_from_source(executor: Any, resolved_source: str) -> None:
    """Load executor weights from an artifact directory or legacy checkpoint.

    Detects whether *resolved_source* is a model-artifact directory
    (has ``manifest.json``) and delegates accordingly.  This helper
    is the single dispatch point used by ``run_offline_eval`` so that
    ``offline.py`` does not need to import from ``models/``.

    Parameters
    ----------
    executor:
        Already-constructed model or executor whose weights are loaded.
    resolved_source:
        Path to an artifact directory or a legacy checkpoint file.
    """
    source = Path(resolved_source)
    if source.is_dir() and (source / "manifest.json").is_file():
        from ehc_sn.model_artifacts import ModelArtifact

        artifact = ModelArtifact.open(source)
        artifact.load_state_into(executor)
    else:
        _hydrate_executor_from_checkpoint(executor, source)


# =============================================================================
def _initialize_eval_runtime(executor: Any) -> None:
    """Initialize evaluation/test runtime when executor exposes setup()."""
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
                "n_samples": result.n_samples,
                "source_context": source_context,
                "loss": loss,
                "has_trace": has_trace,
            }
        )

        row: dict[str, Any] = {
            "case_id": result.case_id,
            "n_samples": result.n_samples,
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
    """Persist one case payload and return one manifest-ready case row.

    Always writes a per-case ``.summary.json`` file with case metadata.
    When ``result.trace`` is a ``TraceTree`` (legacy path), additionally
    writes per-case ``.dense.npz`` and ``.meta.json`` files.
    """
    source_context = _to_jsonable(result.source_context)
    loss = _extract_loss_scalar(result.evaluated)
    has_trace = result.trace is not None
    stem = f"{index:04d}-{_sanitize_filename_component(result.case_id)}"
    row: dict[str, Any] = {
        "case_id": result.case_id,
        "n_samples": result.n_samples,
        "source_context": source_context,
        "loss": loss,
        "has_trace": has_trace,
    }

    # Write per-case JSON summary unconditionally.
    summary_rel = Path("cases") / f"{stem}.summary.json"
    (run_dir / summary_rel).write_text(
        json.dumps(row, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    row["summary_artifact"] = str(summary_rel)

    if not has_trace:
        return row

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
    analysis_artifacts: list[dict[str, object]] | None = None,
    produced_artifacts: Sequence[ProducedArtifact] = (),
    checkpoint_sha256: str = "",
    plan_digest: str = "",
) -> None:
    """Write canonical manifest payloads.

    Publishes ``"schema": _ARTIFACT_SCHEMA`` with typed ``"artifacts"`` dict
    keyed by ``ArtifactKey.manifest_key()``.
    """
    provenance: dict[str, object] = {}
    if model_family is not None:
        provenance["model_family"] = model_family
    if trace_paradigm is not None:
        provenance["trace_paradigm"] = trace_paradigm
    provenance["global_step"] = step
    provenance["epoch"] = epoch
    if checkpoint_sha256:
        provenance["checkpoint_sha256"] = checkpoint_sha256
    if plan_digest:
        provenance["plan_digest"] = plan_digest

    # Build the artifacts inventory from ProducedArtifact records.
    artifacts_dict: dict[str, dict[str, object]] = {}
    for artifact in produced_artifacts:
        key = artifact.key.manifest_key()
        artifacts_dict[key] = {
            "schema_version": artifact.schema_version,
            "path": str(artifact.path),
            "media_type": artifact.media_type,
            "producer_digest": artifact.producer_digest,
            "content_digest": artifact.content_digest,
            "metadata": dict(artifact.metadata),
        }

    # Build dependency lineage from produced-artifact records.
    dependencies: dict[str, list[dict[str, object]]] = {}
    for artifact in produced_artifacts:
        dep_key = artifact.key.manifest_key()
        deps: list[dict[str, object]] = []
        metadata = artifact.metadata or {}
        upstream_keys = metadata.get("upstream_artifact_keys")
        if isinstance(upstream_keys, (list, tuple)):
            for up_key in upstream_keys:
                if isinstance(up_key, dict):
                    deps.append(
                        {
                            "key": str(up_key.get("key", "")),
                            "schema_version": up_key.get("schema_version", 1),
                            "content_digest": str(
                                up_key.get("content_digest", "")
                            ),
                        }
                    )
        if deps:
            dependencies[dep_key] = deps

    if dependencies:
        provenance["dependencies"] = dependencies

    manifest = build_evaluation_artifact_manifest(
        task=task,
        regime_id=regime_id,
        regime_kind=regime_kind,
        phase_kind="diag" if regime_kind == "diagnostic" else "bench",
        trigger_kind=trigger_kind,
        epoch=epoch,
        step=step,
        evaluation=provenance,
        temporal_semantics=temporal_semantics,
        summary=dict(regime_summary),
        cases=manifest_rows,
        artifacts=artifacts_dict,
        capture=capture_info,
        analysis_artifacts=analysis_artifacts,
    )
    (run_dir / _MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Write metrics.json — scalar measurements only, not operational counts.
    _write_regime_metrics(run_dir, regime_summary)


def _write_regime_metrics(
    run_dir: Path,
    summary: dict[str, object],
) -> None:
    """Extract scalar scientific metrics from *summary* and write ``metrics.json``.

    Operational counts (``n_cases``, ``n_samples``, etc.) stay in
    ``manifest.summary``.  Only float-compatible values that are not
    operational counts are written to ``metrics.json``.
    """
    _COUNT_KEYS = frozenset(
        {
            "n_cases",
            "n_samples",
            "n_sequence_completed",
            "n_sequence_eligible",
            "n_token_correct",
            "n_token_total",
            "n_traced_cases",
        }
    )
    metrics: dict[str, float] = {}
    for k, v in summary.items():
        if k in _COUNT_KEYS:
            continue
        if isinstance(v, (int, float)):
            metrics[k] = float(v)
    (run_dir / _METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =============================================================================
def build_evaluation_artifact_manifest(
    *,
    task: str,
    regime_id: str,
    regime_kind: str,
    phase_kind: str,
    trigger_kind: str,
    epoch: int,
    step: int,
    evaluation: dict[str, object],
    temporal_semantics: dict[str, object] | None = None,
    summary: dict[str, object],
    cases: list[dict[str, Any]],
    artifacts: dict[str, dict[str, object]] | None = None,
    capture: dict[str, object] | None = None,
    analysis_artifacts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a canonical evaluation artifact manifest dict.

    This is the single place where the top-level manifest structure is
    assembled.  Callers may add extra keys after calling this function.
    """
    manifest: dict[str, object] = {
        "schema": _ARTIFACT_SCHEMA,
        "status": "complete",
        "task": task,
        "regime_id": regime_id,
        "regime_kind": regime_kind,
        "phase_kind": phase_kind,
        "trigger_kind": trigger_kind,
        "epoch": epoch,
        "step": step,
        "evaluation": evaluation,
        "temporal_semantics": temporal_semantics
        or {
            "rollout_mode": "unknown",
            "carry_policy": "unknown",
            "bptt_chunk_size": None,
            "teacher_forcing": None,
        },
        "summary": _to_jsonable(dict(summary)),
        "cases": cases,
        "artifacts": artifacts or {},
    }
    if capture is not None:
        manifest["capture"] = capture
    if analysis_artifacts:
        manifest["analysis_artifacts"] = analysis_artifacts
    return manifest


# =============================================================================
def load_evaluation_artifact_manifest(
    root: Path,
) -> dict[str, object]:
    """Load and validate one evaluation artifact manifest from *root*.

    This is the **single canonical reader** for evaluation artifact manifests.
    Every consumer (case loading, inspection, reuse, reporting, MLflow)
    **must** route through this function.

    Parameters
    ----------
    root:
        Path to an evaluation artifact directory containing ``manifest.json``
        and ``_SUCCESS``.

    Returns
    -------
    dict
        The validated manifest payload.

    Raises
    ------
    FileNotFoundError
        If ``manifest.json`` is missing.
    RuntimeError
        If ``_SUCCESS`` is missing.
    UnsupportedEvaluationArtifactSchema
        If ``schema`` is absent or does not match ``_ARTIFACT_SCHEMA``.
    RuntimeError
        If ``status != "complete"``.
    """
    root = Path(root)
    manifest_path = root / _MANIFEST_FILENAME
    success_path = root / _SUCCESS_FILENAME

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {_MANIFEST_FILENAME} under {root}.")
    if not success_path.exists():
        raise RuntimeError(
            f"evaluation artifact at {root} is missing {_SUCCESS_FILENAME} "
            f"sentinel (possibly incomplete or corrupted write)."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    actual_schema = manifest.get("schema")
    if actual_schema != _ARTIFACT_SCHEMA:
        raise UnsupportedEvaluationArtifactSchema(
            expected=_ARTIFACT_SCHEMA,
            actual=actual_schema,
            path=manifest_path,
        )

    status = manifest.get("status", "complete")
    if status != "complete":
        raise RuntimeError(
            f"evaluation artifact at {root} has status={status!r} "
            f"(expected 'complete')."
        )

    return manifest


# =============================================================================
def load_artifact_run_cases(run_dir: Path) -> list[LoadedArtifactCase]:
    """Load trace-bearing cases from a persisted artifact run directory."""
    manifest = load_evaluation_artifact_manifest(run_dir)
    return _load_manifest_cases(run_dir, manifest)


# =============================================================================
def resolve_dotted_symbol(symbol_ref: str) -> Any:
    """Resolve one dotted symbol path to a Python object."""
    module_name, symbol_name = symbol_ref.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


# =============================================================================
def resolve_provider(
    provider_ref: str,
    provider_settings: dict[str, Any],
    *,
    batch_size: int | None = None,
) -> Any:
    """Resolve and instantiate a provider class from dotted import path.

    Args:
        provider_ref: Dotted import path to the provider class.
        provider_settings: Keyword arguments for the provider constructor.
        batch_size: Optional first-class batch size.  If given, overrides
            any ``batch_size`` in ``provider_settings``.
    """
    provider_cls = resolve_dotted_symbol(provider_ref)
    if batch_size is not None:
        merged = dict(provider_settings, batch_size=batch_size)
    else:
        merged = provider_settings
    return provider_cls(**merged)


# =============================================================================
def _load_manifest_cases(
    run_dir: Path,
    manifest: dict[str, Any],
) -> list[LoadedArtifactCase]:
    """Load trace-bearing cases from canonical manifest bundle payloads."""
    from ehc_sn.evaluation.trace_artifacts import open_trace_reader

    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a list under key 'cases'.")

    # Resolve a trace reader for this artifact.
    reader = open_trace_reader(run_dir)

    loaded_cases: list[LoadedArtifactCase] = []
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("manifest.json case rows must be dictionaries.")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                "manifest.json case rows require a non-empty case_id."
            )

        trace: TraceTree | None = None
        temporal_semantics = manifest.get("temporal_semantics")

        # Try loading from reader (supports both Zarr and legacy NPZ).
        if reader is not None and reader.has_case(case_id):
            try:
                trace = reader.load_case(case_id)
            except Exception:
                # If reader fails, fall through to legacy NPZ path.
                pass

        # Legacy NPZ path (backward compat: direct dense/meta artifact fields).
        if trace is None:
            has_trace = bool(row.get("has_trace", False))
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
            meta_raw = json.loads(
                (run_dir / meta_rel).read_text(encoding="utf-8")
            )
            if meta_raw is None:
                meta_raw = {}
            if not isinstance(meta_raw, dict):
                raise ValueError(
                    f"Case {case_id!r} meta artifact must decode to a dict."
                )

            # Merge optional task-evidence sidecar.
            task_arrays_rel = row.get("task_arrays_path")
            if isinstance(task_arrays_rel, str):
                task_dense = _read_dense_npz(run_dir / task_arrays_rel)
                dense = {**task_dense, **dense}

            trace = _rehydrate_trace_tree(dense=dense, meta=meta_raw)

        loaded_cases.append(
            LoadedArtifactCase(
                case_id=case_id,
                source_context=row.get("source_context"),
                trace=trace,
                temporal_semantics=temporal_semantics,
            )
        )
    return loaded_cases


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
    "_ARTIFACT_SCHEMA",
    "LoadedArtifactCase",
    "UnsupportedEvaluationArtifactSchema",
    "build_evaluation_artifact_manifest",
    "collect_regime_artifact_bundle",
    "load_artifact_run_cases",
    "load_evaluation_artifact_manifest",
    "persist_regime_artifact_bundle",
    "resolve_provider",
]
