"""Artifact inspection — read completed evaluation artifacts without
constructing a model, provider, or executor.

Provides structured dataclasses and the :func:`inspect_evaluation_artifact`
function for reading v3 evaluation artifacts.  The terminal-facing formatter
:func:`format_inspection_text` produces human-readable output.

Usage::

    from ehc_sn.eval.inspection import inspect_evaluation_artifact

    insp = inspect_evaluation_artifact(
        Path("artifacts/evaluation/hrm_v2/mazehard_n1"),
        case_index=0,
        list_fields=True,
    )
    print(insp.identity.task)
    print(insp.selected_case.fields[0].dtype)

Boundary rules:
    - Must not import from ``experiments/<task>/<family>/``.
    - Must not construct models, providers, or executors.
    - May import ``TraceTree`` from ``traces`` and ``load_artifact_run_cases``
      from ``eval.artifacts`` (both are artifact-level operations).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.eval.artifacts import load_artifact_run_cases
from ehc_sn.experiments._infra import EvaluationIdentity
from ehc_sn.traces.specs import TRACE_PROFILES

# =============================================================================
# Exceptions
# =============================================================================


class EvaluationArtifactError(ValueError):
    """Raised when an evaluation artifact is missing, incomplete, or malformed."""


# =============================================================================
# Data types
# =============================================================================


@dataclass(frozen=True)
class CapturedFieldSummary:
    """Metadata for one captured trace field in an artifact."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    storage: str  # "dense" or "meta"
    minimum: float | None = None
    maximum: float | None = None
    required: bool = False


@dataclass(frozen=True)
class EvaluationCaseSummary:
    """Lightweight per-case metadata derived from the manifest (no dense arrays)."""

    index: int
    case_id: str
    step_count: int | None = None
    halt_step: int | None = None
    truncated: bool | None = None


@dataclass(frozen=True)
class EvaluationCaseInspection:
    """Full inspection of one case including field metadata."""

    case_index: int
    case_id: str
    fields: tuple[CapturedFieldSummary, ...]
    halt_step: int | None = None
    truncated: bool | None = None


@dataclass(frozen=True)
class EvaluationArtifactInspection:
    """Complete inspection result for one evaluation artifact."""

    artifact_root: Path
    identity: EvaluationIdentity | None
    capture_profile: str | None
    capture_profile_version: int | None
    resolved_fields: tuple[str, ...]
    case_count: int
    cases: tuple[EvaluationCaseSummary, ...] = ()
    selected_case: EvaluationCaseInspection | None = None
    manifest_raw: dict[str, Any] | None = None


# =============================================================================
# Constants (mirrored from eval.artifacts — avoids circular import risk)
# =============================================================================

_MANIFEST_FILENAME = "manifest.json"
_SUCCESS_FILENAME = "_SUCCESS"
_REQUIRED_SCHEMA = "ehc_sn.eval.artifact.v3"


# =============================================================================
# Field resolution helpers
# =============================================================================


def _resolve_profile_required_fields(
    manifest: dict[str, Any],
) -> tuple[str, int | None, tuple[str, ...]]:
    """Extract capture profile name, version, and required field set from the manifest.

    Returns ``(profile_name, profile_version, required_field_names)``.
    If no capture block exists, returns ``(None, None, ())``.
    """
    capture = manifest.get("capture")
    if not isinstance(capture, dict):
        return None, None, ()

    profile = capture.get("profile")
    version = capture.get("profile_version")
    if not isinstance(profile, str) or profile not in TRACE_PROFILES:
        return profile, version, ()

    profile_spec = TRACE_PROFILES[profile]
    paradigm = capture.get("paradigm") or manifest.get("trace_paradigm", "act")
    binding = profile_spec.paradigm_fields.get(paradigm)
    if binding is None:
        return profile, version, ()

    return profile, version, tuple(binding.required)


# =============================================================================
# Public API
# =============================================================================


def inspect_evaluation_artifact(
    artifact_root: Path,
    *,
    case_index: int | None = None,
    list_cases: bool = False,
    list_fields: bool = False,
    include_manifest: bool = False,
    expected_experiment_id: str | None = None,
) -> EvaluationArtifactInspection:
    """Inspect a completed evaluation artifact.

    Reads the manifest and optional case data.  Does NOT construct a
    model, provider, or executor.

    Parameters
    ----------
    artifact_root:
        Path to a completed evaluation artifact directory
        (contains ``manifest.json`` and ``_SUCCESS``).
    case_index:
        If provided, load full field metadata for that case index.
    list_cases:
        If True, include per-case summaries (no dense arrays).
    list_fields:
        If True and ``case_index`` is provided, include per-field
        dtype, shape, and range metadata.
    include_manifest:
        If True, include the raw manifest dict in the result.
    expected_experiment_id:
        If provided, the artifact's ``experiment_id`` (or ``task``
        for older v3 artifacts) must match.

    Returns
    -------
    EvaluationArtifactInspection

    Raises
    ------
    EvaluationArtifactError
        On missing manifest, missing ``_SUCCESS``, schema mismatch,
        or case index out of range.
    FileNotFoundError
        If ``artifact_root`` does not exist (not caught).
    """
    artifact_root = artifact_root.resolve()

    # ---- Validate structure -------------------------------------------------
    manifest_path = artifact_root / _MANIFEST_FILENAME
    success_path = artifact_root / _SUCCESS_FILENAME

    if not manifest_path.exists():
        raise EvaluationArtifactError(
            f"No manifest.json found in {artifact_root}."
        )
    if not success_path.exists():
        raise EvaluationArtifactError(
            f"Artifact at {artifact_root} is incomplete "
            f"(missing _SUCCESS sentinel)."
        )

    manifest = json.loads(manifest_path.read_text("utf-8"))

    # ---- Validate schema ----------------------------------------------------
    schema = manifest.get("schema")
    if schema != _REQUIRED_SCHEMA:
        raise EvaluationArtifactError(
            f"Unsupported artifact schema {schema!r}. "
            f"Expected {_REQUIRED_SCHEMA!r}."
        )

    # ---- Validate experiment identity ---------------------------------------
    if expected_experiment_id is not None:
        artifact_id = manifest.get("experiment_id", manifest.get("task"))
        if artifact_id != expected_experiment_id:
            raise EvaluationArtifactError(
                f"Artifact experiment_id={artifact_id!r} does not "
                f"match expected {expected_experiment_id!r}."
            )

    # ---- Extract identity ---------------------------------------------------
    identity = None
    task = manifest.get("task")
    model_family = manifest.get("provenance", {}).get("model_family")
    trace_paradigm = manifest.get("provenance", {}).get("trace_paradigm")
    if task:
        identity = EvaluationIdentity(
            task=task,
            model_family=model_family or "unknown",
            trace_paradigm=trace_paradigm or "unknown",
        )

    # ---- Extract capture provenance -----------------------------------------
    capture_profile, capture_profile_version, required_fields = (
        _resolve_profile_required_fields(manifest)
    )

    resolved_fields = tuple(
        manifest.get("capture", {}).get("resolved_fields", [])
    )

    # ---- Case count ---------------------------------------------------------
    case_count = manifest.get("summary", {}).get("n_cases") or len(
        manifest.get("cases", [])
    )

    # ---- Case summaries -----------------------------------------------------
    cases: tuple[EvaluationCaseSummary, ...] = ()
    if list_cases and isinstance(manifest.get("cases"), list):
        case_rows = []
        for i, row in enumerate(manifest["cases"]):
            steps = row.get("steps")
            halt_step = row.get("halt_step")
            truncated = row.get("truncated")
            case_rows.append(
                EvaluationCaseSummary(
                    index=i,
                    case_id=row.get("case_id", f"case-{i}"),
                    step_count=int(steps) if steps is not None else None,
                    halt_step=(
                        int(halt_step) if halt_step is not None else None
                    ),
                    truncated=(
                        bool(truncated) if truncated is not None else None
                    ),
                )
            )
        cases = tuple(case_rows)

    # ---- Selected case ------------------------------------------------------
    selected_case: EvaluationCaseInspection | None = None
    if case_index is not None:
        loaded = load_artifact_run_cases(artifact_root)
        if case_index < 0 or case_index >= len(loaded):
            raise EvaluationArtifactError(
                f"Case index {case_index} out of range "
                f"(0–{len(loaded) - 1})."
            )
        lc = loaded[case_index]

        # Build field metadata
        fields: list[CapturedFieldSummary] = []

        if list_fields:
            # Dense fields from export(flatten=True)
            dense_map = lc.trace.export(flatten=True, sep="/")
            if isinstance(dense_map, dict):
                for key in sorted(dense_map.keys()):
                    arr = np.asarray(dense_map[key])
                    arr_min = None
                    arr_max = None
                    if arr.size > 0 and arr.dtype.kind != "O":
                        try:
                            arr_min = float(arr.min())
                            arr_max = float(arr.max())
                        except (TypeError, ValueError):
                            pass
                    fields.append(
                        CapturedFieldSummary(
                            name=key,
                            dtype=str(arr.dtype),
                            shape=arr.shape,
                            storage="dense",
                            minimum=arr_min,
                            maximum=arr_max,
                            required=key in required_fields,
                        )
                    )

            # Meta fields
            meta = lc.trace.get_meta()
            if isinstance(meta, dict):
                for key in sorted(meta.keys()):
                    val = meta[key]
                    fields.append(
                        CapturedFieldSummary(
                            name=key,
                            dtype=type(val).__name__,
                            shape=(),
                            storage="meta",
                            required=key in required_fields,
                        )
                    )

        selected_case = EvaluationCaseInspection(
            case_index=case_index,
            case_id=lc.case_id,
            fields=tuple(fields),
        )

    return EvaluationArtifactInspection(
        artifact_root=artifact_root,
        identity=identity,
        capture_profile=capture_profile,
        capture_profile_version=capture_profile_version,
        resolved_fields=resolved_fields,
        case_count=case_count,
        cases=cases,
        selected_case=selected_case,
        manifest_raw=manifest if include_manifest else None,
    )


# =============================================================================
# Formatting
# =============================================================================


def format_inspection_text(
    inspection: EvaluationArtifactInspection,
) -> str:
    """Format an :class:`EvaluationArtifactInspection` as human-readable text.

    Returns a string suitable for printing to a terminal.  Does not
    include color or ANSI escape codes.
    """
    lines: list[str] = []
    lines.append("")

    # Identity block
    ident = inspection.identity
    if ident is not None:
        lines.append(f"  Task:             {ident.task}")
        lines.append(f"  Model family:     {ident.model_family}")
        lines.append(f"  Trace paradigm:   {ident.trace_paradigm}")
    lines.append(f"  Artifact:         {inspection.artifact_root}")

    # Capture block
    if inspection.capture_profile:
        version = (
            f"@{inspection.capture_profile_version}"
            if inspection.capture_profile_version
            else ""
        )
        lines.append(
            f"  Capture profile:  {inspection.capture_profile}{version}"
        )
        if inspection.resolved_fields:
            lines.append(
                f"  Resolved fields:  {len(inspection.resolved_fields)} — "
                f"{', '.join(inspection.resolved_fields[:6])}"
                f"{'…' if len(inspection.resolved_fields) > 6 else ''}"
            )
    lines.append(f"  Cases:            {inspection.case_count}")

    # Case summaries
    if inspection.cases:
        lines.append("")
        header = f"  {'Index':<6} {'Case ID':<35} {'Steps':<8} {'Halted':<8} {'Truncated'}"
        sep_line = f"  {'-'*6} {'-'*35} {'-'*8} {'-'*8} {'-'*9}"
        lines.append(header)
        lines.append(sep_line)
        for c in inspection.cases:
            steps = str(c.step_count) if c.step_count is not None else "-"
            halted = f"step {c.halt_step}" if c.halt_step is not None else "-"
            truncated = "yes" if c.truncated else "no"
            lines.append(
                f"  {c.index:<6} {c.case_id:<35} {steps:<8} {halted:<8} {truncated}"
            )

    # Selected case
    if inspection.selected_case:
        sc = inspection.selected_case
        lines.append("")
        lines.append(f"  Case {sc.case_index}: {sc.case_id}")
        if sc.halt_step is not None:
            lines.append(f"    Halt step:    {sc.halt_step}")
        if sc.truncated is not None:
            lines.append(f"    Truncated:    {'yes' if sc.truncated else 'no'}")
        lines.append(f"    Fields:       {len(sc.fields)}")

        if sc.fields:
            lines.append("")
            fh = f"    {'Field':<40} {'Required':<9} {'Shape':<20} {'Dtype':<10} {'Range'}"
            fs = f"    {'-'*40} {'-'*9} {'-'*20} {'-'*10} {'-'*15}"
            lines.append(fh)
            lines.append(fs)
            for f in sc.fields:
                req = "yes" if f.required else "no"
                shape = str(list(f.shape))
                rng = ""
                if f.minimum is not None and f.maximum is not None:
                    rng = f"[{f.minimum:.4g}, {f.maximum:.4g}]"
                lines.append(
                    f"    {f.name:<40} {req:<9} {shape:<20} {f.dtype:<10} {rng}"
                )

    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CapturedFieldSummary",
    "EvaluationArtifactError",
    "EvaluationArtifactInspection",
    "EvaluationCaseInspection",
    "EvaluationCaseSummary",
    "format_inspection_text",
    "inspect_evaluation_artifact",
]
