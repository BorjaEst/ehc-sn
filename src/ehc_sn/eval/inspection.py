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
    n_samples: int = 1
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
class InspectionGalleryImage:
    """One rendered inspection figure image."""

    sample_index: int
    role: str
    path: Path  # relative to gallery output root
    success: bool
    missing_trace_keys: tuple[str, ...] = ()
    missing_meta_keys: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class InspectionGallery:
    """Gallery result for an inspection run."""

    output_root: Path
    rendered_roles: tuple[str, ...]
    sample_count: int
    images: tuple[InspectionGalleryImage, ...]
    role_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationArtifactInspection:
    """Complete inspection result for one evaluation artifact."""

    artifact_root: Path
    identity: EvaluationIdentity | None
    capture_profile: str | None
    capture_profile_version: int | None
    resolved_fields: tuple[str, ...]
    case_count: int
    sample_count: int = 0
    cases: tuple[EvaluationCaseSummary, ...] = ()
    selected_case: EvaluationCaseInspection | None = None
    manifest_raw: dict[str, Any] | None = None
    gallery: InspectionGallery | None = None


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
    gallery: bool = False,
    gallery_output: Path | None = None,
    max_gallery_samples: int = 8,
    gallery_roles: tuple[str, ...] = ("prediction_reasoning",),
) -> EvaluationArtifactInspection:
    """Inspect a completed evaluation artifact.

    Reads the manifest and optional case data.  Does NOT construct a
    model, provider, or executor.

    When ``gallery=True``, renders evaluation figures for the selected
    cases and writes PNG images to ``gallery_output/images/``.

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
    gallery:
        If True, render evaluation figure images to ``gallery_output``.
    gallery_output:
        Destination directory for the generated inspection artifact.
        Required when ``gallery=True``.
    max_gallery_samples:
        Maximum number of evaluation batches to render (default 8).
    gallery_roles:
        Figure roles to render (e.g. ``("prediction_reasoning",)``).

    Returns
    -------
    EvaluationArtifactInspection

    Raises
    ------
    EvaluationArtifactError
        On missing manifest, missing ``_SUCCESS``, schema mismatch,
        case index out of range, or missing trace cases for gallery.
    ValueError
        If ``gallery=True`` but ``gallery_output`` is ``None``.
    FileNotFoundError
        If ``artifact_root`` does not exist (not caught).
    """
    if gallery and gallery_output is None:
        raise ValueError("gallery_output is required when gallery=True.")

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

    # ---- Case and sample counts --------------------------------------------
    case_count = manifest.get("summary", {}).get("n_cases") or len(
        manifest.get("cases", [])
    )
    sample_count = manifest.get("summary", {}).get("n_samples", 0)

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
                    n_samples=row.get("n_samples", 1),
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

    # ---- Gallery (optional) -------------------------------------------------
    gallery_result: InspectionGallery | None = None
    if gallery and gallery_output is not None:
        gallery_result = _build_gallery(
            artifact_root=artifact_root,
            manifest=manifest,
            task=task or "",
            resolved_fields=resolved_fields,
            max_samples=max_gallery_samples,
            roles=gallery_roles,
            output_root=gallery_output,
        )

    return EvaluationArtifactInspection(
        artifact_root=artifact_root,
        identity=identity,
        capture_profile=capture_profile,
        capture_profile_version=capture_profile_version,
        resolved_fields=resolved_fields,
        case_count=case_count,
        sample_count=sample_count,
        cases=cases,
        selected_case=selected_case,
        manifest_raw=manifest if include_manifest else None,
        gallery=gallery_result,
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
    lines.append(f"  Evaluated batches: {inspection.case_count}")
    lines.append(f"  Evaluated samples: {inspection.sample_count}")

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


# =============================================================================
# Gallery generation
# =============================================================================


def _build_gallery(
    *,
    artifact_root: Path,
    manifest: dict[str, Any],
    task: str,
    resolved_fields: tuple[str, ...],
    max_samples: int,
    roles: tuple[str, ...],
    output_root: Path,
) -> InspectionGallery:
    """Render evaluation figure images from an artifact and write them.

    Dispatches to registered figure templates by ``category=evaluation``,
    ``role``, and ``task``.  Skips roles with no matching registration
    or missing required trace keys.
    """
    import hashlib

    from ehc_sn.figures import (
        REGISTRY,
        FigureContext,
        list_figure_specs,
        render,
    )

    # Ensure built-in figures are registered.
    list_figure_specs()

    output_root = output_root.resolve()
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the eval artifact digest for provenance.
    source_digest = ""
    try:
        source_digest = hashlib.sha256(
            (artifact_root / "manifest.json").read_bytes()
        ).hexdigest()[:16]
    except OSError:
        pass

    # Load cases.
    loaded = load_artifact_run_cases(artifact_root)
    if not loaded:
        raise EvaluationArtifactError("No trace cases available for gallery.")

    n_to_render = min(max_samples, len(loaded))

    # Build a FigureContext for each sample.
    ctx = FigureContext()

    image_records: list[InspectionGalleryImage] = []
    rendered_roles: set[str] = set()

    role_error_records: list[str] = []

    for role in roles:
        # Resolve figure spec for this task + role.
        try:
            spec = REGISTRY.resolve(
                category="evaluation",
                role=role,  # type: ignore[arg-type]
                task=task,
            )
        except KeyError as exc:
            role_error_records.append(str(exc))
            continue

        rendered_roles.add(role)

        for i in range(n_to_render):
            trace = loaded[i].trace
            # Validate required fields are present.
            missing_trace = tuple(
                sorted(spec.trace_keys - set(trace.path_strs))
            )
            missing_meta = tuple(
                sorted(
                    mk for mk in spec.meta_keys if not trace.has_meta_path(mk)
                )
            )
            if missing_trace or missing_meta:
                image_records.append(
                    InspectionGalleryImage(
                        sample_index=i,
                        role=role,
                        path=Path(),
                        success=False,
                        missing_trace_keys=missing_trace,
                        missing_meta_keys=missing_meta,
                    )
                )
                continue

            try:
                fig = render(spec.name, trace, ctx)
                fname = f"{role}_{i:04d}.png"
                fig_path = images_dir / fname
                fig.savefig(
                    fig_path,
                    dpi=150,
                    bbox_inches="tight",
                    facecolor="white",
                )
                import matplotlib.pyplot as plt

                plt.close(fig)
                image_records.append(
                    InspectionGalleryImage(
                        sample_index=i,
                        role=role,
                        path=Path("images") / fname,
                        success=True,
                    )
                )
            except Exception as exc:
                image_records.append(
                    InspectionGalleryImage(
                        sample_index=i,
                        role=role,
                        path=Path(),
                        success=False,
                        error=str(exc),
                    )
                )

    # Write inspection manifest.
    gallery_manifest: dict[str, object] = {
        "schema": "ehc_sn.eval.inspection.v1",
        "source_artifact": str(artifact_root),
        "source_digest": source_digest,
        "task": task,
        "rendered_roles": sorted(rendered_roles),
        "sample_count": n_to_render,
        "images": [
            {
                "sample_index": img.sample_index,
                "role": img.role,
                "path": str(img.path) if img.path else None,
                "success": img.success,
                "missing_trace_keys": list(img.missing_trace_keys),
                "missing_meta_keys": list(img.missing_meta_keys),
                "error": img.error,
            }
            for img in image_records
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(gallery_manifest, indent=2), encoding="utf-8"
    )
    (output_root / "_SUCCESS").write_text("", encoding="utf-8")

    return InspectionGallery(
        output_root=output_root,
        rendered_roles=tuple(sorted(rendered_roles)),
        sample_count=n_to_render,
        images=tuple(image_records),
        role_errors=tuple(role_error_records),
    )


__all__ = [
    "CapturedFieldSummary",
    "EvaluationArtifactError",
    "EvaluationArtifactInspection",
    "EvaluationCaseInspection",
    "EvaluationCaseSummary",
    "InspectionGallery",
    "InspectionGalleryImage",
    "format_inspection_text",
    "inspect_evaluation_artifact",
]
