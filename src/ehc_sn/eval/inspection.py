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
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.eval.artifact_models import (
    RegimeArtifactManifest,
    RegimeArtifactSet,
    _normalize_v1_regime_manifest,
)
from ehc_sn.eval.artifacts import (
    _ARTIFACT_SCHEMA,
    UnsupportedEvaluationArtifactSchema,
    load_artifact_run_cases,
)
from ehc_sn.eval.artifacts import (
    load_evaluation_artifact_manifest as _load_regime_manifest_raw,
)
from ehc_sn.eval.contracts import EvaluationIdentity
from ehc_sn.figures import REGISTRY, list_figure_specs
from ehc_sn.traces.specs import TRACE_PROFILES

# =============================================================================
# Exceptions
# =============================================================================


class EvaluationArtifactError(ValueError):
    """Raised when an evaluation artifact is missing, incomplete, or malformed."""


# =============================================================================
# Diagnostic types
# =============================================================================


class FigureRejectionCode(StrEnum):
    """Categorised reason a figure was rejected or could not be rendered."""

    UNKNOWN_FIGURE = "unknown_figure"
    UNSUPPORTED_CONTRACT = "unsupported_contract"
    UNRESOLVABLE_ROLE = "unresolvable_role"
    MISSING_TRACE_KEYS = "missing_trace_keys"
    MISSING_META_KEYS = "missing_meta_keys"
    RENDER_ERROR = "render_error"


@dataclass(frozen=True)
class FigureDiagnostic:
    """One structured diagnostic for a gallery figure problem.

    Attributes
    ----------
    figure_key:
        Figure name, role, or identifier that caused the diagnostic.
    code:
        Categorised rejection or failure code.
    message:
        Human-readable explanation.
    """

    figure_key: str
    code: FigureRejectionCode
    message: str


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
    requested_figures: tuple[str, ...] = ()
    diagnostics: tuple[FigureDiagnostic, ...] = ()


@dataclass(frozen=True)
class EvaluationArtifactInspection:
    """Complete inspection result for one evaluation artifact."""

    artifact_root: Path
    identity: EvaluationIdentity | None
    manifest: RegimeArtifactManifest | None = None
    capture_profile: str | None = None
    capture_profile_version: int | None = None
    resolved_fields: tuple[str, ...] = ()
    case_count: int = 0
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


# =============================================================================
# Manifest conversion helper
# =============================================================================


def _typed_manifest_to_dict(
    m: RegimeArtifactManifest,
) -> dict[str, Any]:
    """Convert a typed ``RegimeArtifactManifest`` back to a dict.

    Used within ``inspect_evaluation_artifact`` for backward-compatible
    code paths (gallery, profile resolution) that currently consume raw
    dicts.
    """
    d: dict[str, Any] = {
        "schema": m.schema,
        "status": m.status,
        "regime_id": m.regime_id,
        "regime_kind": m.regime_kind,
        "phase_kind": m.phase_kind,
        "trigger_kind": m.trigger_kind,
        "epoch": m.epoch,
        "step": m.step,
        "summary": dict(m.summary),
        "artifacts": dict(m.artifacts),
    }
    if m.task is not None:
        d["task"] = m.task
    if m.capture is not None:
        d["capture"] = dict(m.capture)
    return d


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
    artifact: Path | RegimeArtifactSet,
    *,
    case_index: int | None = None,
    list_cases: bool = False,
    list_fields: bool = False,
    include_manifest: bool = False,
    expected_recipe_id: str | None = None,
    gallery: bool = False,
    gallery_output: Path | None = None,
    max_gallery_samples: int = 8,
    gallery_roles: tuple[str, ...] = (),
) -> EvaluationArtifactInspection:
    """Inspect a completed evaluation artifact.

    Reads the manifest and optional case data.  Does NOT construct a
    model, provider, or executor.

    When ``artifact`` is a ``Path``, it is loaded as a ``RegimeArtifactSet``
    first.  Existing v1 regime artifacts are normalised automatically.

    When ``gallery=True``, renders evaluation figures for the selected
    cases and writes PNG images to ``gallery_output/images/``.

    Parameters
    ----------
    artifact:
        Path to a completed evaluation artifact directory
        (contains ``manifest.json`` and ``_SUCCESS``), or an already-loaded
        ``RegimeArtifactSet``.
    case_index:
        If provided, load full field metadata for that case index.
    list_cases:
        If True, include per-case summaries (no dense arrays).
    list_fields:
        If True and ``case_index`` is provided, include per-field
        dtype, shape, and range metadata.
    include_manifest:
        If True, include the raw manifest dict in the result.
    expected_recipe_id:
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
        Figure roles to render (e.g. ``("prediction",)``).

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
        If ``artifact`` is a ``Path`` and the directory does not exist
        (not caught).
    """
    if gallery and gallery_output is None:
        raise ValueError("gallery_output is required when gallery=True.")

    # --- Load or accept RegimeArtifactSet ------------------------------------
    if isinstance(artifact, RegimeArtifactSet):
        regime = artifact
    else:
        artifact_root = artifact.resolve()
        try:
            regime = RegimeArtifactSet.load(artifact_root)
        except UnsupportedEvaluationArtifactSchema as exc:
            raise EvaluationArtifactError(str(exc)) from exc
        except RuntimeError as exc:
            raise EvaluationArtifactError(str(exc)) from exc

    artifact_root = regime.root
    typed_manifest = regime.manifest

    # --- Convert typed manifest back to raw dict for backward-compat code -----
    manifest = _typed_manifest_to_dict(typed_manifest)

    # ---- Validate experiment identity ---------------------------------------
    if expected_recipe_id is not None:
        artifact_id = manifest.get("experiment_id", typed_manifest.regime_id)
        if artifact_id != expected_recipe_id:
            raise EvaluationArtifactError(
                f"Artifact experiment_id={artifact_id!r} does not "
                f"match expected {expected_recipe_id!r}."
            )

    # ---- Extract identity -----------------------------------------------
    identity = None
    # Prefer the regime manifest's top-level "task" field; fall back to
    # regime_kind (v1 manifests always carry "task").
    task = typed_manifest.task or typed_manifest.regime_kind
    # Derive model_family / trace_paradigm from the evaluation provenance
    # block embedded in the regime manifest.
    eval_block = manifest.get("evaluation", {})
    model_family = None
    trace_paradigm = None
    if isinstance(eval_block, dict):
        model_family = eval_block.get("model_family")
        trace_paradigm = eval_block.get("trace_paradigm")
    if task:
        identity = EvaluationIdentity(
            task=task,
            model_family=model_family or "unknown",
            trace_paradigm=trace_paradigm or "unknown",
        )

    # ---- Capture provenance (from typed manifest) -------------------------
    capture_profile, capture_profile_version, required_fields = (
        _resolve_profile_required_fields(manifest)
    )

    resolved_fields = tuple(
        manifest.get("capture", {}).get("resolved_fields", [])
    )

    # ---- Case and sample counts --------------------------------------------
    case_count = typed_manifest.summary.get("n_cases") or len(
        manifest.get("cases", [])
    )
    sample_count = typed_manifest.summary.get("n_samples", 0)

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
        gallery_requested_figures: tuple[str, ...] = ()
        gallery_specs: tuple = ()
        gallery_pending_diagnostics: list[FigureDiagnostic] = []
        if gallery_roles:
            # CLI explicit roles: resolve to specs under evaluation category.
            specs: list = []
            for role in gallery_roles:
                try:
                    spec = REGISTRY.resolve(
                        category="evaluation",
                        role=role,
                        task=typed_manifest.task
                        or typed_manifest.regime_kind
                        or "",
                    )
                    specs.append(spec)
                except KeyError:
                    gallery_pending_diagnostics.append(
                        FigureDiagnostic(
                            figure_key=role,
                            code=FigureRejectionCode.UNRESOLVABLE_ROLE,
                            message=(
                                f"No registered figure for role={role!r}, "
                                f"task={typed_manifest.task or typed_manifest.regime_kind or ''!r}."
                            ),
                        )
                    )
            gallery_specs = tuple(specs)
        else:
            # Recipe-driven mode: resolve specs from recipe figures.
            (
                gallery_specs,
                gallery_requested_figures,
            ) = _resolve_default_gallery_roles(artifact_root)
        gallery_result = _build_gallery(
            artifact_root=artifact_root,
            manifest=manifest,
            task=typed_manifest.task or typed_manifest.regime_kind or "",
            resolved_fields=resolved_fields,
            max_samples=max_gallery_samples,
            figure_specs=gallery_specs,
            output_root=gallery_output,
            requested_figures=gallery_requested_figures,
            pre_diagnostics=tuple(gallery_pending_diagnostics),
        )

    return EvaluationArtifactInspection(
        artifact_root=artifact_root,
        identity=identity,
        manifest=typed_manifest,
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


def _resolve_default_gallery_roles(
    artifact_root: Path,
) -> tuple[tuple, tuple[str, ...]]:
    """Derive default gallery roles from the recipe's ``[inspection]`` section.

    Uses ``compile_inspection_plan()`` for surface-aware compilation.
    Raises ``EvaluationArtifactError`` when incompatible figures are
    explicitly selected.  Returns empty tuples on graceful degradation.
    """
    # Ensure built-in figures are registered before resolving names.
    list_figure_specs()

    # Read alias from evaluation-manifest.json.
    eval_manifest_path = artifact_root / "evaluation-manifest.json"
    if not eval_manifest_path.exists():
        return (), ()
    try:
        with eval_manifest_path.open() as f:
            eval_manifest = json.load(f)
    except Exception:
        return (), ()

    alias = eval_manifest.get("identity", {}).get("alias")
    if not alias:
        return (), ()

    # Read recipe TOML.
    recipes_dir = Path("config/evaluation/recipes")
    recipe_path = recipes_dir / f"{alias}.toml"
    if not recipe_path.exists():
        return (), ()

    try:
        recipe = tomllib.loads(recipe_path.read_text())
    except Exception:
        return (), ()

    task = recipe.get("task", "arena")
    figure_names = recipe.get("inspection", {}).get("figures", None)

    if not figure_names:
        return (), ()

    from ehc_sn.analysis.compiler import compile_inspection_plan

    plan = compile_inspection_plan(
        figure_names,
        task=task,
        figure_registry=REGISTRY,
    )

    # Raise on explicit incompatibilities.
    if plan.diagnostics:
        diag_lines = "\n".join(
            f"  - {d.figure_key}: {d.message}" for d in plan.diagnostics
        )
        raise EvaluationArtifactError(
            f"The evaluation gallery cannot render "
            f"{len(plan.diagnostics)} requested figure(s):\n"
            f"{diag_lines}\n"
        )

    # Return specs for compatible figures.
    compatible_specs = [REGISTRY.get(f.spec_key) for f in plan.resolved_figures]
    return tuple(compatible_specs), tuple(figure_names)


def _build_gallery(
    *,
    artifact_root: Path,
    manifest: dict[str, Any],
    task: str,
    resolved_fields: tuple[str, ...],
    max_samples: int,
    figure_specs: tuple = (),
    output_root: Path,
    requested_figures: tuple[str, ...] = (),
    pre_diagnostics: tuple[FigureDiagnostic, ...] = (),
) -> InspectionGallery:
    """Render gallery images from concrete ``FigureSpec`` objects.

    Each spec is rendered against every trace case.  No role re-resolution
    occurs — specs obtained during resolution are used directly.
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

    diagnostics: list[FigureDiagnostic] = list(pre_diagnostics)

    for spec in figure_specs:
        rendered_roles.add(spec.name)

        for i in range(n_to_render):
            trace = loaded[i].trace
            from ehc_sn.figures.registry import TaskDataInputs, TraceInputs

            trace_keys: frozenset[str] = frozenset()
            meta_keys: frozenset[str] = frozenset()
            match spec.inputs:
                case TraceInputs(trace_keys=tk, meta_keys=mk):
                    trace_keys = tk
                    meta_keys = mk
                case TaskDataInputs(meta_keys=mk):
                    meta_keys = mk
                case _:
                    pass
            # A trace key is present if it exists as a direct path OR as a
            # prefix of a multi-scale band path (e.g. "diagnostic/lec/cells"
            # matches "diagnostic/lec/cells/freq_0").
            present = set(trace.path_strs)
            for p in list(present):
                # Add all parent prefixes: diagnostic/lec/cells/freq_0 → ...
                parts = p.split("/")
                for end in range(1, len(parts)):
                    present.add("/".join(parts[:end]))
            missing_trace = tuple(sorted(trace_keys - present))
            missing_meta = tuple(
                sorted(mk for mk in meta_keys if not trace.has_meta_path(mk))
            )
            if missing_trace or missing_meta:
                image_records.append(
                    InspectionGalleryImage(
                        sample_index=i,
                        role=spec.name,
                        path=Path(),
                        success=False,
                        missing_trace_keys=missing_trace,
                        missing_meta_keys=missing_meta,
                    )
                )
                if missing_trace:
                    diagnostics.append(
                        FigureDiagnostic(
                            figure_key=f"{spec.name}/case_{i:04d}",
                            code=FigureRejectionCode.MISSING_TRACE_KEYS,
                            message=(
                                f"Missing trace keys: {', '.join(missing_trace)}"
                            ),
                        )
                    )
                if missing_meta:
                    diagnostics.append(
                        FigureDiagnostic(
                            figure_key=f"{spec.name}/case_{i:04d}",
                            code=FigureRejectionCode.MISSING_META_KEYS,
                            message=(
                                f"Missing meta keys: {', '.join(missing_meta)}"
                            ),
                        )
                    )
                continue

            try:
                fig = render(spec.name, trace, ctx)
                fname = f"{spec.name}_{i:04d}.png"
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
                        role=spec.name,
                        path=Path("images") / fname,
                        success=True,
                    )
                )
            except Exception as exc:
                image_records.append(
                    InspectionGalleryImage(
                        sample_index=i,
                        role=spec.name,
                        path=Path(),
                        success=False,
                        error=str(exc),
                    )
                )
                diagnostics.append(
                    FigureDiagnostic(
                        figure_key=f"{spec.name}/case_{i:04d}",
                        code=FigureRejectionCode.RENDER_ERROR,
                        message=str(exc),
                    )
                )

    # Write inspection manifest.
    n_rendered = len(rendered_roles)
    n_errors = sum(1 for img in image_records if not img.success)
    gallery_success = n_rendered > 0 and n_errors == 0

    gallery_manifest: dict[str, object] = {
        "schema": "ehc_sn.eval.inspection.v1",
        "source_artifact": str(artifact_root),
        "source_digest": source_digest,
        "task": task,
        "requested_figures": requested_figures,
        "rendered_roles": sorted(rendered_roles),
        "diagnostics": [
            {"figure_key": d.figure_key, "code": d.code, "message": d.message}
            for d in diagnostics
        ],
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
    if gallery_success:
        (output_root / "_SUCCESS").write_text("", encoding="utf-8")

    return InspectionGallery(
        output_root=output_root,
        rendered_roles=tuple(sorted(rendered_roles)),
        sample_count=n_to_render,
        images=tuple(image_records),
        requested_figures=tuple(requested_figures),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "CapturedFieldSummary",
    "EvaluationArtifactError",
    "EvaluationArtifactInspection",
    "EvaluationCaseInspection",
    "EvaluationCaseSummary",
    "FigureDiagnostic",
    "FigureRejectionCode",
    "InspectionGallery",
    "InspectionGalleryImage",
    "format_inspection_text",
    "inspect_evaluation_artifact",
]
