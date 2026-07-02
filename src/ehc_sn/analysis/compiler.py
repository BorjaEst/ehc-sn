"""Figure-driven evaluation plan compiler.

The compiler resolves a set of requested figure names through the figure,
aggregate, and analysis registries into a complete ``CompiledFigurePlan``
that declares:

- required model views (``model.trace_views()`` names),
- required record fields (``StepRecord`` fields),
- required run metadata keys,
- aggregate consumers to instantiate,
- analysis runners to execute post-evaluation.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ehc_sn.analysis.registries import (
    AnalysisRegistry,
    EvaluationRegistries,
)
from ehc_sn.analysis.specs import AggregateSpec, AnalysisSpec
from ehc_sn.contracts.dependencies import Dependency, DependencyKind
from ehc_sn.evaluation.contracts import (
    ArtifactKey,
    ArtifactKind,
    ArtifactRequirement,
)

if TYPE_CHECKING:
    from ehc_sn.evaluation.contracts import EvaluationExperiment
    from ehc_sn.figures.registry import FigureSpec

# Lazy imports (avoid circular dependencies at module level).
_figure_diagnostics_imported: bool = False
_FigureDiagnostic = None
_FigureRejectionCode = None
_FigureSurface = None
_ArtifactInputs = None
_TraceInputs = None
_TaskDataInputs = None


def _lazy_imports() -> None:
    global _FigureDiagnostic, _FigureRejectionCode, _FigureSurface
    global _ArtifactInputs, _TraceInputs, _TaskDataInputs
    global _figure_diagnostics_imported
    if _figure_diagnostics_imported:
        return
    from ehc_sn.evaluation.inspection import (
        FigureDiagnostic,
        FigureRejectionCode,
    )
    from ehc_sn.figures.registry import ArtifactInputs as _ArtifactInputs
    from ehc_sn.figures.registry import FigureSurface as _FigureSurface
    from ehc_sn.figures.registry import TaskDataInputs as _TaskDataInputs
    from ehc_sn.figures.registry import TraceInputs as _TraceInputs

    _FigureDiagnostic = FigureDiagnostic
    _FigureRejectionCode = FigureRejectionCode
    _figure_diagnostics_imported = True


# =============================================================================
@dataclass(frozen=True)
class FigureCompilationRequest:
    """Input to surface-aware figure compilation."""

    names: tuple[str, ...]
    surface: "Any"  # FigureSurface (lazy)
    task: str
    explicit: bool = True


# =============================================================================
@dataclass(frozen=True)
class CompiledFigure:
    """Per-figure resolved snapshot in a compiled plan.

    Captures the contract kind, required trace/meta keys, and artifact
    dependencies for one requested figure — for provenance and manifest
    recording.
    """

    spec_key: str
    contract_kind: str
    trace_keys: frozenset[str] = field(default_factory=frozenset)
    meta_keys: frozenset[str] = field(default_factory=frozenset)
    artifact_dependencies: tuple[ArtifactRequirement, ...] = field(
        default_factory=tuple
    )


# =============================================================================
@dataclass(frozen=True)
class CompiledFigurePlan:
    """Complete resolved plan for one figure-driven evaluation run.

    All fields are frozen.  ``unsupported`` lists figure names that could
    not be resolved.
    """

    requested_figures: frozenset[str] = field(default_factory=frozenset)
    required_trace_fields: frozenset[str] = field(default_factory=frozenset)
    required_meta_fields: frozenset[str] = field(default_factory=frozenset)
    required_model_views: frozenset[str] = field(default_factory=frozenset)
    required_record_fields: frozenset[str] = field(default_factory=frozenset)
    required_run_metadata: frozenset[str] = field(default_factory=frozenset)
    aggregate_specs: tuple[AggregateSpec, ...] = field(default_factory=tuple)
    analysis_specs: tuple[AnalysisSpec, ...] = field(default_factory=tuple)
    unsupported: frozenset[str] = field(default_factory=frozenset)
    resolved_figures: tuple[CompiledFigure, ...] = field(default_factory=tuple)
    diagnostics: tuple["Any", ...] = field(default_factory=tuple)
    """Structured ``FigureDiagnostic`` entries per figure."""
    surface: "Any | None" = None
    """The ``FigureSurface`` this plan was compiled for, if any."""
    diagnostics: tuple["Any", ...] = field(default_factory=tuple)
    """Structured ``FigureDiagnostic`` entries per figure."""
    surface: "Any | None" = None
    """The ``FigureSurface`` this plan was compiled for, if any."""


# =============================================================================
class FigurePlanError(ValueError):
    """Raised when a figure plan cannot be compiled."""

    def __init__(self, message: str, *, figure: str | None = None) -> None:
        self.figure = figure
        super().__init__(message)


# =============================================================================
def _resolve_artifact_requirements(
    inputs: frozenset[ArtifactRequirement],
    aggregate_registry: Any,
    analysis_registry: Any,
    *,
    visited: set[str] | None = None,
) -> tuple[frozenset[Dependency], list[AggregateSpec], list[AnalysisSpec]]:
    """Walk artifact requirements transitively and collect specs + dependencies.

    Returns ``(dependencies, aggregate_specs, analysis_specs)``.
    """
    if visited is None:
        visited = set()

    deps: set[Dependency] = set()
    agg_specs: list[AggregateSpec] = []
    anl_specs: list[AnalysisSpec] = []

    for req in inputs:
        key_str = f"{req.key.kind.value}/{req.key.name}@{req.schema_version}"
        if key_str in visited:
            continue
        visited.add(key_str)

        if req.key.kind is ArtifactKind.ANALYSIS:
            spec = analysis_registry.get(req.key.name)
            anl_specs.append(spec)

            # Recursively resolve analysis inputs.
            sub_deps, sub_agg, sub_anl = _resolve_artifact_requirements(
                spec.inputs,
                aggregate_registry,
                analysis_registry,
                visited=visited,
            )
            deps.update(sub_deps)
            agg_specs.extend(sub_agg)
            anl_specs.extend(sub_anl)

        elif req.key.kind is ArtifactKind.AGGREGATE:
            spec = aggregate_registry.get(req.key.name)
            agg_specs.append(spec)

            # Collect typed dependencies from the aggregate spec.
            for dep in spec.dependencies:
                deps.add(dep)

        else:
            raise FigurePlanError(
                f"Unsupported artifact kind {req.key.kind} "
                f"in requirement {req.key.manifest_key()}."
            )

    return frozenset(deps), agg_specs, anl_specs


# =============================================================================
def compile_figure_evaluation_plan(
    *,
    experiment: EvaluationExperiment,
    figure_names: Collection[str],
    registries: EvaluationRegistries,
) -> CompiledFigurePlan:
    """Resolve figure names to a complete evaluation plan.

    Steps:
    1. Resolve each figure name to its ``FigureSpec``.
    2. For ``BOUNDED_TRACE`` figures: collect ``trace_keys`` and ``meta_keys``.
    3. For ``ARTIFACT`` figures: transitively resolve analysis → aggregate
       dependencies, collecting model views and record fields.
    4. Separate collected dependencies by kind.

    Args:
        experiment: Pre-loaded evaluation experiment (used for model family
            validation in error messages).
        figure_names: Requested figure names.
        registries: Facade with populated figure/aggregate/analysis registries.

    Returns:
        ``CompiledFigurePlan`` with resolved dependencies.

    Raises:
        ``FigurePlanError`` if a figure is unknown or dependencies are
        unresolvable.
    """
    unsupported: list[str] = []
    required_trace_fields: set[str] = set()
    required_meta_fields: set[str] = set()
    all_deps: set[Dependency] = set()
    aggregate_specs: list[AggregateSpec] = []
    analysis_specs: list[AnalysisSpec] = []
    resolved_figures: list[CompiledFigure] = []

    for name in figure_names:
        if not registries.figures.has(name):
            unsupported.append(name)
            continue

        spec = registries.figures.get(name)

        from ehc_sn.figures.registry import (
            ArtifactInputs,
            TaskDataInputs,
            TraceInputs,
        )

        match spec.inputs:
            case TaskDataInputs(meta_keys=mk):
                required_meta_fields.update(mk)
                resolved_figures.append(
                    CompiledFigure(
                        spec_key=name,
                        contract_kind="task_data",
                        meta_keys=frozenset(mk),
                    )
                )

            case TraceInputs(trace_keys=tk, meta_keys=mk):
                required_trace_fields.update(tk)
                required_meta_fields.update(mk)
                resolved_figures.append(
                    CompiledFigure(
                        spec_key=name,
                        contract_kind="trace",
                        trace_keys=frozenset(tk),
                        meta_keys=frozenset(mk),
                    )
                )

            case ArtifactInputs(artifacts=inputs):
                deps, agg_specs, anl_specs = _resolve_artifact_requirements(
                    inputs,
                    registries.aggregates,
                    registries.analyses,
                )
                all_deps.update(deps)
                aggregate_specs.extend(agg_specs)
                analysis_specs.extend(anl_specs)
                resolved_figures.append(
                    CompiledFigure(
                        spec_key=name,
                        contract_kind="artifact",
                        artifact_dependencies=tuple(inputs),
                    )
                )

            case _:
                unsupported.append(name)

    # Separate typed dependencies by kind.
    required_model_views: set[str] = {
        d.name for d in all_deps if d.kind is DependencyKind.MODEL_VIEW
    }
    required_record_fields: set[str] = {
        d.name for d in all_deps if d.kind is DependencyKind.RECORD_FIELD
    }
    required_run_metadata: set[str] = {
        d.name for d in all_deps if d.kind is DependencyKind.RUN_METADATA
    }

    return CompiledFigurePlan(
        requested_figures=frozenset(figure_names),
        required_trace_fields=frozenset(required_trace_fields),
        required_meta_fields=frozenset(required_meta_fields),
        required_model_views=frozenset(required_model_views),
        required_record_fields=frozenset(required_record_fields),
        required_run_metadata=frozenset(required_run_metadata),
        aggregate_specs=tuple(aggregate_specs),
        analysis_specs=tuple(analysis_specs),
        unsupported=frozenset(unsupported),
        resolved_figures=tuple(resolved_figures),
    )


# =============================================================================
def compile_figure_plan(
    request: FigureCompilationRequest,
    *,
    figure_registry: "Any",
    aggregate_registry: "Any | None" = None,
    analysis_registry: "Any | None" = None,
    capture_profile: "Any | None" = None,
    available_fields: "frozenset[str] | None" = None,
) -> CompiledFigurePlan:
    """Compile a figure plan with surface, task, and capture-profile validation.

    Validates, in order for each requested figure:
      1. Exists in ``figure_registry``.
      2. Supports the requested task (when figure.task is not None).
      3. Supports the requested surface.
      4. For ``ArtifactInputs`` figures: resolves aggregate/analysis deps.
      5. For ``TraceInputs`` / ``TaskDataInputs`` figures with a capture
         profile: verifies declared trace/meta keys are in profile fields.

    ``explicit=True`` (default): unsupported figures produce ``FigureDiagnostic``
    entries in ``diagnostics``.  ``explicit=False``: silently excluded.

    Raises ``FigurePlanError`` if artifact dependencies are unresolvable
    (broken graph, not a selection mismatch).
    """
    _lazy_imports()
    FD = _FigureDiagnostic
    FRC = _FigureRejectionCode

    diagnostics: list = []
    resolved_figures_list: list[CompiledFigure] = []
    unsupported: list[str] = []
    required_trace_fields: set[str] = set()
    required_meta_fields: set[str] = set()
    all_deps: set[Dependency] = set()
    agg_specs: list[AggregateSpec] = []
    anl_specs: list[AnalysisSpec] = []

    for name in request.names:
        # 1. Exists in registry.
        if not figure_registry.has(name):
            if request.explicit:
                diagnostics.append(
                    FD(
                        figure_key=name,
                        code=FRC.UNKNOWN_FIGURE,
                        message=f"Figure {name!r} is not registered.",
                    )
                )
            else:
                unsupported.append(name)
            continue

        spec = figure_registry.get(name)

        # 2. Task match.
        if spec.task is not None and spec.task != request.task:
            if request.explicit:
                diagnostics.append(
                    FD(
                        figure_key=name,
                        code=FRC.TASK_MISMATCH,
                        message=(
                            f"Figure {name!r} is for task {spec.task!r}, "
                            f"not {request.task!r}."
                        ),
                    )
                )
            else:
                unsupported.append(name)
            continue

        # 3. Surface support.
        if request.surface not in spec.allowed_surfaces:
            if request.explicit:
                diagnostics.append(
                    FD(
                        figure_key=name,
                        code=FRC.UNSUPPORTED_CONTRACT,
                        message=(
                            f"Figure {name!r} does not support surface "
                            f"{request.surface.value!r}. "
                            f"Allowed surfaces: {sorted(s.value for s in spec.allowed_surfaces)}."
                        ),
                    )
                )
            else:
                unsupported.append(name)
            continue

        # 4-5. Per-input-contract validation.
        from ehc_sn.figures.registry import ArtifactInputs as AI
        from ehc_sn.figures.registry import TaskDataInputs, TraceInputs

        match spec.inputs:
            case AI(artifacts=inputs):
                if (
                    aggregate_registry is not None
                    and analysis_registry is not None
                ):
                    deps, extra_agg, extra_anl = _resolve_artifact_requirements(
                        inputs,
                        aggregate_registry,
                        analysis_registry,
                    )
                    all_deps.update(deps)
                    agg_specs.extend(extra_agg)
                    anl_specs.extend(extra_anl)
                resolved_figures_list.append(
                    CompiledFigure(
                        spec_key=name,
                        contract_kind="artifact",
                        artifact_dependencies=tuple(inputs),
                    )
                )

            case TraceInputs(trace_keys=tk, meta_keys=mk):
                required_trace_fields.update(tk)
                required_meta_fields.update(mk)
                _validate_capture_compatibility(
                    name,
                    tk,
                    mk,
                    capture_profile,
                    request.task,
                    diagnostics,
                    FD,
                    FRC,
                    available_fields=available_fields,
                )
                resolved_figures_list.append(
                    CompiledFigure(
                        spec_key=name,
                        contract_kind="trace",
                        trace_keys=frozenset(tk),
                        meta_keys=frozenset(mk),
                    )
                )

            case TaskDataInputs(meta_keys=mk):
                required_meta_fields.update(mk)
                _validate_capture_compatibility(
                    name,
                    frozenset(),
                    mk,
                    capture_profile,
                    request.task,
                    diagnostics,
                    FD,
                    FRC,
                    available_fields=available_fields,
                )
                resolved_figures_list.append(
                    CompiledFigure(
                        spec_key=name,
                        contract_kind="task_data",
                        meta_keys=frozenset(mk),
                    )
                )

            case _:
                unsupported.append(name)

    # Separate typed dependencies by kind.
    required_model_views: set[str] = {
        d.name for d in all_deps if d.kind is DependencyKind.MODEL_VIEW
    }
    required_record_fields: set[str] = {
        d.name for d in all_deps if d.kind is DependencyKind.RECORD_FIELD
    }
    required_run_metadata: set[str] = {
        d.name for d in all_deps if d.kind is DependencyKind.RUN_METADATA
    }

    return CompiledFigurePlan(
        requested_figures=frozenset(request.names),
        required_trace_fields=frozenset(required_trace_fields),
        required_meta_fields=frozenset(required_meta_fields),
        required_model_views=frozenset(required_model_views),
        required_record_fields=frozenset(required_record_fields),
        required_run_metadata=frozenset(required_run_metadata),
        aggregate_specs=tuple(agg_specs),
        analysis_specs=tuple(anl_specs),
        unsupported=frozenset(unsupported),
        resolved_figures=tuple(resolved_figures_list),
        diagnostics=tuple(diagnostics),
        surface=request.surface,
    )


def _validate_capture_compatibility(
    name: str,
    trace_keys: frozenset[str],
    meta_keys: frozenset[str],
    capture_profile: "Any | None",
    task: str,
    diagnostics: list,
    FD: type,
    FRC: type,
    *,
    available_fields: frozenset[str] | None = None,
) -> None:
    """Check that all trace/meta keys are covered by the capture profile.

    When ``available_fields`` is provided (from capability resolution),
    checks against those actual fields rather than the profile declaration.
    """
    if capture_profile is None and available_fields is None:
        return

    if available_fields is not None:
        if not available_fields:
            # No capabilities registered yet — can't validate.
            return
        available = available_fields
    else:
        # Fallback: use profile declaration (no capability info).
        paradigm = "tem"
        binding = capture_profile.paradigm_fields.get(paradigm)  # type: ignore[union-attr]
        if binding is None:
            return
        available = set(binding.required) | set(binding.optional)

    missing_trace = trace_keys - available
    if missing_trace:
        diagnostics.append(
            FD(
                figure_key=name,
                code=FRC.MISSING_TRACE_KEYS,
                message=(
                    f"Trace keys not in available fields"
                    f"{' from ' + capture_profile.name if capture_profile is not None else ''}"
                    f": {', '.join(sorted(missing_trace))}."
                ),
            )
        )

    missing_meta = meta_keys - available
    if missing_meta:
        diagnostics.append(
            FD(
                figure_key=name,
                code=FRC.MISSING_META_KEYS,
                message=(
                    f"Meta keys not in available fields"
                    f"{' from ' + capture_profile.name if capture_profile is not None else ''}"
                    f": {', '.join(sorted(missing_meta))}."
                ),
            )
        )


def compile_inspection_plan(
    names: Collection[str],
    *,
    task: str,
    figure_registry: "Any",
    aggregate_registry: "Any | None" = None,
    analysis_registry: "Any | None" = None,
    capture_profile: "Any | None" = None,
) -> CompiledFigurePlan:
    """Compile a plan for ``INSPECTION`` surface figures."""
    _lazy_imports()
    return compile_figure_plan(
        FigureCompilationRequest(
            names=tuple(names),
            surface=_FigureSurface.INSPECTION,
            task=task,
            explicit=True,
        ),
        figure_registry=figure_registry,
        aggregate_registry=aggregate_registry,
        analysis_registry=analysis_registry,
        capture_profile=capture_profile,
    )


def compile_report_plan(
    names: Collection[str],
    *,
    task: str,
    figure_registry: "Any",
    aggregate_registry: "Any | None" = None,
    analysis_registry: "Any | None" = None,
    capture_profile: "Any | None" = None,
) -> CompiledFigurePlan:
    """Compile a plan for ``REPORT`` surface figures."""
    _lazy_imports()
    return compile_figure_plan(
        FigureCompilationRequest(
            names=tuple(names),
            surface=_FigureSurface.REPORT,
            task=task,
            explicit=True,
        ),
        figure_registry=figure_registry,
        aggregate_registry=aggregate_registry,
        analysis_registry=analysis_registry,
        capture_profile=capture_profile,
    )


# ---------------------------------------------------------------------------
__all__ = [
    "CompiledFigure",
    "CompiledFigurePlan",
    "FigureCompilationRequest",
    "FigurePlanError",
    "compile_figure_evaluation_plan",
    "compile_figure_plan",
    "compile_inspection_plan",
    "compile_report_plan",
]
