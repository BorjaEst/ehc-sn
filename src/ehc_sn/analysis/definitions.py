"""Unified evaluation definitions — one validated collection of deployable assets.

Analogous to Dagster's ``Definitions`` object.  Collects the five distributed
registries (figures, aggregates, analyses, recipes, capture profiles) and
provides a single ``validate()`` entry point for cross-registry integrity
checks.

Usage::

    from ehc_sn.analysis.definitions import DEFINITIONS

    diags = DEFINITIONS.validate()
    if diags:
        for d in diags:
            print(f"  [{d.code.value}] {d.figure_key}: {d.message}")
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ehc_sn.evaluation.recipes import EvaluationAlias, EvaluationRecipe
    from ehc_sn.traces.specs import CaptureProfileSpec

_initialized: bool = False
_DEFINITIONS: "EvaluationDefinitions | None" = None


def _initialize() -> "EvaluationDefinitions":
    """Lazily build the module-level ``DEFINITIONS`` singleton."""
    global _initialized, _DEFINITIONS
    if _initialized and _DEFINITIONS is not None:
        return _DEFINITIONS

    # Ensure figures are registered.
    from ehc_sn.figures import list_figure_specs

    list_figure_specs()

    # Ensure aggregate/analysis specs are registered.
    from ehc_sn.analysis.registries import (
        AggregateRegistry,
        AnalysisRegistry,
        EvaluationRegistries,
        register_builtin_analysis_specs,
    )
    from ehc_sn.figures.registry import REGISTRY as _figure_registry

    _agg_registry = AggregateRegistry()
    _anl_registry = AnalysisRegistry()
    _eval_registries = EvaluationRegistries(
        figures=_figure_registry,
        aggregates=_agg_registry,
        analyses=_anl_registry,
    )
    register_builtin_analysis_specs(_eval_registries)

    # Resolve recipes.
    from ehc_sn.evaluation.recipes import (
        _EVALUATION_RECIPE_BINDINGS,
        list_recipes,
        resolve_recipe,
    )

    list_recipes()  # ensure bindings are populated
    _recipes: dict[str, Any] = {}
    for alias, binding in _EVALUATION_RECIPE_BINDINGS.items():
        _recipes[alias.value] = resolve_recipe(alias)

    # Load capture profiles.
    from ehc_sn.traces.specs import TRACE_PROFILES

    _DEFINITIONS = EvaluationDefinitions(
        figures=_figure_registry,
        aggregates=_agg_registry,
        analyses=_anl_registry,
        recipes=dict(_recipes),
        capture_profiles=dict(TRACE_PROFILES),
    )
    _initialized = True
    return _DEFINITIONS


# =============================================================================
@dataclass(frozen=True)
class EvaluationDefinitions:
    """One validated collection of deployable evaluation assets.

    Attributes
    ----------
    figures:
        Registry of ``FigureSpec`` entries.
    aggregates:
        Registry of ``AggregateSpec`` entries.
    analyses:
        Registry of ``AnalysisSpec`` entries.
    recipes:
        Map from alias string to resolved ``EvaluationRecipe``.
    capture_profiles:
        Map from profile name to ``CaptureProfileSpec``.
    """

    figures: Any = None
    aggregates: Any = None
    analyses: Any = None
    recipes: Mapping[str, Any] = field(default_factory=dict)
    capture_profiles: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def validate(self) -> tuple[Any, ...]:
        """Run all cross-registry integrity checks.

        Returns a tuple of ``FigureDiagnostic`` — empty means valid.
        """
        from ehc_sn.analysis.compiler import (
            FigureCompilationRequest,
            compile_figure_plan,
        )
        from ehc_sn.evaluation.contracts import ArtifactKind
        from ehc_sn.evaluation.inspection import (
            FigureDiagnostic,
            FigureRejectionCode,
        )
        from ehc_sn.figures.registry import ArtifactInputs, FigureSurface

        diagnostics: list[Any] = []
        FD = FigureDiagnostic
        FRC = FigureRejectionCode
        FS = FigureSurface

        # 1. Every registered figure name is unique (structural, but check).
        if self.figures is not None:
            seen = set()
            for name in self.figures.list():
                if name in seen:
                    diagnostics.append(
                        FD(
                            figure_key=name,
                            code=FRC.UNKNOWN_FIGURE,
                            message=f"Duplicate figure name {name!r}.",
                        )
                    )
                seen.add(name)

            # 2. Every ArtifactInputs figure's deps reference registered specs.
            for name in self.figures.list():
                spec = self.figures.get(name)
                if not isinstance(spec.inputs, ArtifactInputs):
                    continue
                for req in spec.inputs.artifacts:
                    kind = req.key.kind
                    dep_name = req.key.name
                    try:
                        if kind is ArtifactKind.ANALYSIS:
                            self.analyses.get(dep_name)
                        elif kind is ArtifactKind.AGGREGATE:
                            self.aggregates.get(dep_name)
                        else:
                            diagnostics.append(
                                FD(
                                    figure_key=name,
                                    code=FRC.UNSUPPORTED_CONTRACT,
                                    message=(
                                        f"Unsupported artifact kind "
                                        f"{kind.value!r} for dependency {dep_name!r}."
                                    ),
                                )
                            )
                    except KeyError:
                        diagnostics.append(
                            FD(
                                figure_key=name,
                                code=FRC.MISSING_ARTIFACT_PROVIDER,
                                message=(
                                    f"Artifact dependency {dep_name!r} (kind={kind.value!r}) "
                                    f"not found in registry."
                                ),
                            )
                        )

            # 3. Preview/canonical figure pairs both reference a shared
            #    ``render_*`` function (different plot dispatch, same renderer).
            _shared_renderer_pairs: dict[str, tuple[str, str, str]] = {
                "render_mec_grid_metrics": (
                    "mec_grid_metrics",
                    "mec_grid_metrics_preview",
                    "mec_grid_metrics",
                ),
                "render_hpc_place_metrics": (
                    "hpc_place_metrics",
                    "hpc_place_metrics_preview",
                    "hpc_place_metrics",
                ),
            }
            import ehc_sn.figures.templates.diagnostics.hpc_place_metrics as _hpc_mod
            import ehc_sn.figures.templates.diagnostics.mec_grid_metrics as _mec_mod

            for renderer_name, (
                canonical_name,
                preview_name,
                _,
            ) in _shared_renderer_pairs.items():
                try:
                    canonical = self.figures.get(canonical_name)
                    preview = self.figures.get(preview_name)
                    # Verify they reference different plot functions
                    if canonical.plot is preview.plot:
                        diagnostics.append(
                            FD(
                                figure_key=preview_name,
                                code=FRC.UNSUPPORTED_CONTRACT,
                                message=(
                                    f"Preview {preview_name!r} and canonical "
                                    f"{canonical_name!r} should use different "
                                    f"plot dispatch functions after Phase 3c split."
                                ),
                            )
                        )
                except KeyError:
                    pass

        # 4. Every recipe's inspection figures compile against resolved
        #    model capabilities and capture profile.
        for alias_str, recipe in self.recipes.items():
            figure_names = recipe.config.inspection.figures
            if not figure_names:
                continue

            try:
                # Resolve capabilities.
                paradigm = recipe.config.capture.profile or "tem"
                model_family = (
                    recipe.config.model_family.value
                    if hasattr(recipe.config.model_family, "value")
                    else str(recipe.config.model_family)
                )

                from ehc_sn.traces.capabilities import (
                    resolve_capture_spec,
                )

                capture_profile_obj = (
                    self.capture_profiles.get(recipe.config.capture.profile)
                    if recipe.config.capture.profile
                    else None
                )

                available: frozenset[str] = frozenset()
                if capture_profile_obj is not None:
                    resolved = resolve_capture_spec(
                        capture_profile_obj,
                        paradigm=paradigm,
                        model_family=model_family,
                    )
                    available = resolved.fields
                else:
                    # No capture profile — no field-level validation possible.
                    pass

                plan = compile_figure_plan(
                    FigureCompilationRequest(
                        names=tuple(figure_names),
                        surface=FS.INSPECTION,
                        task=recipe.config.task,
                        explicit=True,
                    ),
                    figure_registry=self.figures,
                    capture_profile=capture_profile_obj,
                    available_fields=available,
                )
                diagnostics.extend(plan.diagnostics)
            except Exception as exc:
                diagnostics.append(
                    FD(
                        figure_key=alias_str,
                        code=FRC.UNSUPPORTED_CONTRACT,
                        message=(
                            f"Recipe {alias_str!r} inspection plan compilation "
                            f"raised: {exc}."
                        ),
                    )
                )

        return tuple(diagnostics)

    @classmethod
    def from_module_state(cls) -> "EvaluationDefinitions":
        """Construct definitions from the current module-level registry singletons."""
        return _initialize()


# Module-level singleton — lazy initialized on first access.
def get_definitions() -> EvaluationDefinitions:
    """Return the module-level ``DEFINITIONS`` singleton, building it on first call."""
    return _initialize()


DEFINITIONS = _initialize()
"""Lazy module-level singleton.  Built on import; safe to import multiple times."""
