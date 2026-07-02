"""Registries for aggregate specs, analysis specs, and the combined ``EvaluationRegistries`` facade.

All registries follow the same pattern as ``ehp_sn.figures.registry.Registry``:
dict-backed, ``register()`` raises on duplicate, ``get()`` raises ``KeyError``
on missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ehc_sn.analysis.specs import AggregateSpec, AnalysisSpec, ConsumerPlacement
from ehc_sn.figures.registry import Registry as FigureRegistry


# =============================================================================
class AggregateRegistry:
    """Registry of named, versioned ``AggregateSpec`` entries."""

    def __init__(self) -> None:
        self._entries: dict[str, AggregateSpec] = {}

    def register(self, spec: AggregateSpec) -> None:
        """Register an aggregate spec.

        Raises ``ValueError`` if a spec with the same name is already
        registered.
        """
        if spec.name in self._entries:
            raise ValueError(
                f"Aggregate spec {spec.name!r} is already registered."
            )
        self._entries[spec.name] = spec

    def get(self, name: str) -> AggregateSpec:
        """Return the registered spec for *name*.

        Raises ``KeyError`` if not found.
        """
        if name not in self._entries:
            raise KeyError(f"Unknown aggregate spec {name!r}.")
        return self._entries[name]

    def has(self, name: str) -> bool:
        """Return whether a spec with *name* is registered."""
        return name in self._entries

    def list(self) -> list[str]:
        """Return sorted list of registered aggregate spec names."""
        return sorted(self._entries.keys())


# =============================================================================
class AnalysisRegistry:
    """Registry of named, versioned ``AnalysisSpec`` entries."""

    def __init__(self) -> None:
        self._entries: dict[str, AnalysisSpec] = {}

    def register(self, spec: AnalysisSpec) -> None:
        """Register an analysis spec.

        Raises ``ValueError`` if a spec with the same name is already
        registered.
        """
        if spec.name in self._entries:
            raise ValueError(
                f"Analysis spec {spec.name!r} is already registered."
            )
        self._entries[spec.name] = spec

    def get(self, name: str) -> AnalysisSpec:
        """Return the registered spec for *name*.

        Raises ``KeyError`` if not found.
        """
        if name not in self._entries:
            raise KeyError(f"Unknown analysis spec {name!r}.")
        return self._entries[name]

    def has(self, name: str) -> bool:
        """Return whether a spec with *name* is registered."""
        return name in self._entries

    def list(self) -> list[str]:
        """Return sorted list of registered analysis spec names."""
        return sorted(self._entries.keys())


# =============================================================================
@dataclass(frozen=True)
class EvaluationRegistries:
    """Facade over all registries used by the figure-plan compiler."""

    figures: FigureRegistry
    aggregates: AggregateRegistry = field(default_factory=AggregateRegistry)
    analyses: AnalysisRegistry = field(default_factory=AnalysisRegistry)


# =============================================================================
# Built-in registration — called once at first use
# =============================================================================

_builtin_registered = False


def register_builtin_analysis_specs(registries: EvaluationRegistries) -> None:
    """Register built-in aggregate and analysis specs.

    Idempotent — safe to call multiple times.
    """
    global _builtin_registered
    if _builtin_registered:
        return
    _builtin_registered = True

    # ── Aggregate specs ─────────────────────────────────────────────────
    # Placeholder — these will raise NotImplementedError until Phase 2.

    from ehc_sn.contracts.dependencies import model_view
    from ehc_sn.evaluation.accumulators import SpatialPopulationAccumulator
    from ehc_sn.evaluation.contracts import (
        AggregateBuildContext,
        ArtifactKey,
        ArtifactKind,
    )
    from ehc_sn.types import ScaleMetadata as _ScaleMetadata

    def build_mec_spatial_population(
        ctx: AggregateBuildContext,
    ) -> SpatialPopulationAccumulator:
        return SpatialPopulationAccumulator(
            name="mec_spatial_population",
            population_view="diagnostic/mec/location_mean",
            # Placeholder values — lifecycle-based spatial metadata binding
            # (SpatialGeometry, ScaleMetadata) is deferred to a follow-up.
            n_locations=1,
            environment_id="",
            geometry=None,
            scale_metadata={
                "placeholder": _ScaleMetadata(
                    band_name="placeholder",
                    feature_dim=1,
                    index=0,
                )
            },
        )

    def build_hpc_spatial_population(
        ctx: AggregateBuildContext,
    ) -> SpatialPopulationAccumulator:
        return SpatialPopulationAccumulator(
            name="hpc_spatial_population",
            population_view="diagnostic/hpc/location_mean",
            n_locations=1,
            environment_id="",
            geometry=None,
            scale_metadata={
                "placeholder": _ScaleMetadata(
                    band_name="placeholder",
                    feature_dim=1,
                    index=0,
                )
            },
        )

    registries.aggregates.register(
        AggregateSpec(
            name="mec_spatial_population",
            schema_version=1,
            factory=build_mec_spatial_population,
            dependencies=frozenset(
                {
                    model_view("diagnostic/mec/location_mean"),
                }
            ),
            placement=ConsumerPlacement.MODEL_DEVICE,
        )
    )

    registries.aggregates.register(
        AggregateSpec(
            name="hpc_spatial_population",
            schema_version=1,
            factory=build_hpc_spatial_population,
            dependencies=frozenset(
                {
                    model_view("diagnostic/hpc/location_mean"),
                }
            ),
            placement=ConsumerPlacement.MODEL_DEVICE,
        )
    )

    # ── Analysis specs ──────────────────────────────────────────────────
    # MEC grid analysis from spatial population aggregates.

    from ehc_sn.analysis.runners import (
        compute_hpc_place_analysis,
        compute_mec_grid_analysis,
    )
    from ehc_sn.evaluation.contracts import ArtifactRequirement

    registries.analyses.register(
        AnalysisSpec(
            name="mec_grid",
            schema_version=1,
            inputs=frozenset(
                {
                    ArtifactRequirement(
                        key=ArtifactKey(
                            ArtifactKind.AGGREGATE, "mec_spatial_population"
                        ),
                        schema_version=1,
                    ),
                }
            ),
            runner=compute_mec_grid_analysis,
            produces=frozenset(
                {ArtifactKey(ArtifactKind.ANALYSIS, "mec_grid")}
            ),
        )
    )

    registries.analyses.register(
        AnalysisSpec(
            name="hpc_place",
            schema_version=1,
            inputs=frozenset(
                {
                    ArtifactRequirement(
                        key=ArtifactKey(
                            ArtifactKind.AGGREGATE, "hpc_spatial_population"
                        ),
                        schema_version=1,
                    ),
                }
            ),
            runner=compute_hpc_place_analysis,
            produces=frozenset(
                {ArtifactKey(ArtifactKind.ANALYSIS, "hpc_place")}
            ),
        )
    )


def _raise_not_implemented(name: str) -> None:
    raise NotImplementedError(
        f"{name} is not implemented yet. "
        f"This is a placeholder spec registered during Phase 1."
    )


# Re-export for convenience
from ehc_sn.analysis.specs import (  # noqa: E402, F811
    AggregateSpec,
    AnalysisSpec,
    ConsumerPlacement,
)
