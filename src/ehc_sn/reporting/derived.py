"""Derived resource registry — maps logical resource names to builders.

The registry is a static mapping.  Each :class:`DerivedResourceSpec`
declares which tasks and model families the builder supports, what it
produces, and its source resource dependencies.  Builders live in
``tasks/<task>/inspection.py`` and are called from ``preparation.py``
via this registry.

Usage::

    from ehc_sn.reporting.derived import DERIVED_RESOURCE_SPECS

    spec = DERIVED_RESOURCE_SPECS["pathway_metrics"]
    df = spec.builder(regime_artifact_set, request)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from ehc_sn.evaluation.artifact_models import RegimeArtifactSet
from ehc_sn.reporting.request import ReportDataRequest


# =============================================================================
@dataclass(frozen=True)
class DerivedResourceSpec:
    """Registration for one derived resource builder.

    Attributes:
        name: Logical resource name used in ``report.toml``.
        tasks: Task families supported by this builder.
        model_families: Model families supported by this builder.
        format: Output format (``"csv"``, ``"parquet"``, ``"json"``).
        relative_path: Path relative to the package root.
        builder: Callable that accepts a regime artifact set and request,
            returns a DataFrame.
        version: Builder schema version.
        source_resources: Names of source resources this builder depends on.
    """

    name: str
    tasks: frozenset[str]
    model_families: frozenset[str]
    format: str
    relative_path: str
    builder: Callable[[RegimeArtifactSet, ReportDataRequest], pd.DataFrame]
    version: int = 1
    source_resources: tuple[str, ...] = field(default_factory=tuple)


# =============================================================================
# Builders — thin wrappers around task inspection logic
# =============================================================================


def _build_arena_pathway_metrics(
    regime: RegimeArtifactSet,
    request: ReportDataRequest,  # noqa: ARG001
) -> pd.DataFrame:
    """Build pathway metrics from an Arena TEM regime artifact.

    Delegates to :func:`ehp_sn.tasks.arena.inspection.build_pathway_metrics_dataframe`.
    """
    from ehc_sn.tasks.arena.inspection import (
        build_pathway_metrics_dataframe,
    )

    return build_pathway_metrics_dataframe(dict(regime.metrics))


# =============================================================================
# Registry — single authoritative mapping
# =============================================================================

DERIVED_RESOURCE_SPECS: Mapping[str, DerivedResourceSpec] = {
    "pathway_metrics": DerivedResourceSpec(
        name="pathway_metrics",
        tasks=frozenset({"arena"}),
        model_families=frozenset({"tem-v1", "tem-v2"}),
        format="csv",
        relative_path="derived/pathway-metrics.csv",
        builder=_build_arena_pathway_metrics,
        version=1,
        source_resources=("metrics",),
    ),
}
