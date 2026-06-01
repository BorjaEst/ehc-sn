"""Registry for figure specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
)

import matplotlib.figure as mpl_figure
import pub_ready_plots as prp
import scienceplots  # noqa: F401 (registers "science", "nature", ...)

from ehc_sn.traces.trace_tree import TraceTree

_ShapeConstraint: TypeAlias = dict[str, object]
"""A dictionary of optional shape constraints for one trace key.

Supported keys:
    ndim (int): Expected number of dimensions.
    last_dim (int): Expected size of the last dimension.
"""

_TemporalConstraint: TypeAlias = dict[str, object]
"""A dictionary of required temporal-semantics field values."""

FigureMaturity: TypeAlias = Literal["experimental", "stable", "deprecated"]
"""Implementation stability of a figure.

``"experimental"``
    Layout, interpretation, or semantics may change.
``"stable"``
    Schema, visual semantics, and interpretation are stable.
``"deprecated"``
    Kept for compatibility; not recommended for new report specs.
"""

FigureSurface: TypeAlias = Literal["training", "diagnostic", "report"]
"""Permitted rendering surfaces for a figure.

``"training"``
    May be emitted during training callbacks or TensorBoard previews.
``"diagnostic"``
    May be rendered manually for debugging or model inspection.
``"report"``
    May be rendered by ``ehc_sn.reporting`` into a ``ReportRun``.
"""

FigureInputContract: TypeAlias = Literal[
    "bounded_trace",
    "evaluation_artifact",
    "offline_artifact",
]
"""Minimum trace fidelity required for a figure to produce correct output.

Values
    bounded_trace
        May be truncated (limited batches), partial metadata.
        Safe for ``FigureGenerationCallback`` (diagnostic capture path).
    evaluation_artifact
        Requires a complete trajectory with all requested trace/metadata
        keys present.  Produced by evaluator regime runs.
    offline_artifact
        Requires a trace loaded from persisted disk artifacts
        (``load_artifact_run_cases``).  May depend on
        artifact-format guarantees (numpy vs torch, deserialization).
"""


@dataclass(frozen=True)
class FigureContext:
    """Context information passed to figure plot functions."""

    # Indices for selecting data from multi-environment, multi-frequency traces
    env_idx: int = 0
    freq_idx: int = 0

    # Render-time sample selection
    sample_idx: int = 0
    """Index of the batch sample to use as the primary displayed item."""
    max_items: int | None = None
    """Maximum number of items shown per multi-sample panel.  ``None`` defers
    to each template's own default cap."""
    max_cells: int | None = None
    """Maximum number of cells shown per cell-level figure.  ``None`` defers
    to each selector's own default cap."""

    # Optional figure customization parameters
    styles: Sequence[str] = field(default_factory=lambda: ["science"])
    layout: Optional[prp.Layout] = None
    dpi: Optional[int] = None

    # Optional training context
    global_step: Optional[int] = None
    split_name: Optional[str] = None


@dataclass(frozen=True)
class FigureSpec:
    """Specification for a named figure."""

    name: str
    plot: Callable[[TraceTree, FigureContext], mpl_figure.Figure]
    default_filename: str
    maturity: FigureMaturity = "experimental"
    allowed_surfaces: set[FigureSurface] = field(
        default_factory=lambda: {"diagnostic"}
    )
    input_contract: FigureInputContract = "evaluation_artifact"
    tags: set[str] = field(default_factory=set)
    trace_keys: set[str] = field(default_factory=set)
    meta_keys: set[str] = field(default_factory=set)
    required_numeric: dict[str, _ShapeConstraint] | None = None
    """Per-key shape constraints checked in addition to key presence.

    Each entry maps a trace key to an optional dict of constraints.
    Supported constraint keys: ``ndim`` (int), ``last_dim`` (int).
    ``None`` or empty dict means no shape validation beyond key presence.
    """
    required_temporal: _TemporalConstraint | None = None
    """Required temporal-semantics field values.

    Each entry maps a ``temporal_semantics`` field name to its required
    value.  Validation only fires when the renderer has access to temporal
    metadata (e.g. from a ``LoadedArtifactCase``).  ``None`` means no
    temporal validation.
    """
    description: str = ""


class Registry:
    """Registry of figure specifications."""

    def __init__(self) -> None:
        self._specs: dict[str, FigureSpec] = {}

    def register(self, spec: FigureSpec) -> None:
        """Register a figure specification.

        Args:
            spec: Figure specification.

        Raises:
            ValueError: If the name is already registered.
        """
        if spec.name in self._specs:
            raise ValueError(f"Figure '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def has(self, name: str) -> bool:
        """Return whether a figure name is registered."""
        return name in self._specs

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name in self._specs

    def get(self, name: str) -> FigureSpec:
        """Return a registered figure spec."""
        if name not in self._specs:
            raise KeyError(f"Unknown figure '{name}'")
        return self._specs[name]

    def validate(self, names: Iterable[str]) -> None:
        """Validate that all names are registered."""
        missing = [name for name in names if name not in self._specs]
        if missing:
            raise ValueError(f"Unknown figures: {', '.join(missing)}")

    def list(self, *, surface: FigureSurface | None = None) -> list[str]:
        """Return sorted list of registered figure names.

        Args:
            surface: Optional allowed-surface filter.  When set, only
                figures whose ``allowed_surfaces`` contain *surface*
                are returned.
        """
        if surface is None:
            return sorted(self._specs.keys())
        return sorted(
            name
            for name, spec in self._specs.items()
            if surface in spec.allowed_surfaces
        )

    def list_specs(  # -------------------------------------------------------
        self,
        *,
        maturity: FigureMaturity | None = None,
        surface: FigureSurface | None = None,
        input_contract: FigureInputContract | None = None,
        tags: set[str] | None = None,
    ) -> list[FigureSpec]:
        """Return sorted list of figure specs, optionally filtered.

        Args:
            maturity: Optional maturity filter.
            surface: Optional allowed-surface filter.  When set, only specs
                whose ``allowed_surfaces`` contain *surface* are returned.
            input_contract: Optional input contract filter.
            tags: Optional tag filter.  When not ``None``, only specs whose
                ``tags`` are a superset of this set are returned.  An empty
                set matches all specs (identity filter).

        Returns:
            List of ``FigureSpec`` objects sorted by name.
        """
        specs = list(self._specs.values())
        if maturity is not None:
            specs = [s for s in specs if s.maturity == maturity]
        if surface is not None:
            specs = [s for s in specs if surface in s.allowed_surfaces]
        if input_contract is not None:
            specs = [s for s in specs if s.input_contract == input_contract]
        if tags is not None:
            specs = [s for s in specs if tags.issubset(s.tags)]
        return sorted(specs, key=lambda s: s.name)

    def required_meta_keys(  # ------------------------------------------------
        self,
        *,
        tags: set[str] | None = None,
    ) -> set[str]:
        """Return the union of required ``meta_keys`` across matched figure specs.

        Args:
            tags: Forwarded to :meth:`list_specs` to select which specs to
                include.  ``None`` queries all registered specs.

        Returns:
            Set of meta-key paths that every rendered trace must carry for
            the selected figure set.
        """
        specs = self.list_specs(tags=tags)
        return set().union(*(s.meta_keys for s in specs))


REGISTRY = Registry()


def _trace_has_numeric_path(trace: TraceTree, path: str) -> bool:
    """Return whether a numeric leaf or numeric subtree exists at ``path`` in ``trace``."""
    if not trace.path_to_index:
        return False
    idx = trace.path_to_index.get(path)
    if idx is not None:
        return bool(trace.leaf_is_numeric[idx])
    prefix = f"{path}/"
    for candidate, candidate_idx in trace.path_to_index.items():
        if (
            candidate.startswith(prefix)
            and trace.leaf_is_numeric[candidate_idx]
        ):
            return True
    return False


def _validate_figure_requirements(
    trace: TraceTree,
    spec: FigureSpec,
    *,
    temporal_semantics: dict[str, object] | None = None,
) -> None:
    """Validate that ``trace`` satisfies the requirements declared in ``spec``.

    Args:
        trace: Rollout trace to validate against ``spec`` requirements.
        spec: Figure specification with ``trace_keys``, ``meta_keys``,
            ``required_numeric``, and ``required_temporal`` declarations.
        temporal_semantics: Optional dict from artifact manifest
            ``temporal_semantics``.  Required when ``spec.required_temporal``
            is set; skipped when ``None``.

    Raises:
        ValueError: If any required key is absent, shape constraint is
            violated, or temporal-semantics requirement is not met.
    """
    # --- Key presence (existing) ---
    if spec.trace_keys:
        missing = [
            p
            for p in sorted(spec.trace_keys)
            if not _trace_has_numeric_path(trace, p)
        ]
        if missing:
            raise ValueError(
                f"Figure '{spec.name}' missing required trace keys: {', '.join(missing)}"
            )
    if spec.meta_keys:
        missing = [
            p for p in sorted(spec.meta_keys) if not trace.has_meta_path(p)
        ]
        if missing:
            raise ValueError(
                f"Figure '{spec.name}' missing required metadata keys: {', '.join(missing)}"
            )

    # --- Shape constraints (required_numeric) ---
    if spec.required_numeric:
        dense_leaves = trace.dense_leaves
        for key, constraints in spec.required_numeric.items():
            if not constraints:
                continue
            arr = _resolve_numeric_leaf(trace, key, dense_leaves)
            if arr is None:
                raise ValueError(
                    f"Figure '{spec.name}' required_numeric key "
                    f"'{key}' not found."
                )
            expected_ndim = constraints.get("ndim")
            if expected_ndim is not None and arr.ndim != expected_ndim:
                raise ValueError(
                    f"Figure '{spec.name}' required_numeric key "
                    f"'{key}': expected ndim={expected_ndim}, "
                    f"got ndim={arr.ndim}."
                )
            expected_last_dim = constraints.get("last_dim")
            if (
                expected_last_dim is not None
                and arr.shape[-1] != expected_last_dim
            ):
                raise ValueError(
                    f"Figure '{spec.name}' required_numeric key "
                    f"'{key}': expected last_dim={expected_last_dim}, "
                    f"got shape={arr.shape}."
                )

    # --- Temporal semantics constraints (required_temporal) ---
    if spec.required_temporal:
        if temporal_semantics is None:
            # No manifest context — skip temporal validation.
            return
        for field, expected in spec.required_temporal.items():
            got = temporal_semantics.get(field)
            if got != expected:
                raise ValueError(
                    f"Figure '{spec.name}' requires "
                    f"temporal_semantics.{field}={expected!r}, "
                    f"but artifact has {got!r}."
                )


def _resolve_numeric_leaf(
    trace: TraceTree,
    key: str,
    dense_leaves: list[Any] | None,
) -> Any | None:
    """Resolve a numeric leaf array from a trace by path string."""
    if dense_leaves is None:
        return None
    idx = trace.path_to_index.get(key)
    if idx is not None and idx < len(dense_leaves):
        leaf = dense_leaves[idx]
        if leaf is not None and _has_shape(leaf):
            return leaf
    return None


def _has_shape(value: Any) -> bool:
    """Return whether *value* has a ``.ndim`` and ``.shape`` attribute."""
    return hasattr(value, "ndim") and hasattr(value, "shape")
