"""Registry for figure specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
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

if TYPE_CHECKING:
    from ehc_sn.evaluation.contracts import ArtifactRequirement

_ShapeConstraint: TypeAlias = dict[str, object]
"""A dictionary of optional shape constraints for one trace key.

Supported keys:
    ndim (int): Expected number of dimensions.
    last_dim (int): Expected size of the last dimension.
"""

_TemporalConstraint: TypeAlias = dict[str, object]
"""A dictionary of required temporal-semantics field values."""

# ── Figure category / role / source taxonomy ────────────────────────────────

FigureCategory: TypeAlias = Literal["task", "evaluation", "diagnostic"]
"""Scientific question category for a figure.

``"task"``
    Shows the task contract — input, target, environment.
    Consumes a corpus sample.  Used by data-gen inspect.
``"evaluation"``
    Shows what the model predicted relative to the task contract.
    Consumes an evaluation artifact.  Used by evaluation inspect.
``"diagnostic"``
    Shows internal model behavior or mechanism.
    Consumes traces, diagnostic artifacts, or an active model run.
"""

FigureRole: TypeAlias = Literal[
    "task_overview",
    "prediction_reasoning",
    "prediction_overlay",
    "residuals_over_steps",
    "hpc_place_metrics",
    "hpc_rate_map_mosaic",
    "mec_grid_metrics",
    "mec_autocorr_mosaic",
    "lec_content_filtering",
    "lec_content_structure_rsa",
    "pfc_latent_dynamics",
    "pfc_path_memory_probe",
    "cell_activity",
]
"""Canonical figure role — the visual family a figure implements.

Distinct from the concrete implementation (``registry key``), which
combines role + task.  Multiple figures can share the same role for
different tasks (e.g. ``prediction_reasoning`` for mazehard vs
goaltrace).
"""

FigureSourceKind: TypeAlias = Literal[
    "task_sample",
    "evaluation_sample",
    "evaluation_run",
    "diagnostic_result",
]
"""Kind of source data a figure consumes.

``"task_sample"``
    A single corpus sample (dataset batch row).  No model involved.
``"evaluation_sample"``
    A single evaluated sample from an evaluation artifact trace.
``"evaluation_run"``
    Aggregate metrics or summary data for a complete evaluation run.
``"diagnostic_result"``
    Output from a separate diagnostic analysis.
"""


FigureMaturity: TypeAlias = Literal["experimental", "stable", "deprecated"]
"""Implementation stability of a figure.

``"experimental"``
    Layout, interpretation, or semantics may change.
``"stable"``
    Schema, visual semantics, and interpretation are stable.
``"deprecated"``
    Kept for compatibility; not recommended for new report specs.
"""


class FigureSurface(StrEnum):
    """Permitted rendering surfaces for a figure.

    ``INSPECTION``
        May be rendered manually for debugging or model inspection.
    ``REPORT``
        May be rendered from evaluation artifacts via ``ehp_sn.figures.render``
        and composed into notebook reports via ``ehp_sn.reporting`` loaders.
    ``TRAINING``
        May be emitted during training callbacks or TensorBoard previews.
    """

    INSPECTION = "inspection"
    REPORT = "report"
    TRAINING = "training"


# ── Figure input variants (tagged union) ──────────────────────────────────


@dataclass(frozen=True)
class TaskDataInputs:
    """Inputs for a TASK_DATA figure: corpus meta-keys only."""

    meta_keys: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class TraceInputs:
    """Inputs for a BOUNDED_TRACE figure: trace keys and optional meta keys."""

    trace_keys: frozenset[str] = field(default_factory=frozenset)
    meta_keys: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ArtifactInputs:
    """Inputs for an ARTIFACT figure: typed analysis/aggregate artifact refs."""

    artifacts: frozenset[ArtifactRequirement] = field(default_factory=frozenset)


FigureInputs = TaskDataInputs | TraceInputs | ArtifactInputs
"""Tagged union of all supported figure input contracts."""


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

    # evaluation-artifact root for probe-backed figures
    artifact_path: Path | None = None
    """Path to the evaluation artifact root directory.

    Probe-backed figures (e.g. ``pfc_path_memory_probe``) use this to
    resolve compact probe artifacts (``probes/*.npz``) relative to the
    evaluation artifact directory.  Set by report loaders or the artifact
    inspection pipeline; ``None`` in diagnostic/training surfaces.
    """
    artifact_data: dict[str, object] | None = None
    """Pre-loaded artifact data for ARTIFACT-contract figures.

    Populated by the ``render`` subcommand or report pipeline.  The dict
    is keyed by ``ArtifactRequirement.key.manifest_key()`` and contains
    loaded NumPy arrays or data dicts.
    """


# _validate_figure_spec is intentionally removed.  The tagged union
# FigureInputs (TaskDataInputs | TraceInputs | ArtifactInputs) makes
# contradictory field combinations structurally unrepresentable.


@dataclass(frozen=True)
class FigureSpec:
    """Specification for a named figure."""

    name: str
    plot: Callable[[TraceTree, FigureContext], mpl_figure.Figure]
    default_filename: str

    # ── Semantic taxonomy ───────────────────────────────────────────────
    category: FigureCategory
    role: FigureRole
    source_kind: FigureSourceKind
    task: str | None = None
    """Task name when figure is task-specific (``None`` = any task)."""
    model_family: str | None = None
    """Model-family identifier when figure is model-specific (``None`` = any)."""

    maturity: FigureMaturity = "experimental"
    allowed_surfaces: set[FigureSurface] = field(
        default_factory=lambda: {FigureSurface.INSPECTION}
    )
    inputs: FigureInputs = field(default_factory=TaskDataInputs)
    tags: set[str] = field(default_factory=set)
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
    version: int = 1
    """Schema version for this figure's dependency declaration and renderer.

    Bump when the figure's ``inputs``, ``trace_keys``, ``meta_keys``, or
    renderer materially change.  Used in the plan digest for reuse matching.
    """

    @property
    def contract_kind(self) -> str:
        """Return a short string identifying the input-contract variant.

        One of ``"task_data"``, ``"trace"``, or ``"artifact"``.
        """
        match self.inputs:
            case TaskDataInputs():
                return "task_data"
            case TraceInputs():
                return "trace"
            case ArtifactInputs():
                return "artifact"


# ── FigureSelection ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FigureSelection:
    """Result of selecting figures by surface or other criteria.

    Attributes
    ----------
    specs:
        Compatible figures.
    excluded:
        Diagnostics for figures that were considered but excluded.
    """

    specs: tuple[FigureSpec, ...] = ()
    excluded: tuple[dict[str, object], ...] = ()

    @classmethod
    def for_surface(
        cls,
        surface: FigureSurface,
        *,
        registry: "Registry",
    ) -> "FigureSelection":
        """Return all registered figures whose ``allowed_surfaces`` contain *surface*."""
        selected: list[FigureSpec] = []
        for name in sorted(registry._specs):
            spec = registry._specs[name]
            if surface in spec.allowed_surfaces:
                selected.append(spec)
        return cls(specs=tuple(selected))

    def exclude_unsupported(
        self,
        *,
        surface: FigureSurface,
    ) -> "FigureSelection":
        """Return a new selection filtered to figures whose inputs
        are compatible with the given *surface*.

        For ``INSPECTION``: excludes ``ArtifactInputs`` figures.
        For ``REPORT`` and ``TRAINING``: any input contract is acceptable.
        """
        compatible: list[FigureSpec] = []
        excluded_list: list[dict[str, object]] = []
        for spec in self.specs:
            if surface is FigureSurface.INSPECTION and isinstance(
                spec.inputs, ArtifactInputs
            ):
                excluded_list.append(
                    {
                        "figure_key": spec.name,
                        "code": "unsupported_contract",
                        "message": (
                            f"Figure {spec.name!r} uses ArtifactInputs, "
                            f"which is not supported on INSPECTION surface."
                        ),
                    }
                )
            else:
                compatible.append(spec)
        return FigureSelection(
            specs=tuple(compatible),
            excluded=tuple(excluded_list),
        )


class Registry:
    """Registry of figure specifications."""

    def __init__(self) -> None:
        self._specs: dict[str, FigureSpec] = {}

    def register(self, spec: FigureSpec) -> None:
        """Register a figure specification.

        Args:
            spec: Figure specification.

        Raises:
            ValueError: If the name is already registered or the spec's
                input-contract validation fails.
            TypeError: If ``category`` is not a ``FigureCategory`` or
                ``input_contract`` is not a ``FigureInputContract``.
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
        input_contract: str | None = None,
        tags: set[str] | None = None,
    ) -> list[FigureSpec]:
        """Return sorted list of figure specs, optionally filtered.

        Args:
            maturity: Optional maturity filter.
            surface: Optional allowed-surface filter.  When set, only specs
                whose ``allowed_surfaces`` contain *surface* are returned.
            input_contract: Optional input contract filter (``"trace"``,
                ``"task_data"``, ``"artifact"``).
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
            specs = [s for s in specs if s.contract_kind == input_contract]
        if tags is not None:
            specs = [s for s in specs if tags.issubset(s.tags)]
        return sorted(specs, key=lambda s: s.name)

    # ── New category/role/task filtering ─────────────────────────────────

    def list_by_category(
        self,
        category: FigureCategory,
        *,
        role: FigureRole | None = None,
        task: str | None = None,
        source_kind: FigureSourceKind | None = None,
        maturity: FigureMaturity | None = None,
    ) -> list[FigureSpec]:
        """Return specs matching *category* and optional filters.

        Parameters
        ----------
        category:
            Required category.
        role:
            If set, only specs with this role.
        task:
            If set, only specs whose ``task`` matches (``None``-task specs
            match any task).
        source_kind:
            If set, only specs with this source kind.
        maturity:
            If set, only specs with this maturity.

        Returns
        -------
        List of ``FigureSpec`` objects sorted by name.
        """
        specs: list[FigureSpec] = []
        for s in self._specs.values():
            if s.category != category:
                continue
            if role is not None and s.role != role:
                continue
            if task is not None and s.task is not None and s.task != task:
                continue
            if source_kind is not None and s.source_kind != source_kind:
                continue
            if maturity is not None and s.maturity != maturity:
                continue
            specs.append(s)
        return sorted(specs, key=lambda s: s.name)

    def resolve(
        self,
        *,
        category: FigureCategory,
        role: FigureRole,
        task: str,
    ) -> FigureSpec:
        """Return exactly one spec matching *category*, *role*, and *task*.

        When multiple specs match (e.g. a deprecated figure alongside a
        current figure with the same role), the non-deprecated spec is
        preferred.  Raises ``KeyError`` if no spec matches.

        Raises
        ------
        KeyError
            If no spec matches the combination.
        """
        candidates: list[FigureSpec] = []
        for s in self._specs.values():
            if s.category == category and s.role == role and s.task == task:
                candidates.append(s)
        if not candidates:
            raise KeyError(
                f"No figure for category={category!r}, role={role!r}, "
                f"task={task!r}. "
                f"Available: {sorted(self._specs)}."
            )
        # Prefer non-deprecated over deprecated when multiple match.
        non_deprecated = [c for c in candidates if c.maturity != "deprecated"]
        return non_deprecated[0] if non_deprecated else candidates[0]

    def available_roles(
        self,
        category: FigureCategory,
        *,
        task: str | None = None,
    ) -> set[str]:
        """Return the set of roles for a given *category* and optional *task*.

        Parameters
        ----------
        category:
            Required category.
        task:
            If set, only roles for specs whose ``task`` matches
            (``None``-task specs are always included).

        Returns
        -------
        Set of role strings.
        """
        roles: set[str] = set()
        for s in self._specs.values():
            if s.category != category:
                continue
            if task is not None and s.task is not None and s.task != task:
                continue
            roles.add(s.role)
        return roles

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

        def _collect_meta(s: FigureSpec) -> set[str]:
            match s.inputs:
                case TaskDataInputs(meta_keys=mk) | TraceInputs(meta_keys=mk):
                    return mk if isinstance(mk, set) else set(mk)
                case _:
                    return set()

        return set().union(*(_collect_meta(s) for s in specs))


REGISTRY = Registry()


def _resolve_input_kind(spec: FigureSpec) -> str:
    """Return the input kind literal for display / logging."""
    return spec.contract_kind


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
    # --- Key presence ---
    trace_keys: frozenset[str] = frozenset()
    meta_keys: frozenset[str] = frozenset()
    match spec.inputs:
        case TraceInputs(trace_keys=tk, meta_keys=mk):
            trace_keys = tk
            meta_keys = mk
        case TaskDataInputs(meta_keys=mk):
            meta_keys = mk
        case ArtifactInputs():
            pass  # no trace/meta keys to validate

    if trace_keys:
        missing = [
            p
            for p in sorted(trace_keys)
            if not _trace_has_numeric_path(trace, p)
        ]
        if missing:
            raise ValueError(
                f"Figure '{spec.name}' missing required trace keys: {', '.join(missing)}"
            )
    if meta_keys:
        missing = [p for p in sorted(meta_keys) if not trace.has_meta_path(p)]
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
