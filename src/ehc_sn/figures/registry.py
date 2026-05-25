"""Registry for figure specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, TypeAlias

import matplotlib.figure as mpl_figure
import pub_ready_plots as prp
import scienceplots  # noqa: F401 (registers "science", "nature", ...)

from ehc_sn.traces.trace_tree import TraceTree

FigureKind: TypeAlias = Literal["dev", "diagnostic", "report"]
"""Classification for the intended lifecycle and stability of a figure."""


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
    kind: FigureKind = "diagnostic"
    tags: set[str] = field(default_factory=set)
    trace_keys: set[str] = field(default_factory=set)
    meta_keys: set[str] = field(default_factory=set)
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

    def list(self, *, kind: FigureKind | None = None) -> list[str]:
        """Return sorted list of registered figure names.

        Args:
            kind: Optional figure kind filter.
        """
        if kind is None:
            return sorted(self._specs.keys())
        return sorted(name for name, spec in self._specs.items() if spec.kind == kind)


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
        if candidate.startswith(prefix) and trace.leaf_is_numeric[candidate_idx]:
            return True
    return False


def _validate_figure_requirements(trace: TraceTree, spec: FigureSpec) -> None:
    """Validate that ``trace`` satisfies the key requirements declared in ``spec``.

    Args:
        trace: Rollout trace to validate against ``spec`` requirements.
        spec: Figure specification with ``trace_keys`` and ``meta_keys`` declarations.

    Raises:
        ValueError: If any required numeric or metadata key is absent from the trace.
    """
    if spec.trace_keys:
        missing = [p for p in sorted(spec.trace_keys) if not _trace_has_numeric_path(trace, p)]
        if missing:
            raise ValueError(f"Figure '{spec.name}' missing required trace keys: {', '.join(missing)}")
    if spec.meta_keys:
        missing = [p for p in sorted(spec.meta_keys) if not trace.has_meta_path(p)]
        if missing:
            raise ValueError(f"Figure '{spec.name}' missing required metadata keys: {', '.join(missing)}")
