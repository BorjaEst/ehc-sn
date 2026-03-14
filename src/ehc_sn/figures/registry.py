"""Registry for figure specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, TypeAlias

import matplotlib.figure as mpl_figure
import pub_ready_plots as prp
import scienceplots  # noqa: F401 (registers "science", "nature", ...)

from ehc_sn.rollouts.trace_tree import TraceTree

FigureKind: TypeAlias = Literal["dev", "diagnostic", "report"]
"""Classification for the intended lifecycle and stability of a figure."""


@dataclass(frozen=True)
class FigureContext:
    """Context information passed to figure plot functions."""

    # Indices for selecting data from multi-environment, multi-frequency traces
    env_idx: int = 0
    freq_idx: int = 0

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
