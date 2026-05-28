"""Figures package — public API.

Usage::

    from ehc_sn.figures import render, REGISTRY, FigureContext, FigureSpec, list_figures

The root public API (``list_figures`` and ``render``) auto-registers built-in
figures on first call, so an explicit ``register_builtin_figures()`` call is
not required when going through the root API.
"""

from __future__ import annotations

import matplotlib.figure as mpl_figure

from ehc_sn.figures.register import register_builtin_figures
from ehc_sn.figures.registry import (
    REGISTRY,
    FigureContext,
    FigureInputContract,
    FigureSpec,
    _validate_figure_requirements,
)
from ehc_sn.traces.trace_tree import TraceTree

_registered = False


def _ensure_registered() -> None:
    """Register built-in figures on first use of the root API."""
    global _registered
    if not _registered:
        register_builtin_figures()
        _registered = True


def list_figures(
    *,
    kind: str | None = None,
    input_contract: FigureInputContract | None = None,
) -> list[str]:
    """Return registered figure names, optionally filtered.

    Args:
        kind: Optional figure kind filter.
        input_contract: Optional input-contract filter.

    Returns:
        Sorted list of matching figure names.
    """
    return [
        spec.name
        for spec in list_figure_specs(
            kind=kind,  # type: ignore[arg-type]
            input_contract=input_contract,
        )
    ]


def list_figure_specs(
    *,
    kind: FigureKind | None = None,
    input_contract: FigureInputContract | None = None,
) -> list[FigureSpec]:
    """Return registered figure specs, optionally filtered.

    Args:
        kind: Optional figure kind filter.
        input_contract: Optional input contract filter.

    Returns:
        Sorted list of matching ``FigureSpec`` objects.
    """
    _ensure_registered()
    return REGISTRY.list_specs(kind=kind, input_contract=input_contract)


def render(
    name: str,
    trace: TraceTree,
    ctx: FigureContext | None = None,
    *,
    temporal_semantics: dict[str, object] | None = None,
) -> mpl_figure.Figure:
    """Look up a registered figure by name and render it.

    Validates that the trace satisfies the numeric and metadata key requirements
    declared by the spec before plotting.  Built-in figures are auto-registered
    on first call.

    Args:
        name: Registered figure name.
        trace: Rollout trace (``TraceTree``).
        ctx: Optional figure context; defaults to ``FigureContext()``.
        temporal_semantics: Optional manifest temporal-semantics block.
            Passed through to ``_validate_figure_requirements`` for
            ``required_temporal`` checks when available.

    Returns:
        Matplotlib ``Figure``.
    """
    _ensure_registered()
    if ctx is None:
        ctx = FigureContext()
    spec = REGISTRY.get(name)
    _validate_figure_requirements(
        trace, spec, temporal_semantics=temporal_semantics
    )
    return spec.plot(trace, ctx)


__all__ = [
    "FigureContext",
    "FigureInputContract",
    "FigureSpec",
    "REGISTRY",
    "list_figure_specs",
    "list_figures",
    "register_builtin_figures",
    "render",
]
