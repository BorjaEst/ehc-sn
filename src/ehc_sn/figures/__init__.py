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
    FigureSpec,
    _validate_figure_requirements,
)
from ehc_sn.traces.trace_tree import TraceTree

_registered = False


# =============================================================================
def _ensure_registered() -> None:
    """Register built-in figures on first use of the root API."""
    global _registered
    if not _registered:
        register_builtin_figures()
        _registered = True


# =============================================================================
def list_figures(
    *,
    surface: str | None = None,
    input_contract: str | None = None,
) -> list[str]:
    """Return registered figure names, optionally filtered.

    Args:
        surface: Optional allowed-surface filter (e.g. ``"report"``).
        input_contract: Optional input-contract filter (e.g. ``"trace"``).

    Returns:
        Sorted list of matching figure names.
    """
    _ensure_registered()
    return [
        spec.name
        for spec in REGISTRY.list_specs(
            surface=surface,  # type: ignore[arg-type]
            input_contract=input_contract,
        )
    ]


# =============================================================================
def list_figure_specs(
    *,
    maturity: str | None = None,
    surface: str | None = None,
    input_contract: str | None = None,
) -> list[FigureSpec]:
    """Return registered figure specs, optionally filtered.

    Args:
        maturity: Optional maturity filter (e.g. ``"stable"``).
        surface: Optional allowed-surface filter (e.g. ``"report"``).
        input_contract: Optional input contract filter (e.g. ``"trace"``).

    Returns:
        Sorted list of matching ``FigureSpec`` objects.
    """
    _ensure_registered()
    return REGISTRY.list_specs(
        maturity=maturity,  # type: ignore[arg-type]
        surface=surface,  # type: ignore[arg-type]
        input_contract=input_contract,
    )


# =============================================================================
def render(
    name: str,
    trace: TraceTree | None = None,
    ctx: FigureContext | None = None,
    *,
    temporal_semantics: dict[str, object] | None = None,
) -> mpl_figure.Figure:
    """Look up a registered figure by name and render it.

    For ``BOUNDED_TRACE`` figures, *trace* must be a valid ``TraceTree``.
    For ``ARTIFACT`` figures, data is loaded from ``ctx.artifact_data``.
    Built-in figures are auto-registered on first call.
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

    # Skip trace validation for ARTIFACT figures — they consume artifact data.
    from ehc_sn.figures.registry import ArtifactInputs

    if not isinstance(spec.inputs, ArtifactInputs) and trace is not None:
        _validate_figure_requirements(
            trace, spec, temporal_semantics=temporal_semantics
        )
    elif isinstance(spec.inputs, ArtifactInputs) and ctx.artifact_data is None:
        raise ValueError(
            f"Figure '{name}' requires ARTIFACT input but "
            f"ctx.artifact_data is None."
        )

    return spec.plot(trace, ctx)


# =============================================================================
__all__ = [
    "FigureContext",
    "FigureSpec",
    "REGISTRY",
    "list_figure_specs",
    "list_figures",
    "register_builtin_figures",
    "render",
]
