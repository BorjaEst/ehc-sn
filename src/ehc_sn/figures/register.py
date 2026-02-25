"""Registration helpers for built-in figures."""

from __future__ import annotations

from ehc_sn.figures.modules import dummy
from ehc_sn.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register built-in figure specifications."""

    # Overall dummy figure
    if not REGISTRY.has("dummy"):
        REGISTRY.register(
            FigureSpec(
                name="dummy",
                description="dummy figure for testing",
                plot=dummy.plot,
                default_filename="dummy",
                tags={"episode"},
                trace_keys=set(),
                extras_keys=set(),
            )
        )
