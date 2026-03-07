"""Registration helpers for built-in figures."""

from __future__ import annotations

from ehc_sn.figures.modules import dummy, evolution, overlay
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

    # Hierarchical overlay figure for MazeHard: GT vs model overlays for N samples.
    if not REGISTRY.has("overlay"):
        REGISTRY.register(
            FigureSpec(
                name="overlay",
                description="MazeHard overlays: N samples with GT vs model paths",
                plot=overlay.plot,
                default_filename="overlay",
                tags={"paper", "mazehard"},
                trace_keys={"act/halted", "pred/solution_overlay"},
                extras_keys={"inputs", "labels"},
            )
        )

    if not REGISTRY.has("evolution"):
        REGISTRY.register(
            FigureSpec(
                name="evolution",
                description=("MazeHard prediction evolution: GT + per-step argmax overlays for one sample"),
                plot=evolution.plot,
                default_filename="evolution",
                tags={"mazehard"},
                trace_keys={"act/halted", "pred/solution_overlay"},
                extras_keys={"inputs", "labels"},
            )
        )
