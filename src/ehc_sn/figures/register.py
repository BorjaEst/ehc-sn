"""Registration helpers for built-in figures."""

from __future__ import annotations

from ehc_sn.figures.modules import (
    dummy,
    evolution,
    feature_cells,
    grid_cells,
    hpc_overview,
    lec_overview,
    mec_overview,
    overlay,
    place_cells,
)
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

    if not REGISTRY.has("lec_feature_cells"):
        REGISTRY.register(
            FigureSpec(
                name="lec_feature_cells",
                aliases=("feature_cells",),
                description="LEC feature/cell timeseries diagnostic",
                plot=feature_cells.plot,
                default_filename="lec-feature-cells",
                tags={"lec"},
                trace_keys={"world_step/observation", "output/features"},
                extras_keys=set(),
            )
        )

    if not REGISTRY.has("lec_overview"):
        REGISTRY.register(
            FigureSpec(
                name="lec_overview",
                description="LEC overview with parameters and per-frequency activations",
                plot=lec_overview.plot,
                default_filename="lec-overview",
                tags={"lec"},
                trace_keys={"world_step/observation"},
                extras_keys={"lec"},
            )
        )

    if not REGISTRY.has("mec_grid_cells"):
        REGISTRY.register(
            FigureSpec(
                name="mec_grid_cells",
                aliases=("grid_cells",),
                description="Single-frequency MEC grid-cell spatial maps and autocorrelograms",
                plot=grid_cells.plot,
                default_filename="mec-grid-cells",
                tags={"mec"},
                trace_keys={"world_step/location_ids"},
                extras_keys=set(),
            )
        )

    if not REGISTRY.has("mec_overview"):
        REGISTRY.register(
            FigureSpec(
                name="mec_overview",
                description="Multi-frequency MEC overview",
                plot=mec_overview.plot,
                default_filename="mec-overview",
                tags={"mec"},
                trace_keys={"world_step/location_ids"},
                extras_keys=set(),
            )
        )

    if not REGISTRY.has("hpc_place_cells"):
        REGISTRY.register(
            FigureSpec(
                name="hpc_place_cells",
                aliases=("place_cells",),
                description="Single-frequency HPC place-cell spatial maps and autocorrelograms",
                plot=place_cells.plot,
                default_filename="hpc-place-cells",
                tags={"hpc"},
                trace_keys={"world_step/location_ids"},
                extras_keys=set(),
            )
        )

    if not REGISTRY.has("hpc_overview"):
        REGISTRY.register(
            FigureSpec(
                name="hpc_overview",
                description="Multi-frequency HPC overview with memory panels",
                plot=hpc_overview.plot,
                default_filename="hpc-overview",
                tags={"hpc"},
                trace_keys={"world_step/location_ids"},
                extras_keys=set(),
            )
        )
