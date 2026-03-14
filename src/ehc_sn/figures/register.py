"""Registration helpers for built-in figures."""

from __future__ import annotations

from ehc_sn.figures.modules import (
    dummy,
    evolution,
    hpc_cells,
    hpc_summary,
    lec_pipeline,
    lec_summary,
    mec_cells,
    mec_summary,
    overlay,
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
                kind="dev",
                tags={"episode"},
                trace_keys=set(),
                meta_keys=set(),
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
                kind="report",
                tags={"paper", "mazehard"},
                trace_keys={
                    "act/halted",
                    "pred/solution_overlay",
                },
                meta_keys={"inputs", "labels"},
            )
        )

    if not REGISTRY.has("evolution"):
        REGISTRY.register(
            FigureSpec(
                name="evolution",
                description=("MazeHard prediction evolution: GT + per-step argmax overlays for one sample"),
                plot=evolution.plot,
                default_filename="evolution",
                kind="diagnostic",
                tags={"mazehard"},
                trace_keys={
                    "act/halted",
                    "pred/solution_overlay",
                },
                meta_keys={"inputs", "labels"},
            )
        )

    if not REGISTRY.has("lec_summary"):
        REGISTRY.register(
            FigureSpec(
                name="lec_summary",
                description="LEC overview with observations and per-frequency activations",
                plot=lec_summary.plot,
                default_filename="lec-overview",
                kind="diagnostic",
                tags={"lec"},
                trace_keys={
                    "world_step/observation",
                    "diagnostic/lec/cells",
                },
                meta_keys={"lec/filter/alpha_sigmoid", "lec/w_f_sigmoid"},
            )
        )

    if not REGISTRY.has("mec_summary"):
        REGISTRY.register(
            FigureSpec(
                name="mec_summary",
                description="Multi-frequency MEC overview",
                plot=mec_summary.plot,
                default_filename="mec-overview",
                kind="diagnostic",
                tags={"mec"},
                trace_keys={
                    "world_step/location_ids",
                    "diagnostic/mec/location_mean",
                },
                meta_keys={"environments"},
            )
        )

    if not REGISTRY.has("hpc_summary"):
        REGISTRY.register(
            FigureSpec(
                name="hpc_summary",
                description="Multi-frequency HPC overview with memory panels",
                plot=hpc_summary.plot,
                default_filename="hpc-overview",
                kind="diagnostic",
                tags={"hpc"},
                trace_keys={
                    "world_step/location_ids",
                    "diagnostic/hpc/location_mean",
                    "diagnostic/hpc/memory",
                },
                meta_keys={"environments"},
            )
        )

    if not REGISTRY.has("lec_pipeline"):
        REGISTRY.register(
            FigureSpec(
                name="lec_pipeline",
                description="Single-frequency LEC activation detail diagnostic",
                plot=lec_pipeline.plot,
                default_filename="lec-feature-cells",
                kind="diagnostic",
                tags={"lec"},
                trace_keys={
                    "world_step/observation",
                    "diagnostic/lec/cells",
                },
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("mec_cells"):
        REGISTRY.register(
            FigureSpec(
                name="mec_cells",
                description="Single-frequency MEC grid-cell spatial maps and autocorrelograms",
                plot=mec_cells.plot,
                default_filename="mec-grid-cells",
                kind="diagnostic",
                tags={"mec"},
                trace_keys={
                    "world_step/location_ids",
                    "diagnostic/mec/location_mean",
                },
                meta_keys={"environments"},
            )
        )

    if not REGISTRY.has("hpc_cells"):
        REGISTRY.register(
            FigureSpec(
                name="hpc_cells",
                description="Single-frequency HPC place-cell spatial maps and autocorrelograms",
                plot=hpc_cells.plot,
                default_filename="hpc-place-cells",
                kind="diagnostic",
                tags={"hpc"},
                trace_keys={
                    "world_step/location_ids",
                    "diagnostic/hpc/location_mean",
                },
                meta_keys={"environments"},
            )
        )
