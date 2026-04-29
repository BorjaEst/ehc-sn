"""Registration helpers for built-in figures."""

from __future__ import annotations

from ehc_sn.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register built-in figure specifications (lazy template imports)."""
    # Templates are imported inside this function so that importing
    # ``ehc_sn.figures`` does not eagerly pull in all template modules.
    from ehc_sn.figures.templates import dummy, evolution, hpc_cells, hpc_summary  # noqa: PLC0415
    from ehc_sn.figures.templates import lec_pipeline, lec_summary, mec_cells, mec_summary, overlay  # noqa: PLC0415

    # Use path constants from selector modules to avoid duplicating literals.
    from ehc_sn.figures.selectors.mazehard import (  # noqa: PLC0415
        META_KEY_GT_OVERLAY,
        META_KEY_INPUT_IDS,
        TRACE_KEY_HALTED,
        TRACE_KEY_PRED_OVERLAY,
    )
    from ehc_sn.figures.selectors.lec import (  # noqa: PLC0415
        META_KEY_LEC_ALPHA,
        META_KEY_LEC_WF,
        TRACE_KEY_LEC_CELLS,
        TRACE_KEY_LEC_FILTERED,
        TRACE_KEY_OBSERVATION,
    )
    from ehc_sn.figures.selectors.mec import (  # noqa: PLC0415
        META_KEY_ENVIRONMENTS as MEC_META_ENVIRONMENTS,
        TRACE_KEY_LOCATION_IDS as MEC_TRACE_LOCATION_IDS,
        TRACE_KEY_MEC_CELLS,
    )
    from ehc_sn.figures.selectors.hpc import (  # noqa: PLC0415
        META_KEY_ENVIRONMENTS as HPC_META_ENVIRONMENTS,
        TRACE_KEY_LOCATION_IDS as HPC_TRACE_LOCATION_IDS,
        TRACE_KEY_HPC_CELLS,
        TRACE_KEY_HPC_MEMORY,
    )

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

    if not REGISTRY.has("overlay"):
        REGISTRY.register(
            FigureSpec(
                name="overlay",
                description="MazeHard overlays: N samples with GT vs model paths",
                plot=overlay.plot,
                default_filename="overlay",
                kind="report",
                tags={"paper", "mazehard"},
                trace_keys={TRACE_KEY_HALTED, TRACE_KEY_PRED_OVERLAY},
                meta_keys={META_KEY_INPUT_IDS, META_KEY_GT_OVERLAY},
            )
        )

    if not REGISTRY.has("evolution"):
        REGISTRY.register(
            FigureSpec(
                name="evolution",
                description="MazeHard prediction evolution: GT + per-step argmax overlays for one sample",
                plot=evolution.plot,
                default_filename="evolution",
                kind="diagnostic",
                tags={"mazehard"},
                trace_keys={TRACE_KEY_HALTED, TRACE_KEY_PRED_OVERLAY},
                meta_keys={META_KEY_INPUT_IDS, META_KEY_GT_OVERLAY},
            )
        )

    if not REGISTRY.has("lec_summary"):
        REGISTRY.register(
            FigureSpec(
                name="lec_summary",
                description="LEC overview with observations and per-frequency activations",
                plot=lec_summary.plot,
                default_filename="lec_summary",
                kind="diagnostic",
                tags={"lec"},
                trace_keys={TRACE_KEY_OBSERVATION, TRACE_KEY_LEC_CELLS},
                meta_keys={META_KEY_LEC_ALPHA, META_KEY_LEC_WF},
            )
        )

    if not REGISTRY.has("mec_summary"):
        REGISTRY.register(
            FigureSpec(
                name="mec_summary",
                description="Multi-frequency MEC overview",
                plot=mec_summary.plot,
                default_filename="mec_summary",
                kind="diagnostic",
                tags={"mec"},
                trace_keys={MEC_TRACE_LOCATION_IDS, TRACE_KEY_MEC_CELLS},
                meta_keys={MEC_META_ENVIRONMENTS},
            )
        )

    if not REGISTRY.has("hpc_summary"):
        REGISTRY.register(
            FigureSpec(
                name="hpc_summary",
                description="Multi-frequency HPC overview with memory panels",
                plot=hpc_summary.plot,
                default_filename="hpc_summary",
                kind="diagnostic",
                tags={"hpc"},
                trace_keys={HPC_TRACE_LOCATION_IDS, TRACE_KEY_HPC_CELLS, TRACE_KEY_HPC_MEMORY},
                meta_keys={HPC_META_ENVIRONMENTS},
            )
        )

    if not REGISTRY.has("lec_pipeline"):
        REGISTRY.register(
            FigureSpec(
                name="lec_pipeline",
                description="Single-frequency LEC activation detail diagnostic",
                plot=lec_pipeline.plot,
                default_filename="lec_pipeline",
                kind="diagnostic",
                tags={"lec"},
                trace_keys={TRACE_KEY_OBSERVATION, TRACE_KEY_LEC_CELLS, TRACE_KEY_LEC_FILTERED},
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("mec_cells"):
        REGISTRY.register(
            FigureSpec(
                name="mec_cells",
                description="Single-frequency MEC grid-cell spatial maps and spatial autocorr",
                plot=mec_cells.plot,
                default_filename="mec_cells",
                kind="diagnostic",
                tags={"mec"},
                trace_keys={MEC_TRACE_LOCATION_IDS, TRACE_KEY_MEC_CELLS},
                meta_keys={MEC_META_ENVIRONMENTS},
            )
        )

    if not REGISTRY.has("hpc_cells"):
        REGISTRY.register(
            FigureSpec(
                name="hpc_cells",
                description="Single-frequency HPC place-cell spatial maps and spatial autocorr",
                plot=hpc_cells.plot,
                default_filename="hpc_cells",
                kind="diagnostic",
                tags={"hpc"},
                trace_keys={HPC_TRACE_LOCATION_IDS, TRACE_KEY_HPC_CELLS},
                meta_keys={HPC_META_ENVIRONMENTS},
            )
        )
