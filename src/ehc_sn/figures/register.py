"""Registration helpers for built-in figures."""

from __future__ import annotations

from ehc_sn.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register built-in figure specifications (lazy template imports)."""
    # Templates are imported inside this function so that importing
    # ``ehc_sn.figures`` does not eagerly pull in all template modules.
    from ehc_sn.figures.selectors.arena_tem import (  # noqa: PLC0415
        META_KEY_TARGET_OBS_ID as TEM_META_KEY_TARGET_OBS_ID,
    )
    from ehc_sn.figures.selectors.arena_tem import (
        TRACE_KEY_PRED_ANCESTRAL as TEM_TRACE_KEY_PRED_ANCESTRAL,
    )
    from ehc_sn.figures.selectors.arena_tem import (
        TRACE_KEY_PRED_INFERENCE as TEM_TRACE_KEY_PRED_INFERENCE,
    )
    from ehc_sn.figures.selectors.arena_tem import (
        TRACE_KEY_PRED_RETRIEVED as TEM_TRACE_KEY_PRED_RETRIEVED,
    )
    from ehc_sn.figures.selectors.hpc import (  # noqa: PLC0415
        META_KEY_ENVIRONMENTS as HPC_META_ENVIRONMENTS,
    )
    from ehc_sn.figures.selectors.hpc import (
        TRACE_KEY_HPC_CELLS,
        TRACE_KEY_HPC_MEMORY,
    )
    from ehc_sn.figures.selectors.hpc import (
        TRACE_KEY_LOCATION_IDS as HPC_TRACE_LOCATION_IDS,
    )
    from ehc_sn.figures.selectors.lec import (  # noqa: PLC0415
        META_KEY_LEC_ALPHA,
        META_KEY_LEC_WF,
        TRACE_KEY_LEC_CELLS,
        TRACE_KEY_LEC_FILTERED,
        TRACE_KEY_OBSERVATION,
    )

    # Use path constants from selector modules to avoid duplicating literals.
    from ehc_sn.figures.selectors.mazehard import (  # noqa: PLC0415
        META_KEY_GT_OVERLAY,
        META_KEY_INPUT_IDS,
        TRACE_KEY_HALTED,
        TRACE_KEY_PRED_OVERLAY,
    )
    from ehc_sn.figures.selectors.mec import (  # noqa: PLC0415
        META_KEY_ENVIRONMENTS as MEC_META_ENVIRONMENTS,
    )
    from ehc_sn.figures.selectors.mec import (
        TRACE_KEY_LOCATION_IDS as MEC_TRACE_LOCATION_IDS,
    )
    from ehc_sn.figures.selectors.mec import (
        TRACE_KEY_MEC_CELLS,
    )
    from ehc_sn.figures.templates import (  # noqa: PLC0415
        dummy,
        evolution,
        halting,
        hidden_norm,
        hpc_cells,
        hpc_summary,
        lec_pipeline,
        lec_summary,
        mec_cells,
        mec_summary,
        occupancy,
        overlay,
        tem_prediction_overlay,
    )
    from ehc_sn.figures.templates.value_evolution import (  # noqa: PLC0415
        plot as _value_evolution_plot,
    )

    if not REGISTRY.has("dummy"):
        REGISTRY.register(
            FigureSpec(
                name="dummy",
                description="dummy figure for testing",
                plot=dummy.plot,
                default_filename="dummy",
                kind="dev",
                input_contract="bounded_trace",
                tags={"episode"},
                trace_keys={"act/halted"},
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
                input_contract="offline_artifact",
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
                input_contract="evaluation_artifact",
                tags={"mazehard"},
                trace_keys={TRACE_KEY_HALTED, TRACE_KEY_PRED_OVERLAY},
                meta_keys={META_KEY_INPUT_IDS, META_KEY_GT_OVERLAY},
            )
        )

    if not REGISTRY.has("tem_prediction_overlay"):
        REGISTRY.register(
            FigureSpec(
                name="tem_prediction_overlay",
                description=(
                    "TEM observation_id: GT vs predicted "
                    "(inference / retrieved / ancestral) for N samples"
                ),
                plot=tem_prediction_overlay.plot,
                default_filename="tem_prediction_overlay",
                kind="diagnostic",
                input_contract="evaluation_artifact",
                tags={"tem", "arena"},
                trace_keys={
                    TEM_TRACE_KEY_PRED_INFERENCE,
                    TEM_TRACE_KEY_PRED_RETRIEVED,
                    TEM_TRACE_KEY_PRED_ANCESTRAL,
                },
                meta_keys={TEM_META_KEY_TARGET_OBS_ID},
            )
        )

    if not REGISTRY.has("halting_timeline"):
        REGISTRY.register(
            FigureSpec(
                name="halting_timeline",
                description="Binary halting signal heatmap over time",
                plot=halting.plot,
                default_filename="halting_timeline",
                kind="diagnostic",
                input_contract="bounded_trace",
                tags={"hrm", "ehc", "reasoning", "halting"},
                trace_keys={"act/halted"},
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("q_value_evolution"):
        REGISTRY.register(
            FigureSpec(
                name="q_value_evolution",
                description="Q-values over rollout steps (RL/EHC-reason/HRM-v2)",
                plot=_value_evolution_plot,
                default_filename="q_value_evolution",
                kind="diagnostic",
                input_contract="bounded_trace",
                tags={"hrm", "ehc", "rl", "reasoning", "value"},
                trace_keys={"value/q_values"},
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("halt_logit_evolution"):
        REGISTRY.register(
            FigureSpec(
                name="halt_logit_evolution",
                description="Halt/continue logits over rollout steps (ACT/HRM-v1)",
                plot=_value_evolution_plot,
                default_filename="halt_logit_evolution",
                kind="diagnostic",
                input_contract="bounded_trace",
                tags={"hrm", "act", "reasoning", "halting"},
                trace_keys={"value/q_logits"},
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("occupancy_histogram"):
        REGISTRY.register(
            FigureSpec(
                name="occupancy_histogram",
                description=(
                    "Occupancy histogram — reducer-summary diagnostic "
                    "(TensorBoard path only; not compatible with FigureGenerationCallback)"
                ),
                plot=occupancy.plot,
                default_filename="occupancy_histogram",
                kind="diagnostic",
                input_contract="evaluation_artifact",
                tags={"tem", "ehc", "spatial", "summary"},
                trace_keys={"diagnostic/occupancy"},
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("hidden_norm_histogram"):
        REGISTRY.register(
            FigureSpec(
                name="hidden_norm_histogram",
                description=(
                    "Hidden-state norm histogram — reducer-summary diagnostic "
                    "(TensorBoard path only; not compatible with FigureGenerationCallback)"
                ),
                plot=hidden_norm.plot,
                default_filename="hidden_norm_histogram",
                kind="diagnostic",
                input_contract="evaluation_artifact",
                tags={"tem", "ehc", "spatial", "summary"},
                trace_keys={
                    "diagnostic/hidden_norms",
                    "diagnostic/hidden_norms_density",
                },
                meta_keys=set(),
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
                input_contract="evaluation_artifact",
                tags={"lec", "ehc"},
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
                input_contract="evaluation_artifact",
                tags={"mec", "ehc"},
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
                input_contract="evaluation_artifact",
                tags={"hpc", "ehc"},
                trace_keys={
                    HPC_TRACE_LOCATION_IDS,
                    TRACE_KEY_HPC_CELLS,
                    TRACE_KEY_HPC_MEMORY,
                },
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
                input_contract="evaluation_artifact",
                tags={"lec", "ehc"},
                trace_keys={
                    TRACE_KEY_OBSERVATION,
                    TRACE_KEY_LEC_CELLS,
                    TRACE_KEY_LEC_FILTERED,
                },
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
                input_contract="evaluation_artifact",
                tags={"mec", "ehc"},
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
                input_contract="evaluation_artifact",
                tags={"hpc", "ehc"},
                trace_keys={HPC_TRACE_LOCATION_IDS, TRACE_KEY_HPC_CELLS},
                meta_keys={HPC_META_ENVIRONMENTS},
            )
        )

    # --- Post-registration validation: every bounded_trace figure must  ---
    #     declare non-empty trace_keys so the callback can derive required
    #     keys from the registry.
    for spec in REGISTRY.list_specs(input_contract="bounded_trace"):
        if not spec.trace_keys:
            raise ValueError(
                f"bounded_trace figure {spec.name!r} declares empty trace_keys; "
                "every bounded_trace figure must declare at least one trace_key "
                "so FigureGenerationCallback can derive required keys."
            )
