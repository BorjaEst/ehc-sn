"""Registration helpers for built-in figures."""

from __future__ import annotations

from ehc_sn.figures.registry import REGISTRY, FigureSpec
from ehc_sn.figures.templates import mazehard_solution_overlay


def register_builtin_figures() -> None:
    """Register built-in figure specifications (lazy template imports)."""
    # Templates are imported inside this function so that importing
    # ``ehc_sn.figures`` does not eagerly pull in all template modules.
    from ehc_sn.figures.selectors.arena_task import (
        ARENA_TRACE_KEY_OBSERVATION_IDS,
        ARENA_TRACE_KEY_REVISIT_MASK,
        ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS,
        ARENA_TRACE_KEY_WALL_MASK,
    )
    from ehc_sn.figures.selectors.arena_tem import (
        TEM_META_KEY_TARGET_OBS_ID,
        TEM_TRACE_KEY_PRED_ANCESTRAL,
        TEM_TRACE_KEY_PRED_INFERENCE,
        TEM_TRACE_KEY_PRED_RETRIEVED,
    )
    from ehc_sn.figures.selectors.hpc import (
        HPC_META_KEY_ENVIRONMENTS,
        HPC_TRACE_KEY_CELLS,
        HPC_TRACE_KEY_LOCATION_IDS,
        HPC_TRACE_KEY_MEMORY,
    )
    from ehc_sn.figures.selectors.lec import (
        LEC_META_KEY_ALPHA,
        LEC_META_KEY_WF,
        LEC_TRACE_KEY_CELLS,
        LEC_TRACE_KEY_FILTERED,
        LEC_TRACE_KEY_LOCATION_IDS,
        LEC_TRACE_KEY_OBSERVATION,
    )
    from ehc_sn.figures.selectors.mazehard import (
        MAZEHARD_META_KEY_GT_OVERLAY,
        MAZEHARD_META_KEY_INPUT_IDS,
        MAZEHARD_TRACE_KEY_HALTED,
        MAZEHARD_TRACE_KEY_PRED_OVERLAY,
    )
    from ehc_sn.figures.selectors.mec import (
        MEC_META_KEY_ENVIRONMENTS,
        MEC_TRACE_KEY_CELLS,
        MEC_TRACE_KEY_LOCATION_IDS,
    )
    from ehc_sn.figures.templates import (
        arena_prediction_overlay,
        arena_task_layout,
        halt_logit_evolution,
        halting_timeline,
        hidden_norm,
        hpc_place_metrics,
        hpc_rate_map_mosaic,
        lec_activity_trajectory,
        lec_content_filtering,
        lec_content_structure_rsa,
        lec_observation_tuning,
        mazehard_prediction_evolution,
        mec_autocorr_mosaic,
        mec_grid_metrics,
        occupancy,
        pfc_latent_dynamics,
        q_value_evolution,
    )

    if not REGISTRY.has("mazehard_solution_overlay"):
        REGISTRY.register(
            FigureSpec(
                name="mazehard_solution_overlay",
                description="MazeHard overlays: N samples with GT vs model paths",
                plot=mazehard_solution_overlay.plot,
                default_filename="mazehard_solution_overlay",
                maturity="stable",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="offline_artifact",
                tags={"paper", "mazehard"},
                trace_keys={
                    MAZEHARD_TRACE_KEY_HALTED,
                    MAZEHARD_TRACE_KEY_PRED_OVERLAY,
                },
                meta_keys={
                    MAZEHARD_META_KEY_INPUT_IDS,
                    MAZEHARD_META_KEY_GT_OVERLAY,
                },
            )
        )

    if not REGISTRY.has("mazehard_prediction_evolution"):
        REGISTRY.register(
            FigureSpec(
                name="mazehard_prediction_evolution",
                description="MazeHard prediction evolution: GT + per-step argmax overlays for one sample",
                plot=mazehard_prediction_evolution.plot,
                default_filename="mazehard_prediction_evolution",
                maturity="experimental",
                allowed_surfaces={"diagnostic"},
                input_contract="evaluation_artifact",
                tags={"mazehard"},
                trace_keys={
                    MAZEHARD_TRACE_KEY_HALTED,
                    MAZEHARD_TRACE_KEY_PRED_OVERLAY,
                },
                meta_keys={
                    MAZEHARD_META_KEY_INPUT_IDS,
                    MAZEHARD_META_KEY_GT_OVERLAY,
                },
            )
        )

    if not REGISTRY.has("arena_prediction_overlay"):
        REGISTRY.register(
            FigureSpec(
                name="arena_prediction_overlay",
                description=(
                    "Per-step argmax prediction mazehard_solution_overlay: GT vs predicted "
                    "observation IDs (inference / retrieved / ancestral) "
                    "across the full episode. Mismatched cells are outlined "
                    "in black. Family-neutral; compatible with TEM-style "
                    "and EHC-style Arena traces. Does NOT show confidence "
                    "or pathway uncertainty."
                ),
                plot=arena_prediction_overlay.plot,
                default_filename="arena_prediction_overlay",
                maturity="stable",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="evaluation_artifact",
                tags={"arena"},
                trace_keys={
                    TEM_TRACE_KEY_PRED_INFERENCE,
                    TEM_TRACE_KEY_PRED_RETRIEVED,
                    TEM_TRACE_KEY_PRED_ANCESTRAL,
                },
                meta_keys={TEM_META_KEY_TARGET_OBS_ID},
            )
        )

    if not REGISTRY.has("arena_task_layout"):
        REGISTRY.register(
            FigureSpec(
                name="arena_task_layout",
                description=(
                    "Arena task overview: topology, observation map, "
                    "trajectory, and revisit markers."
                ),
                plot=arena_task_layout.plot,
                default_filename="arena_task_layout",
                maturity="stable",
                allowed_surfaces={"report"},
                input_contract="offline_artifact",
                tags={"arena", "task-context"},
                trace_keys={
                    ARENA_TRACE_KEY_WALL_MASK,
                    ARENA_TRACE_KEY_OBSERVATION_IDS,
                    ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS,
                    ARENA_TRACE_KEY_REVISIT_MASK,
                },
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("halting_timeline"):
        REGISTRY.register(
            FigureSpec(
                name="halting_timeline",
                description="Binary halting signal heatmap over time",
                plot=halting_timeline.plot,
                default_filename="halting_timeline",
                maturity="stable",
                allowed_surfaces={"training", "diagnostic"},
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
                plot=q_value_evolution.plot,
                default_filename="q_value_evolution",
                maturity="experimental",
                allowed_surfaces={"training", "diagnostic"},
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
                plot=halt_logit_evolution.plot,
                default_filename="halt_logit_evolution",
                maturity="experimental",
                allowed_surfaces={"training", "diagnostic"},
                input_contract="bounded_trace",
                tags={"hrm", "act", "reasoning", "halting"},
                trace_keys={"value/q_logits"},
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("pfc_latent_dynamics"):
        REGISTRY.register(
            FigureSpec(
                name="pfc_latent_dynamics",
                description=(
                    "HRM H/L latent dynamics: state norm and delta "
                    "over rollout time"
                ),
                plot=pfc_latent_dynamics.plot,
                default_filename="pfc_latent_dynamics",
                maturity="stable",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="offline_artifact",
                tags={"hrm", "dynamics", "latent"},
                trace_keys={"pfc/z_H", "pfc/z_L"},
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
                maturity="stable",
                allowed_surfaces={"training", "diagnostic"},
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
                maturity="experimental",
                allowed_surfaces={"training", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"tem", "ehc", "spatial", "summary"},
                trace_keys={
                    "diagnostic/hidden_norms",
                    "diagnostic/hidden_norms_density",
                },
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("lec_content_filtering"):
        REGISTRY.register(
            FigureSpec(
                name="lec_content_filtering",
                description=(
                    "LEC / x content-state filtering per frequency band: "
                    "content-state activity, filtered diagnostic state, "
                    "and alpha_f/w_f gate parameters when available. "
                    "Mechanism diagnostic for the sensory/content stream, "
                    "not a spatial-cell diagnostic."
                ),
                plot=lec_content_filtering.plot,
                default_filename="lec_content_filtering",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"lec", "ehc", "report"},
                trace_keys={LEC_TRACE_KEY_OBSERVATION, LEC_TRACE_KEY_CELLS},
                meta_keys={LEC_META_KEY_ALPHA, LEC_META_KEY_WF},
            )
        )

    if not REGISTRY.has("mec_grid_metrics"):
        REGISTRY.register(
            FigureSpec(
                name="mec_grid_metrics",
                description=(
                    "Quantitative gridness and spacing by frequency band. "
                    "Replaces qualitative autocorrelogram inspection with "
                    "per-cell metric distributions."
                ),
                plot=mec_grid_metrics.plot,
                default_filename="mec_grid_metrics",
                maturity="experimental",
                allowed_surfaces={"report"},
                input_contract="evaluation_artifact",
                tags={"mec", "ehc", "report"},
                trace_keys={MEC_TRACE_KEY_LOCATION_IDS, MEC_TRACE_KEY_CELLS},
                meta_keys={MEC_META_KEY_ENVIRONMENTS},
            )
        )

    if not REGISTRY.has("mec_autocorr_mosaic"):
        REGISTRY.register(
            FigureSpec(
                name="mec_autocorr_mosaic",
                description=(
                    "Population autocorrelogram mosaic showing grid-like "
                    "periodicity across many MEC cells, ordered by gridness "
                    "score per frequency band."
                ),
                plot=mec_autocorr_mosaic.plot,
                default_filename="mec_autocorr_mosaic",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"mec", "ehc", "report"},
                trace_keys={MEC_TRACE_KEY_LOCATION_IDS, MEC_TRACE_KEY_CELLS},
                meta_keys={MEC_META_KEY_ENVIRONMENTS},
            )
        )

    if not REGISTRY.has("hpc_place_metrics"):
        REGISTRY.register(
            FigureSpec(
                name="hpc_place_metrics",
                description=(
                    "Quantitative place-cell metrics: spatial information "
                    "distribution, field-center coverage, and top place-like "
                    "rate-map examples."
                ),
                plot=hpc_place_metrics.plot,
                default_filename="hpc_place_metrics",
                maturity="experimental",
                allowed_surfaces={"report"},
                input_contract="evaluation_artifact",
                tags={"hpc", "ehc", "report"},
                trace_keys={HPC_TRACE_KEY_LOCATION_IDS, HPC_TRACE_KEY_CELLS},
                meta_keys={HPC_META_KEY_ENVIRONMENTS},
            )
        )

    if not REGISTRY.has("hpc_rate_map_mosaic"):
        REGISTRY.register(
            FigureSpec(
                name="hpc_rate_map_mosaic",
                description=(
                    "Population rate-map mosaic for HPC cells, ordered by "
                    "spatial information descending.  Shows place-like rate "
                    "maps across many cells."
                ),
                plot=hpc_rate_map_mosaic.plot,
                default_filename="hpc_rate_map_mosaic",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"hpc", "ehc", "report"},
                trace_keys={HPC_TRACE_KEY_LOCATION_IDS, HPC_TRACE_KEY_CELLS},
                meta_keys={HPC_META_KEY_ENVIRONMENTS},
            )
        )

    if not REGISTRY.has("lec_activity_trajectory"):
        REGISTRY.register(
            FigureSpec(
                name="lec_activity_trajectory",
                description=(
                    "LEC activity trajectory: observation IDs, location IDs, "
                    "and LEC cell/filtered activity heatmaps over the episode. "
                    "Descriptive only — does not claim content selectivity."
                ),
                plot=lec_activity_trajectory.plot,
                default_filename="lec_activity_trajectory",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"lec", "ehc", "report"},
                trace_keys={
                    LEC_TRACE_KEY_CELLS,
                    LEC_TRACE_KEY_OBSERVATION,
                    LEC_TRACE_KEY_LOCATION_IDS,
                },
                meta_keys={TEM_META_KEY_TARGET_OBS_ID},
            )
        )

    if not REGISTRY.has("lec_observation_tuning"):
        REGISTRY.register(
            FigureSpec(
                name="lec_observation_tuning",
                description=(
                    "LEC observation-ID tuning matrix. Tests whether LEC "
                    "units are selective for observation/content identity."
                ),
                plot=lec_observation_tuning.plot,
                default_filename="lec_observation_tuning",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"lec", "ehc", "report"},
                trace_keys={
                    LEC_TRACE_KEY_CELLS,
                    LEC_TRACE_KEY_OBSERVATION,
                    LEC_TRACE_KEY_LOCATION_IDS,
                },
                meta_keys={TEM_META_KEY_TARGET_OBS_ID},
            )
        )

    if not REGISTRY.has("lec_content_structure_rsa"):
        REGISTRY.register(
            FigureSpec(
                name="lec_content_structure_rsa",
                description=(
                    "Cross-system (LEC / MEC / HPC) representational "
                    "similarity analysis. Tests whether LEC is organised "
                    "by observation identity, MEC by location identity, "
                    "and HPC shows mixed / conjunctive organisation."
                ),
                plot=lec_content_structure_rsa.plot,
                default_filename="lec_content_structure_rsa",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"lec", "ehc", "report"},
                trace_keys={
                    LEC_TRACE_KEY_CELLS,
                    LEC_TRACE_KEY_CELLS,
                    LEC_TRACE_KEY_CELLS,
                    LEC_TRACE_KEY_OBSERVATION,
                    LEC_TRACE_KEY_LOCATION_IDS,
                },
                meta_keys={TEM_META_KEY_TARGET_OBS_ID},
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
