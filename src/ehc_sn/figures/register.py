"""Registration helpers for built-in figures."""

from __future__ import annotations

from ehc_sn.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register built-in figure specifications (lazy template imports)."""
    # Templates are imported inside this function so that importing
    # ``ehc_sn.figures`` does not eagerly pull in all template modules.
    from ehc_sn.figures.templates.diagnostics import (
        h_l_residuals_over_steps,
        hpc_place_metrics,
        hpc_rate_map_mosaic,
        lec_content_filtering,
        lec_content_structure_rsa,
        mec_autocorr_mosaic,
        mec_grid_metrics,
        pfc_latent_dynamics,
        pfc_path_memory_probe,
    )
    from ehc_sn.figures.templates.evaluation import (
        prediction_overlay_arena,
        prediction_reasoning_goaltrace,
        prediction_reasoning_mazehard,
        prediction_reasoning_routebind,
    )
    from ehc_sn.figures.templates.task import (
        task_overview_arena,
        task_overview_goaltrace,
        task_overview_mazehard,
        task_overview_routebind,
    )
    from ehc_sn.traces.keys import (
        ARENA_TRACE_KEY_OBSERVATION_IDS,
        ARENA_TRACE_KEY_REVISIT_MASK,
        ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS,
        ARENA_TRACE_KEY_WALL_MASK,
        GOALTRACE_META_KEY_CURRENT_FLAG,
        GOALTRACE_META_KEY_GOAL_FLAG,
        GOALTRACE_META_KEY_NODE_MASK,
        GOALTRACE_META_KEY_OBSERVATION_ID,
        GOALTRACE_META_KEY_SUCCESSOR_INDICES,
        GOALTRACE_META_KEY_SUCCESSOR_MASK,
        GOALTRACE_META_KEY_TARGET_FIELD,
        GOALTRACE_META_KEY_WEIGHT,
        GOALTRACE_TRACE_KEY_FIRING_FIELD,
        HPC_TRACE_KEY_CELLS,
        LEC_META_KEY_ALPHA,
        LEC_META_KEY_WF,
        LEC_TRACE_KEY_CELLS,
        LEC_TRACE_KEY_FILTERED,
        MAZEHARD_META_KEY_GT_OVERLAY,
        MAZEHARD_META_KEY_INPUT_IDS,
        MAZEHARD_TRACE_KEY_HALTED,
        MAZEHARD_TRACE_KEY_PRED_OVERLAY,
        MEC_TRACE_KEY_CELLS,
        META_KEY_ENVIRONMENTS,
        PFC_TRACE_KEY_Z_H,
        PFC_TRACE_KEY_Z_L,
        ROUTEBIND_META_KEY_CANVAS_HEIGHT,
        ROUTEBIND_META_KEY_CANVAS_WIDTH,
        ROUTEBIND_META_KEY_CELL_MASK,
        ROUTEBIND_META_KEY_CELL_TYPE,
        ROUTEBIND_META_KEY_GOAL_FLAG,
        ROUTEBIND_META_KEY_N_OBSERVATIONS,
        ROUTEBIND_META_KEY_OBSERVATION_ID,
        ROUTEBIND_META_KEY_START_FLAG,
        ROUTEBIND_META_KEY_TARGET_TRAJECTORY,
        ROUTEBIND_META_KEY_TARGET_WAYPOINT,
        TEM_META_KEY_TARGET_OBS_ID,
        TEM_TRACE_KEY_PRED_PATH,
        TEM_TRACE_KEY_PRED_POST,
        TEM_TRACE_KEY_PRED_RECALL,
        WORLD_TRACE_KEY_LOCATION_IDS,
        WORLD_TRACE_KEY_OBSERVATION,
    )

    if not REGISTRY.has("task_overview_mazehard"):
        REGISTRY.register(
            FigureSpec(
                name="task_overview_mazehard",
                description="MazeHard task layout: input grid + target path for case-level orientation",
                category="task",
                role="task_overview",
                source_kind="task_sample",
                task="mazehard",
                plot=task_overview_mazehard.plot,
                default_filename="task_overview_mazehard",
                maturity="stable",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="offline_artifact",
                tags={"mazehard", "task-context"},
                trace_keys=set(),
                meta_keys={
                    MAZEHARD_META_KEY_INPUT_IDS,
                    MAZEHARD_META_KEY_GT_OVERLAY,
                },
            )
        )

    if not REGISTRY.has("prediction_reasoning_mazehard"):
        REGISTRY.register(
            FigureSpec(
                name="prediction_reasoning_mazehard",
                description=(
                    "MazeHard prediction reasoning: ground-truth path "
                    "vs. per-deliberation-step prediction mosaic with "
                    "halt/truncation markers and path-IoU metrics"
                ),
                category="evaluation",
                role="prediction_reasoning",
                source_kind="evaluation_sample",
                task="mazehard",
                plot=prediction_reasoning_mazehard.plot,
                default_filename="prediction_reasoning_mazehard",
                maturity="experimental",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="evaluation_artifact",
                tags={"mazehard", "reasoning", "prediction"},
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

    if not REGISTRY.has("prediction_reasoning_goaltrace"):
        REGISTRY.register(
            FigureSpec(
                name="prediction_reasoning_goaltrace",
                description=(
                    "Goaltrace prediction reasoning: target firing field "
                    "vs. per-deliberation-step prediction mosaic with "
                    "halt markers and field-MSE metrics"
                ),
                category="evaluation",
                role="prediction_reasoning",
                source_kind="evaluation_sample",
                task="goaltrace",
                plot=prediction_reasoning_goaltrace.plot,
                default_filename="prediction_reasoning_goaltrace",
                maturity="experimental",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="evaluation_artifact",
                tags={"goaltrace", "reasoning", "prediction"},
                trace_keys={
                    GOALTRACE_TRACE_KEY_FIRING_FIELD,
                },
                meta_keys=frozenset(
                    {
                        GOALTRACE_META_KEY_OBSERVATION_ID,
                        GOALTRACE_META_KEY_WEIGHT,
                        GOALTRACE_META_KEY_CURRENT_FLAG,
                        GOALTRACE_META_KEY_GOAL_FLAG,
                        GOALTRACE_META_KEY_NODE_MASK,
                        GOALTRACE_META_KEY_TARGET_FIELD,
                        GOALTRACE_META_KEY_SUCCESSOR_INDICES,
                        GOALTRACE_META_KEY_SUCCESSOR_MASK,
                    }
                ),
            )
        )

    if not REGISTRY.has("prediction_reasoning_routebind"):
        REGISTRY.register(
            FigureSpec(
                name="prediction_reasoning_routebind",
                description=(
                    "Routebind prediction reasoning: target trajectory field "
                    "vs. per-deliberation-step prediction mosaic with "
                    "halt markers and field-MSE metrics"
                ),
                category="evaluation",
                role="prediction_reasoning",
                source_kind="evaluation_sample",
                task="routebind",
                plot=prediction_reasoning_routebind.plot,
                default_filename="prediction_reasoning_routebind",
                maturity="experimental",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="evaluation_artifact",
                tags={"routebind", "reasoning", "prediction"},
                trace_keys={
                    "routebind/trajectory_field",
                },
                meta_keys=frozenset(
                    {
                        ROUTEBIND_META_KEY_CELL_TYPE,
                        ROUTEBIND_META_KEY_OBSERVATION_ID,
                        ROUTEBIND_META_KEY_START_FLAG,
                        ROUTEBIND_META_KEY_GOAL_FLAG,
                        ROUTEBIND_META_KEY_CELL_MASK,
                        ROUTEBIND_META_KEY_TARGET_TRAJECTORY,
                        ROUTEBIND_META_KEY_TARGET_WAYPOINT,
                    }
                ),
            )
        )

    if not REGISTRY.has("pfc_path_memory_probe"):
        REGISTRY.register(
            FigureSpec(
                name="pfc_path_memory_probe",
                description=(
                    "Linear decoding of target-path information from "
                    "z_H and z_L across recurrent steps — "
                    "probe-backed evidence for working memory content"
                ),
                category="diagnostic",
                role="pfc_path_memory_probe",
                source_kind="evaluation_sample",
                task=None,
                plot=pfc_path_memory_probe.plot,
                default_filename="pfc_path_memory_probe",
                maturity="experimental",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="offline_artifact",
                tags={"hrm", "probe", "working_memory"},
                trace_keys=set(),
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("prediction_overlay_arena"):
        REGISTRY.register(
            FigureSpec(
                name="prediction_overlay_arena",
                description=(
                    "Per-step argmax prediction overlay: GT vs predicted "
                    "observation IDs (inference / retrieved / ancestral) "
                    "across the full episode. Mismatched cells are outlined "
                    "in black. Family-neutral; compatible with TEM-style "
                    "and EHP-style Arena traces. Does NOT show confidence "
                    "or pathway uncertainty."
                ),
                category="evaluation",
                role="prediction_overlay",
                source_kind="evaluation_sample",
                task="arena",
                plot=prediction_overlay_arena.plot,
                default_filename="prediction_overlay_arena",
                maturity="stable",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="evaluation_artifact",
                tags={"arena"},
                trace_keys={
                    TEM_TRACE_KEY_PRED_POST,
                    TEM_TRACE_KEY_PRED_RECALL,
                    TEM_TRACE_KEY_PRED_PATH,
                },
                meta_keys={TEM_META_KEY_TARGET_OBS_ID},
            )
        )

    if not REGISTRY.has("task_overview_arena"):
        REGISTRY.register(
            FigureSpec(
                name="task_overview_arena",
                description=(
                    "Task input figure: arena topology, observation map, "
                    "trajectory, and revisit markers. "
                    "Not a model diagnostic."
                ),
                category="task",
                role="task_overview",
                source_kind="task_sample",
                task="arena",
                plot=task_overview_arena.plot,
                default_filename="task_overview_arena",
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

    if not REGISTRY.has("pfc_latent_dynamics"):
        REGISTRY.register(
            FigureSpec(
                name="pfc_latent_dynamics",
                description=(
                    "HRM H/L latent dynamics: state norm and delta "
                    "over rollout time"
                ),
                category="diagnostic",
                role="pfc_latent_dynamics",
                source_kind="evaluation_sample",
                task=None,
                plot=pfc_latent_dynamics.plot,
                default_filename="pfc_latent_dynamics",
                maturity="stable",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="offline_artifact",
                tags={"hrm", "dynamics", "latent"},
                trace_keys={PFC_TRACE_KEY_Z_H, PFC_TRACE_KEY_Z_L},
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("h_l_residuals_over_steps"):
        REGISTRY.register(
            FigureSpec(
                name="h_l_residuals_over_steps",
                description=(
                    "HRM H/L residual dynamics: forward residuals, "
                    "cosine similarity, and H/L separation over "
                    "fixed-budget recurrent rollout steps"
                ),
                category="diagnostic",
                role="residuals_over_steps",
                source_kind="evaluation_sample",
                task=None,
                plot=h_l_residuals_over_steps.plot,
                default_filename="h_l_residuals_over_steps",
                maturity="experimental",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="offline_artifact",
                tags={"hrm", "dynamics", "latent"},
                trace_keys={PFC_TRACE_KEY_Z_H, PFC_TRACE_KEY_Z_L},
                meta_keys=set(),
            )
        )

    if not REGISTRY.has("lec_content_filtering"):
        REGISTRY.register(
            FigureSpec(
                name="lec_content_filtering",
                description=(
                    "LEC transformation cascade: sensory code → "
                    "EMA-filtered state → final LEC cells "
                    "(mean sub + ReLU + L2 norm + sigmoid(w_f) scaling). "
                    "Mechanism diagnostic for the sensory/content stream, "
                    "not a spatial-cell diagnostic."
                ),
                category="diagnostic",
                role="lec_content_filtering",
                source_kind="evaluation_sample",
                task=None,
                plot=lec_content_filtering.plot,
                default_filename="lec_content_filtering",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"lec", "ehp", "report"},
                trace_keys={
                    WORLD_TRACE_KEY_OBSERVATION,
                    LEC_TRACE_KEY_CELLS,
                    LEC_TRACE_KEY_FILTERED,
                },
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
                category="diagnostic",
                role="mec_grid_metrics",
                source_kind="evaluation_sample",
                task=None,
                plot=mec_grid_metrics.plot,
                default_filename="mec_grid_metrics",
                maturity="experimental",
                allowed_surfaces={"report"},
                input_contract="evaluation_artifact",
                tags={"mec", "ehp", "report"},
                trace_keys={WORLD_TRACE_KEY_LOCATION_IDS, MEC_TRACE_KEY_CELLS},
                meta_keys={META_KEY_ENVIRONMENTS},
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
                category="diagnostic",
                role="mec_autocorr_mosaic",
                source_kind="evaluation_sample",
                task=None,
                plot=mec_autocorr_mosaic.plot,
                default_filename="mec_autocorr_mosaic",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"mec", "ehp", "report"},
                trace_keys={WORLD_TRACE_KEY_LOCATION_IDS, MEC_TRACE_KEY_CELLS},
                meta_keys={META_KEY_ENVIRONMENTS},
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
                category="diagnostic",
                role="hpc_place_metrics",
                source_kind="evaluation_sample",
                task=None,
                plot=hpc_place_metrics.plot,
                default_filename="hpc_place_metrics",
                maturity="experimental",
                allowed_surfaces={"report"},
                input_contract="evaluation_artifact",
                tags={"hpc", "ehp", "report"},
                trace_keys={WORLD_TRACE_KEY_LOCATION_IDS, HPC_TRACE_KEY_CELLS},
                meta_keys={META_KEY_ENVIRONMENTS},
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
                category="diagnostic",
                role="hpc_rate_map_mosaic",
                source_kind="evaluation_sample",
                task=None,
                plot=hpc_rate_map_mosaic.plot,
                default_filename="hpc_rate_map_mosaic",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"hpc", "ehp", "report"},
                trace_keys={WORLD_TRACE_KEY_LOCATION_IDS, HPC_TRACE_KEY_CELLS},
                meta_keys={META_KEY_ENVIRONMENTS},
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
                    "and HPC shows mixed / conjunctive organization."
                ),
                category="diagnostic",
                role="lec_content_structure_rsa",
                source_kind="evaluation_sample",
                task=None,
                plot=lec_content_structure_rsa.plot,
                default_filename="lec_content_structure_rsa",
                maturity="experimental",
                allowed_surfaces={"report", "diagnostic"},
                input_contract="evaluation_artifact",
                tags={"lec", "ehp", "report"},
                trace_keys={
                    LEC_TRACE_KEY_CELLS,
                    MEC_TRACE_KEY_CELLS,
                    HPC_TRACE_KEY_CELLS,
                    WORLD_TRACE_KEY_OBSERVATION,
                    WORLD_TRACE_KEY_LOCATION_IDS,
                },
                meta_keys={TEM_META_KEY_TARGET_OBS_ID},
            )
        )

    if not REGISTRY.has("task_overview_routebind"):
        REGISTRY.register(
            FigureSpec(
                name="task_overview_routebind",
                description=(
                    "Routebind task overview: input layout \u2192 oracle "
                    "trajectory field. Task-context figure only; no "
                    "dense model trace required."
                ),
                category="task",
                role="task_overview",
                source_kind="task_sample",
                task="routebind",
                plot=task_overview_routebind.plot,
                default_filename="task_overview_routebind",
                maturity="experimental",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="offline_artifact",
                tags={"routebind", "task-context"},
                trace_keys=frozenset(),
                meta_keys=frozenset(
                    {
                        ROUTEBIND_META_KEY_CELL_TYPE,
                        ROUTEBIND_META_KEY_GOAL_FLAG,
                        ROUTEBIND_META_KEY_OBSERVATION_ID,
                        ROUTEBIND_META_KEY_START_FLAG,
                        ROUTEBIND_META_KEY_TARGET_TRAJECTORY,
                        ROUTEBIND_META_KEY_TARGET_WAYPOINT,
                    }
                ),
            )
        )

    if not REGISTRY.has("task_overview_goaltrace"):
        REGISTRY.register(
            FigureSpec(
                name="task_overview_goaltrace",
                description=(
                    "Goaltrace task overview: oracle input weights → target "
                    "prospective firing field transformation.  Task-context "
                    "figure only; no dense model trace required."
                ),
                category="task",
                role="task_overview",
                source_kind="task_sample",
                task="goaltrace",
                plot=task_overview_goaltrace.plot,
                default_filename="task_overview_goaltrace",
                maturity="experimental",
                allowed_surfaces={"diagnostic", "report"},
                input_contract="offline_artifact",
                tags={"goaltrace", "task-context"},
                trace_keys=frozenset(),
                meta_keys=frozenset(
                    {
                        GOALTRACE_META_KEY_OBSERVATION_ID,
                        GOALTRACE_META_KEY_WEIGHT,
                        GOALTRACE_META_KEY_CURRENT_FLAG,
                        GOALTRACE_META_KEY_GOAL_FLAG,
                        GOALTRACE_META_KEY_NODE_MASK,
                        GOALTRACE_META_KEY_TARGET_FIELD,
                        GOALTRACE_META_KEY_SUCCESSOR_INDICES,
                        GOALTRACE_META_KEY_SUCCESSOR_MASK,
                    }
                ),
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
