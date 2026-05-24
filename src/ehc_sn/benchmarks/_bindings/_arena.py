"""Arena protocol owner for model-comparison benchmark bindings."""

from __future__ import annotations

from typing import Any

from ehc_sn.adapters.arena.ehc import (
    ArenaEHCAdapterSettings,
    ArenaEHCTaskBinding,
    ArenaEHCV1BridgeAdapter,
)
from ehc_sn.adapters.arena.tem import (
    ArenaTEMAdapterSettings,
    ArenaTEMTaskBinding,
    ArenaTEMV1BridgeAdapter,
    ArenaTEMV2BridgeAdapter,
)
from ehc_sn.benchmarks.contracts import (
    ArenaScoreReport,
    ArtifactManifest,
    MazeHardScoreReport,
    ModelComparisonBinding,
    ModelComparisonExecution,
    ModelComparisonExecutionBundle,
    ModelComparisonExecutionResources,
    TrackRecipe,
    validate_model_comparison_pair,
)
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryController,
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.objectives.ehc import EHCObjective, EHCObjectiveConfig
from ehc_sn.objectives.tem import TEMObjective, TEMObjectiveConfig
from ehc_sn.rollouts.runtime import RecurrentRunner
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability
from ehc_sn.tasks.arena.providers import ArenaReplayProvider

from ._shared import (
    binding_config,
    bridge_only_parameter_groups,
    load_frozen_model,
    normalize_model_family,
)

_ARENA_MODEL_FAMILIES: frozenset[str] = frozenset(
    {"tem-v1", "tem-v2", "ehc-v1"}
)


# =============================================================================
class SharedArenaReplayModelComparisonBinding(ModelComparisonBinding):
    """Shared Arena replay benchmark binding for TEM v1/v2 and EHC v1."""

    def bind_model_comparison(
        self,
        manifest: ArtifactManifest,
        recipe: TrackRecipe,
    ) -> ModelComparisonExecutionBundle:
        validate_model_comparison_pair(manifest, recipe)
        if recipe.track_id != "arena-struct":
            raise ValueError(
                "SharedArenaReplayModelComparisonBinding only supports track "
                f"'arena-struct', got {recipe.track_id!r}."
            )
        if recipe.capability_kind != "arena_replay":
            raise ValueError(
                "SharedArenaReplayModelComparisonBinding requires "
                "capability_kind='arena_replay'."
            )

        model_family = normalize_model_family(manifest.model_family)
        if model_family not in _ARENA_MODEL_FAMILIES:
            raise ValueError(
                "Arena shared binding only supports model families "
                f"{sorted(_ARENA_MODEL_FAMILIES)!r}, got {model_family!r}."
            )

        family_binding = binding_config(recipe, model_family)
        model, _, _ = load_frozen_model(
            model_family,
            manifest,
        )

        def create_execution() -> ModelComparisonExecution:
            bridge = _build_arena_bridge(
                model_family, model, recipe, family_binding
            )
            controller = ReplayTrajectoryController(
                bridge,
                ReplayTrajectoryControllerConfig(window_size=None),
                ArenaReplayCapability(),
            )
            return ModelComparisonExecution(
                bridge=bridge,
                controller=controller,
                objective=_build_arena_objective(model_family),
                runner=RecurrentRunner(),
                bridge_parameter_groups=bridge_only_parameter_groups(
                    model,
                    bridge,
                ),
            )

        return ModelComparisonExecutionBundle(
            model=model,
            create_execution=create_execution,
            resources=ModelComparisonExecutionResources(
                replay_provider_factory=ArenaReplayProvider,
                controller_step_options={},
            ),
            score_aggregator=_aggregate_arena_score_reports,
        )


# =============================================================================
def _build_arena_bridge(
    model_family: str,
    model: object,
    recipe: TrackRecipe,
    family_binding: dict[str, Any],
) -> object:
    adapter_cfg = _resolve_arena_adapter_config(recipe, family_binding, model)
    if model_family == "tem-v1":
        settings = ArenaTEMAdapterSettings.model_validate(adapter_cfg)
        return ArenaTEMV1BridgeAdapter(model, settings)
    if model_family == "tem-v2":
        settings = ArenaTEMAdapterSettings.model_validate(adapter_cfg)
        return ArenaTEMV2BridgeAdapter(model, settings)
    if model_family == "ehc-v1":
        settings = ArenaEHCAdapterSettings.model_validate(adapter_cfg)
        return ArenaEHCV1BridgeAdapter(model, settings)
    raise ValueError(f"Unsupported Arena model family: {model_family!r}.")


# =============================================================================
def _resolve_arena_adapter_config(
    recipe: TrackRecipe,
    family_binding: dict[str, Any],
    model: object,
) -> dict[str, Any]:
    adapter_cfg = dict(family_binding.get("adapter", {}))
    if "observation_dim" not in adapter_cfg:
        bridge_cfg = recipe.bridge.model_dump(exclude_none=True)
        data_cfg = recipe.data.model_dump(exclude_none=True)
        for source in (family_binding, bridge_cfg, data_cfg):
            value = source.get("observation_dim")
            if value is not None:
                adapter_cfg["observation_dim"] = int(value)
                break
    if "observation_dim" not in adapter_cfg:
        raise ValueError(
            "Arena shared binding requires observation_dim in either "
            "recipe.bindings[model_family].adapter, recipe.bridge, or "
            "recipe.data."
        )

    if "action_count" not in adapter_cfg:
        default_action_count = getattr(
            model.config, "transition_action_count", None
        )
        if default_action_count is None:
            default_action_count = family_binding.get("action_count")
        if default_action_count is None:
            raise ValueError(
                "Arena shared binding requires action_count when the model "
                "config does not expose transition_action_count."
            )
        adapter_cfg["action_count"] = int(default_action_count)

    if "encoder" not in adapter_cfg and "encoder" in family_binding:
        adapter_cfg["encoder"] = family_binding["encoder"]
    if "decoder" not in adapter_cfg and "decoder" in family_binding:
        adapter_cfg["decoder"] = family_binding["decoder"]
    return adapter_cfg


# =============================================================================
def _build_arena_objective(model_family: str) -> object:
    if model_family in {"tem-v1", "tem-v2"}:
        return TEMObjective(
            TEMObjectiveConfig(),
            task_binding=ArenaTEMTaskBinding(),
        )
    if model_family == "ehc-v1":
        return EHCObjective(
            EHCObjectiveConfig(),
            task_binding=ArenaEHCTaskBinding(),
        )
    raise ValueError(f"Unsupported Arena model family: {model_family!r}.")


# =============================================================================
def _aggregate_arena_score_reports(
    score_reports: tuple[ArenaScoreReport | MazeHardScoreReport, ...],
) -> ArenaScoreReport:
    if not score_reports:
        raise ValueError(
            "Arena score aggregation requires at least one report."
        )
    if not all(
        isinstance(report, ArenaScoreReport) for report in score_reports
    ):
        raise TypeError("Arena score aggregation received a non-Arena score.")

    reports = tuple(score_reports)
    zero = reports[0].correct_all.new_zeros(())
    correct_all = zero
    count_all = zero
    correct_revisit = zero
    count_revisit = zero

    for report in reports:
        correct_all = correct_all + report.correct_all
        count_all = count_all + report.count_all
        correct_revisit = correct_revisit + report.correct_revisit
        count_revisit = count_revisit + report.count_revisit

    return ArenaScoreReport(
        accuracy_all=correct_all / count_all.clamp_min(1.0),
        accuracy_revisit=correct_revisit / count_revisit.clamp_min(1.0),
        correct_all=correct_all,
        count_all=count_all,
        correct_revisit=correct_revisit,
        count_revisit=count_revisit,
    )


# =============================================================================
__all__ = ["SharedArenaReplayModelComparisonBinding"]
