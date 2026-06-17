"""Model constructor for MazeHard × HRM-v2.

Shared by both training and evaluation paths.
"""

from __future__ import annotations

from ehc_sn.adapters.hrm import (
    MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS,
    MazeHardHRMAdapterSettings,
    MazeHardHRMV2BridgeAdapter,
    MazeHardHRMV2HybridTaskBinding,
    build_mazehard_hrm_trace_meta,
)
from ehc_sn.controllers.deliberation.actor_critic import (
    DeliberationACController,
    DeliberationACControllerConfig,
)
from ehc_sn.lightning.modules.actor_critic import (
    ActorCriticBindings,
    ActorCriticComponentConfigs,
    ActorCriticConfig,
    ActorCriticModule,
    ActorCriticTrainingConfig,
)
from ehc_sn.metrics.routes.rl import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, ModelSettingsV2
from ehc_sn.objectives.hybrid_rl import (
    HybridRLLossConfig,
    HybridRLObjective,
)
from ehc_sn.tasks.mazehard.reward import (
    MazeHardRewardConfig,
    MazeHardRewardProjector,
)
from ehc_sn.tasks.mazehard.runtime import (
    MazeHardRuntime,
    MazeHardRuntimeConfig,
)
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.actor_critic import (
    TD0ActorCriticBatchBuilder,
    ZeroBootstrapActorCriticValidationScorer,
)
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.schedules import SchedulerConfig

from .config import MazeHardHRMV2ModelConfig


def build_mazehard_hrm_v2_model(
    config: MazeHardHRMV2ModelConfig,
    *,
    training_config: ActorCriticTrainingConfig | None = None,
    execution: MazeHardRuntimeConfig | None = None,
    scheduler: SchedulerConfig | None = None,
    supervised_only_warmup_steps: int | None = None,
) -> ActorCriticModule:
    """Construct an ActorCriticModule for MazeHard × HRM-v2.

    Parameters
    ----------
    config:
        Model configuration (components + architecture path).
    execution:
        Execution policy (runtime config). Required for both training and
        evaluation when the module is used with a Trainer.
    training_config:
        Training-only optimiser and reward config.  ``None`` during
        evaluation-only construction.  When provided, ``num_slots`` is
        read from ``training_config.num_slots`` in ``setup()``.
    """
    components: ActorCriticComponentConfigs = config.components  # type: ignore[assignment]

    bindings = ActorCriticBindings(
        model_cls=HRModelV2,
        model_settings_cls=ModelSettingsV2,
        adapter_cls=MazeHardHRMV2BridgeAdapter,
        adapter_settings_cls=MazeHardHRMAdapterSettings,
        controller_cls=DeliberationACController,
        controller_config_cls=DeliberationACControllerConfig,
        objective_cls=HybridRLObjective,
        objective_config_cls=HybridRLLossConfig,
        task_binding_cls=MazeHardHRMV2HybridTaskBinding,
        learner_cls=TD0ActorCriticBatchBuilder,
        val_scorer_cls=ZeroBootstrapActorCriticValidationScorer,
        optimizer_cls=AdamATan2,
        optimizer_config_cls=AdamATan2Config,
        runtime_config_cls=MazeHardRuntimeConfig,
        reward_config_cls=MazeHardRewardConfig,
        runtime_cls=MazeHardRuntime,
        reward_projector_cls=MazeHardRewardProjector,
        trace_fields=MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS,
        build_trace_meta_fn=build_mazehard_hrm_trace_meta,
        step_routes=RL_STEP_ROUTES,
        episode_routes=RL_EPISODE_ROUTES,
        hidden_state_fields=HRM_HIDDEN_STATE_FIELDS,
    )
    return ActorCriticModule(
        config=ActorCriticConfig(
            model_config_path=config.model_config_path,
            scheduler=scheduler or SchedulerConfig(),
            supervised_only_warmup_steps=supervised_only_warmup_steps or 0,
        ),
        component_configs=components,
        bindings=bindings,
        execution=execution,
        training_config=training_config,
    )


__all__ = ["build_mazehard_hrm_v2_model"]
