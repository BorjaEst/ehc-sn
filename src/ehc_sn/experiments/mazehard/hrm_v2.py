"""MazeHard + HRM v2 experiment (hybrid actor-critic).

Composition:
    model: HRModelV2
    adapter: MazeHardHRMV2BridgeAdapter
    controller: DeliberationACController
    objective: HybridRLObjective
    regime module: ActorCriticModule
"""

from __future__ import annotations

from typing import Any

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
)
from ehc_sn.metrics.routes.rl import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, ModelSettingsV2
from ehc_sn.objectives.hybrid_rl import (
    HybridRLLossConfig,
    HybridRLObjective,
)
from ehc_sn.tasks.mazehard.capabilities.deliberation import (
    MazeHardDeliberationCapability,
    MazeHardDeliberationConfig,
)
from ehc_sn.tasks.mazehard.reward import (
    MazeHardRewardConfig,
    MazeHardRewardProjector,
)
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.actor_critic import (
    TD0ActorCriticBatchBuilder,
    ZeroBootstrapActorCriticValidationScorer,
)
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2, AdamATan2Config


def build_experiment(raw_config: dict) -> ActorCriticModule:
    """Parse config and return an instantiated ActorCriticModule.

    Args:
        raw_config: Flat TOML-like mapping with keys matching
            ``ActorCriticConfig`` fields.

    Returns:
        Instantiated :class:`~ehc_sn.lightning.modules.actor_critic.ActorCriticModule`.
    """
    field_names = set(ActorCriticConfig.model_fields)
    filtered = {k: v for k, v in raw_config.items() if k in field_names}
    config = ActorCriticConfig(**filtered)
    component_configs = ActorCriticComponentConfigs(
        adapter=MazeHardHRMAdapterSettings.model_validate(
            raw_config["adapter"]
        ),
        deliberation=MazeHardDeliberationConfig.model_validate(
            raw_config["deliberation"]
        ),
        reward=MazeHardRewardConfig.model_validate(raw_config["reward"]),
        controller=DeliberationACControllerConfig.model_validate(
            raw_config["controller"]
        ),
        objective=HybridRLLossConfig.model_validate(raw_config["objective"]),
        optimizer_supervised=AdamATan2Config.model_validate(
            raw_config["optimizer_supervised"]
        ),
        optimizer_rl=AdamATan2Config.model_validate(raw_config["optimizer_rl"]),
        optimizer_qv=AdamATan2Config.model_validate(raw_config["optimizer_qv"]),
        runtime=HRMRuntimeConfig.model_validate(raw_config["runtime"]),
    )
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
        deliberation_config_cls=MazeHardDeliberationConfig,
        reward_config_cls=MazeHardRewardConfig,
        reward_projector_cls=MazeHardRewardProjector,
        capability_cls=MazeHardDeliberationCapability,
        trace_fields=MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS,
        build_trace_meta_fn=build_mazehard_hrm_trace_meta,
        step_routes=RL_STEP_ROUTES,
        episode_routes=RL_EPISODE_ROUTES,
        hidden_state_fields=HRM_HIDDEN_STATE_FIELDS,
    )
    return ActorCriticModule(
        config=config,
        component_configs=component_configs,
        bindings=bindings,
    )


__all__ = ["build_experiment"]
