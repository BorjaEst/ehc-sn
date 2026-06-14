"""SeqMaze + HRM v2 experiment (hybrid actor-critic).

Composition:
    model: HRModelV2
    adapter: SeqMazeHRMV2BridgeAdapter
    controller: DeliberationACController
    objective: HybridRLObjective
    regime module: ActorCriticModule
"""

from __future__ import annotations

from typing import Any

from ehc_sn.adapters.hrm import (
    SEQMAZE_HRM_ACTOR_CRITIC_TRACE_FIELDS,
    SeqMazeAdapterSettings,
    SeqMazeHRMV2BridgeAdapter,
    SeqMazeHRMV2HybridTaskBinding,
    build_seqmaze_hrm_actor_critic_trace_meta,
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
from ehc_sn.tasks.seqmaze.reward import (
    SeqMazeRewardConfig,
    SeqMazeRewardProjector,
)
from ehc_sn.tasks.seqmaze.runtime import (
    SeqMazeRuntime,
    SeqMazeRuntimeConfig,
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
            ``ActorCriticConfig`` fields plus ``[adapter]``,
            ``[deliberation]``, ``[reward]``, ``[controller]``,
            ``[objective]``, optimizer sections, ``[runtime]``.

    Returns:
        Instantiated
        :class:`~ehc_sn.lightning.modules.actor_critic.ActorCriticModule`.
    """
    field_names = set(ActorCriticConfig.model_fields)
    filtered = {k: v for k, v in raw_config.items() if k in field_names}
    config = ActorCriticConfig(**filtered)
    component_configs = ActorCriticComponentConfigs(
        adapter=SeqMazeAdapterSettings.model_validate(raw_config["adapter"]),
        runtime=SeqMazeRuntimeConfig.model_validate(raw_config["deliberation"]),
        reward=SeqMazeRewardConfig.model_validate(raw_config["reward"]),
        controller=DeliberationACControllerConfig.model_validate(
            raw_config["controller"]
        ),
        objective=HybridRLLossConfig.model_validate(raw_config["objective"]),
        optimizer_supervised=AdamATan2Config.model_validate(
            raw_config["optimizer_supervised"]
        ),
        optimizer_rl=AdamATan2Config.model_validate(raw_config["optimizer_rl"]),
        optimizer_qv=AdamATan2Config.model_validate(raw_config["optimizer_qv"]),
        hrm_runtime=HRMRuntimeConfig.model_validate(raw_config["runtime"]),
    )

    # Derive constants from adapter config for task binding
    n_max = component_configs.adapter.n_max

    bindings = ActorCriticBindings(
        model_cls=HRModelV2,
        model_settings_cls=ModelSettingsV2,
        adapter_cls=SeqMazeHRMV2BridgeAdapter,
        adapter_settings_cls=SeqMazeAdapterSettings,
        controller_cls=DeliberationACController,
        controller_config_cls=DeliberationACControllerConfig,
        objective_cls=HybridRLObjective,
        objective_config_cls=HybridRLLossConfig,
        task_binding_cls=lambda: SeqMazeHRMV2HybridTaskBinding(n_max=n_max),
        learner_cls=TD0ActorCriticBatchBuilder,
        val_scorer_cls=ZeroBootstrapActorCriticValidationScorer,
        optimizer_cls=AdamATan2,
        optimizer_config_cls=AdamATan2Config,
        runtime_config_cls=SeqMazeRuntimeConfig,
        reward_config_cls=SeqMazeRewardConfig,
        runtime_cls=SeqMazeRuntime,
        reward_projector_cls=SeqMazeRewardProjector,
        trace_fields=SEQMAZE_HRM_ACTOR_CRITIC_TRACE_FIELDS,
        build_trace_meta_fn=build_seqmaze_hrm_actor_critic_trace_meta,
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
