"""Model constructor for SeqMaze × HRM-v2."""

from __future__ import annotations

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
    ActorCriticTrainingConfig,
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
from ehc_sn.training.optim import AdamATan2, AdamATan2Config

from .config import SeqMazeHRMV2ModelConfig


def build_seqmaze_hrm_v2_model(
    config: SeqMazeHRMV2ModelConfig,
) -> ActorCriticModule:
    """Construct an ActorCriticModule for SeqMaze × HRM-v2."""
    components: ActorCriticComponentConfigs = config.components  # type: ignore[assignment]
    adapter_settings: SeqMazeAdapterSettings = components.adapter  # type: ignore[assignment]
    n_max = adapter_settings.n_max

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
        config=ActorCriticConfig(model_config_path=config.model_config_path),
        component_configs=components,
        bindings=bindings,
    )


def build_experiment(
    config: ActorCriticConfig,
    components: ActorCriticComponentConfigs,
    deliberation: SeqMazeRuntimeConfig,
    training: ActorCriticTrainingConfig | None = None,
) -> ActorCriticModule:
    """Backward-compat builder — delegates to ``build_seqmaze_hrm_v2_model``."""
    from .config import SeqMazeHRMV2ComponentConfigs, SeqMazeHRMV2ModelConfig

    model_config = SeqMazeHRMV2ModelConfig(
        model_config_path=config.model_config_path,
        components=SeqMazeHRMV2ComponentConfigs(
            adapter=components.adapter,
            controller=components.controller,
            objective=components.objective,
        ),
    )
    module = build_seqmaze_hrm_v2_model(model_config)
    if training is not None:
        module._deliberation = deliberation
        module._training_config = training
    return module


__all__ = ["build_experiment", "build_seqmaze_hrm_v2_model"]
