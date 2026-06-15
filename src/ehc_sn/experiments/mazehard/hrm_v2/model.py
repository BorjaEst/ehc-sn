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

from .config import MazeHardHRMV2ModelConfig


def build_mazehard_hrm_v2_model(
    config: MazeHardHRMV2ModelConfig,
) -> ActorCriticModule:
    """Construct an ActorCriticModule for MazeHard × HRM-v2.

    No training config — the returned module is safe for eval.
    Training config is attached by the training experiment builder.
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
            global_batch_size=1,
        ),
        component_configs=components,
        bindings=bindings,
    )


def build_experiment(
    config: ActorCriticConfig,
    components: ActorCriticComponentConfigs,
    deliberation: MazeHardRuntimeConfig,
    training: ActorCriticTrainingConfig | None = None,
) -> ActorCriticModule:
    """Backward-compat builder — delegates to ``build_mazehard_hrm_v2_model``.

    Matches the legacy typed signature used by ``artifacts.py``.
    """
    from .config import MazeHardHRMV2ComponentConfigs, MazeHardHRMV2ModelConfig

    model_config = MazeHardHRMV2ModelConfig(
        model_config_path=config.model_config_path,
        components=MazeHardHRMV2ComponentConfigs(
            adapter=components.adapter,
            controller=components.controller,
            objective=components.objective,
        ),
    )
    module = build_mazehard_hrm_v2_model(model_config)
    if training is not None:
        module._deliberation = deliberation
    return module


__all__ = ["build_experiment", "build_mazehard_hrm_v2_model"]
