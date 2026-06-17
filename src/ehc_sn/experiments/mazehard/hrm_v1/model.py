"""Model constructor for MazeHard × HRM-v1.

Shared by both training and evaluation paths.
"""

from __future__ import annotations

from ehc_sn.adapters.hrm import (
    MAZE_HARD_HRM_ACT_TRACE_FIELDS,
    MazeHardHRMAdapterSettings,
    MazeHardHRMV1ACTTaskBinding,
    MazeHardHRMV1BridgeAdapter,
    build_mazehard_hrm_trace_meta,
)
from ehc_sn.controllers.deliberation.act import (
    ACTController,
    ACTControllerConfig,
)
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedBindings,
    ACTSupervisedComponentConfigs,
    ACTSupervisedConfig,
    ACTSupervisedModule,
    ACTSupervisedTrainingConfig,
)
from ehc_sn.metrics.routes.act import ACT_EPISODE_ROUTES, ACT_STEP_ROUTES
from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettingsV1
from ehc_sn.objectives.act import ACTObjective, ACTObjectiveConfig
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.stabilization import TargetNetworkConfig

from .config import MazeHardHRMV1ModelConfig


def build_mazehard_hrm_v1_model(
    config: MazeHardHRMV1ModelConfig,
    *,
    training_config: ACTSupervisedTrainingConfig | None = None,
    execution: HRMRuntimeConfig | None = None,
    scheduler: SchedulerConfig | None = None,
    supervised_only_warmup_steps: int | None = None,
    target_network: TargetNetworkConfig | None = None,
) -> ACTSupervisedModule:
    """Construct an ACTSupervisedModule for MazeHard × HRM-v1.

    Training config and execution are both constructor parameters —
    no post-construction mutation needed.  When ``training_config`` is
    provided the module is training-capable; when ``None`` it is safe
    for evaluation without an optimizer.

    Args:
        config: Model-level config (components only).
        training_config: Training-only settings (optimizer, runtime).
            ``None`` during evaluation-only construction.
        execution: Execution policy for eval-time rollout bounds.
            Used only when ``training_config`` is ``None``.
    """
    components: ACTSupervisedComponentConfigs = config.components  # type: ignore[assignment]

    bindings = ACTSupervisedBindings(
        model_cls=HRModelV1,
        model_settings_cls=ModelSettingsV1,
        adapter_cls=MazeHardHRMV1BridgeAdapter,
        adapter_settings_cls=MazeHardHRMAdapterSettings,
        controller_cls=ACTController,
        controller_config_cls=ACTControllerConfig,
        objective_cls=ACTObjective,
        objective_config_cls=ACTObjectiveConfig,
        task_binding_cls=MazeHardHRMV1ACTTaskBinding,
        optimizer_cls=AdamATan2,
        optimizer_config_cls=AdamATan2Config,
        trace_fields=MAZE_HARD_HRM_ACT_TRACE_FIELDS,
        build_trace_meta_fn=build_mazehard_hrm_trace_meta,
        step_routes=ACT_STEP_ROUTES,
        episode_routes=ACT_EPISODE_ROUTES,
        hidden_state_fields=HRM_HIDDEN_STATE_FIELDS,
    )
    return ACTSupervisedModule(
        config=ACTSupervisedConfig(
            model_config_path=config.model_config_path,
            scheduler=scheduler or SchedulerConfig(),
            supervised_only_warmup_steps=supervised_only_warmup_steps or 0,
            target_network=target_network or TargetNetworkConfig(),
        ),
        component_configs=components,
        bindings=bindings,
        training_config=training_config,
        execution=execution,
    )


__all__ = ["build_mazehard_hrm_v1_model"]
