"""Model constructor for Goaltrace × HRM-v1.

Shared by both training and evaluation paths.
"""

from __future__ import annotations

from ehc_sn.adapters.hrm import (
    GoaltraceAdapterSettings,
    GoaltraceHRMV1ACTTaskBinding,
    GoaltraceHRMV1BridgeAdapter,
    build_goaltrace_hrm_trace_meta,
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
from ehc_sn.objectives.continuous_field import (
    ContinuousFieldObjective,
    ContinuousFieldObjectiveConfig,
)
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.stabilization import TargetNetworkConfig

from .config import GoaltraceHRMV1ModelConfig


def build_goaltrace_hrm_v1_model(
    config: GoaltraceHRMV1ModelConfig,
    *,
    training_config: ACTSupervisedTrainingConfig | None = None,
    execution: HRMRuntimeConfig | None = None,
    scheduler: SchedulerConfig | None = None,
    supervised_only_warmup_steps: int | None = None,
    target_network: TargetNetworkConfig | None = None,
    simple_supervised: bool = True,
) -> ACTSupervisedModule:
    """Construct an ACTSupervisedModule for Goaltrace × HRM-v1.

    Uses the generic ``ACTSupervisedModule`` — ``_training_step_simple``
    delegates loss computation to the bound objective, which handles
    MSE field loss (``ContinuousFieldObjective``) naturally.

    Args:
        config: Model-level config (components only).
        training_config: Training-only settings (optimizer, runtime).
            ``None`` during evaluation-only construction.
        execution: Execution policy for eval-time rollout bounds.
            Used only when ``training_config`` is ``None``.
        simple_supervised: When True (default), bypasses ACT rollout
            and uses single-pass training with objective-delegated loss.
    """
    components: ACTSupervisedComponentConfigs = config.components  # type: ignore[assignment]

    bindings = ACTSupervisedBindings(
        model_cls=HRModelV1,
        model_settings_cls=ModelSettingsV1,
        adapter_cls=GoaltraceHRMV1BridgeAdapter,
        adapter_settings_cls=GoaltraceAdapterSettings,
        controller_cls=ACTController,
        controller_config_cls=ACTControllerConfig,
        objective_cls=ContinuousFieldObjective,
        objective_config_cls=ContinuousFieldObjectiveConfig,
        task_binding_cls=GoaltraceHRMV1ACTTaskBinding,
        optimizer_cls=AdamATan2,
        optimizer_config_cls=AdamATan2Config,
        trace_fields=(),
        build_trace_meta_fn=build_goaltrace_hrm_trace_meta,
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
            simple_supervised=simple_supervised,
        ),
        component_configs=components,
        bindings=bindings,
        training_config=training_config,
        execution=execution,
    )


__all__ = [
    "build_goaltrace_hrm_v1_model",
]
