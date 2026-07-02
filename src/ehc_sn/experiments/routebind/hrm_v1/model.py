"""Model constructor for Routebind × HRM-v1.

Shared by both training and evaluation paths.
"""

from __future__ import annotations

from ehc_sn.adapters.hrm import (
    ROUTEBIND_HRM_ACT_TRACE_FIELDS,
    RoutebindHRMAdapterSettings,
    RoutebindHRMV1BridgeAdapter,
    build_routebind_hrm_trace_meta,
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
from ehc_sn.metrics.routes.continuous_field import (
    CONTINUOUS_FIELD_EPISODE_ROUTES,
    CONTINUOUS_FIELD_STEP_ROUTES,
)
from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettingsV1
from ehc_sn.objectives.composites.act import (
    ACTSupervisedScorer,
)
from ehc_sn.objectives.task.routebind import (
    RoutebindTrajectoryBootstrapEvaluator,
)
from ehc_sn.targets.halt import (
    FieldQualityHaltTarget,
    FieldQualityHaltTargetConfig,
    HaltTargetBuilder,
)
from ehc_sn.tasks.routebind.supervision import build_routebind_supervision
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2, AdamATan2Config

from .config import RoutebindHRMV1ModelConfig


def build_routebind_hrm_v1_model(
    config: RoutebindHRMV1ModelConfig,
    *,
    regime_config: ACTSupervisedConfig | None = None,
    training_config: ACTSupervisedTrainingConfig | None = None,
    execution: HRMRuntimeConfig | None = None,
) -> ACTSupervisedModule:
    """Construct an ACTSupervisedModule for Routebind × HRM-v1.

    Uses the generic ``ACTSupervisedModule`` with field-modality scoring.
    The routebind branch in `_build_step_input` detects
    ``target_trajectory`` / ``cell_mask`` on the supervision struct and
    uses ``trajectory_field`` as the prediction.

    Args:
        config: Model-level config (components only).
        regime_config: ACT regime configuration (halt_disabled_steps,
            target_network, single_step).  ``None`` during evaluation —
            safe defaults are used.
        training_config: Training-only settings (optimizer, runtime).
            ``None`` during evaluation-only construction.
        execution: Execution policy for evaluation-time rollout bounds.
            Used only when ``training_config`` is ``None``.
    """
    components: ACTSupervisedComponentConfigs = config.components

    bindings = ACTSupervisedBindings(
        model_cls=HRModelV1,
        model_settings_cls=ModelSettingsV1,
        adapter_cls=RoutebindHRMV1BridgeAdapter,
        adapter_settings_cls=RoutebindHRMAdapterSettings,
        controller_cls=ACTController,
        controller_config_cls=ACTControllerConfig,
        task_evaluator_cls=RoutebindTrajectoryBootstrapEvaluator,
        task_evaluator_config_cls=object,
        optimizer_cls=AdamATan2,
        optimizer_config_cls=AdamATan2Config,
        trace_fields=ROUTEBIND_HRM_ACT_TRACE_FIELDS,
        build_trace_meta_fn=build_routebind_hrm_trace_meta,
        step_routes=CONTINUOUS_FIELD_STEP_ROUTES,
        episode_routes=CONTINUOUS_FIELD_EPISODE_ROUTES,
        hidden_state_fields=HRM_HIDDEN_STATE_FIELDS,
        supervision_builder=build_routebind_supervision,
    )
    _regime = (
        regime_config if regime_config is not None else ACTSupervisedConfig()
    )
    return ACTSupervisedModule(
        config=_regime.model_copy(
            update={"model_config_path": config.model_config_path},
        ),
        component_configs=components,
        bindings=bindings,
        training_config=training_config,
        execution=execution,
    )


__all__ = [
    "build_routebind_hrm_v1_model",
]
