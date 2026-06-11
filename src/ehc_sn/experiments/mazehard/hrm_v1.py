"""MazeHard + HRM v1 experiment (ACT-supervised).

Composition:
    model: HRModelV1
    adapter: MazeHardHRMV1BridgeAdapter
    controller: ACTController
    objective: ACTObjective
    regime module: ACTSupervisedModule
"""

from __future__ import annotations

from typing import Any

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
)
from ehc_sn.metrics.routes.act import ACT_EPISODE_ROUTES, ACT_STEP_ROUTES
from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettingsV1
from ehc_sn.objectives.act import ACTObjective, ACTObjectiveConfig
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2, AdamATan2Config


def build_experiment(raw_config: dict) -> ACTSupervisedModule:
    """Parse config and return an instantiated ACTSupervisedModule.

    Args:
        raw_config: Flat TOML-like mapping with keys matching
            ``ACTSupervisedConfig`` fields.

    Returns:
        Instantiated :class:`~ehc_sn.lightning.modules.act_supervised.ACTSupervisedModule`.
    """
    field_names = set(ACTSupervisedConfig.model_fields)
    filtered = {k: v for k, v in raw_config.items() if k in field_names}
    config = ACTSupervisedConfig(**filtered)
    component_configs = ACTSupervisedComponentConfigs(
        adapter=MazeHardHRMAdapterSettings.model_validate(
            raw_config["adapter"]
        ),
        controller=ACTControllerConfig.model_validate(raw_config["controller"]),
        objective=ACTObjectiveConfig.model_validate(raw_config["objective"]),
        optimizer=AdamATan2Config.model_validate(raw_config["optimizer"]),
        runtime=HRMRuntimeConfig.model_validate(raw_config["runtime"]),
    )
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
        config=config,
        component_configs=component_configs,
        bindings=bindings,
    )


__all__ = ["build_experiment"]
