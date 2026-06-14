"""SeqMaze + HRM v1 experiment (ACT-supervised).

Composition:
    model: HRModelV1
    adapter: SeqMazeHRMV1BridgeAdapter
    controller: ACTController
    objective: ACTObjective
    regime module: ACTSupervisedModule
    validation scorer: SeqMazeValidationScorer
"""

from __future__ import annotations

from typing import Any

from ehc_sn.adapters.hrm import (
    SeqMazeAdapterSettings,
    SeqMazeHRMV1ACTTaskBinding,
    SeqMazeHRMV1BridgeAdapter,
    build_seqmaze_hrm_trace_meta,
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
from ehc_sn.tasks.seqmaze.evaluation import SeqMazeValidationScorer
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2, AdamATan2Config


def build_experiment(raw_config: dict) -> ACTSupervisedModule:
    """Parse config and return an instantiated ACTSupervisedModule.

    Args:
        raw_config: Flat TOML-like mapping with keys matching
            ``ACTSupervisedConfig`` fields and an ``[adapter]`` section.

    Returns:
        Instantiated :class:`~ehc_sn.lightning.modules.act_supervised.ACTSupervisedModule`.
    """
    field_names = set(ACTSupervisedConfig.model_fields)
    filtered = {k: v for k, v in raw_config.items() if k in field_names}
    config = ACTSupervisedConfig(**filtered)
    component_configs = ACTSupervisedComponentConfigs(
        adapter=SeqMazeAdapterSettings.model_validate(raw_config["adapter"]),
        controller=ACTControllerConfig.model_validate(raw_config["controller"]),
        objective=ACTObjectiveConfig.model_validate(raw_config["objective"]),
        optimizer=AdamATan2Config.model_validate(raw_config["optimizer"]),
        runtime=HRMRuntimeConfig.model_validate(raw_config["runtime"]),
    )

    # Derive vocabulary constants from adapter settings
    n_max = component_configs.adapter.n_max
    t_max = component_configs.adapter.t_max
    eos_id = n_max
    pad_id = n_max + 1

    bindings = ACTSupervisedBindings(
        model_cls=HRModelV1,
        model_settings_cls=ModelSettingsV1,
        adapter_cls=SeqMazeHRMV1BridgeAdapter,
        adapter_settings_cls=SeqMazeAdapterSettings,
        controller_cls=ACTController,
        controller_config_cls=ACTControllerConfig,
        objective_cls=ACTObjective,
        objective_config_cls=ACTObjectiveConfig,
        task_binding_cls=lambda: SeqMazeHRMV1ACTTaskBinding(n_max=n_max),
        optimizer_cls=AdamATan2,
        optimizer_config_cls=AdamATan2Config,
        trace_fields=(),
        build_trace_meta_fn=build_seqmaze_hrm_trace_meta,
        step_routes=ACT_STEP_ROUTES,
        episode_routes=ACT_EPISODE_ROUTES,
        hidden_state_fields=HRM_HIDDEN_STATE_FIELDS,
        task_scorer_factory=lambda: SeqMazeValidationScorer(
            eos_id=eos_id, pad_id=pad_id, n_max=n_max
        ),
    )
    return ACTSupervisedModule(
        config=config,
        component_configs=component_configs,
        bindings=bindings,
    )


__all__ = ["build_experiment"]
