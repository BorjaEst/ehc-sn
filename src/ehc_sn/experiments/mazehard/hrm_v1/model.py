"""Model constructor for MazeHard × HRM-v1.

Shared by both training and evaluation paths.
Constructs the ACTSupervisedModule with ``training_config=None``,
so the module is usable for inference without optimizers.
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

from .config import MazeHardHRMV1ModelConfig


def build_mazehard_hrm_v1_model(
    config: MazeHardHRMV1ModelConfig,
    *,
    runtime: HRMRuntimeConfig | None = None,
) -> ACTSupervisedModule:
    """Construct an ACTSupervisedModule for MazeHard × HRM-v1.

    No training config is provided — the returned module is suitable for
    both training (training config added by the training experiment builder)
    and evaluation (training config remains ``None``).

    Args:
        config: Model-level config (components only).
        runtime: Optional runtime config for eval-time rollout bounds.
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
            global_batch_size=1,
        ),
        component_configs=components,
        bindings=bindings,
        runtime=runtime,
    )


def build_experiment(
    config: ACTSupervisedConfig,
    components: ACTSupervisedComponentConfigs,
    training: ACTSupervisedTrainingConfig | None = None,
    *,
    runtime: HRMRuntimeConfig | None = None,
) -> ACTSupervisedModule:
    """Backward-compat builder — delegates to ``build_mazehard_hrm_v1_model``.

    Matches the legacy typed signature used by ``artifacts.py``.
    """
    from .config import MazeHardHRMV1ComponentConfigs, MazeHardHRMV1ModelConfig

    model_config = MazeHardHRMV1ModelConfig(
        model_config_path=config.model_config_path,
        components=MazeHardHRMV1ComponentConfigs(
            adapter=components.adapter,
            controller=components.controller,
            objective=components.objective,
        ),
    )
    module = build_mazehard_hrm_v1_model(model_config, runtime=runtime)
    if training is not None:
        module._training_config = training
    return module


__all__ = ["build_experiment", "build_mazehard_hrm_v1_model"]
