"""Model constructor for SeqMaze × HRM-v1."""

from __future__ import annotations

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
    ACTSupervisedTrainingConfig,
)
from ehc_sn.metrics.routes.act import ACT_EPISODE_ROUTES, ACT_STEP_ROUTES
from ehc_sn.models.hrm.hrm_v1 import HRModelV1, ModelSettingsV1
from ehc_sn.objectives.act import ACTObjective, ACTObjectiveConfig
from ehc_sn.tasks.seqmaze.evaluation import SeqMazeValidationScorer
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.stabilization import TargetNetworkConfig

from .config import SeqMazeHRMV1ModelConfig


def build_seqmaze_hrm_v1_model(
    config: SeqMazeHRMV1ModelConfig,
    *,
    training_config: ACTSupervisedTrainingConfig | None = None,
    execution: HRMRuntimeConfig | None = None,
    scheduler: SchedulerConfig | None = None,
    supervised_only_warmup_steps: int | None = None,
    target_network: TargetNetworkConfig | None = None,
) -> ACTSupervisedModule:
    """Construct an ACTSupervisedModule for SeqMaze × HRM-v1.

    Training config and execution are both constructor parameters —
    no post-construction mutation needed.

    Args:
        config: Model-level configuration (components + architecture path).
        training_config: Training-only settings (optimizer, runtime).
            ``None`` during evaluation-only construction.
        execution: Execution policy for eval-time rollout bounds.
            Used only when ``training_config`` is ``None``.

    Returns:
        Instantiated ACTSupervisedModule.
    """
    components: ACTSupervisedComponentConfigs = config.components  # type: ignore[assignment]
    adapter_settings: SeqMazeAdapterSettings = components.adapter  # type: ignore[assignment]
    n_max = adapter_settings.n_max
    t_max = adapter_settings.t_max
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


__all__ = ["build_seqmaze_hrm_v1_model"]
