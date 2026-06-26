"""Model constructor for SeqMaze × HRM-v2."""

from __future__ import annotations

from ehc_sn.adapters.hrm import (
    SEQMAZE_HRM_Q_HALTING_TRACE_FIELDS,
    SeqMazeAdapterSettings,
    SeqMazeHRMV2BridgeAdapter,
    build_seqmaze_hrm_q_halting_trace_meta,
)
from ehc_sn.controllers.deliberation.q_halting import (
    DeliberationQHaltingController,
    DeliberationQHaltingControllerConfig,
)
from ehc_sn.lightning.modules.q_halting import (
    QHaltingBindings,
    QHaltingComponentConfigs,
    QHaltingConfig,
    QHaltingModule,
    QHaltingTrainingConfig,
)
from ehc_sn.metrics.routes.rl import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.models.hrm.hrm_v2 import HRModelV2, ModelSettingsV2
from ehc_sn.objectives.composites.hybrid_rl import (
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
from ehc_sn.tasks.seqmaze.supervision import build_seqmaze_supervision
from ehc_sn.traces.specs import HRM_HIDDEN_STATE_FIELDS
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.q_halting import (
    TD0QHaltingBatchBuilder,
    ZeroBootstrapQHaltingValidationScorer,
)

from .config import SeqMazeHRMV2ModelConfig


def build_seqmaze_hrm_v2_model(
    config: SeqMazeHRMV2ModelConfig,
    *,
    training_config: QHaltingTrainingConfig | None = None,
    execution: SeqMazeRuntimeConfig | None = None,
) -> QHaltingModule:
    """Construct an QHaltingModule for SeqMaze × HRM-v2.

    Parameters
    ----------
    config:
        Model configuration (components + architecture path).
    execution:
        Execution policy (runtime config). Required for both training and
        evaluation when the module is used with a Trainer.  Passed
        through to the module constructor.
    training_config:
        Training-only optimiser and reward config.  ``None`` during
        evaluation-only construction.  When provided, ``num_slots`` is
        read from ``training_config.num_slots`` in ``setup()``.
    """
    components: QHaltingComponentConfigs = config.components  # type: ignore[assignment]
    adapter_settings: SeqMazeAdapterSettings = components.adapter  # type: ignore[assignment]
    n_max = adapter_settings.n_max

    bindings = QHaltingBindings(
        model_cls=HRModelV2,
        model_settings_cls=ModelSettingsV2,
        adapter_cls=SeqMazeHRMV2BridgeAdapter,
        adapter_settings_cls=SeqMazeAdapterSettings,
        controller_cls=DeliberationQHaltingController,
        controller_config_cls=DeliberationQHaltingControllerConfig,
        objective_cls=HybridRLObjective,
        objective_config_cls=HybridRLLossConfig,
        learner_cls=TD0QHaltingBatchBuilder,
        val_scorer_cls=ZeroBootstrapQHaltingValidationScorer,
        optimizer_cls=AdamATan2,
        optimizer_config_cls=AdamATan2Config,
        runtime_config_cls=SeqMazeRuntimeConfig,
        reward_config_cls=SeqMazeRewardConfig,
        runtime_cls=SeqMazeRuntime,
        reward_projector_cls=SeqMazeRewardProjector,
        trace_fields=SEQMAZE_HRM_Q_HALTING_TRACE_FIELDS,
        build_trace_meta_fn=build_seqmaze_hrm_q_halting_trace_meta,
        step_routes=RL_STEP_ROUTES,
        episode_routes=RL_EPISODE_ROUTES,
        hidden_state_fields=HRM_HIDDEN_STATE_FIELDS,
        supervision_builder=build_seqmaze_supervision,
        token_weight_builder=None,
    )
    return QHaltingModule(
        config=QHaltingConfig(
            model_config_path=config.model_config_path,
            halt_disabled_steps=(
                training_config.halt_disabled_steps if training_config else 0
            ),
        ),
        component_configs=components,
        bindings=bindings,
        execution=execution,
        training_config=training_config,
    )


__all__ = ["build_seqmaze_hrm_v2_model"]
