"""Model constructor for Arena × TEM-v1."""

from __future__ import annotations

from ehc_sn.adapters.tem import (
    ARENA_TEM_TRACE_FIELDS,
    ArenaTEMAdapterSettings,
    ArenaTEMV1BridgeAdapter,
    build_arena_tem_trace_meta,
)
from ehc_sn.lightning.modules.variational_replay import (
    TEMTrainingConfig,
    VariationalReplayBindings,
    VariationalReplayComponentConfigs,
    VariationalReplayConfig,
    VariationalReplayModule,
)
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMModelV1
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability
from ehc_sn.tasks.arena.supervision import (
    build_arena_supervision,
    build_arena_tem_objective_supervision,
)
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.tem import RuntimeConfig as TEMRuntimeConfig

from .config import ArenaTEMV1ModelConfig


def build_arena_tem_v1_model(
    config: ArenaTEMV1ModelConfig,
    *,
    training_config: TEMTrainingConfig | None = None,
    execution: TEMRuntimeConfig | None = None,
    scheduler: SchedulerConfig | None = None,
) -> VariationalReplayModule:
    """Construct a VariationalReplayModule for Arena × TEM-v1."""
    components: VariationalReplayComponentConfigs = config.components  # type: ignore[assignment]

    bindings = VariationalReplayBindings(
        model_cls=TEMModelV1,
        model_settings_cls=ModelSettingsV1,
        adapter_cls=ArenaTEMV1BridgeAdapter,
        adapter_settings_cls=ArenaTEMAdapterSettings,
        trace_fields=ARENA_TEM_TRACE_FIELDS,
        build_trace_meta_fn=build_arena_tem_trace_meta,
        replay_runtime_factory=ArenaReplayCapability,
        supervision_builder=build_arena_supervision,
        scoring_supervision_builder=build_arena_tem_objective_supervision,
    )
    return VariationalReplayModule(
        config=VariationalReplayConfig(
            model_config_path=config.model_config_path,
            scheduler=scheduler or SchedulerConfig(),
        ),
        component_configs=components,
        bindings=bindings,
        training_config=training_config,
        execution=execution,
    )


__all__ = ["build_arena_tem_v1_model"]
