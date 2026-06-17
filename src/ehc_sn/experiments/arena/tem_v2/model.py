"""Model constructor for Arena × TEM-v2."""

from __future__ import annotations

from ehc_sn.adapters.tem import (
    ARENA_TEM_TRACE_FIELDS,
    ArenaTEMAdapterSettings,
    ArenaTEMTaskBinding,
    ArenaTEMV2BridgeAdapter,
    build_arena_tem_trace_meta,
)
from ehc_sn.lightning.modules.variational_replay import (
    TEMTrainingConfig,
    VariationalReplayBindings,
    VariationalReplayComponentConfigs,
    VariationalReplayConfig,
    VariationalReplayModule,
)
from ehc_sn.models.tem.tem_v2 import ModelSettingsV2, TEMModelV2
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.tem import RuntimeConfig as TEMRuntimeConfig

from .config import ArenaTEMV2ModelConfig


def build_arena_tem_v2_model(
    config: ArenaTEMV2ModelConfig,
    *,
    training_config: TEMTrainingConfig | None = None,
    execution: TEMRuntimeConfig | None = None,
    scheduler: SchedulerConfig | None = None,
) -> VariationalReplayModule:
    components: VariationalReplayComponentConfigs = config.components  # type: ignore[assignment]

    bindings = VariationalReplayBindings(
        model_cls=TEMModelV2,
        model_settings_cls=ModelSettingsV2,
        adapter_cls=ArenaTEMV2BridgeAdapter,
        adapter_settings_cls=ArenaTEMAdapterSettings,
        trace_fields=ARENA_TEM_TRACE_FIELDS,
        build_trace_meta_fn=build_arena_tem_trace_meta,
        replay_runtime_factory=ArenaReplayCapability,
        task_binding_factory=ArenaTEMTaskBinding,
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


__all__ = ["build_arena_tem_v2_model"]
