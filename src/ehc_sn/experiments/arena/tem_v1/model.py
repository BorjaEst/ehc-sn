"""Model constructor for Arena × TEM-v1."""

from __future__ import annotations

from ehc_sn.adapters.tem import (
    ARENA_TEM_TRACE_FIELDS,
    ArenaTEMAdapterSettings,
    ArenaTEMTaskBinding,
    ArenaTEMV1BridgeAdapter,
    build_arena_tem_trace_meta,
)
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.lightning.modules.variational_replay import (
    VariationalReplayBindings,
    VariationalReplayComponentConfigs,
    VariationalReplayConfig,
    VariationalReplayModule,
)
from ehc_sn.models.tem.tem_v1 import ModelSettingsV1, TEMModelV1
from ehc_sn.objectives.tem import TEMObjectiveConfig
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability

from .config import ArenaTEMV1ModelConfig


def build_arena_tem_v1_model(
    config: ArenaTEMV1ModelConfig,
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
        task_binding_factory=ArenaTEMTaskBinding,
    )
    return VariationalReplayModule(
        config=VariationalReplayConfig(
            model_config_path=config.model_config_path,
        ),
        component_configs=components,
        bindings=bindings,
    )


def build_experiment(
    config: VariationalReplayConfig,
    components: VariationalReplayComponentConfigs,
) -> VariationalReplayModule:
    """Backward-compat builder — delegates to ``build_arena_tem_v1_model``."""
    return build_arena_tem_v1_model(
        ArenaTEMV1ModelConfig(
            model_config_path=config.model_config_path,
            components=type(ArenaTEMV1ModelConfig.model_fields["components"])(
                adapter=components.adapter,
                controller=components.controller,
                objective=components.objective,
            ),
        )
    )


__all__ = ["build_experiment", "build_arena_tem_v1_model"]
