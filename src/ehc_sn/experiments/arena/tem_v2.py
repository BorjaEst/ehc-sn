"""Arena + TEM v2 experiment.

Composition:
    model: TEMModelV2
    adapter: ArenaTEMV2BridgeAdapter
    controller: ReplayTrajectoryController
    objective: TEMObjective
    regime module: VariationalReplayModule
"""

from __future__ import annotations

from typing import Any

from ehc_sn.adapters.tem import (
    ARENA_TEM_TRACE_FIELDS,
    ArenaTEMAdapterSettings,
    ArenaTEMTaskBinding,
    ArenaTEMV2BridgeAdapter,
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
from ehc_sn.models.tem.tem_v2 import ModelSettingsV2, TEMModelV2
from ehc_sn.objectives.tem import TEMObjectiveConfig
from ehc_sn.tasks.arena.capabilities.replay import ArenaReplayCapability


def build_experiment(raw_config: dict) -> VariationalReplayModule:
    """Parse config and return an instantiated VariationalReplayModule.

    Args:
        raw_config: Flat TOML-like mapping with keys matching
            ``VariationalReplayConfig`` fields.

    Returns:
        Instantiated :class:`~ehc_sn.lightning.modules.variational_replay.VariationalReplayModule`.
    """
    field_names = set(VariationalReplayConfig.model_fields)
    filtered = {k: v for k, v in raw_config.items() if k in field_names}
    config = VariationalReplayConfig(**filtered)
    component_configs = VariationalReplayComponentConfigs(
        adapter=ArenaTEMAdapterSettings.model_validate(raw_config["adapter"]),
        controller=ReplayTrajectoryControllerConfig.model_validate(
            raw_config["controller"]
        ),
        objective=TEMObjectiveConfig.model_validate(raw_config["objective"]),
    )
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
        config=config,
        component_configs=component_configs,
        bindings=bindings,
    )


__all__ = ["build_experiment"]
