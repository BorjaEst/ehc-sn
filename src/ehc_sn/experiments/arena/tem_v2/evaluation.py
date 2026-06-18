"""Evaluation executor assembly for Arena × TEM-v2."""

from __future__ import annotations

from ehc_sn.lightning.modules.variational_replay import (
    VariationalReplayModule,
)

from .config import ArenaTEMV2EvaluationExperimentConfig
from .model import build_arena_tem_v2_model


def build_arena_tem_v2_evaluation_executor(
    config: ArenaTEMV2EvaluationExperimentConfig,
) -> VariationalReplayModule:
    return build_arena_tem_v2_model(
        config.model,
        execution=config.execution,
    )


__all__ = ["build_arena_tem_v2_evaluation_executor"]
