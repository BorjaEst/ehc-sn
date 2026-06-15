"""Evaluation executor assembly for Arena × TEM-v1."""

from __future__ import annotations

from ehc_sn.lightning.modules.variational_replay import (
    VariationalReplayModule,
)

from .config import ArenaTEMV1EvaluationExperimentConfig
from .model import build_arena_tem_v1_model


def build_arena_tem_v1_evaluation_executor(
    config: ArenaTEMV1EvaluationExperimentConfig,
) -> VariationalReplayModule:
    return build_arena_tem_v1_model(config.model)


__all__ = ["build_arena_tem_v1_evaluation_executor"]
