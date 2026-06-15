"""Evaluation executor assembly for SeqMaze × HRM-v2."""

from __future__ import annotations

from ehc_sn.lightning.modules.actor_critic import ActorCriticModule

from .config import SeqMazeHRMV2EvaluationExperimentConfig
from .model import build_seqmaze_hrm_v2_model


def build_seqmaze_hrm_v2_evaluation_executor(
    config: SeqMazeHRMV2EvaluationExperimentConfig,
) -> ActorCriticModule:
    return build_seqmaze_hrm_v2_model(config.model)


__all__ = ["build_seqmaze_hrm_v2_evaluation_executor"]
