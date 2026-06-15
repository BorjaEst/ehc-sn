"""Evaluation executor assembly for MazeHard × HRM-v2."""

from __future__ import annotations

from ehc_sn.lightning.modules.actor_critic import ActorCriticModule

from .config import MazeHardHRMV2EvaluationExperimentConfig
from .model import build_mazehard_hrm_v2_model


def build_mazehard_hrm_v2_evaluation_executor(
    config: MazeHardHRMV2EvaluationExperimentConfig,
) -> ActorCriticModule:
    """Build an evaluation executor for MazeHard × HRM-v2.

    The returned module has ``_training_config is None``.
    """
    return build_mazehard_hrm_v2_model(config.model)


__all__ = ["build_mazehard_hrm_v2_evaluation_executor"]
