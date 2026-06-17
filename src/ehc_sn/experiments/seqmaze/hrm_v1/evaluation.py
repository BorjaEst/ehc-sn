"""Evaluation executor assembly for SeqMaze × HRM-v1."""

from __future__ import annotations

from ehc_sn.lightning.modules.act_supervised import ACTSupervisedModule

from .config import SeqMazeHRMV1EvaluationExperimentConfig
from .model import build_seqmaze_hrm_v1_model


def build_seqmaze_hrm_v1_evaluation_executor(
    config: SeqMazeHRMV1EvaluationExperimentConfig,
) -> ACTSupervisedModule:
    return build_seqmaze_hrm_v1_model(
        config.model,
        execution=config.execution,
    )


__all__ = ["build_seqmaze_hrm_v1_evaluation_executor"]
