"""Evaluation executor assembly for MazeHard × HRM-v1.

Uses the shared model builder.  No training config is attached.
"""

from __future__ import annotations

from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedModule,
)

from .config import MazeHardHRMV1EvaluationExperimentConfig
from .model import build_mazehard_hrm_v1_model


def build_mazehard_hrm_v1_evaluation_executor(
    config: MazeHardHRMV1EvaluationExperimentConfig,
) -> ACTSupervisedModule:
    """Build an evaluation executor for MazeHard \u00d7 HRM-v1."""
    return build_mazehard_hrm_v1_model(
        config.model,
        execution=config.execution,
    )


__all__ = ["build_mazehard_hrm_v1_evaluation_executor"]
