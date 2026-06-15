"""Evaluation executor assembly for MazeHard × HRM-v1.

Uses the shared model builder.  No training config is attached.
"""

from __future__ import annotations

from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedModule,
)
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig

from .config import MazeHardHRMV1EvaluationExperimentConfig
from .model import build_mazehard_hrm_v1_model


def build_mazehard_hrm_v1_evaluation_executor(
    config: MazeHardHRMV1EvaluationExperimentConfig,
    *,
    runtime: HRMRuntimeConfig | None = None,
) -> ACTSupervisedModule:
    """Build an evaluation executor for MazeHard × HRM-v1.

    The returned module has ``_training_config is None`` — it is safe
    to use for inference without constructing optimizers.
    """
    return build_mazehard_hrm_v1_model(config.model, runtime=runtime)


__all__ = ["build_mazehard_hrm_v1_evaluation_executor"]
