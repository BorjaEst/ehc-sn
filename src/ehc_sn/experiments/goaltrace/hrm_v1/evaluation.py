"""Evaluation executor assembly for Goaltrace × HRM-v1."""

from __future__ import annotations

from ehc_sn.lightning.modules.act_supervised import ACTSupervisedModule

from .config import GoaltraceHRMV1EvaluationExperimentConfig
from .model import build_goaltrace_hrm_v1_model


def build_goaltrace_hrm_v1_evaluation_executor(
    config: GoaltraceHRMV1EvaluationExperimentConfig,
) -> ACTSupervisedModule:
    """Build an evaluation-only executor for Goaltrace × HRM-v1.

    Args:
        config: Evaluation configuration (model only, no training config).

    Returns:
        Instantiated ACTSupervisedModule in eval mode.
    """
    return build_goaltrace_hrm_v1_model(
        config.model,
        execution=config.execution,
    )


__all__ = ["build_goaltrace_hrm_v1_evaluation_executor"]
