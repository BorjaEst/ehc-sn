"""Configuration schemas for the Goaltrace × HRM-v1 experiment pairing.

Hierarchy (training):

    GoaltraceHRMV1TrainingExperimentConfig
    ├── model: GoaltraceHRMV1ModelConfig
    │   └── components: GoaltraceHRMV1ComponentConfigs
    │       ├── adapter: GoaltraceHRMAdapterSettings
    │       ├── controller: ACTControllerConfig
    │       └── objective: ACTSupervisedScorerConfig
    ├── training: ACTSupervisedTrainingConfig
    ├── data: DatamoduleConfig
    ├── trainer: TrainerConfig
    ├── checkpointing: CheckpointingConfig
    └── logging: LoggerSettings
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import GoaltraceHRMAdapterSettings
from ehc_sn.controllers.deliberation.act import ACTControllerConfig
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.evaluation.invocation import EvaluationOptions
from ehc_sn.experiments._infra import (
    CheckpointingConfig,
    TrainerConfig,
)
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedConfig,
    ACTSupervisedTrainingConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.composites.act import ACTSupervisedScorerConfig
from ehc_sn.objectives.task.goaltrace import GoaltraceTaskEvaluatorConfig
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig

# =============================================================================
# Component configuration (task–model binding)
# =============================================================================


class GoaltraceHRMV1ComponentConfigs(BaseModel, extra="forbid"):
    """Component-level config for Goaltrace × HRM-v1."""

    adapter: GoaltraceHRMAdapterSettings = Field(
        ...,
        description="Goaltrace adapter settings for HRM v1.",
    )
    controller: ACTControllerConfig = Field(
        ...,
        description="ACT deliberation controller configuration "
        "(bypassed when single_step=true).",
    )
    task_evaluator: GoaltraceTaskEvaluatorConfig = Field(
        ...,
        description="Goaltrace task-evaluator configuration.",
    )
    scorer: ACTSupervisedScorerConfig = Field(
        default_factory=ACTSupervisedScorerConfig,
        description="ACT loss composition configuration.",
    )


# =============================================================================
# Model configuration (computational structure only)
# =============================================================================


class GoaltraceHRMV1ModelConfig(BaseModel, extra="forbid"):
    """Model-level config for Goaltrace × HRM-v1."""

    model_config_path: Path = Field(
        ...,
        description="Path to the HRM v1 model architecture TOML file.",
    )
    components: GoaltraceHRMV1ComponentConfigs = Field(
        ...,
        description="Task-family component binding "
        "(adapter, controller, objective).",
    )


# =============================================================================
# Experiment-level configurations
# =============================================================================


class GoaltraceHRMV1TrainingExperimentConfig(BaseModel, extra="forbid"):
    """Full training application configuration for Goaltrace × HRM-v1."""

    model: GoaltraceHRMV1ModelConfig = Field(
        ...,
        description="Model structure (components + architecture path).",
    )
    regime: ACTSupervisedConfig = Field(
        default_factory=ACTSupervisedConfig,
        description="ACT regime configuration (halt_disabled_steps, "
        "target_network, single_step).",
    )
    training: ACTSupervisedTrainingConfig = Field(
        ...,
        description="ACT supervised training configuration.",
    )
    execution: Optional[HRMRuntimeConfig] = Field(
        default=None,
        description="Execution policy (validation safety limits).",
    )
    data: DatamoduleConfig = Field(
        ...,
        description="Dataset and DataLoader settings.",
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Lightning Trainer settings.",
    )
    checkpointing: CheckpointingConfig = Field(
        default_factory=CheckpointingConfig,
        description="Checkpointing and diagnostic settings.",
    )
    logging: LoggerSettings = Field(
        default_factory=LoggerSettings,
        description="Logging and TensorBoard settings.",
    )


class GoaltraceHRMV1EvaluationOptions(EvaluationOptions):
    """Pair-specific evaluation options for Goaltrace × HRM-v1.

    Currently empty — all evaluation options inherited from recipe defaults.
    """

    pass


__all__ = [
    "GoaltraceHRMV1ComponentConfigs",
    "GoaltraceHRMV1EvaluationOptions",
    "GoaltraceHRMV1ModelConfig",
    "GoaltraceHRMV1TrainingExperimentConfig",
]
