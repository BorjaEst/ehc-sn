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
from ehc_sn.experiments._infra import CheckpointingConfig, TrainerConfig
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedTrainingConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.composites.act import ACTSupervisedScorerConfig
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.stabilization import TargetNetworkConfig

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
    objective: ACTSupervisedScorerConfig = Field(
        ...,
        description="ACTSupervisedScorer configuration "
        "(bypassed when single_step=true).",
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
    training: ACTSupervisedTrainingConfig = Field(
        ...,
        description="ACT supervised training configuration.",
    )
    execution: Optional[HRMRuntimeConfig] = Field(
        default=None,
        description="Execution policy (validation safety limits).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    supervised_only_warmup_steps: int = Field(
        default=0,
        description="Optimizer steps with learned halting disabled. "
        "Ignored when single_step=true.",
    )
    single_step: bool = Field(
        default=True,
        description="Bypass ACT rollout and use a single forward pass with "
        "MSE field loss.  Set to false after implementing ACT deliberation "
        "for goaltrace.",
    )
    target_network: TargetNetworkConfig = Field(
        default_factory=TargetNetworkConfig,
        description="Optional EMA-lagged target network config.",
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


class GoaltraceHRMV1EvaluationExperimentConfig(BaseModel, extra="forbid"):
    """Full evaluation application configuration for Goaltrace × HRM-v1."""

    model: GoaltraceHRMV1ModelConfig = Field(
        ...,
        description="Model structure (components only).",
    )
    execution: Optional[HRMRuntimeConfig] = Field(
        default=None,
        description="Execution policy for eval-time rollout bounds.",
    )


__all__ = [
    "GoaltraceHRMV1ComponentConfigs",
    "GoaltraceHRMV1EvaluationExperimentConfig",
    "GoaltraceHRMV1ModelConfig",
    "GoaltraceHRMV1TrainingExperimentConfig",
]
