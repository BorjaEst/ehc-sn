"""Configuration schemas for the SeqMaze × HRM-v1 experiment pairing.

Hierarchy (training):

    SeqMazeHRMV1TrainingExperimentConfig
    ├── model: SeqMazeHRMV1ModelConfig
    │   └── components: SeqMazeHRMV1ComponentConfigs
    │       ├── adapter: SeqMazeAdapterSettings
    │       ├── controller: ACTControllerConfig
    │       └── objective: ACTSupervisedScorerConfig
    ├── training: ACTSupervisedTrainingConfig
    ├── data: DatamoduleConfig
    ├── trainer: TrainerConfig
    ├── checkpointing: CheckpointingConfig
    └── logging: LoggerSettings

Hierarchy (evaluation):

    SeqMazeHRMV1EvaluationExperimentConfig
    └── model: SeqMazeHRMV1ModelConfig (same as above)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import SeqMazeAdapterSettings
from ehc_sn.controllers.deliberation.act import ACTControllerConfig
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.experiments._infra import (
    CaptureConfig,
    CheckpointingConfig,
    ProviderConfig,
    RegimeConfig,
    TrainerConfig,
)
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedConfig,
    ACTSupervisedTrainingConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.composites.act import ACTSupervisedScorerConfig
from ehc_sn.objectives.task.seqmaze import SeqMazeTaskEvaluatorConfig
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig

# =============================================================================
# Component configuration (task–model binding)
# =============================================================================


class SeqMazeHRMV1ComponentConfigs(BaseModel, extra="forbid"):
    """Component-level config for SeqMaze × HRM-v1."""

    adapter: SeqMazeAdapterSettings = Field(
        ...,
        description="SeqMaze adapter settings for HRM v1.",
    )
    controller: ACTControllerConfig = Field(
        ...,
        description="ACT deliberation controller configuration.",
    )
    task_evaluator: SeqMazeTaskEvaluatorConfig = Field(
        ...,
        description="SeqMaze task-evaluator configuration.",
    )
    scorer: ACTSupervisedScorerConfig = Field(
        default_factory=ACTSupervisedScorerConfig,
        description="ACT loss composition configuration.",
    )


# =============================================================================
# Model configuration (computational structure only)
# =============================================================================


class SeqMazeHRMV1ModelConfig(BaseModel, extra="forbid"):
    """Model-level config for SeqMaze × HRM-v1."""

    model_config_path: Path = Field(
        ...,
        description="Path to the HRM v1 model architecture TOML file.",
    )
    components: SeqMazeHRMV1ComponentConfigs = Field(
        ...,
        description="Task-family component binding (adapter, controller, objective).",
    )


# =============================================================================
# Experiment-level configurations
# =============================================================================


class SeqMazeHRMV1TrainingExperimentConfig(BaseModel, extra="forbid"):
    """Full training application configuration for SeqMaze × HRM-v1."""

    model: SeqMazeHRMV1ModelConfig = Field(
        ...,
        description="Model structure (components + architecture path).",
    )
    regime: ACTSupervisedConfig = Field(
        default_factory=ACTSupervisedConfig,
        description="ACT regime configuration (halt_disabled_steps, "
        "target_network).",
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
    use_token_weights: bool = Field(
        default=False,
        description="Whether to apply task-provided per-token weights when "
        "computing token supervision loss.  Passed through to "
        "ACTSupervisedTrainingConfig for the ACT rollout path.",
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Lightning Trainer settings.",
    )
    checkpointing: CheckpointingConfig = Field(
        default_factory=CheckpointingConfig,
        description="Checkpoint, weight-init, and diagnostic settings.",
    )
    logging: Optional[LoggerSettings] = Field(
        default=None,
        description="TensorBoard logger settings.",
    )


class SeqMazeHRMV1EvaluationExperimentConfig(BaseModel, extra="forbid"):
    """Full evaluation application configuration for SeqMaze × HRM-v1."""

    model: SeqMazeHRMV1ModelConfig = Field(
        ...,
        description="Model structure (components only).",
    )
    execution: Optional[HRMRuntimeConfig] = Field(
        default=None,
        description="Execution policy for eval-time rollout bounds.",
    )
    provider: ProviderConfig = Field(
        ...,
        description="Evaluation data provider specification.",
    )
    regime: RegimeConfig = Field(
        ...,
        description="Evaluation regime identity.",
    )
    capture: CaptureConfig = Field(
        default_factory=lambda: CaptureConfig(),
        description="Trace capture policy.",
    )


# =============================================================================
__all__ = [
    "SeqMazeHRMV1ComponentConfigs",
    "SeqMazeHRMV1ModelConfig",
    "SeqMazeHRMV1TrainingExperimentConfig",
    "SeqMazeHRMV1EvaluationExperimentConfig",
]
