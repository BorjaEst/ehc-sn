"""Configuration schemas for the Arena × TEM-v1 experiment pairing.

Hierarchy (training):

    ArenaTEMV1TrainingExperimentConfig
    ├── model: ArenaTEMV1ModelConfig
    │   └── components: ArenaTEMV1ComponentConfigs
    │       ├── adapter: ArenaTEMAdapterSettings
    │       ├── controller: ReplayTrajectoryControllerConfig
    │       └── objective: TEMObjectiveConfig
    ├── training: TEMTrainingConfig
    ├── execution: TEMRuntimeConfig
    ├── data: DatamoduleConfig
    ├── trainer: TrainerConfig
    ├── checkpointing: CheckpointingConfig
    └── logging: LoggerSettings

Hierarchy (evaluation):

    ArenaTEMV1EvaluationExperimentConfig
    └── model: ArenaTEMV1ModelConfig (same as above)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.tem import ArenaTEMAdapterSettings
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.experiments._infra import CheckpointingConfig, TrainerConfig
from ehc_sn.lightning.modules.variational_replay import (
    TEMTrainingConfig,
    VariationalReplayConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.tem import TEMObjectiveConfig
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.tem import RuntimeConfig as TEMRuntimeConfig

# =============================================================================
# Component configuration (task–model binding)
# =============================================================================


class ArenaTEMV1ComponentConfigs(BaseModel, extra="forbid"):
    """Component-level config for Arena × TEM-v1."""

    adapter: ArenaTEMAdapterSettings = Field(
        ...,
        description="Arena adapter settings for TEM v1.",
    )
    controller: ReplayTrajectoryControllerConfig = Field(
        ...,
        description="Replay-trajectory controller configuration.",
    )
    objective: TEMObjectiveConfig = Field(
        ...,
        description="TEM objective (loss) configuration.",
    )


# =============================================================================
# Model configuration (computational structure only)
# =============================================================================


class ArenaTEMV1ModelConfig(BaseModel, extra="forbid"):
    """Model-level config for Arena × TEM-v1."""

    model_config_path: Path = Field(
        ...,
        description="Path to the TEM v1 model architecture TOML file.",
    )
    components: ArenaTEMV1ComponentConfigs = Field(
        ...,
        description="Task-family component binding (adapter, controller, objective).",
    )


# =============================================================================
# Experiment-level configurations
# =============================================================================


class ArenaTEMV1TrainingExperimentConfig(BaseModel, extra="forbid"):
    """Full training application configuration for Arena × TEM-v1."""

    model: ArenaTEMV1ModelConfig = Field(
        ...,
        description="Model structure (components + architecture path).",
    )
    training: TEMTrainingConfig = Field(
        ...,
        description="TEM training configuration (optimizer).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    data: DatamoduleConfig = Field(
        ...,
        description="Dataset and DataLoader settings.",
    )
    execution: TEMRuntimeConfig = Field(
        default_factory=TEMRuntimeConfig,
        description="TEM runtime / execution configuration.",
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


class ArenaTEMV1EvaluationExperimentConfig(BaseModel, extra="forbid"):
    """Full evaluation application configuration for Arena × TEM-v1."""

    model: ArenaTEMV1ModelConfig = Field(
        ...,
        description="Model structure (components only).",
    )
    execution: Optional[TEMRuntimeConfig] = Field(
        default=None,
        description="Execution policy for eval-time runtime dynamics. "
        "None skips runtime configuration (not valid for actual eval).",
    )


# =============================================================================
__all__ = [
    "ArenaTEMV1ComponentConfigs",
    "ArenaTEMV1ModelConfig",
    "ArenaTEMV1TrainingExperimentConfig",
    "ArenaTEMV1EvaluationExperimentConfig",
]
