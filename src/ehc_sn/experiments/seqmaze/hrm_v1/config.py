"""Configuration schemas for the SeqMaze × HRM-v1 experiment pairing.

Hierarchy (training):

    SeqMazeHRMV1TrainingExperimentConfig
    ├── model: SeqMazeHRMV1ModelConfig
    │   └── components: SeqMazeHRMV1ComponentConfigs
    │       ├── adapter: SeqMazeAdapterSettings
    │       ├── controller: ACTControllerConfig
    │       └── objective: ACTObjectiveConfig
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
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import SeqMazeAdapterSettings
from ehc_sn.controllers.deliberation.act import ACTControllerConfig
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.lightning.callbacks.checkpoint import CheckpointSettings
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedTrainingConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.act import ACTObjectiveConfig
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.stabilization import TargetNetworkConfig

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
    objective: ACTObjectiveConfig = Field(
        ...,
        description="ACT supervised objective configuration.",
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
# Trainer / infra config
# =============================================================================


class TrainerConfig(BaseModel, extra="forbid"):
    """Lightning Trainer configuration fields."""

    accelerator: Literal["auto", "gpu", "cpu"] = Field(
        default="gpu",
        description="Trainer accelerator setting.",
    )
    strategy: Literal["auto", "ddp"] = Field(
        default="ddp",
        description="Trainer DDP strategy.",
    )
    devices: int = Field(
        default=1,
        ge=1,
        description="Number of devices per node.",
    )
    num_nodes: int = Field(
        default=1,
        ge=1,
        description="Number of nodes.",
    )
    precision: str = Field(
        default="16-mixed",
        description="Training precision.",
    )
    max_steps: int = Field(
        default=200000,
        ge=1,
        description="Maximum training steps.",
    )
    val_check_interval: int = Field(
        default=500,
        ge=1,
        description="Validation check interval in steps.",
    )
    log_every_n_steps: int = Field(
        default=10,
        ge=1,
        description="Log metrics every N steps.",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress bar.",
    )
    limit_val_batches: int | float = Field(
        default=1.0,
        description="Validation batches (int=N, float=fraction).",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="RNG seed for reproducibility.",
    )
    find_unused_parameters: bool = Field(
        default=False,
        description="Enable DDP find_unused_parameters.",
    )


class CheckpointingConfig(BaseModel, extra="forbid"):
    """Checkpoint and weight-init settings."""

    checkpoint: Optional[CheckpointSettings] = Field(
        default=None,
        description="Model checkpoint settings.",
    )
    resume_from: Optional[str] = Field(
        default=None,
        description="Checkpoint path to resume full trainer state.",
    )
    init_weights_from: Optional[str] = Field(
        default=None,
        description="Checkpoint path for model-weight initialization only.",
    )
    init_weights_groups: list[str] = Field(
        default_factory=lambda: ["all"],
        description="Named weight groups to hydrate from init_weights_from.",
    )
    supervised_only_warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Steps with learned halting disabled.",
    )
    diagnostic_level: Literal["minimal", "standard", "research"] = Field(
        default="standard",
        description="Instrumentation tier for diagnostic logging.",
    )
    non_finite_policy: Literal["drop", "raise"] = Field(
        default="raise",
        description="Policy for NaN/Inf diagnostics.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    target_network: TargetNetworkConfig = Field(
        default_factory=TargetNetworkConfig,
        description="EMA-lagged target network config.",
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
    training: ACTSupervisedTrainingConfig = Field(
        ...,
        description="ACT supervised training configuration.",
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


# =============================================================================
__all__ = [
    "SeqMazeHRMV1ComponentConfigs",
    "SeqMazeHRMV1ModelConfig",
    "SeqMazeHRMV1TrainingExperimentConfig",
    "SeqMazeHRMV1EvaluationExperimentConfig",
    "TrainerConfig",
    "CheckpointingConfig",
]
