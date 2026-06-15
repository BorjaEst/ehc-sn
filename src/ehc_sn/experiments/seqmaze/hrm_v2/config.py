"""Configuration schemas for the SeqMaze × HRM-v2 experiment pairing.

Hierarchy (training):

    SeqMazeHRMV2TrainingExperimentConfig
    ├── model: SeqMazeHRMV2ModelConfig
    │   └── components: SeqMazeHRMV2ComponentConfigs
    │       ├── adapter: SeqMazeAdapterSettings
    │       ├── controller: DeliberationACControllerConfig
    │       └── objective: HybridRLLossConfig
    ├── training: ActorCriticTrainingConfig
    ├── data: DatamoduleConfig
    ├── trainer: TrainerConfig
    ├── checkpointing: CheckpointingConfig
    └── logging: LoggerSettings

Hierarchy (evaluation):

    SeqMazeHRMV2EvaluationExperimentConfig
    └── model: SeqMazeHRMV2ModelConfig (same as above)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import SeqMazeAdapterSettings
from ehc_sn.controllers.deliberation.actor_critic import (
    DeliberationACControllerConfig,
)
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.lightning.callbacks.checkpoint import CheckpointSettings
from ehc_sn.lightning.modules.actor_critic import (
    ActorCriticTrainingConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig
from ehc_sn.training.schedules import SchedulerConfig

# =============================================================================
# Component configuration (task–model binding)
# =============================================================================


class SeqMazeHRMV2ComponentConfigs(BaseModel, extra="forbid"):
    """Component-level config for SeqMaze × HRM-v2."""

    adapter: SeqMazeAdapterSettings = Field(
        ...,
        description="SeqMaze adapter settings for HRM v2.",
    )
    controller: DeliberationACControllerConfig = Field(
        ...,
        description="Deliberation actor-critic controller configuration.",
    )
    objective: HybridRLLossConfig = Field(
        ...,
        description="Hybrid RL loss configuration.",
    )


# =============================================================================
# Model configuration (computational structure only)
# =============================================================================


class SeqMazeHRMV2ModelConfig(BaseModel, extra="forbid"):
    """Model-level config for SeqMaze × HRM-v2."""

    model_config_path: Path = Field(
        ...,
        description="Path to the HRM v2 model architecture TOML file.",
    )
    components: SeqMazeHRMV2ComponentConfigs = Field(
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
        default=5000,
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


# =============================================================================
# Experiment-level configurations
# =============================================================================


class SeqMazeHRMV2TrainingExperimentConfig(BaseModel, extra="forbid"):
    """Full training application configuration for SeqMaze × HRM-v2."""

    model: SeqMazeHRMV2ModelConfig = Field(
        ...,
        description="Model structure (components + architecture path).",
    )
    training: ActorCriticTrainingConfig = Field(
        ...,
        description="Actor-critic training configuration.",
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


class SeqMazeHRMV2EvaluationExperimentConfig(BaseModel, extra="forbid"):
    """Full evaluation application configuration for SeqMaze × HRM-v2."""

    model: SeqMazeHRMV2ModelConfig = Field(
        ...,
        description="Model structure (components only).",
    )


# =============================================================================
__all__ = [
    "SeqMazeHRMV2ComponentConfigs",
    "SeqMazeHRMV2ModelConfig",
    "SeqMazeHRMV2TrainingExperimentConfig",
    "SeqMazeHRMV2EvaluationExperimentConfig",
    "TrainerConfig",
    "CheckpointingConfig",
]
