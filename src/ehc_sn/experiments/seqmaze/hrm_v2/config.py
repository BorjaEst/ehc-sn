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
from typing import Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import SeqMazeAdapterSettings
from ehc_sn.controllers.deliberation.actor_critic import (
    DeliberationACControllerConfig,
)
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.experiments._infra import CheckpointingConfig, TrainerConfig
from ehc_sn.lightning.modules.actor_critic import ActorCriticTrainingConfig
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig
from ehc_sn.tasks.seqmaze.runtime import SeqMazeRuntimeConfig
from ehc_sn.training.hrm import ValidationRuntimeConfig
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
# Deliberation configuration (task-owned execution limits)
# =============================================================================


class SeqMazeDeliberationConfig(BaseModel, extra="forbid"):
    """SeqMaze deliberation / execution policy settings.

    Controls how the AC controller runs during inference and evaluation.
    Not part of model structure -- does not affect parameter shapes or
    checkpoint compatibility.
    """

    halt_action: int = Field(
        default=0,
        ge=0,
        description="Action index that signals episode termination.",
    )
    episode_horizon: int = Field(
        default=16,
        ge=1,
        description="Maximum number of deliberation steps per episode.",
    )
    validation: Optional[ValidationRuntimeConfig] = Field(
        default=None,
        description="Runner-owned safety limits (max steps, seed). "
        "Defaults from the runtime config when None.",
    )

    def to_runtime_config(self) -> SeqMazeRuntimeConfig:
        """Convert to the runtime config consumed by SeqMazeRuntime."""
        return SeqMazeRuntimeConfig(
            halt_action=self.halt_action,
            episode_horizon=self.episode_horizon,
            validation=self.validation or ValidationRuntimeConfig(),
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
    execution: SeqMazeDeliberationConfig = Field(
        default_factory=SeqMazeDeliberationConfig,
        description="Deliberation execution policy (halt_action, episode_horizon).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    supervised_only_warmup_steps: int = Field(
        default=5000,
        description="Optimizer steps with learned halting disabled.",
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
    execution: SeqMazeDeliberationConfig = Field(
        default_factory=SeqMazeDeliberationConfig,
        description="Deliberation execution policy (halt_action, episode_horizon).",
    )


# =============================================================================
__all__ = [
    "SeqMazeDeliberationConfig",
    "SeqMazeHRMV2ComponentConfigs",
    "SeqMazeHRMV2ModelConfig",
    "SeqMazeHRMV2TrainingExperimentConfig",
    "SeqMazeHRMV2EvaluationExperimentConfig",
]
