"""Configuration schemas for the MazeHard × HRM-v2 experiment pairing.

Hierarchy (training):

    MazeHardHRMV2TrainingExperimentConfig
    ├── model: MazeHardHRMV2ModelConfig
    │   └── components: MazeHardHRMV2ComponentConfigs
    │       ├── adapter: MazeHardHRMAdapterSettings
    │       ├── controller: DeliberationQHaltingControllerConfig | None
    │       └── objective: HybridRLLossConfig
    ├── execution: MazeHardDeliberationConfig
    │   ├── halt_action
    │   └── episode_horizon
    ├── training: QHaltingTrainingConfig
    │   ├── optimizer
    │   └── runtime
    ├── data: DatamoduleConfig
    ├── trainer: TrainerConfig
    ├── checkpointing: CheckpointingConfig
    └── logging: LoggerSettings

Hierarchy (evaluation):

    MazeHardHRMV2EvaluationExperimentConfig
    ├── model: MazeHardHRMV2ModelConfig (same as above)
    └── execution: MazeHardDeliberationConfig (same as above)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import MazeHardHRMAdapterSettings
from ehc_sn.controllers.deliberation.q_halting import (
    DeliberationQHaltingControllerConfig,
)
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.experiments._infra import (
    CaptureConfig,
    CheckpointingConfig,
    ProviderConfig,
    RegimeConfig,
    TrainerConfig,
)
from ehc_sn.lightning.modules.q_halting import (
    QHaltingTrainingConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.composites.hybrid_rl import HybridRLLossConfig
from ehc_sn.tasks.mazehard.runtime import MazeHardRuntimeConfig
from ehc_sn.training.hrm import ValidationRuntimeConfig

# =============================================================================
# Deliberation configuration (task-owned execution limits)
# =============================================================================


class MazeHardDeliberationConfig(BaseModel, extra="forbid"):
    """MazeHard deliberation / execution policy settings.

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

    def to_runtime_config(self) -> MazeHardRuntimeConfig:
        """Convert to the runtime config consumed by MazeHardRuntime."""
        return MazeHardRuntimeConfig(
            halt_action=self.halt_action,
            episode_horizon=self.episode_horizon,
            validation=self.validation or ValidationRuntimeConfig(),
        )


# =============================================================================
# Model configuration (computational structure only)
# =============================================================================


class MazeHardHRMV2ComponentConfigs(BaseModel, extra="forbid"):
    """Component-level config for MazeHard × HRM-v2."""

    adapter: MazeHardHRMAdapterSettings = Field(
        ...,
        description="MazeHard adapter settings for HRM v2.",
    )
    controller: DeliberationQHaltingControllerConfig | None = Field(
        default=None,
        description="Deliberation actor-critic controller configuration. "
        "None disables the controller during evaluation.",
    )
    objective: HybridRLLossConfig = Field(
        ...,
        description="Hybrid RL loss configuration.",
    )


class MazeHardHRMV2ModelConfig(BaseModel, extra="forbid"):
    """Model-level config for MazeHard × HRM-v2.

    Contains computational structure and the model architecture TOML
    path. Deliberation and training configs are peers at the experiment
    level.
    """

    model_config_path: Path = Field(
        ...,
        description="Path to the HRM v2 model architecture TOML file.",
    )
    components: MazeHardHRMV2ComponentConfigs = Field(
        ...,
        description="Task-family component binding (adapter, controller, objective).",
    )


# =============================================================================
# Experiment-level configurations
# =============================================================================


class MazeHardHRMV2TrainingExperimentConfig(BaseModel, extra="forbid"):
    """Full training application configuration for MazeHard × HRM-v2."""

    model: MazeHardHRMV2ModelConfig = Field(
        ...,
        description="Model structure (components + architecture path).",
    )
    training: QHaltingTrainingConfig = Field(
        ...,
        description="Actor-critic training configuration (optimizers, reward). "
        "Runtime/execution policy moved to deliberation.validation.",
    )
    execution: MazeHardDeliberationConfig = Field(
        default_factory=MazeHardDeliberationConfig,
        description="Execution / deliberation policy. Defaults are safe for training.",
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


class MazeHardHRMV2EvaluationExperimentConfig(BaseModel, extra="forbid"):
    """Full evaluation application configuration for MazeHard × HRM-v2."""

    model: MazeHardHRMV2ModelConfig = Field(
        ...,
        description="Model structure (components only).",
    )
    execution: MazeHardDeliberationConfig = Field(
        default_factory=MazeHardDeliberationConfig,
        description="Execution / deliberation policy. Defaults are safe for evaluation.",
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
    "MazeHardDeliberationConfig",
    "MazeHardHRMV2ComponentConfigs",
    "MazeHardHRMV2ModelConfig",
    "MazeHardHRMV2TrainingExperimentConfig",
    "MazeHardHRMV2EvaluationExperimentConfig",
]
