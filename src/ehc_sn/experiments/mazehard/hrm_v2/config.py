"""Configuration schemas for the MazeHard × HRM-v2 experiment pairing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import MazeHardHRMAdapterSettings
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
from ehc_sn.training.stabilization import TargetNetworkConfig

# =============================================================================
# Deliberation configuration (task-owned execution limits)
# =============================================================================


class MazeHardDeliberationConfig(BaseModel, extra="forbid"):
    """MazeHard deliberation / execution policy settings."""

    halt_action: int = Field(default=0, ge=0)
    episode_horizon: int = Field(default=16, ge=1)


# =============================================================================
# Model configuration
# =============================================================================


class MazeHardHRMV2ComponentConfigs(BaseModel, extra="forbid"):
    """Component-level config for MazeHard × HRM-v2."""

    adapter: MazeHardHRMAdapterSettings
    controller: DeliberationACControllerConfig | None = None
    objective: HybridRLLossConfig


class MazeHardHRMV2ModelConfig(BaseModel, extra="forbid"):
    """Model-level config — computational structure only."""

    model_config_path: Path
    components: MazeHardHRMV2ComponentConfigs


# =============================================================================
# Infra stubs
# =============================================================================


class TrainerConfig(BaseModel, extra="forbid"):
    accelerator: Literal["auto", "gpu", "cpu"] = "gpu"
    strategy: Literal["auto", "ddp"] = "ddp"
    devices: int = 1
    num_nodes: int = 1
    precision: str = "16-mixed"
    max_steps: int = 200000
    val_check_interval: int = 0
    log_every_n_steps: int = 10
    enable_progress_bar: bool = True
    limit_val_batches: int | float = 1.0
    seed: int = 42


class CheckpointingConfig(BaseModel, extra="forbid"):
    checkpoint: Optional[CheckpointSettings] = None
    resume_from: Optional[str] = None
    init_weights_from: Optional[str] = None
    init_weights_groups: list[str] = ["all"]
    supervised_only_warmup_steps: int = 5000
    diagnostic_level: Literal["minimal", "standard", "research"] = "standard"
    non_finite_policy: Literal["drop", "raise"] = "raise"
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoint_every_eval: bool = False


# =============================================================================
# Experiment-level configurations
# =============================================================================


class MazeHardHRMV2TrainingExperimentConfig(BaseModel, extra="forbid"):
    model: MazeHardHRMV2ModelConfig
    deliberation: MazeHardDeliberationConfig
    training: ActorCriticTrainingConfig
    data: DatamoduleConfig
    trainer: TrainerConfig = TrainerConfig()
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    logging: Optional[LoggerSettings] = None


class MazeHardHRMV2EvaluationExperimentConfig(BaseModel, extra="forbid"):
    model: MazeHardHRMV2ModelConfig
    deliberation: MazeHardDeliberationConfig
