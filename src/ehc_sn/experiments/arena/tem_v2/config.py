"""Configuration schemas for the Arena × TEM-v2 experiment pairing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.tem import ArenaTEMAdapterSettings
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.lightning.callbacks.checkpoint import CheckpointSettings
from ehc_sn.lightning.modules.variational_replay import (
    VariationalReplayConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.tem import TEMObjectiveConfig
from ehc_sn.training.schedules import SchedulerConfig


class ArenaTEMV2ComponentConfigs(BaseModel, extra="forbid"):
    adapter: ArenaTEMAdapterSettings
    controller: ReplayTrajectoryControllerConfig
    objective: TEMObjectiveConfig


class ArenaTEMV2ModelConfig(BaseModel, extra="forbid"):
    model_config_path: Path
    components: ArenaTEMV2ComponentConfigs


class TrainerConfig(BaseModel, extra="forbid"):
    accelerator: Literal["auto", "gpu", "cpu"] = "gpu"
    strategy: Literal["auto", "ddp"] = "ddp"
    devices: int = 1
    num_nodes: int = 1
    precision: str = "16-mixed"
    max_steps: int = 200000
    val_check_interval: int = 500
    log_every_n_steps: int = 10
    enable_progress_bar: bool = True
    limit_val_batches: int | float = 1.0
    seed: int = 42


class CheckpointingConfig(BaseModel, extra="forbid"):
    checkpoint: Optional[CheckpointSettings] = None
    resume_from: Optional[str] = None
    init_weights_from: Optional[str] = None
    init_weights_groups: list[str] = ["all"]
    diagnostic_level: Literal["minimal", "standard", "research"] = "standard"
    non_finite_policy: Literal["drop", "raise"] = "raise"
    scheduler: SchedulerConfig = SchedulerConfig()


class ArenaTEMV2TrainingExperimentConfig(BaseModel, extra="forbid"):
    model: ArenaTEMV2ModelConfig
    training: VariationalReplayConfig
    data: DatamoduleConfig
    trainer: TrainerConfig = TrainerConfig()
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    logging: Optional[LoggerSettings] = None


class ArenaTEMV2EvaluationExperimentConfig(BaseModel, extra="forbid"):
    model: ArenaTEMV2ModelConfig
