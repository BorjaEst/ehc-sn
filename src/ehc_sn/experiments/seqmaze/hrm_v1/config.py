"""Configuration schemas for the SeqMaze × HRM-v1 experiment pairing."""

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


class SeqMazeHRMV1ComponentConfigs(BaseModel, extra="forbid"):
    adapter: SeqMazeAdapterSettings
    controller: ACTControllerConfig
    objective: ACTObjectiveConfig


class SeqMazeHRMV1ModelConfig(BaseModel, extra="forbid"):
    model_config_path: Path
    components: SeqMazeHRMV1ComponentConfigs


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
    supervised_only_warmup_steps: int = 0
    diagnostic_level: Literal["minimal", "standard", "research"] = "standard"
    non_finite_policy: Literal["drop", "raise"] = "raise"
    scheduler: SchedulerConfig = SchedulerConfig()
    target_network: TargetNetworkConfig = TargetNetworkConfig()


class SeqMazeHRMV1TrainingExperimentConfig(BaseModel, extra="forbid"):
    model: SeqMazeHRMV1ModelConfig
    training: ACTSupervisedTrainingConfig
    data: DatamoduleConfig
    trainer: TrainerConfig = TrainerConfig()
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    logging: Optional[LoggerSettings] = None


class SeqMazeHRMV1EvaluationExperimentConfig(BaseModel, extra="forbid"):
    model: SeqMazeHRMV1ModelConfig
