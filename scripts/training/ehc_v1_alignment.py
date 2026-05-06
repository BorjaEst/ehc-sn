"""Phase-3 EHC v1 alignment training entrypoint.

Trains only model.pfc_to_hpc using strict arena/mazehard batch alternation.
All other model parameters are frozen.  Optionally loads from two pre-trained
checkpoints (phase-1 arena + phase-2 mazehard).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Optional

import torch
from lightning.pytorch import Trainer, seed_everything
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.adapters.arena.ehc import ArenaEHCAdapterSettings
from ehc_sn.adapters.mazehard.ehc import MazeHardEHCAdapterSettings
from ehc_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from ehc_sn.callbacks.metrics import TrainingMetricsCallback
from ehc_sn.controllers.deliberation.actor_critic import DeliberationACControllerConfig
from ehc_sn.controllers.replay.trajectory import ReplayTrajectoryControllerConfig
from ehc_sn.data.alternating import AlignmentDatamodule, AlignmentDatamoduleConfig
from ehc_sn.lightning.ehc.ehc_v1_alignment import ModelConfig_EHC_V1_Alignment, TrainingModel
from ehc_sn.lightning.hrm.core.runtime import RuntimeConfig
from ehc_sn.logging.tensorboard import Logger, LoggerSettings
from ehc_sn.objectives import EHCObjectiveConfig
from ehc_sn.objectives.hybrid_rl import HybridRLLossConfig
from ehc_sn.tasks.mazehard.capabilities.deliberation import MazeHardDeliberationConfig
from ehc_sn.training.distributed import resolve_effective_world_size, resolve_trainer_strategy, validate_batch_size_divisibility
from ehc_sn.training.optim import AdamATan2Config
from ehc_sn.training.schedules import SchedulerConfig

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

CONFIGURATION_PATH = os.environ.get("EHC_V1_ALIGNMENT_CONFIGURATION_PATH", "config/training.ehc-v1-alignment.toml")


# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True):
    """CLI-composable settings for the phase-3 alignment trainer."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    # -- Phase gate -----------------------------------------------------------------------
    alignment_phase: Literal["initial_alignment"] = Field(
        default="initial_alignment",
        description="Phase gate. Must be 'initial_alignment'.",
    )

    # -- Names and tracking ---------------------------------------------------------------
    project_name: Optional[str] = Field(default=None)
    run_name: Optional[str] = Field(default=None)

    # -- Shared model architecture --------------------------------------------------------
    model_config_path: Path = Field(
        ...,
        description="Path to the shared EHC v1 model TOML file.",
    )

    # -- Checkpoint handoff ---------------------------------------------------------------
    arena_ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to the phase-1 arena checkpoint (.ckpt). None = fresh init.",
    )
    mazehard_ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to the phase-2 mazehard checkpoint (.ckpt). None = fresh init.",
    )

    # -- Arena task -----------------------------------------------------------------------
    arena_adapter: ArenaEHCAdapterSettings = Field(
        ...,
        description="Arena bridge adapter settings.",
    )
    arena_controller: ReplayTrajectoryControllerConfig = Field(
        default_factory=ReplayTrajectoryControllerConfig,
        description="Arena replay trajectory controller config.",
    )
    arena_objective: EHCObjectiveConfig = Field(
        default_factory=EHCObjectiveConfig,
        description="EHC predictive objective config for the arena task.",
    )

    # -- MazeHard task --------------------------------------------------------------------
    mazehard_adapter: MazeHardEHCAdapterSettings = Field(
        default_factory=MazeHardEHCAdapterSettings,
        description="MazeHard bridge adapter settings.",
    )
    mazehard_deliberation: MazeHardDeliberationConfig = Field(
        ...,
        description="MazeHard deliberation config.",
    )
    mazehard_controller: DeliberationACControllerConfig = Field(
        default_factory=DeliberationACControllerConfig,
        description="MazeHard deliberation AC controller config.",
    )
    mazehard_objective: HybridRLLossConfig = Field(
        ...,
        description="MazeHard hybrid RL objective config.",
    )

    # -- Optimizer & scheduling -----------------------------------------------------------
    optimizer: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Single optimizer for pfc_to_hpc parameters.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Validation runner safety settings.",
    )

    # -- Datasets -------------------------------------------------------------------------
    arena_dataset_path: Path = Field(..., description="Path to the processed arena dataset root.")
    mazehard_dataset_path: Path = Field(..., description="Path to the processed mazehard dataset root.")
    global_batch_size: int = Field(..., description="Global batch size across all devices.")
    num_workers: int = Field(default=4)
    prefetch_factor: int = Field(default=2)
    pin_memory: bool = Field(default=True)
    persistent_workers: bool = Field(default=True)
    augment: bool = Field(default=True)
    seed: int = Field(default=42)

    # -- Training control -----------------------------------------------------------------
    max_epochs: Optional[int] = Field(default=50)
    max_steps: int = Field(default=100000)
    log_every_n_steps: int = Field(default=10)
    val_check_interval: int = Field(default=500)
    enable_progress_bar: bool = Field(default=True)
    limit_val_batches: int = Field(default=10)

    # -- Callbacks ------------------------------------------------------------------------
    logger: Optional[LoggerSettings] = Field(default_factory=LoggerSettings)
    checkpoint: Optional[CheckpointSettings] = Field(default_factory=CheckpointSettings)

    # -- Distributed training -------------------------------------------------------------
    trainer_accelerator: Literal["auto", "gpu", "cpu"] = Field(default="gpu")
    trainer_strategy: Literal["auto", "ddp"] = Field(default="ddp")
    trainer_devices: int = Field(default=1)
    trainer_num_nodes: int = Field(default=1)
    trainer_precision: str = Field(default="16-mixed")

    # -- Checkpointing --------------------------------------------------------------------
    checkpoint_path: Optional[str] = Field(default=None)

    # -- Aggregate settings ---------------------------------------------------------------
    @property
    def alignment_config(self) -> ModelConfig_EHC_V1_Alignment:
        """Compose ModelConfig_EHC_V1_Alignment from leaf settings."""
        return ModelConfig_EHC_V1_Alignment.model_validate(self, from_attributes=True)

    @property
    def datamodule_config(self) -> AlignmentDatamoduleConfig:
        """Compose AlignmentDatamoduleConfig from leaf settings."""
        return AlignmentDatamoduleConfig.model_validate(self, from_attributes=True)


# =================================================================================================
if __name__ == "__main__":
    defaults_from_path = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
    settings = RunArguments(**defaults_from_path)
    world_size = resolve_effective_world_size(
        settings.trainer_strategy,
        settings.trainer_devices,
        settings.trainer_num_nodes,
    )
    validate_batch_size_divisibility(settings.global_batch_size, world_size)

    seed_everything(settings.seed)

    callbacks_list = [TrainingMetricsCallback()]
    if settings.checkpoint is not None:
        callbacks_list.append(CheckpointCallback(settings.checkpoint))

    trainer = Trainer(
        logger=Logger(settings.logger) if settings.logger is not None else None,
        callbacks=callbacks_list if callbacks_list else None,
        accelerator=settings.trainer_accelerator,
        strategy=resolve_trainer_strategy(settings.trainer_strategy, world_size, find_unused_parameters=True),
        devices=settings.trainer_devices,
        num_nodes=settings.trainer_num_nodes,
        precision=settings.trainer_precision,
        max_epochs=settings.max_epochs,
        max_steps=settings.max_steps,
        val_check_interval=settings.val_check_interval,
        limit_val_batches=settings.limit_val_batches,
        log_every_n_steps=settings.log_every_n_steps,
        enable_progress_bar=settings.enable_progress_bar,
    )

    trainer.fit(
        model=TrainingModel(settings.alignment_config),
        datamodule=AlignmentDatamodule(settings.datamodule_config),
        ckpt_path=settings.checkpoint_path,
    )
