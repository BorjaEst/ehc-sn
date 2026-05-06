""" """

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Optional

import torch
from lightning.pytorch import Trainer, seed_everything
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.adapters.mazehard.ehc import MazeHardEHCAdapterSettings
from ehc_sn.adapters.mazehard.hrm.core import coerce_maze_hard_batch
from ehc_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from ehc_sn.callbacks.diagnostics import DiagnosticsCallback, DiagnosticsSettings
from ehc_sn.callbacks.figures import FigureCallbackSettings, FiguresCallback
from ehc_sn.callbacks.metrics import TrainingMetricsCallback
from ehc_sn.controllers.deliberation.actor_critic import DeliberationACControllerConfig
from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.ehc.ehc_v1_mazehard import ModelConfig_EHC_V1_MazeHard, TrainingModel
from ehc_sn.lightning.hrm.core.runtime import RuntimeConfig
from ehc_sn.logging.tensorboard import Logger, LoggerSettings
from ehc_sn.objectives import HybridRLLossConfig
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

CONFIGURATION_PATH = os.environ.get("EHC_V1_MAZEHARD_CONFIGURATION_PATH", "config/training.ehc-v1-mazehard.toml")


# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True):
    """ """

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,) -> tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    # -- Phase gate -----------------------------------------------------------------------
    mazehard_phase: Literal["mazehard_controller_pretrain"] = Field(
        default="mazehard_controller_pretrain",
        description="MazeHard controller pretrain phase gate.  Must be 'mazehard_controller_pretrain'.",
    )

    # -- Names and tracking ---------------------------------------------------------------
    project_name: Optional[str] = Field(default=None)
    run_name: Optional[str] = Field(default=None)

    # -- Model architecture and data ------------------------------------------------------
    model_config_path: Path = Field(
        ...,
        description="Path to the EHC v1 MazeHard model configuration TOML file.",
    )
    adapter: MazeHardEHCAdapterSettings = Field(
        default_factory=MazeHardEHCAdapterSettings,
        description="Adapter settings for the MazeHard+EHC bridge.",
    )
    deliberation: MazeHardDeliberationConfig = Field(
        ...,
        description="Deliberation capability config (halt_action, episode_horizon).",
    )
    controller: DeliberationACControllerConfig = Field(
        default_factory=DeliberationACControllerConfig,
        description="Deliberation actor-critic controller configuration.",
    )
    objective: HybridRLLossConfig = Field(
        ...,
        description="Hybrid RL objective configuration.",
    )

    # -- Optimizers & scheduling ----------------------------------------------------------
    optimizer_ctrl: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimizer for controller params (PFC backbone + encoder + decoder).",
    )
    optimizer_heads: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimizer for head params (pfc.estimator + STR).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config applied to both optimizers.",
    )
    supervised_only_warmup_steps: int = Field(
        default=5000,
        ge=0,
        description="Steps during which only opt_ctrl trains.",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Validation runner safety settings.",
    )

    # -- Data settings -------------------------------------------------------------------
    dataset_path: Path = Field(..., description="Path to the processed MazeHard dataset directory.")
    seed: int = Field(42)
    augment: bool = Field(True)
    global_batch_size: int = Field(..., description="Global batch size across all devices.")
    num_workers: int = Field(4)
    prefetch_factor: int = Field(2)
    pin_memory: bool = Field(True)
    persistent_workers: bool = Field(True)

    # -- Training control ----------------------------------------------------------------
    max_epochs: int = Field(...)

    # -- Callbacks -----------------------------------------------------------------------
    logger: Optional[LoggerSettings] = Field(default_factory=LoggerSettings)
    checkpoint: Optional[CheckpointSettings] = Field(default_factory=CheckpointSettings)
    figures: Optional[FigureCallbackSettings] = Field(default_factory=FigureCallbackSettings)
    diagnostic_level: Literal["minimal", "standard", "research"] = Field(default="standard")

    # -- Lightning Trainer kwargs --------------------------------------------------------
    max_steps: int = Field(default=200000)
    log_every_n_steps: int = Field(default=10)
    check_val_every_n_epoch: Optional[int] = Field(default=None)
    val_check_interval: int = Field(default=1000)
    enable_progress_bar: bool = Field(default=True)

    # -- Distributed training ------------------------------------------------------------
    trainer_accelerator: Literal["auto", "gpu", "cpu"] = Field(default="gpu")
    trainer_strategy: Literal["auto", "ddp"] = Field(default="ddp")
    trainer_devices: int = Field(default=1)
    trainer_num_nodes: int = Field(default=1)
    trainer_precision: str = Field(default="16-mixed")

    # -- Checkpointing -------------------------------------------------------------------
    checkpoint_path: Optional[str] = Field(default=None)
    checkpoint_every_eval: bool = Field(default=False)
    limit_val_batches: int = Field(default=10)
    eval_save_outputs: list[str] = Field(default_factory=list)

    # -- Aggregate settings --------------------------------------------------------------
    @property
    def ehc_mazehard_config(self) -> ModelConfig_EHC_V1_MazeHard:
        """Compose ModelConfig_EHC_V1_MazeHard from leaf settings."""
        return ModelConfig_EHC_V1_MazeHard.model_validate(self, from_attributes=True)

    @property
    def datamodule(self) -> DatamoduleConfig:
        """Compose DatamoduleConfig from leaf settings."""
        return DatamoduleConfig.model_validate(self, from_attributes=True)

    @property
    def diagnostics(self) -> DiagnosticsSettings:
        """Compose DiagnosticsSettings from leaf settings."""
        return DiagnosticsSettings.model_validate(self, from_attributes=True)


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
    if settings.figures is not None and settings.figures.enabled:
        callbacks_list.append(FiguresCallback(settings.figures))
    if settings.diagnostic_level != "minimal":
        callbacks_list.append(DiagnosticsCallback(settings.diagnostics))

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
        check_val_every_n_epoch=settings.check_val_every_n_epoch,
        val_check_interval=settings.val_check_interval,
        limit_val_batches=settings.limit_val_batches,
        log_every_n_steps=settings.log_every_n_steps,
        enable_progress_bar=settings.enable_progress_bar,
    )

    trainer.fit(
        model=TrainingModel(settings.ehc_mazehard_config),
        datamodule=Datamodule(settings.datamodule, transform=coerce_maze_hard_batch),
        ckpt_path=settings.checkpoint_path,
    )
