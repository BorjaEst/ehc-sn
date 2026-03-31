""" """

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from lightning.pytorch import Trainer, seed_everything
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource

from ehc_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from ehc_sn.callbacks.diagnostics import DiagnosticsCallback, DiagnosticsSettings
from ehc_sn.callbacks.figures import FigureCallbackSettings, FiguresCallback
from ehc_sn.callbacks.metrics import TrainingMetricsCallback
from ehc_sn.controllers.tem import TEMControllerConfig
from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.envs.dungeon_walk import EnvConfig
from ehc_sn.heads.tem import TEMLossConfig
from ehc_sn.logging.tensorboard import Logger, LoggerSettings
from ehc_sn.models import tem_v1
from ehc_sn.models.tem_v1 import ModelSettings_V1
from ehc_sn.runtimes.training.tem_v1 import ModelConfig_TEM_V1, RuntimeConfig, TrainingModel
from ehc_sn.training.distributed import resolve_effective_world_size, resolve_trainer_strategy, validate_batch_size_divisibility
from ehc_sn.training.optim import AdamConfig
from ehc_sn.training.schedules import SchedulerConfig

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
CONFIGURATION_PATH = os.environ.get("TEM_V1_CONFIGURATION_PATH", "config/tem-dungeons.v1.toml")


# =================================================================================================
# Settings Model
# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True):
    """ """

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
        """Customize settings source order.

        Pydantic Settings supports multiple value sources; we explicitly place
        the CLI first so that command-line overrides always win.
        """
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    # ---------------------------------------------------------------------------------------------
    # Names and tracking
    project_name: Optional[str] = Field(
        default=None,
        description=("Project name. If not set, it defaults to the capitalized name of the dataset " "(e.g. `MATH` -> `Math ACT-torch`)."),
    )
    run_name: Optional[str] = Field(
        default=None,
        description=(
            "Run name. If not set, it defaults to `<arch_name> <random_slug>` " "(e.g. `TemV1 2x128 4L 16H 0.1D ACT-torch cool-slug`)."
        ),
    )

    # ---------------------------------------------------------------------------------------------
    # Model architecture and data
    model: ModelSettings_V1 = Field(
        ...,
        description="TEM v1 model architecture settings (PFC + embedding/LM-head).",
    )
    environment: EnvConfig = Field(
        ...,
        description="Environment configuration (max_steps, seq_length, vocab_size, halt_action).",
    )
    controller: TEMControllerConfig = Field(
        ...,
        description="TEM controller configuration (exploration probability).",
    )
    loss: TEMLossConfig = Field(
        ...,
        description="TEM loss head configuration (observation, latent, regularization",
    )

    # ~~ Optimizers & scheduling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    optimizer: AdamConfig = Field(
        default_factory=AdamConfig,
        description="Adam optimizer settings (learning_rate, betas, eps, weight_decay).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler settings (scheduler_type, warmup_steps, total_steps).",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="TEM runtime dynamics schedule settings applied inside the training loop.",
    )

    # ---------------------------------------------------------------------------------------------
    # Data settings (flat fields composed into DatamoduleConfig)
    dataset_path: Path = Field(
        ...,
        description="Path to the processed dataset directory (contains index.jsonl + NPZ files).",
    )
    seed: int = Field(
        42,
        description="RNG seed for training-split dihedral augmentation and reproducibility.",
    )
    augment: bool = Field(
        True,
        description="Apply RandomDihedral augmentation to training samples.",
    )
    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. " "The per-device batch size is computed as `global_batch_size // world_size`."
        ),
    )
    num_workers: int = Field(
        4,
        description="Number of workers for DataLoader.",
    )
    prefetch_factor: int = Field(
        2,
        description="Number of batches to prefetch per worker.",
    )
    pin_memory: bool = Field(
        True,
        description="Whether to pin memory in DataLoader.",
    )
    persistent_workers: bool = Field(
        True,
        description="Whether to keep DataLoader workers alive between epochs.",
    )

    # ---------------------------------------------------------------------------------------------
    # Training control settings (passed as top-level settings for ease of CLI overrides)
    epochs: int = Field(
        ...,
        description="Total number of epochs to train.",
    )

    # ---------------------------------------------------------------------------------------------
    # Core settings for model, data, and training configuration (passed as configs to modules)
    logger: Optional[LoggerSettings] = Field(
        default_factory=LoggerSettings,
        description="TensorBoard logger settings.",
    )
    checkpoint: Optional[CheckpointSettings] = Field(
        default_factory=CheckpointSettings,
        description="Model checkpoint settings.",
    )
    figures: Optional[FigureCallbackSettings] = Field(
        default_factory=FigureCallbackSettings,
        description="Figure generation callback settings.",
    )
    diagnostic_level: Literal["minimal", "standard", "research"] = Field(
        default="standard",
        description=(
            "Instrumentation tier. 'minimal': only training metrics. "
            "'standard': training metrics + model health diagnostics. "
            "'research': all available diagnostic signals."
        ),
    )

    # ---------------------------------------------------------------------------------------------
    # Training control settings (passed as kwargs to Lightning Trainer)
    max_steps: int = Field(
        default=200000,
        description="Maximum training steps.",
    )
    log_every_n_steps: int = Field(
        default=10,
        description="Log metrics every N steps.",
    )
    check_val_every_n_epoch: Optional[int] = Field(
        default=None,
        description=(
            "Validation scheduling mode. Set to None to validate based on total training batches "
            "across epochs (i.e., use val_check_interval as a global step interval)."
        ),
    )
    val_check_interval: int = Field(
        default=1000,
        description="Validation check interval (in training steps).",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress bar during training.",
    )

    # ---------------------------------------------------------------------------------------------
    # Distributed training settings (explicitly passed to Lightning Trainer)
    trainer_accelerator: Literal["auto", "gpu", "cpu"] = Field(
        default="gpu",
        description="Trainer accelerator setting. Use 'gpu' for HAICORE multi-GPU runs.",
    )
    trainer_strategy: Literal["auto", "ddp"] = Field(
        default="ddp",
        description="Trainer strategy setting. Use 'ddp' for SLURM multi-GPU runs.",
    )
    trainer_devices: int = Field(
        default=1,
        description="Number of devices per node for the Trainer (per process when using SLURM tasks).",
    )
    trainer_num_nodes: int = Field(
        default=1,
        description="Number of nodes for distributed training.",
    )
    trainer_precision: str = Field(
        default="16-mixed",
        description=(
            "Lightning Trainer precision. '32-true' = full fp32 (paper-parity default). " "Use 'bf16-mixed' for throughput on Ampere+."
        ),
    )

    # ---------------------------------------------------------------------------------------------
    # Checkpointing and evaluation settings (passed as kwargs to Trainer and Checkpoint callback)
    checkpoint_path: Optional[str] = Field(
        default=None,
        description=("Path to save checkpoints and logs. " "If not set, it defaults to `checkpoints/<project_name>/<run_name>`."),
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Whether to checkpoint the model after every evaluation.",
    )
    limit_val_batches: int = Field(
        default=10,
        description="Cap validation to N batches per validation run.",
    )
    eval_save_outputs: list[str] = Field(
        default_factory=list,
        description="Evaluation output keys saved as tensors in the checkpoint directory.",
    )

    # ---------------------------------------------------------------------------------------------
    # Aggregate settings (compose leaf settings for modules)
    @property
    def tem_config(self) -> ModelConfig_TEM_V1:
        """Compose ModelConfig_TEM_V1 from leaf settings."""
        return ModelConfig_TEM_V1.model_validate(self, from_attributes=True)

    @property
    def datamodule(self) -> DatamoduleConfig:
        """Compose DatamoduleConfig from leaf settings."""
        return DatamoduleConfig.model_validate(self, from_attributes=True)

    @property
    def diagnostics(self) -> DiagnosticsSettings:
        """Compose DiagnosticsSettings from leaf settings."""
        return DiagnosticsSettings.model_validate(self, from_attributes=True)


# =================================================================================================
# Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    # Load defaults from TOML, then parse settings.
    # CLI arguments override TOML values; Pydantic defaults fill in anything missing.
    defaults_from_path = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
    settings = RunArguments(**defaults_from_path)
    world_size = resolve_effective_world_size(
        settings.trainer_strategy,
        settings.trainer_devices,
        settings.trainer_num_nodes,
    )
    validate_batch_size_divisibility(settings.global_batch_size, world_size)

    # Seed everything for reproducibility.
    seed_everything(settings.seed)

    # Prepare callbacks: checkpointing + optional figure generation.
    callbacks_list = [TrainingMetricsCallback()]
    if settings.checkpoint is not None:
        callbacks_list.append(CheckpointCallback(settings.checkpoint))
    if settings.figures is not None and settings.figures.enabled:
        callbacks_list.append(FiguresCallback(settings.figures))
    if settings.diagnostic_level != "minimal":
        callbacks_list.append(DiagnosticsCallback(settings.diagnostics))

    # Build the PyTorch Lightning Trainer.
    # This wires together logging, callbacks, and training control.
    trainer = Trainer(
        # Logger + callbacks handle metrics/hparams, figures, and checkpointing.
        logger=Logger(settings.logger) if settings.logger is not None else None,
        callbacks=callbacks_list if callbacks_list else None,
        # Lightning Trainer kwargs (extracted from config)
        accelerator=settings.trainer_accelerator,
        strategy=resolve_trainer_strategy(settings.trainer_strategy, world_size, find_unused_parameters=True),
        devices=settings.trainer_devices,
        num_nodes=settings.trainer_num_nodes,
        precision=settings.trainer_precision,
        max_steps=settings.max_steps,
        check_val_every_n_epoch=settings.check_val_every_n_epoch,
        val_check_interval=settings.val_check_interval,
        limit_val_batches=settings.limit_val_batches,
        log_every_n_steps=settings.log_every_n_steps,
        enable_progress_bar=settings.enable_progress_bar,
    )

    # Start training.
    # - The LightningModule wraps the TEM model and defines the training loop.
    # - The DataModule constructs loaders for the puzzle/maze dataset.
    trainer.fit(
        # Lightning module: training step, optimizer and schedule setup.
        model=TrainingModel(settings.tem_config),
        # Data module: dataset + DataLoader construction.
        datamodule=Datamodule(settings.datamodule, transform=None),
        # Optional: resume training from a checkpoint.
        ckpt_path=settings.checkpoint_path,
    )
