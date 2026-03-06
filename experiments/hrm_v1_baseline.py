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
from ehc_sn.callbacks.figures import FigureCallbackSettings, FiguresCallback
from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.logging.tensorboard import Logger, LoggerSettings
from ehc_sn.models import hrm_v1
from ehc_sn.models.hrm_v1 import ModelConfig_HRM_V1, ModelSettings_V1, TrainingModel
from ehc_sn.training.act_controller import ACTControllerConfig
from ehc_sn.training.act_head import ACTLossConfig
from ehc_sn.training.optim import AdamATan2Config
from ehc_sn.training.schedules import SchedulerConfig

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")
CONFIGURATION_PATH = os.environ.get("HRM_V1_CONFIGURATION_PATH", "config/defaults_hrm-mazehard.toml")


# =================================================================================================
# Settings Model
# =================================================================================================
class RunArguments(BaseSettings, extra="forbid", cli_parse_args=True):
    """ """

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:  # fmt: skip
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
        description=(
            "Project name. If not set, it defaults to the capitalized name of the dataset "
            "(e.g. `MATH` -> `Math ACT-torch`)."
        ),
    )
    run_name: Optional[str] = Field(
        default=None,
        description=(
            "Run name. If not set, it defaults to `<arch_name> <random_slug>` "
            "(e.g. `HrmV1 2x128 4L 16H 0.1D ACT-torch cool-slug`)."
        ),
    )

    # ---------------------------------------------------------------------------------------------
    # Model architecture and data
    model: ModelSettings_V1 = Field(
        ...,
        description="HRM v1 model architecture settings (PFC + embedding/LM-head).",
    )
    act_controller: ACTControllerConfig = Field(
        ...,
        description=(
            "Configuration for the ACT controller, which manages halting and partial resets"
            "during training. "
            "The keys in `act_controller` are passed to the ACTController constructor."
        ),
    )
    loss: ACTLossConfig = Field(
        ...,
        description="Loss config. The keys in `loss` are passed to the loss head constructor.",
    )
    optimizer: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description=(
            "Main optimizer config for model parameters (e.g. Adam). "
            "The keys in `optim_main` are passed to the optimizer constructor."
        ),
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description=(
            "Learning rate scheduler config. If not set, no learning rate scheduling is applied. "
            "The keys in `scheduler` are passed to the scheduler constructor."
        ),
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
            "Global batch size across all devices. "
            "The per-device batch size is computed as `global_batch_size // world_size`."
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
        default=100,
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

    # ---------------------------------------------------------------------------------------------
    # Checkpointing and evaluation settings (passed as kwargs to Trainer and Checkpoint callback)
    checkpoint_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to save checkpoints and logs. "
            "If not set, it defaults to `checkpoints/<project_name>/<run_name>`."
        ),
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Whether to checkpoint the model after every evaluation.",
    )
    limit_val_batches: int = Field(
        default=1,
        description="Cap validation to N batches per validation run.",
    )
    eval_save_outputs: List[str] = Field(
        default_factory=list,
        description="Evaluation output keys saved as tensors in the checkpoint directory.",
    )

    # ---------------------------------------------------------------------------------------------
    # Aggregate settings (compose leaf settings for modules)
    @property
    def hrm_config(self) -> ModelConfig_HRM_V1:
        """Compose ModelConfig_HRM_V1 from leaf settings."""
        return ModelConfig_HRM_V1.model_validate(self, from_attributes=True)

    @property
    def datamodule(self) -> DatamoduleConfig:
        """Compose DatamoduleConfig from leaf settings."""
        return DatamoduleConfig.model_validate(self, from_attributes=True)


# =================================================================================================
def _validate_global_batch_size(settings: RunArguments) -> None:
    """Validate that global_batch_size is divisible by world_size for distributed training."""
    slurm_world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_JOB_ID"):
        world_size = slurm_world_size
    elif settings.trainer_strategy == "ddp":
        world_size = settings.trainer_devices * settings.trainer_num_nodes
    else:
        world_size = 1

    if world_size <= 0:
        raise ValueError("World size must be a positive integer.")
    if settings.global_batch_size % world_size != 0:
        raise ValueError(
            "global_batch_size must be divisible by world_size. "
            f"Got global_batch_size={settings.global_batch_size}, world_size={world_size}."
        )


# =================================================================================================
# Main Entrypoint
# =================================================================================================
if __name__ == "__main__":
    # Load defaults from TOML, then parse settings.
    # CLI arguments override TOML values; Pydantic defaults fill in anything missing.
    defaults_from_path = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
    settings = RunArguments(**defaults_from_path)
    _validate_global_batch_size(settings)

    # Seed everything for reproducibility.
    seed_everything(settings.seed)

    # Prepare callbacks: checkpointing + optional figure generation.
    callbacks_list = []
    if settings.checkpoint is not None:
        callbacks_list.append(CheckpointCallback(settings.checkpoint))
    if settings.figures is not None and settings.figures.enabled:
        callbacks_list.append(FiguresCallback(settings.figures))

    # Build the PyTorch Lightning Trainer.
    # This wires together logging, callbacks, and training control.
    trainer = Trainer(
        # Logger + callbacks handle metrics/hparams, figures, and checkpointing.
        logger=Logger(settings.logger) if settings.logger is not None else None,
        callbacks=callbacks_list if callbacks_list else None,
        # Lightning Trainer kwargs (extracted from config)
        accelerator=settings.trainer_accelerator,
        strategy=settings.trainer_strategy,
        devices=settings.trainer_devices,
        num_nodes=settings.trainer_num_nodes,
        max_steps=settings.max_steps,
        check_val_every_n_epoch=settings.check_val_every_n_epoch,
        val_check_interval=settings.val_check_interval,
        limit_val_batches=settings.limit_val_batches,
        log_every_n_steps=settings.log_every_n_steps,
        enable_progress_bar=settings.enable_progress_bar,
    )

    # Start training.
    # - The LightningModule wraps the HRM model and defines the training loop.
    # - The DataModule constructs loaders for the puzzle/maze dataset.
    trainer.fit(
        # Lightning module: training step, optimizer and schedule setup.
        model=TrainingModel(settings.hrm_config),
        # Data module: dataset + DataLoader construction.
        datamodule=Datamodule(settings.datamodule, transform=hrm_v1.supervised_maze_tokenize),
        # Optional: resume training from a checkpoint.
        ckpt_path=settings.checkpoint_path,
    )
