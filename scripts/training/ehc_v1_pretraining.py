"""Unified EHC v1 training entrypoint.

Supports two public modes (selected by the ``mode`` field in the config):
    spatial_pretrain    — arena replay, EHC variational objective.
    controller_pretrain — MazeHard deliberation, hybrid RL.

Configuration path: EHC_V1_CONFIGURATION_PATH (default: config/training.ehc-v1.toml).
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

from ehc_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from ehc_sn.callbacks.diagnostics import DiagnosticsCallback, DiagnosticsSettings
from ehc_sn.callbacks.eval_regimes import EvaluationRegimesCallback, EvaluationRegimesCallbackSettings
from ehc_sn.callbacks.figures import FigureCallbackSettings, FiguresCallback
from ehc_sn.callbacks.metrics import TrainingMetricsCallback
from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.ehc.ehc_v1 import (
    ModelConfig_EHC_V1,
    TrainingModel,
    parse_ehc_v1_config,
)
from ehc_sn.logging.tensorboard import Logger, LoggerSettings
from ehc_sn.training.distributed import (
    resolve_effective_world_size,
    resolve_trainer_strategy,
    validate_batch_size_divisibility,
)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
CONFIGURATION_PATH = os.environ.get("EHC_V1_CONFIGURATION_PATH", "config/training.ehc-v1.toml")


# =================================================================================================
# Run settings (common to both modes)
# =================================================================================================
class RunArguments(BaseSettings, extra="allow", cli_parse_args=True):
    """Common training script arguments. Mode-specific model settings are read from TOML."""

    @classmethod
    def settings_customise_sources(  # ------------------------------------------------------------
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        return CliSettingsSource(settings_cls), *extra

    # -- Mode ----------------------------------------------------------------------------------
    mode: Literal["spatial_pretrain", "controller_pretrain"] = Field(
        ...,
        description="EHC training mode. Determines adapter, controller, objective, and optimizer families.",
    )

    # -- Names and tracking --------------------------------------------------------------------
    project_name: Optional[str] = Field(
        default=None,
        description="Optional project label retained in the entry-point settings for external launchers or downstream metadata.",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Optional run label retained in the entry-point settings for external launchers or downstream metadata.",
    )

    # -- Data ----------------------------------------------------------------------------------
    dataset_path: Path = Field(
        ...,
        description="Path to the processed dataset directory.",
    )
    seed: int = Field(
        default=42,
        description="RNG seed for reproducibility across dataloading, augmentation, and trainer initialization.",
    )
    augment: bool = Field(
        default=True,
        description="Enable dataset-provided training augmentation when the datamodule supports it.",
    )
    global_batch_size: int = Field(
        ...,
        description="Global batch size across all devices. The per-device batch size is derived from world size.",
    )
    num_workers: int = Field(
        default=4,
        description="Number of DataLoader worker processes.",
    )
    prefetch_factor: int = Field(
        default=2,
        description="Number of batches each DataLoader worker prefetches ahead of consumption.",
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin DataLoader host memory before transfer to the accelerator.",
    )
    persistent_workers: bool = Field(
        default=True,
        description="Keep DataLoader workers alive across epochs instead of recreating them each time.",
    )

    # -- Training control ----------------------------------------------------------------------
    max_epochs: int = Field(
        ...,
        description="Maximum number of training epochs.",
    )
    max_steps: int = Field(
        default=200000,
        description="Maximum number of optimizer steps before training stops.",
    )
    log_every_n_steps: int = Field(
        default=10,
        description="Log trainer metrics every N training steps.",
    )
    check_val_every_n_epoch: Optional[int] = Field(
        default=None,
        description="Epoch-based validation cadence. Set to None to rely on val_check_interval instead.",
    )
    val_check_interval: int = Field(
        default=1000,
        description="Step-based validation cadence used when validation is not scheduled purely by epoch.",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show the Lightning progress bar during fit.",
    )

    # -- Callbacks -----------------------------------------------------------------------------
    logger: Optional[LoggerSettings] = Field(
        default_factory=LoggerSettings,
        description="TensorBoard logger settings for this run.",
    )
    checkpoint: Optional[CheckpointSettings] = Field(
        default_factory=CheckpointSettings,
        description="Model checkpoint callback settings.",
    )
    figures: Optional[FigureCallbackSettings] = Field(
        default_factory=FigureCallbackSettings,
        description="Figure-generation callback settings.",
    )
    eval_regimes: Optional[EvaluationRegimesCallbackSettings] = Field(
        default=None,
        description="Named diagnostic evaluation regime settings run by the evaluation callback.",
    )
    diagnostic_level: Literal["minimal", "standard", "research"] = Field(
        default="standard",
        description=(
            "Instrumentation tier. 'minimal' keeps only core training metrics; "
            "'standard' adds routine diagnostics; 'research' enables the fullest diagnostic surface."
        ),
    )

    # -- Distributed ---------------------------------------------------------------------------
    trainer_accelerator: Literal["auto", "gpu", "cpu"] = Field(
        default="gpu",
        description="Lightning Trainer accelerator selection.",
    )
    trainer_strategy: Literal["auto", "ddp"] = Field(
        default="ddp",
        description="Lightning Trainer distributed strategy. Use 'ddp' for multi-process GPU training.",
    )
    trainer_devices: int = Field(
        default=1,
        description="Number of devices per node assigned to the Trainer.",
    )
    trainer_num_nodes: int = Field(
        default=1,
        description="Number of nodes participating in distributed training.",
    )
    trainer_precision: str = Field(
        default="16-mixed",
        description="Lightning precision mode, such as '16-mixed', 'bf16-mixed', or '32-true'.",
    )

    # -- Checkpointing -------------------------------------------------------------------------
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional checkpoint path to resume from via Trainer.fit(ckpt_path=...).",
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Reserved flag for evaluation-triggered checkpointing; currently not consumed by this entrypoint.",
    )
    limit_val_batches: int = Field(
        default=10,
        description="Maximum number of validation batches to run in each validation pass.",
    )
    eval_save_outputs: list[str] = Field(
        default_factory=list,
        description="Reserved list of evaluation artifact keys to persist; currently not consumed by this entrypoint.",
    )

    # ---------------------------------------------------------------------------------------------
    # Aggregate settings (compose leaf settings for modules)

    @property
    def datamodule_config(self) -> DatamoduleConfig:
        return DatamoduleConfig.model_validate(self, from_attributes=True)

    @property
    def diagnostics(self) -> DiagnosticsSettings:
        return DiagnosticsSettings.model_validate(self, from_attributes=True)


# =================================================================================================
# Entrypoint
# =================================================================================================
if __name__ == "__main__":
    raw = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
    settings = RunArguments(**raw)
    # Merge CLI-overridden run-level values (e.g. --mode) and any extra model-config
    # overrides captured via extra="allow" back into raw, so parse_ehc_v1_config sees them.
    _effective = {**raw, "mode": settings.mode, **settings.model_extra}
    ehc_config = parse_ehc_v1_config(_effective)
    world_size = resolve_effective_world_size(
        settings.trainer_strategy,
        settings.trainer_devices,
        settings.trainer_num_nodes,
    )
    validate_batch_size_divisibility(settings.global_batch_size, world_size)

    # Seed everything for reproducibility.
    seed_everything(settings.seed)

    # Datamodule: spatial_pretrain uses plain loader; controller_pretrain applies MazeHard coercion.
    if settings.mode == "controller_pretrain":
        from ehc_sn.adapters.mazehard.hrm.core import coerce_maze_hard_batch

        transform = coerce_maze_hard_batch
    else:
        transform = None

    # Prepare callbacks: checkpointing + optional figure generation.
    callbacks_list = [TrainingMetricsCallback()]
    if settings.checkpoint is not None:
        callbacks_list.append(CheckpointCallback(settings.checkpoint))
    if settings.eval_regimes is not None:
        callbacks_list.append(EvaluationRegimesCallback(settings.eval_regimes))
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
        max_epochs=settings.max_epochs,
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
        model=TrainingModel(ehc_config),
        # Data module: dataset + DataLoader construction.
        datamodule=Datamodule(settings.datamodule_config, transform=transform),
        # Optional: resume training from a checkpoint.
        ckpt_path=settings.checkpoint_path,
    )
