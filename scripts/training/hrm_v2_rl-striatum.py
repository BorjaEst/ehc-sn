""" """

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Optional

import torch
from lightning.pytorch import Trainer, seed_everything
from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ehc_sn.adapters.mazehard.hrm import MazeHardHRMAdapterSettings
from ehc_sn.callbacks.checkpoint import CheckpointCallback, CheckpointSettings
from ehc_sn.callbacks.diagnostics import (
    DiagnosticsCallback,
    DiagnosticsSettings,
)
from ehc_sn.callbacks.evaluation import (
    EvaluationRegimesCallback,
    EvaluationRegimesCallbackSettings,
)
from ehc_sn.callbacks.figures import (
    FigureGenerationCallback,
    FigureGenerationSettings,
)
from ehc_sn.callbacks.lr_monitor import (
    LearningRateMonitor,
    LearningRateMonitorSettings,
)
from ehc_sn.callbacks.metrics import MetricsCallback
from ehc_sn.controllers.deliberation.actor_critic import (
    DeliberationACControllerConfig,
)
from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.hrm.core._base import (
    VALID_INIT_GROUPS,
    load_weights_from_checkpoint,
)
from ehc_sn.lightning.hrm.core.runtime import RuntimeConfig
from ehc_sn.lightning.hrm.hrm_v2 import HRMV2ModelConfig, HRMV2TrainingModel
from ehc_sn.logging.tensorboard import Logger, LoggerSettings
from ehc_sn.objectives import HybridRLLossConfig
from ehc_sn.tasks.mazehard.capabilities.deliberation import (
    MazeHardDeliberationConfig,
)
from ehc_sn.tasks.mazehard.reward import MazeHardRewardConfig
from ehc_sn.tasks.mazehard.runtime import coerce_maze_hard_batch
from ehc_sn.training.distributed import (
    resolve_effective_world_size,
    resolve_trainer_strategy,
    validate_batch_size_divisibility,
)
from ehc_sn.training.optim import AdamATan2Config
from ehc_sn.training.schedules import SchedulerConfig

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
CONFIGURATION_PATH = os.environ.get(
    "HRM_V2_CONFIGURATION_PATH", "config/training/hrm-v2-default.toml"
)


# =============================================================================
# Settings Model
# =============================================================================
class RunArguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    """Settings for HRM v2 deliberation actor-critic training run."""

    model_config = SettingsConfigDict(extra="forbid")

    @classmethod
    def settings_customise_sources(  # ----------------------------------------
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings source order.

        Pydantic Settings supports multiple value sources; we explicitly place
        the CLI first so that command-line overrides always win.
        """
        extra = [
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ]
        return CliSettingsSource(settings_cls), *extra

    # -------------------------------------------------------------------------
    # Names and tracking
    project_name: Optional[str] = Field(
        default=None,
        description=(
            "Project name. If not set, it defaults to the capitalized name of "
            "the dataset (e.g. `MATH` -> `Math ACT-torch`)."
        ),
    )
    run_name: Optional[str] = Field(
        default=None,
        description=(
            "Run name. If not set, it defaults to `<arch_name> <random_slug>` "
            "(e.g. `HrmV2 2x128 4L 16H 0.1D ACT-torch cool-slug`)."
        ),
    )

    # --------------------------------------------------------------------------
    # Model architecture and data
    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file that specifies the HRM v2 architecture.",
    )
    adapter: MazeHardHRMAdapterSettings = Field(
        default_factory=MazeHardHRMAdapterSettings,
        description="Adapter settings for the MazeHard environment and HRM v2 model.",
    )
    deliberation: MazeHardDeliberationConfig = Field(
        ...,
        description="Deliberation capability config (halt_action, episode_horizon) for the MazeHard deliberation path.",
    )
    reward: MazeHardRewardConfig = Field(
        default_factory=MazeHardRewardConfig,
        description="Reward configuration for MazeHard stop-time projection.",
    )
    controller: DeliberationACControllerConfig = Field(
        default_factory=DeliberationACControllerConfig,
        description="Deliberation actor-critic controller configuration (policy settings).",
    )
    objective: HybridRLLossConfig = Field(
        ...,
        description="Hybrid RL objective configuration (loss function, discount factor, loss coefficients).",
    )

    # ~~ Optimizers & scheduling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    optimizer_supervised: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimizer for supervised parameters (PFC + embeddings + Token head).",
    )
    optimizer_rl: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimizer for RL parameters (STR actor-critic only).",
    )
    optimizer_qv: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimizer for vmPFC parameters (pfc.estimator only).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config applied to both optimizers.",
    )
    supervised_only_warmup_steps: int = Field(
        default=5000,
        ge=0,
        description=(
            "Number of optimizer steps during which only the supervised optimizer trains. "
            "STR and vmPFC are frozen; allow_halt=False forces full deliberation. "
            "Prevents 'halt immediately' collapse before PFC representations are informative."
        ),
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="HRM runtime-owned validation safety settings.",
    )

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Training control settings (passed as top-level settings for ease of CLI overrides)
    max_epochs: int = Field(
        ...,
        description="Total number of epochs to train.",
    )

    # -------------------------------------------------------------------------
    # Core settings for model, data, and training configuration (passed as configs to modules)
    logger: Optional[LoggerSettings] = Field(
        default_factory=LoggerSettings,
        description="TensorBoard logger settings.",
    )
    lr_monitor: Optional[LearningRateMonitorSettings] = Field(
        default=None,
        description="Optional LearningRateMonitor callback settings.",
    )
    checkpoint: Optional[CheckpointSettings] = Field(
        default_factory=CheckpointSettings,
        description="Model checkpoint settings.",
    )
    eval_regimes: Optional[EvaluationRegimesCallbackSettings] = Field(
        default=None,
        description="Optional named replay-evaluation regime callback settings.",
    )
    figures: Optional[FigureGenerationSettings] = Field(
        default=None,
        description="Optional standalone figure generation callback settings.",
    )
    diagnostic_level: Literal["minimal", "standard", "research"] = Field(
        default="standard",
        description=(
            "Instrumentation tier. 'minimal': only training metrics. "
            "'standard': training metrics + model health diagnostics. "
            "'research': all available diagnostic signals."
        ),
    )

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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
            "Lightning Trainer precision. '32-true' = full fp32 (paper-parity default). "
            "Use 'bf16-mixed' for throughput on Ampere+."
        ),
    )

    # -------------------------------------------------------------------------
    # Checkpointing and evaluation settings (passed as kwargs to Trainer and Checkpoint callback)
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Optional checkpoint path to resume full trainer state via Trainer.fit(ckpt_path=...).",
    )
    init_weights_from: Optional[str] = Field(
        default=None,
        description="Optional checkpoint path for model-weight initialization only. Distinct from resume_from_checkpoint.",
    )
    init_weights_groups: list[str] = Field(
        default_factory=lambda: ["all"],
        description="Named HRM semantic groups to hydrate from init_weights_from. Valid groups: pfc_core, striatum, all.",
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Whether to checkpoint the model after every evaluation.",
    )
    limit_val_batches: float = Field(
        default=1.0,
        description=(
            "Fraction of validation batches to run. Use 1.0 for full validation "
            "coverage; values < 1.0 cap the run."
        ),
    )
    eval_save_outputs: list[str] = Field(
        default_factory=list,
        description="Evaluation output keys saved as tensors in the checkpoint directory.",
    )

    @model_validator(mode="after")
    def _validate_transfer_init_options(self) -> "RunArguments":
        if (
            self.resume_from_checkpoint is not None
            and self.init_weights_from is not None
        ):
            raise ValueError(
                "resume_from_checkpoint and init_weights_from are mutually exclusive."
            )
        if self.init_weights_from is not None:
            if not self.init_weights_groups:
                raise ValueError("init_weights_groups must be non-empty.")
            unknown = [
                g
                for g in self.init_weights_groups
                if g not in VALID_INIT_GROUPS
            ]
            if unknown:
                raise ValueError(
                    f"Unknown init_weights_groups: {unknown!r}. "
                    f"Valid groups: {sorted(VALID_INIT_GROUPS)!r}."
                )
        return self

    # -------------------------------------------------------------------------
    # Aggregate settings (compose leaf settings for modules)
    @property
    def hrm_config(self) -> HRMV2ModelConfig:
        """Compose HRMV2ModelConfig from leaf settings."""
        return HRMV2ModelConfig.model_validate(self, from_attributes=True)

    @property
    def datamodule(self) -> DatamoduleConfig:
        """Compose DatamoduleConfig from leaf settings."""
        return DatamoduleConfig.model_validate(self, from_attributes=True)

    @property
    def diagnostics(self) -> DiagnosticsSettings:
        """Compose DiagnosticsSettings from leaf settings."""
        return DiagnosticsSettings.model_validate(self, from_attributes=True)


# =============================================================================
# Main Entrypoint
# =============================================================================
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

    # Build logger first so callback wiring can follow the same gate.
    logger = Logger(settings.logger) if settings.logger is not None else None

    # Prepare callbacks: metrics + checkpointing + optional diagnostics/LR telemetry.
    callbacks_list = [MetricsCallback()]
    if settings.eval_regimes is not None:
        callbacks_list.append(EvaluationRegimesCallback(settings.eval_regimes))
    if settings.checkpoint is not None:
        callbacks_list.append(CheckpointCallback(settings.checkpoint))
    if settings.diagnostic_level != "minimal":
        callbacks_list.append(DiagnosticsCallback(settings.diagnostics))
    if settings.lr_monitor is not None:
        callbacks_list.append(LearningRateMonitor(settings.lr_monitor))
    if settings.figures is not None:
        callbacks_list.append(FigureGenerationCallback(settings.figures))

    # Build the PyTorch Lightning Trainer.
    # This wires together logging, callbacks, and training control.
    trainer = Trainer(
        # Logger + callbacks handle metrics/hparams and checkpointing.
        logger=logger,
        callbacks=callbacks_list if callbacks_list else None,
        # Lightning Trainer kwargs (extracted from config)
        accelerator=settings.trainer_accelerator,
        strategy=resolve_trainer_strategy(
            settings.trainer_strategy, world_size, find_unused_parameters=True
        ),
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
    training_model = HRMV2TrainingModel(settings.hrm_config)

    # Optional: initialize model weights from a checkpoint without restoring
    # trainer/optimizer/scheduler state.
    if settings.init_weights_from is not None:
        loaded_keys = load_weights_from_checkpoint(
            training_model.model,
            settings.init_weights_from,
            settings.init_weights_groups,
        )
        print(
            f"[init_weights_from] Loaded {len(loaded_keys)} parameter keys "
            f"(groups={settings.init_weights_groups}) from {settings.init_weights_from!r}."
        )

    trainer.fit(
        # Lightning module: training step, optimizer and schedule setup.
        model=training_model,
        # Data module: dataset + DataLoader construction.
        datamodule=Datamodule(
            settings.datamodule, transform=coerce_maze_hard_batch
        ),
        # Optional: resume training from a checkpoint.
        ckpt_path=settings.resume_from_checkpoint,
    )
