"""Training entrypoint for HRM v1 SeqMaze (ACT-supervised)."""

from __future__ import annotations

import os
import tomllib
import warnings
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ehc_sn.controllers.deliberation.act import ACTControllerConfig
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.experiments.seqmaze.hrm_v1 import (
    build_experiment as build_seqmaze_v1,
)
from ehc_sn.lightning.callbacks.checkpoint import CheckpointSettings
from ehc_sn.lightning.callbacks.diagnostics import DiagnosticsSettings
from ehc_sn.lightning.callbacks.evaluation import (
    EvaluationRegimesCallbackSettings,
)
from ehc_sn.lightning.callbacks.figures import FigureGenerationSettings
from ehc_sn.lightning.callbacks.lr_monitor import (
    LearningRateMonitorSettings,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives import ACTObjectiveConfig
from ehc_sn.tasks.seqmaze.runtime import extract_seqmaze_task_input
from ehc_sn.training.hrm import (
    VALID_INIT_GROUPS,
    RuntimeConfig,
    load_weights_from_checkpoint,
)
from ehc_sn.training.optim import AdamATan2Config
from ehc_sn.training.runner import TrainingEntrypointSpec, run_training
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.stabilization import TargetNetworkConfig

# Suppress Lightning's manual-optimization checkpoint warning.
warnings.filterwarnings(
    "ignore",
    message=".*ModelCheckpoint with manual optimization.*pre-optimization.*",
    category=UserWarning,
)


CONFIGURATION_PATH = os.environ.get(
    "HRM_V1_CONFIGURATION_PATH", "config/training/hrm-v2-seqmaze-vram8gib.toml"
)


# =============================================================================
# Settings Model
# =============================================================================
class RunArguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    """Common training script arguments. Mode-specific model settings are read from TOML."""

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
        extra = [init_settings, env_settings, dotenv_settings, file_secret_settings]  # fmt: skip
        return CliSettingsSource(settings_cls), *extra

    # -------------------------------------------------------------------------
    # Names and tracking
    project_name: Optional[str] = Field(
        default=None,
        description="Project name. If not set, it defaults to the capitalized "
        "name of the dataset (e.g. `MATH` -> `Math ACT-torch`).",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Run name. If not set, it defaults to `<arch_name> <random_slug>` "
        "(e.g. `HrmV1 2x128 4L 16H 0.1D ACT-torch cool-slug`).",
    )

    # -------------------------------------------------------------------------
    # Model architecture and data
    model_config_path: Path = Field(
        ...,
        description="Path to the HRM v1 model configuration TOML file.",
    )
    adapter: dict = Field(
        ...,
        description="SeqMaze adapter settings that binds the HRM core "
        "(n_max, t_max, k_max, edge_encoding, hidden_size).",
    )
    controller: ACTControllerConfig = Field(
        ...,
        description=(
            "Configuration for the ACT controller, which manages halting and "
            "partial resets during training. The keys in `controller` are "
            "passed to the ACTController constructor."
        ),
    )
    objective: ACTObjectiveConfig = Field(
        ...,
        description="Objective config. The keys in `objective` are passed to "
        "the ACT objective constructor.",
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
            "Learning rate scheduler config. If not set, no learning rate "
            "scheduling is applied. The keys in `scheduler` are passed to the "
            "scheduler constructor."
        ),
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="HRM runtime-owned validation safety settings.",
    )
    target_network: TargetNetworkConfig = Field(
        default_factory=TargetNetworkConfig,
        description="Optional EMA-lagged target network config for "
        "q_continue bootstrap stabilization.",
    )
    supervised_only_warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of optimizer steps during which learned halting "
        "is disabled (allow_halt=False). Pattern-matched from "
        "HRM-v2 warmup phase.",
    )

    # -------------------------------------------------------------------------
    # Data settings
    dataset_path: Path = Field(
        ...,
        description="Path to the processed dataset directory (contains "
        "index.jsonl + NPZ files).",
    )
    seed: int = Field(
        42,
        description="RNG seed for training-split dihedral augmentation and "
        "reproducibility.",
    )
    augment: bool = Field(
        False,
        init=False,  # Augmentation not implemented for seqmaze.
        description="Not used for seqmaze.",
    )
    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. The per-device batch size "
            "is computed as `global_batch_size // world_size`."
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
    # Core settings
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
    diagnostic_level: str = Field(
        default="standard",
        description=(
            "Instrumentation tier. 'minimal': only training metrics. "
            "'standard': training metrics + model health diagnostics. "
            "'research': all available diagnostic signals."
        ),
    )
    non_finite_policy: str = Field(
        default="raise",
        description=(
            "Policy for NaN/Inf scalar diagnostic values. Set to 'raise' "
            "to fail fast instead of silently dropping NaN values."
        ),
    )

    # -------------------------------------------------------------------------
    # Training control
    max_steps: int = Field(
        default=200000,
        description="Maximum training steps.",
    )
    log_every_n_steps: int = Field(
        default=10,
        description="Log metrics every N steps.",
    )
    val_check_interval: int = Field(
        default=0,
        description="Validation check interval (in training steps)."
        " 0 disables validation during training.",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress bar during training.",
    )

    # -------------------------------------------------------------------------
    # Distributed training
    trainer_accelerator: str = Field(
        default="gpu",
        description="Trainer accelerator setting. Use 'gpu' for HAICORE "
        "multi-GPU runs.",
    )
    trainer_strategy: str = Field(
        default="ddp",
        description="Trainer strategy setting. Use 'ddp' for SLURM "
        "multi-GPU runs.",
    )
    trainer_devices: int = Field(
        default=1,
        description="Number of devices per node for the Trainer "
        "(per process when using SLURM tasks).",
    )
    trainer_num_nodes: int = Field(
        default=1,
        description="Number of nodes for distributed training.",
    )
    trainer_precision: str = Field(
        default="16-mixed",
        description=(
            "Lightning Trainer precision. '32-true' = full fp32 (paper-parity "
            "default). Use 'bf16-mixed' for throughput on Ampere+."
        ),
    )

    # -------------------------------------------------------------------------
    # Checkpointing and evaluation
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Optional checkpoint path to resume full trainer state via "
        "Trainer.fit(ckpt_path=...).",
    )
    init_weights_from: Optional[str] = Field(
        default=None,
        description="Optional checkpoint path for model-weight initialization "
        "only. Distinct from resume_from_checkpoint.",
    )
    init_weights_groups: list[str] = Field(
        default_factory=lambda: ["all"],
        description="Named HRM semantic groups to hydrate from "
        "init_weights_from. Valid groups: pfc_core, striatum, all.",
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Whether to checkpoint the model after every evaluation.",
    )
    limit_val_batches: int | float = Field(
        default=1.0,
        description="Validation batches to run. ``int`` = N batches; "
        "``float`` = fraction of validation set (1.0 = 100%).",
    )
    eval_save_outputs: list[str] = Field(
        default_factory=list,
        description="Evaluation output keys saved as tensors in the "
        "checkpoint directory.",
    )

    @model_validator(mode="after")
    def _validate_transfer_init_options(self) -> "RunArguments":
        if (
            self.resume_from_checkpoint is not None
            and self.init_weights_from is not None
        ):
            raise ValueError(
                "resume_from_checkpoint and init_weights_from are mutually "
                "exclusive."
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
    defaults_from_path = tomllib.load(Path(CONFIGURATION_PATH).open("rb"))
    settings = RunArguments(**defaults_from_path)
    spec = TrainingEntrypointSpec(
        build_experiment=build_seqmaze_v1,
        datamodule_transform=extract_seqmaze_task_input,
        find_unused_parameters=False,
        load_weights_from_checkpoint=load_weights_from_checkpoint,
    )
    run_training(settings, spec)
