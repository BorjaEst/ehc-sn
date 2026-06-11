"""Training entrypoint for TEM v2 experiments."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ehc_sn.adapters.tem import ArenaTEMAdapterSettings
from ehc_sn.controllers.replay.trajectory import (
    ReplayTrajectoryControllerConfig,
)
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.data.manifest import read_manifest
from ehc_sn.experiments.arena.tem_v2 import build_experiment as build_tem_v2
from ehc_sn.lightning.callbacks.checkpoint import CheckpointSettings
from ehc_sn.lightning.callbacks.diagnostics import DiagnosticsSettings
from ehc_sn.lightning.callbacks.evaluation import (
    EvaluationRegimesCallbackSettings,
)
from ehc_sn.lightning.callbacks.figures import FigureGenerationSettings
from ehc_sn.lightning.callbacks.lr_monitor import (
    LearningRateMonitorSettings,
)
from ehc_sn.lightning.modules.variational_replay import (
    VariationalReplayConfig as TEMV2ModelConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives import TEMObjectiveConfig
from ehc_sn.training.optim import AdamConfig
from ehc_sn.training.runner import TrainingEntrypointSpec, run_training
from ehc_sn.training.schedules import SchedulerConfig
from ehc_sn.training.tem import (
    VALID_INIT_GROUPS,
    RuntimeConfig,
    load_weights_from_checkpoint,
)

CONFIGURATION_PATH = os.environ.get(
    "TEM_V2_CONFIGURATION_PATH", "config/training/tem-v2-default-vram8gib.toml"
)


# =============================================================================
# Settings Model
# =============================================================================
class RunArguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    """CLI and TOML settings for TEM v2 training runs."""

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
        description=(
            "Project name. If not set, it defaults to the capitalized name of the dataset "
            "(for example `Dungeon` -> `Dungeon`)."
        ),
    )
    run_name: Optional[str] = Field(
        default=None,
        description=(
            "Run name. If not set, it defaults to `<arch_name> <random_slug>` "
            "(for example `tem-v2 cool-slug`)."
        ),
    )

    # -------------------------------------------------------------------------
    # Model architecture and data
    model_config_path: Path = Field(
        ...,
        description="Path to the model configuration TOML file that specifies the TEM v2 architecture.",
    )
    adapter: ArenaTEMAdapterSettings = Field(
        ...,
        description="Settings for the arena bridge adapter that binds TEM v2 to task inputs/outputs.",
    )
    controller: ReplayTrajectoryControllerConfig = Field(
        ...,
        description="Replay trajectory controller configuration (window_size for fixed-window TBPTT).",
    )
    objective: TEMObjectiveConfig = Field(
        default_factory=TEMObjectiveConfig,
        description="TEM objective configuration (observation, latent, regularization).",
    )

    # -------------------------------------------------------------------------
    # Optimizers & scheduling
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
    non_finite_policy: Literal["drop", "raise"] = Field(
        default="drop",
        description=(
            "Policy for NaN/Inf scalar diagnostic values. Set to 'raise' "
            "to fail fast instead of silently dropping NaN values."
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
        description="Named TEM semantic groups to hydrate from init_weights_from. Valid groups: spatial_memory, path_integration, sensory_binding, all.",
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Whether to checkpoint the model after every evaluation.",
    )
    limit_val_batches: int | float = Field(
        default=10,
        description="Validation batches to run. ``int`` = N batches; ``float`` = "
        "fraction of validation set (1.0 = 100%).",
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

    @model_validator(mode="after")
    def _validate_action_count_with_corpus(self) -> "RunArguments":
        """Assert that ``adapter.action_count`` matches the corpus manifest.

        Reads ``manifest.json`` from ``dataset_path`` and checks the stored
        ``action_count`` field agrees with the model adapter's setting.  This
        catches mismatches when switching between corpora with different action
        spaces (e.g. square dungeon v1 = 5 actions, hex openfield = 7 actions).
        """
        manifest = read_manifest(self.dataset_path)
        corpus_action_count = int(manifest["action_count"])
        if corpus_action_count != self.adapter.action_count:
            raise ValueError(
                f"Action-count mismatch: corpus at {self.dataset_path} declares "
                f"action_count={corpus_action_count} but adapter config has "
                f"action_count={self.adapter.action_count}. "
                f"Update the training config to match the target corpus."
            )
        return self

    # -------------------------------------------------------------------------
    # Aggregate settings (compose leaf settings for modules)
    @property
    def tem_config(self) -> TEMV2ModelConfig:
        """Compose TEMV2ModelConfig from leaf settings."""
        return TEMV2ModelConfig.model_validate(self, from_attributes=True)

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
        build_experiment=build_tem_v2,
        datamodule_transform=None,
        find_unused_parameters=True,
        load_weights_from_checkpoint=load_weights_from_checkpoint,
    )
    run_training(settings, spec)
