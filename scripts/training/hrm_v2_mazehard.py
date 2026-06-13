"""Training entrypoint for HRM v2 (deliberation actor-critic) experiments."""

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

from ehc_sn.adapters.hrm import MazeHardHRMAdapterSettings
from ehc_sn.controllers.deliberation.actor_critic import (
    DeliberationACControllerConfig,
)
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.experiments.mazehard.hrm_v2 import build_experiment as build_hrm_v2
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
from ehc_sn.objectives import HybridRLLossConfig
from ehc_sn.tasks.mazehard.evaluators.step import (
    MazeHardStepEvaluatorConfig,
)
from ehc_sn.tasks.mazehard.reward import MazeHardRewardConfig
from ehc_sn.tasks.mazehard.runtime import coerce_maze_hard_batch
from ehc_sn.training.hrm import (
    VALID_INIT_GROUPS,
    RuntimeConfig,
    load_weights_from_checkpoint,
)
from ehc_sn.training.optim import AdamATan2Config
from ehc_sn.training.runner import TrainingEntrypointSpec, run_training
from ehc_sn.training.schedules import SchedulerConfig

CONFIGURATION_PATH = os.environ.get(
    "HRM_V2_CONFIGURATION_PATH", "config/training/hrm-v2-default-vram8gib.toml"
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
    deliberation: MazeHardStepEvaluatorConfig = Field(
        ...,
        description="Step evaluator config (halt_action, episode_horizon) for the MazeHard deliberation path.",
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
        description="Named HRM semantic groups to hydrate from init_weights_from. Valid groups: pfc_core, striatum, all.",
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Whether to checkpoint the model after every evaluation.",
    )
    limit_val_batches: int | float = Field(
        default=1.0,
        description=(
            "Validation batches to run. ``int`` = N batches; "
            "``float`` = fraction of validation set (1.0 = 100%)."
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
        build_experiment=build_hrm_v2,
        datamodule_transform=coerce_maze_hard_batch,
        find_unused_parameters=True,
        load_weights_from_checkpoint=load_weights_from_checkpoint,
    )
    run_training(settings, spec)
