"""Configuration schemas for the MazeHard × HRM-v1 experiment pairing.

Hierarchy (training):

    MazeHardHRMV1TrainingExperimentConfig
    ├── model: MazeHardHRMV1ModelConfig
    │   └── components: MazeHardHRMV1ComponentConfigs
    │       ├── adapter: MazeHardHRMAdapterSettings
    │       ├── controller: ACTControllerConfig
    │       └── objective: ACTObjectiveConfig
    ├── deliberation: MazeHardDeliberationConfig
    │   ├── halt_action
    │   └── episode_horizon
    ├── training: ACTSupervisedTrainingConfig
    │   ├── optimizer
    │   └── runtime
    ├── data: DatamoduleConfig
    ├── trainer: TrainerSettings
    ├── checkpointing: CheckpointSettings
    └── logging: LoggerSettings

Hierarchy (evaluation):

    MazeHardHRMV1EvaluationExperimentConfig
    ├── model: MazeHardHRMV1ModelConfig (same as above)
    ├── deliberation: MazeHardDeliberationConfig (same as above)
    ├── checkpoint: EvaluationCheckpointConfig
    ├── dataset: EvaluationDatasetConfig
    ├── evaluation: EvaluationConfig
    └── output: EvaluationOutputConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import MazeHardHRMAdapterSettings
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

# =============================================================================
# Deliberation configuration (execution policy, not model structure)
# =============================================================================


class MazeHardDeliberationConfig(BaseModel, extra="forbid"):
    """MazeHard deliberation / execution policy settings.

    Controls how the ACT controller runs during inference and evaluation.
    Not part of model structure — does not affect parameter shapes or
    checkpoint compatibility.
    """

    halt_action: int = Field(
        default=0,
        ge=0,
        description="Action index that signals episode termination.",
    )
    episode_horizon: int = Field(
        default=16,
        ge=1,
        description="Maximum number of deliberation steps per episode.",
    )


# =============================================================================
# Model configuration (computational structure only)
# =============================================================================


class MazeHardHRMV1ComponentConfigs(BaseModel, extra="forbid"):
    """Component-level config for MazeHard × HRM-v1.

    Explicit per-task-family schema — not a generic wrapper.
    """

    adapter: MazeHardHRMAdapterSettings = Field(
        ...,
        description="MazeHard adapter settings for HRM v1.",
    )
    controller: ACTControllerConfig = Field(
        ...,
        description="ACT deliberation controller configuration.",
    )
    objective: ACTObjectiveConfig = Field(
        ...,
        description="ACT supervised objective configuration.",
    )


class MazeHardHRMV1ModelConfig(BaseModel, extra="forbid"):
    """Model-level config for MazeHard × HRM-v1.

    Contains computational structure and the model architecture TOML
    path.  Deliberation and training configs are peers at the experiment
    level.
    """

    model_config_path: Path = Field(
        ...,
        description="Path to the HRM v1 model architecture TOML file.",
    )
    components: MazeHardHRMV1ComponentConfigs = Field(
        ...,
        description="Task-family component binding (adapter, controller, objective).",
    )


# =============================================================================
# Trainer / infra stubs (structured configs, minimal validation for now)
# =============================================================================


class TrainerConfig(BaseModel, extra="forbid"):
    """Lightning Trainer configuration fields."""

    accelerator: Literal["auto", "gpu", "cpu"] = Field(
        default="gpu",
        description="Trainer accelerator setting.",
    )
    strategy: Literal["auto", "ddp"] = Field(
        default="ddp",
        description="Trainer DDP strategy.",
    )
    devices: int = Field(
        default=1,
        description="Number of devices per node.",
    )
    num_nodes: int = Field(
        default=1,
        description="Number of nodes.",
    )
    precision: str = Field(
        default="16-mixed",
        description="Training precision.",
    )
    max_steps: int = Field(
        default=200000,
        description="Maximum training steps.",
    )
    val_check_interval: int = Field(
        default=500,
        description="Validation check interval in steps.",
    )
    log_every_n_steps: int = Field(
        default=10,
        description="Log metrics every N steps.",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress bar.",
    )
    limit_val_batches: int | float = Field(
        default=1.0,
        description="Validation batches (int=N, float=fraction).",
    )
    seed: int = Field(
        default=42,
        description="RNG seed for reproducibility.",
    )
    find_unused_parameters: bool = Field(
        default=False,
        description="Enable DDP unused-parameter detection for conditional forward graphs.",
    )


class CheckpointingConfig(BaseModel, extra="forbid"):
    """Checkpoint and weight-init settings."""

    checkpoint: Optional[CheckpointSettings] = Field(
        default_factory=CheckpointSettings,
        description="Model checkpoint settings.",
    )
    resume_from: Optional[str] = Field(
        default=None,
        description="Checkpoint path to resume full trainer state.",
    )
    init_weights_from: Optional[str] = Field(
        default=None,
        description="Checkpoint path for model-weight initialization only.",
    )
    init_weights_groups: list[str] = Field(
        default_factory=lambda: ["all"],
        description="Named HRM semantic groups to hydrate.",
    )
    supervised_only_warmup_steps: int = Field(
        default=500,
        ge=0,
        description="Steps with learned halting disabled.",
    )
    eval_save_outputs: list[str] = Field(
        default_factory=list,
        description="Eval output keys saved as tensors.",
    )
    diagnostic_level: Literal["minimal", "standard", "research"] = Field(
        default="standard",
        description="Instrumentation tier.",
    )
    non_finite_policy: Literal["drop", "raise"] = Field(
        default="raise",
        description="Policy for NaN/Inf diagnostics.",
    )

    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config.",
    )
    target_network: TargetNetworkConfig = Field(
        default_factory=TargetNetworkConfig,
        description="EMA-lagged target network config.",
    )
    checkpoint_every_eval: bool = Field(
        default=False,
        description="Checkpoint after every evaluation.",
    )


# =============================================================================
# Experiment-level configurations
# =============================================================================


class MazeHardHRMV1TrainingExperimentConfig(BaseModel, extra="forbid"):
    """Full training application configuration for MazeHard × HRM-v1.

    Nested structure.  Use :meth:`model_validate` for TOML loading.
    CLI overrides are passed via ``_cli_parse_args`` at the script
    boundary when using :class:`BaseSettings` wrapper (see training
    script).
    """

    # --- Top-level sections --------------------------------------------------
    model: MazeHardHRMV1ModelConfig
    deliberation: MazeHardDeliberationConfig = MazeHardDeliberationConfig()
    training: ACTSupervisedTrainingConfig
    data: DatamoduleConfig = Field(
        ...,
        description="Dataset and DataLoader settings.",
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Lightning Trainer settings.",
    )
    checkpointing: CheckpointingConfig = Field(
        default_factory=CheckpointingConfig,
        description="Checkpoint, weight-init, and diagnostic settings.",
    )
    logging: Optional[LoggerSettings] = Field(
        default_factory=LoggerSettings,
        description="TensorBoard logger settings.",
    )


class MazeHardHRMV1EvaluationExperimentConfig(BaseModel, extra="forbid"):
    """Full evaluation application configuration for MazeHard × HRM-v1."""

    model: MazeHardHRMV1ModelConfig = Field(
        ...,
        description="Model structure (components only).",
    )
    deliberation: MazeHardDeliberationConfig = Field(
        default_factory=MazeHardDeliberationConfig,
        description="Execution / deliberation policy. Defaults are safe for evaluation.",
    )


# =============================================================================
__all__ = [
    "MazeHardDeliberationConfig",
    "MazeHardHRMV1ComponentConfigs",
    "MazeHardHRMV1ModelConfig",
    "MazeHardHRMV1TrainingExperimentConfig",
    "MazeHardHRMV1EvaluationExperimentConfig",
    "TrainerConfig",
    "CheckpointingConfig",
]
