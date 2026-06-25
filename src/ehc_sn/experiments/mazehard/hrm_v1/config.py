"""Configuration schemas for the MazeHard × HRM-v1 experiment pairing.

Hierarchy (training):

    MazeHardHRMV1TrainingExperimentConfig
    ├── model: MazeHardHRMV1ModelConfig
    │   └── components: MazeHardHRMV1ComponentConfigs
    │       ├── adapter: MazeHardHRMAdapterSettings
    │       ├── controller: ACTControllerConfig
    │       └── objective: ACTSupervisedScorerConfig
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
    ├── checkpoint: EvaluationCheckpointConfig
    ├── dataset: EvaluationDatasetConfig
    ├── evaluation: EvaluationConfig
    └── output: EvaluationOutputConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import MazeHardHRMAdapterSettings
from ehc_sn.controllers.deliberation.act import ACTControllerConfig
from ehc_sn.data.datamodules import DatamoduleConfig
from ehc_sn.experiments._infra import (
    CaptureConfig,
    CheckpointingConfig,
    ProviderConfig,
    RegimeConfig,
    TrainerConfig,
)
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedConfig,
    ACTSupervisedTrainingConfig,
)
from ehc_sn.logging.tensorboard import LoggerSettings
from ehc_sn.objectives.composites.act import ACTSupervisedScorerConfig
from ehc_sn.objectives.task.mazehard import MazeHardTaskEvaluatorConfig
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig

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
    task_evaluator: MazeHardTaskEvaluatorConfig = Field(
        ...,
        description="MazeHard task-evaluator configuration.",
    )
    scorer: ACTSupervisedScorerConfig = Field(
        default_factory=ACTSupervisedScorerConfig,
        description="ACT loss composition configuration.",
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

    model: MazeHardHRMV1ModelConfig = Field(
        ...,
        description="Model structure (components + architecture path).",
    )
    regime: ACTSupervisedConfig = Field(
        default_factory=ACTSupervisedConfig,
        description="ACT regime configuration (halt_disabled_steps, "
        "target_network).",
    )
    training: ACTSupervisedTrainingConfig = Field(
        ...,
        description="ACT supervised training configuration (optimizer, runtime). "
        "Deliberation settings are separate at the model level.",
    )
    execution: Optional[HRMRuntimeConfig] = Field(
        default=None,
        description="Execution policy (validation safety limits).",
    )
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
    execution: Optional[HRMRuntimeConfig] = Field(
        default=None,
        description="Execution policy for eval-time rollout bounds.",
    )
    provider: ProviderConfig = Field(
        ...,
        description="Evaluation data provider specification.",
    )
    regime: RegimeConfig = Field(
        ...,
        description="Evaluation regime identity.",
    )
    capture: CaptureConfig = Field(
        default_factory=lambda: CaptureConfig(),
        description="Trace capture policy.",
    )


# =============================================================================
__all__ = [
    "MazeHardHRMV1ComponentConfigs",
    "MazeHardHRMV1ModelConfig",
    "MazeHardHRMV1TrainingExperimentConfig",
    "MazeHardHRMV1EvaluationExperimentConfig",
]
