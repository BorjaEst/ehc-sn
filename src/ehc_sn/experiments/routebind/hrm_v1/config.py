"""Configuration schemas for the Routebind × HRM-v1 experiment pairing.

Hierarchy (training):

    RoutebindHRMV1TrainingExperimentConfig
    ├── model: RoutebindHRMV1ModelConfig
    │   └── components: RoutebindHRMV1ComponentConfigs
    │       ├── adapter: RoutebindHRMAdapterSettings
    │       ├── controller: ACTControllerConfig
    │       └── objective: ACTSupervisedScorerConfig
    ├── training: ACTSupervisedTrainingConfig
    ├── data: DatamoduleConfig
    ├── trainer: TrainerConfig
    ├── checkpointing: CheckpointingConfig
    └── logging: LoggerSettings
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ehc_sn.adapters.hrm import RoutebindHRMAdapterSettings
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
from ehc_sn.objectives.task.routebind import RoutebindTaskEvaluatorConfig
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig


# =============================================================================
class RoutebindHRMV1ComponentConfigs(BaseModel, extra="forbid"):
    """Component-level config for Routebind × HRM-v1."""

    adapter: RoutebindHRMAdapterSettings = Field(
        ...,
        description="Routebind adapter settings for HRM v1.",
    )
    controller: ACTControllerConfig = Field(
        ...,
        description="ACT deliberation controller configuration "
        "(bypassed when single_step=true).",
    )
    task_evaluator: RoutebindTaskEvaluatorConfig = Field(
        ...,
        description="Routebind task-evaluator configuration.",
    )
    scorer: ACTSupervisedScorerConfig = Field(
        default_factory=ACTSupervisedScorerConfig,
        description="ACT loss composition configuration.",
    )


# =============================================================================
class RoutebindHRMV1ModelConfig(BaseModel, extra="forbid"):
    """Model-level config for Routebind × HRM-v1."""

    model_config_path: Path = Field(
        ...,
        description="Path to the HRM v1 model architecture TOML file.",
    )
    components: RoutebindHRMV1ComponentConfigs = Field(
        ...,
        description="Task-family component binding "
        "(adapter, controller, objective).",
    )


# =============================================================================
class RoutebindHRMV1TrainingExperimentConfig(BaseModel, extra="forbid"):
    """Full training application configuration for Routebind × HRM-v1."""

    model: RoutebindHRMV1ModelConfig = Field(
        ...,
        description="Model structure (components + architecture path).",
    )
    regime: ACTSupervisedConfig = Field(
        default_factory=ACTSupervisedConfig,
        description="ACT regime configuration (halt_disabled_steps, "
        "target_network, single_step).",
    )
    training: ACTSupervisedTrainingConfig = Field(
        ...,
        description="ACT supervised training configuration.",
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
        description="Checkpointing and diagnostic settings.",
    )
    logging: LoggerSettings = Field(
        default_factory=LoggerSettings,
        description="Logging and TensorBoard settings.",
    )


# =============================================================================
class RoutebindHRMV1EvaluationExperimentConfig(BaseModel, extra="forbid"):
    """Full evaluation application configuration for Routebind × HRM-v1."""

    model: RoutebindHRMV1ModelConfig = Field(
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
    "RoutebindHRMV1ComponentConfigs",
    "RoutebindHRMV1EvaluationExperimentConfig",
    "RoutebindHRMV1ModelConfig",
    "RoutebindHRMV1TrainingExperimentConfig",
]
