"""Training entrypoint for HRM v1 SeqMaze (ACT-supervised)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ehc_sn.experiments.seqmaze.hrm_v1.config import (
    SeqMazeHRMV1TrainingExperimentConfig,
)
from ehc_sn.experiments.seqmaze.hrm_v1.training import (
    build_seqmaze_hrm_v1_training_experiment,
)
from ehc_sn.training.runner import run_training

CONFIG_ENV_VAR = "HRM_V1_CONFIGURATION_PATH"
DEFAULT_CONFIG = "config/training/hrm-v1-seqmaze-vram8gib.toml"


# =============================================================================
# CLI settings wrapper — TOML provides defaults, CLI overrides via dotted paths
# =============================================================================
class CliSettings(BaseSettings):
    """CLI entry point for HRM v1 SeqMaze training.

    Loads defaults from TOML, then applies CLI overrides using
    dotted-path syntax: ``--experiment.training.optimizer.lr 0.001``.
    """

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        extra="forbid",
        cli_ignore_unknown_args=False,
    )

    experiment: SeqMazeHRMV1TrainingExperimentConfig = Field(
        ...,
        description="Experiment configuration. Defaults are loaded from a TOML file, "
        "but can be overridden via CLI using dotted paths, e.g. "
        "`--experiment.training.optimizer.lr 0.001`.",
    )


# =============================================================================
# Main Entrypoint
# =============================================================================
if __name__ == "__main__":
    toml_path = Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG))
    toml_defaults = tomllib.loads(toml_path.read_text())
    settings = CliSettings(experiment=toml_defaults)
    experiment = build_seqmaze_hrm_v1_training_experiment(settings.experiment)
    run_training(experiment)
