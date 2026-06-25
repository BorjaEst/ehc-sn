"""MazeHard × HRM-v2 experiment package.

Task-family composition: nested config schemas, shared model builder,
training experiment assembly, and evaluation executor construction.
"""

from .config import (
    MazeHardDeliberationConfig,
    MazeHardHRMV2ComponentConfigs,
    MazeHardHRMV2EvaluationExperimentConfig,
    MazeHardHRMV2ModelConfig,
    MazeHardHRMV2TrainingExperimentConfig,
)
from .evaluation import build_mazehard_hrm_v2_evaluation_experiment
from .model import build_mazehard_hrm_v2_model
from .training import build_mazehard_hrm_v2_training_experiment

__all__ = [
    "MazeHardDeliberationConfig",
    "MazeHardHRMV2ComponentConfigs",
    "MazeHardHRMV2ModelConfig",
    "MazeHardHRMV2TrainingExperimentConfig",
    "MazeHardHRMV2EvaluationExperimentConfig",
    "build_mazehard_hrm_v2_model",
    "build_mazehard_hrm_v2_training_experiment",
    "build_mazehard_hrm_v2_evaluation_experiment",
]
