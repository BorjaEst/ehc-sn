"""Routebind × HRM-v1 experiment package.

Training and evaluation experiment builders for the routebind task family
with HRM v1.

Stable surface:

- :class:`RoutebindHRMV1ComponentConfigs` — component-level config.
- :class:`RoutebindHRMV1ModelConfig` — model-level config.
- :class:`RoutebindHRMV1TrainingExperimentConfig` — full training config.
- :func:`build_routebind_hrm_v1_training_experiment` — training experiment builder.
- :func:`build_routebind_hrm_v1_model` — model constructor.
- :func:`build_routebind_hrm_v1_evaluation_experiment` — evaluation experiment builder.
"""

from .config import (
    RoutebindHRMV1ComponentConfigs,
    RoutebindHRMV1ModelConfig,
    RoutebindHRMV1TrainingExperimentConfig,
)
from .evaluation import build_routebind_hrm_v1_evaluation_experiment
from .model import build_routebind_hrm_v1_model
from .training import build_routebind_hrm_v1_training_experiment

__all__ = [
    "RoutebindHRMV1ComponentConfigs",
    "RoutebindHRMV1ModelConfig",
    "RoutebindHRMV1TrainingExperimentConfig",
    "build_routebind_hrm_v1_evaluation_experiment",
    "build_routebind_hrm_v1_model",
    "build_routebind_hrm_v1_training_experiment",
]
