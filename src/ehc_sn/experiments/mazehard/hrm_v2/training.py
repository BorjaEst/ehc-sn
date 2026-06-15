"""Training experiment assembly for MazeHard × HRM-v2."""

from __future__ import annotations

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.modules.actor_critic import ActorCriticTrainingConfig
from ehc_sn.tasks.mazehard.reward import MazeHardRewardConfig
from ehc_sn.tasks.mazehard.runtime import (
    MazeHardRuntimeConfig,
    coerce_maze_hard_batch,
)
from ehc_sn.training.runner import TrainingExperiment

from .config import MazeHardHRMV2TrainingExperimentConfig
from .model import build_mazehard_hrm_v2_model


def build_mazehard_hrm_v2_training_experiment(
    config: MazeHardHRMV2TrainingExperimentConfig,
) -> TrainingExperiment:
    """Build a complete training experiment for MazeHard × HRM-v2."""
    module = build_mazehard_hrm_v2_model(config.model)
    module._deliberation = MazeHardRuntimeConfig(
        halt_action=config.deliberation.halt_action,
        episode_horizon=config.deliberation.episode_horizon,
    )
    module._training_config = ActorCriticTrainingConfig(
        optimizer_supervised=config.training.optimizer_supervised,
        optimizer_rl=config.training.optimizer_rl,
        optimizer_qv=config.training.optimizer_qv,
        reward=MazeHardRewardConfig.model_validate(config.training.reward),
        hrm_runtime=config.training.hrm_runtime,
    )
    datamodule = Datamodule(
        DatamoduleConfig(
            dataset_path=config.data.dataset_path,
            global_batch_size=config.data.global_batch_size,
            num_workers=config.data.num_workers,
            prefetch_factor=config.data.prefetch_factor,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            augment=config.data.augment,
            seed=config.data.seed,
        ),
        transform=coerce_maze_hard_batch,
    )
    return TrainingExperiment(
        module=module,
        datamodule=datamodule,
        trainer=config.trainer,
    )


__all__ = ["build_mazehard_hrm_v2_training_experiment"]
