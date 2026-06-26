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
    module = build_mazehard_hrm_v2_model(
        config.model,
        execution=config.execution.to_runtime_config(),
        training_config=ActorCriticTrainingConfig(
            optimizer_supervised=config.training.optimizer_supervised,
            optimizer_rl=config.training.optimizer_rl,
            optimizer_qv=config.training.optimizer_qv,
            reward=MazeHardRewardConfig.model_validate(config.training.reward),
            num_slots=config.data.num_slots,
            halt_disabled_steps=config.training.halt_disabled_steps,
            scheduler=config.training.scheduler,
        ),
    )
    datamodule = Datamodule(
        DatamoduleConfig(
            dataset_path=config.data.dataset_path,
            num_slots=config.data.num_slots,
            eval_batch_size=config.data.eval_batch_size,
            num_workers=config.data.num_workers,
            prefetch_factor=config.data.prefetch_factor,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            augment=config.data.augment,
            seed=config.data.seed,
        ),
        transform=coerce_maze_hard_batch,
    )
    datamodule.attach_source_provider(lambda: module._train_source)
    return TrainingExperiment(
        module=module,
        datamodule=datamodule,
        trainer=config.trainer,
        checkpointing=config.checkpointing,
        logging=config.logging,
    )


__all__ = ["build_mazehard_hrm_v2_training_experiment"]
