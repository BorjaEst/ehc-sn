"""Training experiment assembly for SeqMaze × HRM-v2."""

from __future__ import annotations

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.modules.actor_critic import ActorCriticTrainingConfig
from ehc_sn.tasks.seqmaze.reward import SeqMazeRewardConfig
from ehc_sn.training.runner import TrainingExperiment

from .config import SeqMazeHRMV2TrainingExperimentConfig
from .model import build_seqmaze_hrm_v2_model


def build_seqmaze_hrm_v2_training_experiment(
    config: SeqMazeHRMV2TrainingExperimentConfig,
) -> TrainingExperiment:
    """Build a complete training experiment for SeqMaze × HRM-v2."""
    module = build_seqmaze_hrm_v2_model(config.model)
    module._deliberation = config.deliberation.to_runtime_config()
    module._training_config = ActorCriticTrainingConfig(
        optimizer_supervised=config.training.optimizer_supervised,
        optimizer_rl=config.training.optimizer_rl,
        optimizer_qv=config.training.optimizer_qv,
        reward=SeqMazeRewardConfig.model_validate(config.training.reward),
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
        transform=None,
    )
    return TrainingExperiment(
        module=module,
        datamodule=datamodule,
        trainer=config.trainer,
        checkpointing=config.checkpointing,
        logging=config.logging,
    )


__all__ = ["build_seqmaze_hrm_v2_training_experiment"]
