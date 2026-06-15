"""Training experiment assembly for SeqMaze × HRM-v1."""

from __future__ import annotations

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedTrainingConfig,
)
from ehc_sn.training.runner import TrainingExperiment

from .config import SeqMazeHRMV1TrainingExperimentConfig
from .model import build_seqmaze_hrm_v1_model


def build_seqmaze_hrm_v1_training_experiment(
    config: SeqMazeHRMV1TrainingExperimentConfig,
) -> TrainingExperiment:
    """Build a complete training experiment for SeqMaze × HRM-v1."""
    module = build_seqmaze_hrm_v1_model(config.model)
    module._training_config = ACTSupervisedTrainingConfig(
        optimizer=config.training.optimizer,
        runtime=config.training.runtime,
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
    )


__all__ = ["build_seqmaze_hrm_v1_training_experiment"]
