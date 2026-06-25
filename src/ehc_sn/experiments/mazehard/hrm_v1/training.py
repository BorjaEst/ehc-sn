"""Training experiment assembly for MazeHard × HRM-v1.

Combines the shared model builder with training-specific machinery:
datamodule, training config (optimizer, runtime), and trainer settings.
Returns a ``TrainingExperiment`` consumed by the generic runner.
"""

from __future__ import annotations

from pathlib import Path

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedModule,
    ACTSupervisedTrainingConfig,
)
from ehc_sn.tasks.mazehard.runtime import coerce_maze_hard_batch
from ehc_sn.training.hrm import RuntimeConfig as HRMRuntimeConfig
from ehc_sn.training.runner import TrainingExperiment

from .config import (
    MazeHardHRMV1TrainingExperimentConfig,
)
from .model import build_mazehard_hrm_v1_model


def build_mazehard_hrm_v1_training_experiment(
    config: MazeHardHRMV1TrainingExperimentConfig,
) -> TrainingExperiment:
    """Build a complete training experiment for MazeHard × HRM-v1.

    Args:
        config: Full nested training configuration.

    Returns:
        A ``TrainingExperiment`` ready for the generic runner.
    """
    transfer_training = ACTSupervisedTrainingConfig(
        optimizer=config.training.optimizer,
        scheduler=config.training.scheduler,
        num_slots=config.data.num_slots,
    )
    module = build_mazehard_hrm_v1_model(
        config.model,
        regime_config=config.regime,
        training_config=transfer_training,
        execution=config.execution,
    )

    # Construct the datamodule.
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


__all__ = ["TrainingExperiment", "build_mazehard_hrm_v1_training_experiment"]
