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
    # Construct training config (optimizer only — runtime/execution policy
    # is passed separately) and pass both through the model builder.
    training_config = ACTSupervisedTrainingConfig(
        optimizer=config.training.optimizer,
        num_slots=config.data.num_slots,
        gradient_clip_val=getattr(config.trainer, "gradient_clip_val", None),
    )
    module = build_seqmaze_hrm_v1_model(
        config.model,
        training_config=training_config,
        execution=config.execution,
        scheduler=config.scheduler,
        supervised_only_warmup_steps=config.supervised_only_warmup_steps,
        target_network=config.target_network,
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
        transform=None,
    )
    datamodule.attach_source_provider(lambda: module._train_source)
    return TrainingExperiment(
        module=module,
        datamodule=datamodule,
        trainer=config.trainer,
        checkpointing=config.checkpointing,
        logging=config.logging,
    )


__all__ = ["build_seqmaze_hrm_v1_training_experiment"]
