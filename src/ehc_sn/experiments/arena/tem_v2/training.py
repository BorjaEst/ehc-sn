"""Training experiment assembly for Arena × TEM-v2."""

from __future__ import annotations

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.training.runner import TrainingExperiment

from .config import ArenaTEMV2TrainingExperimentConfig
from .model import build_arena_tem_v2_model


def build_arena_tem_v2_training_experiment(
    config: ArenaTEMV2TrainingExperimentConfig,
) -> TrainingExperiment:
    module = build_arena_tem_v2_model(config.model)
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


__all__ = ["build_arena_tem_v2_training_experiment"]
