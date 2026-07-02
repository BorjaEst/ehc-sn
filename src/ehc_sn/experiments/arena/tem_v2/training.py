"""Training experiment assembly for Arena × TEM-v2."""

from __future__ import annotations

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.modules.variational_replay import TEMTrainingConfig
from ehc_sn.model_artifacts.assembly import ModelAssembly
from ehc_sn.training.runner import (
    TrainingArtifactSpec,
    TrainingExperiment,
)

from .config import ArenaTEMV2TrainingExperimentConfig
from .model import build_arena_tem_v2_model

_TEM_V2_CAPABILITIES = frozenset(
    {"recurrent-step", "spatial-latents", "episodic-memory"}
)


def build_arena_tem_v2_training_experiment(
    config: ArenaTEMV2TrainingExperimentConfig,
) -> TrainingExperiment:
    module = build_arena_tem_v2_model(
        config.model,
        training_config=TEMTrainingConfig(
            optimizer=config.training.optimizer,
            scheduler=config.training.scheduler,
            num_slots=config.data.num_slots,
        ),
        execution=config.execution,
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
    artifact_spec = TrainingArtifactSpec(
        model_family="tem-v2",
        model_type="tem-v2",
        capabilities=_TEM_V2_CAPABILITIES,
        resolved_assembly_config=ModelAssembly(
            model_family="tem-v2",
            model_type="tem-v2",
            core=module.model.config.model_dump(
                mode="python", exclude_none=True
            ),
            adapter=config.model.components.adapter.model_dump(
                mode="python", exclude_none=True
            ),
            controller=config.model.components.controller.model_dump(
                mode="python", exclude_none=True
            ),
        ),
    )
    return TrainingExperiment(
        module=module,
        datamodule=datamodule,
        trainer=config.trainer,
        checkpointing=config.checkpointing,
        logging=config.logging,
        artifact_spec=artifact_spec,
    )


__all__ = ["build_arena_tem_v2_training_experiment"]
