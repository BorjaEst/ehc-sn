"""Training experiment assembly for SeqMaze × HRM-v2."""

from __future__ import annotations

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.modules.actor_critic import ActorCriticTrainingConfig
from ehc_sn.model_artifacts.assembly import ModelAssembly
from ehc_sn.tasks.seqmaze.reward import SeqMazeRewardConfig
from ehc_sn.training.runner import (
    TrainingArtifactSpec,
    TrainingExperiment,
)

_HRM_V2_CAPABILITIES = frozenset({"deliberative", "analytic-future-state"})

from .config import SeqMazeHRMV2TrainingExperimentConfig
from .model import build_seqmaze_hrm_v2_model


def build_seqmaze_hrm_v2_training_experiment(
    config: SeqMazeHRMV2TrainingExperimentConfig,
) -> TrainingExperiment:
    """Build a complete training experiment for SeqMaze × HRM-v2."""
    module = build_seqmaze_hrm_v2_model(
        config.model,
        execution=config.execution.to_runtime_config(),
        training_config=ActorCriticTrainingConfig(
            optimizer_supervised=config.training.optimizer_supervised,
            optimizer_rl=config.training.optimizer_rl,
            optimizer_qv=config.training.optimizer_qv,
            reward=SeqMazeRewardConfig.model_validate(config.training.reward),
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
        transform=None,
    )
    datamodule.attach_source_provider(lambda: module._train_source)
    artifact_spec = TrainingArtifactSpec(
        model_family="hrm-v2",
        model_type="hrm-v2",
        capabilities=_HRM_V2_CAPABILITIES,
        resolved_assembly_config=ModelAssembly(
            model_family="hrm-v2",
            model_type="hrm-v2",
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


__all__ = ["build_seqmaze_hrm_v2_training_experiment"]
