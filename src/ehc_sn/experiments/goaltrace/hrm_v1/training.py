"""Training experiment assembly for Goaltrace × HRM-v1."""

from __future__ import annotations

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.modules.act_supervised import (
    ACTSupervisedTrainingConfig,
)
from ehc_sn.model_artifacts.assembly import ModelAssembly
from ehc_sn.training.runner import (
    TrainingArtifactSpec,
    TrainingExperiment,
)

_HRM_V1_CAPABILITIES = frozenset({"deliberative", "analytic-future-state"})

from .config import GoaltraceHRMV1TrainingExperimentConfig
from .model import build_goaltrace_hrm_v1_model


def build_goaltrace_hrm_v1_training_experiment(
    config: GoaltraceHRMV1TrainingExperimentConfig,
) -> TrainingExperiment:
    """Build a complete training experiment for Goaltrace × HRM-v1."""
    training_config = ACTSupervisedTrainingConfig(
        optimizer=config.training.optimizer,
        scheduler=config.training.scheduler,
        num_slots=config.data.num_slots,
        gradient_clip_val=getattr(config.trainer, "gradient_clip_val", None),
    )
    module = build_goaltrace_hrm_v1_model(
        config.model,
        regime_config=config.regime,
        training_config=training_config,
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
        model_family="hrm-v1",
        model_type="hrm-v1",
        capabilities=_HRM_V1_CAPABILITIES,
        resolved_assembly_config=ModelAssembly(
            model_family="hrm-v1",
            model_type="hrm-v1",
            core=module.model.config.model_dump(
                mode="python", exclude_none=True
            ),
            adapter=config.model.components.adapter.model_dump(
                mode="python", exclude_none=True
            ),
            controller=config.model.components.controller.model_dump(
                mode="python", exclude_none=True
            ),
            inference_policy={
                "acceptable_error": config.model.components.task_evaluator.acceptable_error,
                "halt_temperature": config.model.components.task_evaluator.halt_temperature,
            },
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


__all__ = ["build_goaltrace_hrm_v1_training_experiment"]
