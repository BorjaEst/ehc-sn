"""Shared training-orchestration runner for EHP-SN entry-point scripts.

Owns the duplicated ~300 lines of orchestration that were previously
copy-pasted across four training scripts.  Scripts become thin wrappers
that select a ``TrainingEntrypointSpec`` and delegate to :func:`run_training`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from lightning.pytorch import Trainer, seed_everything
from pydantic import BaseModel

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.lightning.callbacks.checkpoint import (
    CheckpointCallback,
    CheckpointSettings,
)
from ehc_sn.lightning.callbacks.diagnostics import (
    DiagnosticsCallback,
    DiagnosticsSettings,
)
from ehc_sn.lightning.callbacks.evaluation import (
    EvaluationRegimesCallback,
    EvaluationRegimesCallbackSettings,
)
from ehc_sn.lightning.callbacks.figures import (
    FigureGenerationCallback,
    FigureGenerationSettings,
)
from ehc_sn.lightning.callbacks.lr_monitor import (
    LearningRateMonitor,
    LearningRateMonitorSettings,
)
from ehc_sn.lightning.callbacks.metrics import MetricsCallback
from ehc_sn.lightning.callbacks.progress import StepProgressBar
from ehc_sn.logging.tensorboard import Logger, LoggerSettings
from ehc_sn.training.distributed import (
    resolve_effective_world_size,
    resolve_trainer_strategy,
    validate_batch_size_divisibility,
)

# =============================================================================
# Entry-point specification — captures what differs per experiment family
# =============================================================================


@dataclass(frozen=True)
class TrainingEntrypointSpec:
    """What differs between the four training entry-point scripts.

    Parameters
    ----------
    build_experiment:
        Experiment-builder function.  Receives the parsed settings object
        (or a dict dump thereof, depending on migration stage) and returns
        a ``LightningModule``.
    datamodule_transform:
        Optional callable applied to each batch by the ``Datamodule``.
        ``None`` for TEM-family experiments; ``coerce_maze_hard_batch``
        for HRM-family experiments.
    find_unused_parameters:
        Whether to pass ``find_unused_parameters=True`` to DDP strategy.
        ``False`` for HRM v1; ``True`` for HRM v2 and TEM v1/v2.
    load_weights_from_checkpoint:
        Function that hydrates model weights from a checkpoint by semantic
        group.  Import from ``ehc_sn.training.tem`` or ``ehc_sn.training.hrm``
        depending on model family.
    """

    build_experiment: Callable[[Any], Any]
    datamodule_transform: Callable | None = None
    find_unused_parameters: bool = False
    load_weights_from_checkpoint: Callable | None = None


# =============================================================================
# Optimise PyTorch for modern GPU hardware
# =============================================================================


def _configure_torch() -> None:
    """Set PyTorch performance flags — safe to call even without GPUs."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


# =============================================================================
# Internal helpers
# =============================================================================


def _build_callbacks(  # ------------------------------------------------------
    checkpointing_config: Any | None = None,
    *,
    eval_regimes: EvaluationRegimesCallbackSettings | None = None,
    diagnostics: DiagnosticsSettings | None = None,
    lr_monitor: LearningRateMonitorSettings | None = None,
    figures: FigureGenerationSettings | None = None,
) -> list[object]:
    """Construct callback list from structured config.

    Parameters
    ----------
    checkpointing_config:
        Object with ``checkpoint``, ``diagnostic_level``, and ``diagnostics``
        attributes.  Supplied by the migrated path (``experiment.checkpointing``).
    eval_regimes, diagnostics, lr_monitor, figures:
        Keyword-only overrides used by the legacy path only.
    """
    callbacks: list[object] = [MetricsCallback(), StepProgressBar()]

    if eval_regimes is not None:
        callbacks.append(EvaluationRegimesCallback(eval_regimes))

    if checkpointing_config is not None:
        ckpt: CheckpointSettings | None = getattr(
            checkpointing_config, "checkpoint", None
        )
        if ckpt is not None:
            callbacks.append(CheckpointCallback(ckpt))

        diag_level: str = getattr(
            checkpointing_config, "diagnostic_level", "minimal"
        )
        diag: DiagnosticsSettings | None = (
            diagnostics
            if diagnostics is not None
            else getattr(checkpointing_config, "diagnostics", None)
        )
        if diag_level != "minimal" and diag is not None:
            callbacks.append(DiagnosticsCallback(diag))

    if lr_monitor is not None:
        callbacks.append(LearningRateMonitor(lr_monitor))

    if figures is not None:
        callbacks.append(FigureGenerationCallback(figures))

    return callbacks


def _build_logger(  # ---------------------------------------------------------
    logger_settings: LoggerSettings | None = None,
) -> object | None:
    """Construct TensorBoard logger if configured."""
    if logger_settings is not None:
        return Logger(logger_settings)
    return None


def _build_trainer_from_config(  # --------------------------------------------
    trainer_config: Any,
    *,
    logger: object | None = None,
    callbacks: list[object] | None = None,
    world_size: int = 1,
    find_unused_parameters: bool = False,
) -> Trainer:
    """Construct the PyTorch Lightning ``Trainer`` from a typed config object."""
    return Trainer(
        logger=logger,
        callbacks=callbacks if callbacks else None,
        accelerator=getattr(trainer_config, "accelerator", "gpu"),
        strategy=resolve_trainer_strategy(
            getattr(trainer_config, "strategy", "ddp"),
            world_size,
            find_unused_parameters=find_unused_parameters,
        ),
        devices=getattr(trainer_config, "devices", 1),
        num_nodes=getattr(trainer_config, "num_nodes", 1),
        precision=getattr(trainer_config, "precision", "16-mixed"),
        max_epochs=-1,
        max_steps=getattr(trainer_config, "max_steps", 200000),
        val_check_interval=getattr(trainer_config, "val_check_interval", 500),
        check_val_every_n_epoch=None,
        limit_val_batches=getattr(trainer_config, "limit_val_batches", 1.0),
        log_every_n_steps=getattr(trainer_config, "log_every_n_steps", 10),
        enable_progress_bar=getattr(
            trainer_config, "enable_progress_bar", True
        ),
    )


# =============================================================================
# Application container (training)
# =============================================================================


@dataclass(frozen=True)
class TrainingExperiment:
    """Fully assembled training experiment.

    Produced by a task-family ``build_training_experiment`` function and
    consumed by ``run_training``.  Trainer, checkpointing, and logging
    configs are passed through to shared orchestration helpers.
    """

    module: Any  # LightningModule
    datamodule: Any  # LightningDataModule
    trainer: Any | None = None  # TrainerConfig
    checkpointing: Any | None = None  # CheckpointingConfig
    logging: Any | None = None  # LoggerSettings


# =============================================================================
# Public runner
# =============================================================================


def run_training(  # ----------------------------------------------------------
    experiment_or_settings: BaseModel | TrainingExperiment,
    spec: TrainingEntrypointSpec | None = None,
) -> None:
    """Orchestrate a full training run.

    Two dispatch paths:

    **New path** (migrated families):
        ``experiment_or_settings`` is a :class:`TrainingExperiment`` with a
        pre-assembled module and datamodule.  ``spec`` must be ``None``.

    **Legacy path** (unmigrated families):
        ``experiment_or_settings`` is a ``BaseModel`` settings object and
        ``spec`` is a ``TrainingEntrypointSpec``.  The module and datamodule
        are constructed from the spec and settings.

    Parameters
    ----------
    experiment_or_settings:
        Either a pre-assembled ``TrainingExperiment`` (new path) or a
        ``BaseModel`` settings object (legacy path).
    spec:
        ``TrainingEntrypointSpec`` required for legacy path, ``None`` for
        new path.
    """
    # ---- Dual-path dispatch -------------------------------------------------
    if isinstance(experiment_or_settings, TrainingExperiment):
        if spec is not None:
            raise TypeError("spec must be None when passing TrainingExperiment")
        experiment = experiment_or_settings
        _run_training_experiment(experiment)
        return

    # ---- Legacy path --------------------------------------------------------
    if spec is None:
        raise TypeError("spec is required when passing settings (BaseModel)")
    settings: BaseModel = experiment_or_settings  # type: ignore[assignment]

    _configure_torch()

    world_size = resolve_effective_world_size(
        getattr(settings, "trainer_strategy", "ddp"),
        getattr(settings, "trainer_devices", 1),
        getattr(settings, "trainer_num_nodes", 1),
    )
    validate_batch_size_divisibility(
        getattr(settings, "global_batch_size", 1), world_size
    )
    seed_everything(getattr(settings, "seed", 42))

    logger = _build_logger(
        logger_settings=getattr(settings, "logger", None),
    )
    callbacks = _build_callbacks(
        checkpointing_config=settings,
        eval_regimes=getattr(settings, "eval_regimes", None),
        diagnostics=getattr(settings, "diagnostics", None),
        lr_monitor=getattr(settings, "lr_monitor", None),
        figures=getattr(settings, "figures", None),
    )
    trainer = _build_trainer_from_config(
        settings,
        logger=logger,
        callbacks=callbacks,
        world_size=world_size,
        find_unused_parameters=spec.find_unused_parameters,
    )

    # Build the experiment (LightningModule)
    training_model = spec.build_experiment(settings)

    # Optional weight-init hydration from a checkpoint
    load_weights = spec.load_weights_from_checkpoint
    init_from: str | None = getattr(settings, "init_weights_from", None)
    if init_from is not None and load_weights is not None:
        init_groups: list[str] = getattr(
            settings, "init_weights_groups", ["all"]
        )
        loaded_keys = load_weights(
            training_model.model,
            init_from,
            init_groups,
        )
        print(
            f"[init_weights_from] Loaded {len(loaded_keys)} parameter keys "
            f"(groups={init_groups}) from {init_from!r}."
        )

    # Build the data module
    datamodule = Datamodule(
        DatamoduleConfig.model_validate(settings, from_attributes=True),
        transform=spec.datamodule_transform,
    )

    # Launch training
    trainer.fit(
        model=training_model,
        datamodule=datamodule,
        ckpt_path=getattr(settings, "resume_from_checkpoint", None),
    )


def _run_training_experiment(  # ---------------------------------------------
    experiment: TrainingExperiment,
) -> None:
    """Execute training from a pre-assembled ``TrainingExperiment``."""
    _configure_torch()
    seed_everything(getattr(experiment.trainer, "seed", 42))

    world_size = resolve_effective_world_size(
        getattr(experiment.trainer, "strategy", "ddp"),
        getattr(experiment.trainer, "devices", 1),
        getattr(experiment.trainer, "num_nodes", 1),
    )

    logger = _build_logger(
        logger_settings=experiment.logging,
    )
    callbacks = _build_callbacks(
        checkpointing_config=experiment.checkpointing,
    )
    trainer = _build_trainer_from_config(
        experiment.trainer,
        logger=logger,
        callbacks=callbacks,
        world_size=world_size,
        find_unused_parameters=getattr(
            experiment.trainer, "find_unused_parameters", False
        ),
    )

    ckpt_path: str | None = None
    if experiment.checkpointing is not None:
        ckpt_path = getattr(experiment.checkpointing, "resume_from", None)

    trainer.fit(
        model=experiment.module,
        datamodule=experiment.datamodule,
        ckpt_path=ckpt_path,
    )


# =============================================================================
__all__ = [
    "TrainingEntrypointSpec",
    "TrainingExperiment",
    "run_training",
]
