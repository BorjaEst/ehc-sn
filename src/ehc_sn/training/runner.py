"""Shared training-orchestration runner for EHC-SN entry-point scripts.

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
    settings: BaseModel,
) -> list[object]:
    """Construct callback list from readable settings fields.

    Each callback is optional; the method checks the appropriate settings
    field and creates the callback only when configured.
    """
    callbacks: list[object] = [MetricsCallback(), StepProgressBar()]

    eval_regimes: EvaluationRegimesCallbackSettings | None = getattr(
        settings, "eval_regimes", None
    )
    if eval_regimes is not None:
        callbacks.append(EvaluationRegimesCallback(eval_regimes))

    checkpoint: CheckpointSettings | None = getattr(
        settings, "checkpoint", None
    )
    if checkpoint is not None:
        callbacks.append(CheckpointCallback(checkpoint))

    diag_level = getattr(settings, "diagnostic_level", "minimal")
    diagnostics: DiagnosticsSettings | None = getattr(
        settings, "diagnostics", None
    )
    if diag_level != "minimal" and diagnostics is not None:
        callbacks.append(DiagnosticsCallback(diagnostics))

    lr_monitor: LearningRateMonitorSettings | None = getattr(
        settings, "lr_monitor", None
    )
    if lr_monitor is not None:
        callbacks.append(LearningRateMonitor(lr_monitor))

    figures: FigureGenerationSettings | None = getattr(
        settings, "figures", None
    )
    if figures is not None:
        callbacks.append(FigureGenerationCallback(figures))

    return callbacks


def _build_logger(  # ---------------------------------------------------------
    settings: BaseModel,
) -> object | None:
    """Construct TensorBoard logger if configured."""
    logger_settings: LoggerSettings | None = getattr(settings, "logger", None)
    if logger_settings is not None:
        return Logger(logger_settings)
    return None


def _build_trainer(  # --------------------------------------------------------
    settings: BaseModel,
    callbacks: list[object],
    world_size: int,
    find_unused_parameters: bool,
) -> Trainer:
    """Construct the PyTorch Lightning ``Trainer`` from flat settings fields."""
    return Trainer(
        logger=_build_logger(settings),
        callbacks=callbacks if callbacks else None,
        accelerator=getattr(settings, "trainer_accelerator", "gpu"),
        strategy=resolve_trainer_strategy(
            getattr(settings, "trainer_strategy", "ddp"),
            world_size,
            find_unused_parameters=find_unused_parameters,
        ),
        devices=getattr(settings, "trainer_devices", 1),
        num_nodes=getattr(settings, "trainer_num_nodes", 1),
        precision=getattr(settings, "trainer_precision", "16-mixed"),
        max_epochs=-1,
        max_steps=getattr(settings, "max_steps", 200000),
        val_check_interval=getattr(settings, "val_check_interval", 0),
        check_val_every_n_epoch=None,
        limit_val_batches=getattr(settings, "limit_val_batches", 1.0),
        log_every_n_steps=getattr(settings, "log_every_n_steps", 10),
        enable_progress_bar=getattr(settings, "enable_progress_bar", True),
    )


# =============================================================================
# Public runner
# =============================================================================


def run_training(  # ----------------------------------------------------------
    settings: BaseModel,
    spec: TrainingEntrypointSpec,
) -> None:
    """Orchestrate a full training run.

    Parameters
    ----------
    settings:
        Parsed settings object (e.g. ``RunArguments`` or a future typed
        experiment settings class).  Required fields are accessed via
        ``getattr`` so the runner remains compatible with any settings class
        that exposes the expected attribute names.
    spec:
        Per-experiment-family parameterisation (builder, transform, DDP
        flags, weight-loader).
    """
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

    callbacks = _build_callbacks(settings)
    trainer = _build_trainer(
        settings, callbacks, world_size, spec.find_unused_parameters
    )

    # Build the experiment (LightningModule)
    training_model = spec.build_experiment(settings.model_dump())

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


# =============================================================================
__all__ = [
    "TrainingEntrypointSpec",
    "run_training",
]
