"""Shared training-orchestration runner for EHP-SN entry-point scripts.

Owns the duplicated ~300 lines of orchestration that were previously
copy-pasted across four training scripts.  Scripts become thin wrappers
that select a ``TrainingEntrypointSpec`` and delegate to :func:`run_training`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from pydantic import BaseModel

from ehc_sn.data.datamodules import Datamodule, DatamoduleConfig
from ehc_sn.experiments._infra import CheckpointingConfig
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
from ehc_sn.model_artifacts import (
    ArtifactMetadata,
    CheckpointStateSource,
    InMemoryStateSource,
    deduplicate_state_dict,
    publish_model_artifact,
)
from ehc_sn.model_artifacts.assembly import ModelAssembly
from ehc_sn.training.distributed import (
    resolve_effective_world_size,
    resolve_trainer_strategy,
    validate_num_slots_divisibility,
)

# =============================================================================
# Artifact and callback role types
# =============================================================================


@dataclass(frozen=True)
class TrainingArtifactSpec:
    """Semantic model metadata provided by the experiment to the runner.

    The runner uses these values when publishing the artifact after
    training; it never hardcodes model-family-specific strings.
    """

    model_family: str
    model_type: str
    capabilities: frozenset[str]
    resolved_assembly_config: ModelAssembly
    """Frozen evaluation-assembly configuration.

    A ``ModelAssembly`` instance with ``core``, ``adapter``, and optionally
    ``controller`` and ``inference_policy`` sections.  Constructed by
    experiment builders and forwarded to ``publish_model_artifact()``.
    """


@dataclass(frozen=True)
class TrainingCallbacks:
    """Callback bundle with explicit role assignments.

    ``periodic_checkpoint`` owns recovery and history snapshots.
    ``selection_checkpoint`` owns metric-based best-checkpoint selection
    and is consulted by ``state_source="best"``.
    """

    callbacks: tuple[Callback, ...] = ()
    periodic_checkpoint: ModelCheckpoint | None = None
    selection_checkpoint: ModelCheckpoint | None = None


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
        group.  Import from ``ehp_sn.training.tem`` or ``ehp_sn.training.hrm``
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


def _build_periodic_checkpoint(  # ---------------------------------------------
    config: Any,
) -> ModelCheckpoint | None:
    """Construct a periodic (history-snapshot) ``ModelCheckpoint``.

    Unmonitored, keeps all snapshots (``save_top_k=-1``).
    """
    if not getattr(config, "enabled", True):
        return None
    return ModelCheckpoint(
        dirpath=getattr(config, "dirpath", None),
        filename=getattr(config, "filename", "{step:08d}"),
        monitor=None,
        save_top_k=-1,
        every_n_train_steps=getattr(config, "every_n_train_steps", 1000),
        save_last=getattr(config, "save_last", True),
        save_on_exception=getattr(config, "save_on_exception", True),
        auto_insert_metric_name=False,
    )


def _build_selection_checkpoint(  # -------------------------------------------
    config: Any,
) -> ModelCheckpoint | None:
    """Construct a metric-based selection ``ModelCheckpoint``.

    Fails fast if ``enabled=True`` but ``monitor`` is not set.
    """
    if not getattr(config, "enabled", True):
        return None
    monitor = getattr(config, "monitor", None)
    if monitor is None:
        raise ValueError(
            "checkpointing.selection.monitor is required when "
            "selection checkpointing is enabled."
        )
    return ModelCheckpoint(
        dirpath=getattr(config, "dirpath", None),
        filename=getattr(config, "filename", "best-step={step:08d}"),
        monitor=monitor,
        mode=getattr(config, "mode", "max"),
        save_top_k=getattr(config, "save_top_k", 1),
        save_last=False,
        auto_insert_metric_name=False,
    )


def _build_legacy_checkpoint(  # ----------------------------------------------
    checkpointing_config: Any,
) -> ModelCheckpoint | None:
    """Construct a checkpoint callback from the legacy ``checkpoint`` field."""
    ckpt: CheckpointSettings | None = getattr(
        checkpointing_config, "checkpoint", None
    )
    if ckpt is not None:
        return CheckpointCallback(ckpt)
    return None


def _build_callbacks(  # ------------------------------------------------------
    checkpointing_config: Any | None = None,
    *,
    eval_regimes: EvaluationRegimesCallbackSettings | None = None,
    diagnostics: DiagnosticsSettings | None = None,
    lr_monitor: LearningRateMonitorSettings | None = None,
    figures: FigureGenerationSettings | None = None,
) -> TrainingCallbacks:
    """Construct callback list and role assignments from structured config.

    Parameters
    ----------
    checkpointing_config:
        ``CheckpointingConfig`` (migrated path) or a ``BaseModel`` settings
        object with a legacy ``checkpoint`` attribute (legacy path).
    eval_regimes, diagnostics, lr_monitor, figures:
        Keyword-only overrides used by the legacy path only.

    Returns
    -------
    TrainingCallbacks
        Bundle with the flat callback tuple for the Trainer and role-identified
        checkpoint references for post-training artifact publication.
    """
    callbacks: list[object] = [MetricsCallback(), StepProgressBar()]
    periodic: ModelCheckpoint | None = None
    selection: ModelCheckpoint | None = None

    if eval_regimes is not None:
        callbacks.append(EvaluationRegimesCallback(eval_regimes))

    if checkpointing_config is not None:
        # ---- Migrated path: isinstance type-discrimination ------------------

        if isinstance(checkpointing_config, CheckpointingConfig):
            periodic = _build_periodic_checkpoint(checkpointing_config.periodic)
            selection = _build_selection_checkpoint(
                checkpointing_config.selection
            )
            if periodic is not None:
                callbacks.append(periodic)
            if selection is not None:
                callbacks.append(selection)
        # ---- Legacy path: single checkpoint callback ------------------------
        else:
            legacy = _build_legacy_checkpoint(checkpointing_config)
            if legacy is not None:
                periodic = legacy
                callbacks.append(legacy)

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

    return TrainingCallbacks(
        callbacks=tuple(callbacks),
        periodic_checkpoint=periodic,
        selection_checkpoint=selection,
    )


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
        gradient_clip_val=getattr(trainer_config, "gradient_clip_val", None),
    )


# =============================================================================
# Application container (training)
# =============================================================================


@dataclass(frozen=True)
class TrainingExperiment:
    """Fully assembled training experiment.

    Produced by a task-family ``build_training_experiment`` function and
    consumed by ``run_training``.  Trainer, checkpointing, logging,
    artifact spec, and callback roles are passed through to shared
    orchestration helpers.
    """

    module: Any  # LightningModule
    datamodule: Any  # LightningDataModule
    trainer: Any | None = None  # TrainerConfig
    checkpointing: Any | None = None  # CheckpointingConfig
    logging: Any | None = None  # LoggerSettings
    artifact_spec: TrainingArtifactSpec | None = None


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
    validate_num_slots_divisibility(
        getattr(settings, "num_slots", 1), world_size
    )
    seed_everything(getattr(settings, "seed", 42))

    logger = _build_logger(
        logger_settings=getattr(settings, "logger", None),
    )
    training_callbacks = _build_callbacks(
        checkpointing_config=settings,
        eval_regimes=getattr(settings, "eval_regimes", None),
        diagnostics=getattr(settings, "diagnostics", None),
        lr_monitor=getattr(settings, "lr_monitor", None),
        figures=getattr(settings, "figures", None),
    )
    trainer = _build_trainer_from_config(
        settings,
        logger=logger,
        callbacks=list(training_callbacks.callbacks),
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
    training_callbacks = _build_callbacks(
        checkpointing_config=experiment.checkpointing,
    )
    trainer = _build_trainer_from_config(
        experiment.trainer,
        logger=logger,
        callbacks=list(training_callbacks.callbacks),
        world_size=world_size,
        find_unused_parameters=getattr(
            experiment.trainer, "find_unused_parameters", False
        ),
    )

    # Optional weight-init hydration from a checkpoint
    if experiment.checkpointing is not None:
        init_from: str | None = getattr(
            experiment.checkpointing, "init_weights_from", None
        )
        if init_from is not None:
            init_groups: list[str] = getattr(
                experiment.checkpointing, "init_weights_groups", ["all"]
            )
            loaded_keys = experiment.module.load_weights_from_checkpoint(
                init_from,
                init_groups,
            )
            print(
                f"[init_weights_from] Loaded {len(loaded_keys)} parameter keys "
                f"(groups={init_groups}) from {init_from!r}."
            )

    ckpt_path: str | None = None
    if experiment.checkpointing is not None:
        ckpt_path = getattr(experiment.checkpointing, "resume_from", None)

    trainer.fit(
        model=experiment.module,
        datamodule=experiment.datamodule,
        ckpt_path=ckpt_path,
    )

    # ---- Publish model artifact after successful training -------------------
    if experiment.artifact_spec is None:
        return

    checkpointing = experiment.checkpointing
    if checkpointing is None:
        return

    publication = getattr(checkpointing, "artifact", None)
    if publication is None or not publication.enabled:
        return

    trainer.strategy.barrier("before-model-artifact-publication")

    if not trainer.is_global_zero:
        return

    artifact_dest: Path
    if publication.destination is not None:
        artifact_dest = Path(publication.destination).resolve()
    else:
        artifact_dest = (
            Path("artifacts/models")
            / experiment.artifact_spec.model_family
            / f"run-{trainer.global_step:06d}"
        )

    state_source, selection_metadata = _resolve_artifact_state_source(
        policy=publication.state_source,
        callbacks=training_callbacks,
        module=experiment.module,
    )

    revision: str | None = None
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
    except Exception:
        pass

    metric = selection_metadata.get("metric")
    mode = selection_metadata.get("mode")
    score = selection_metadata.get("score")

    publish_model_artifact(
        destination=artifact_dest,
        state_source=state_source,
        assembly_config=experiment.artifact_spec.resolved_assembly_config,
        metadata=ArtifactMetadata(
            model_family=experiment.artifact_spec.model_family,
            model_type=experiment.artifact_spec.model_type,
            capabilities=experiment.artifact_spec.capabilities,
            selection_metric=metric,
            selection_mode=mode,
            selection_score=score,
            training_epoch=getattr(trainer, "current_epoch", None),
            training_global_step=getattr(trainer, "global_step", None),
            source_revision=revision,
        ),
    )

    print(f"Model artifact published to {artifact_dest}")


def _resolve_artifact_state_source(
    *,
    policy: str,
    callbacks: TrainingCallbacks | None,
    module: object,
) -> tuple[CheckpointStateSource | InMemoryStateSource, dict[str, object]]:
    """Resolve the artifact state source and selection metadata from the policy.

    Falls back to ``final-state`` with a warning when the requested checkpoint
    (best or last) is unavailable — e.g. validation never ran or no periodic
    checkpoint was saved.

    Returns
    -------
    (state_source, selection_metadata)
        *selection_metadata* is a dict with keys ``kind``, and optionally
        ``metric``, ``mode``, ``score``.
    """
    import warnings

    if policy == "best":
        if callbacks is None or callbacks.selection_checkpoint is None:
            raise ValueError(
                "Artifact state_source is 'best', but no selection checkpoint "
                "callback was configured."
            )
        cb = callbacks.selection_checkpoint
        best_path: str | None = getattr(cb, "best_model_path", None)
        if best_path:
            best_score = getattr(cb, "best_model_score", None)
            return CheckpointStateSource(Path(best_path)), {
                "kind": "validation_metric",
                "metric": getattr(cb, "monitor", None),
                "mode": getattr(cb, "mode", None),
                "score": float(best_score) if best_score is not None else None,
            }
        warnings.warn(
            "Artifact state_source is 'best', but the selection callback "
            "did not produce a best checkpoint "
            f"(monitor={getattr(cb, 'monitor', None)!r}). "
            "Falling back to final-state.",
            stacklevel=2,
        )

    if policy == "last":
        if callbacks is None or callbacks.periodic_checkpoint is None:
            raise ValueError(
                "Artifact state_source is 'last', but no periodic checkpoint "
                "callback was configured."
            )
        cb = callbacks.periodic_checkpoint
        last_path: str | None = getattr(cb, "last_model_path", None)
        if not last_path and hasattr(cb, "best_model_path"):
            last_path = cb.best_model_path
        if last_path:
            return CheckpointStateSource(Path(last_path)), {
                "kind": "last_checkpoint",
            }
        warnings.warn(
            "Artifact state_source is 'last', but the periodic callback "
            "did not produce a last checkpoint. "
            "Falling back to final-state.",
            stacklevel=2,
        )

    # final-state (primary or fallback)
    return InMemoryStateSource(
        state_dict=deduplicate_state_dict(
            module.state_dict() if hasattr(module, "state_dict") else {}
        )
    ), {"kind": "final_state"}


# =============================================================================
__all__ = [
    "TrainingArtifactSpec",
    "TrainingCallbacks",
    "TrainingEntrypointSpec",
    "TrainingExperiment",
    "run_training",
]
