"""Training-completion artifact packaging — publishes a self-describing model artifact.

Called once after ``trainer.fit()`` completes, for the explicitly selected
checkpoint (best or final).  Does not run during intermediate checkpoint
callbacks.

Boundary rules
--------------
- Must not import from ``evaluation/``, ``lightning/``, or ``experiments/<task>/<family>/``.
- Uses generic ``nn.Module``, ``BaseModel`` — no concrete model class dependency.
"""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn

from ehc_sn.model_artifacts import (
    ArtifactMetadata,
    ModelArtifact,
    deduplicate_state_dict,
)


def publish_training_artifact(
    *,
    destination: Path,
    module: nn.Module,
    model_settings: BaseModel,
    model_family: str,
    model_type: str,
    capabilities: frozenset[str] | None = None,
    selection_metric: str | None = None,
    selection_mode: str | None = None,
    selection_score: float | None = None,
    training_epoch: int | None = None,
    training_global_step: int | None = None,
    source_revision: str | None = None,
) -> ModelArtifact:
    """Package a trained model as a self-describing local artifact.

    Parameters
    ----------
    destination:
        Output directory for the artifact (created if missing).
    module:
        Trained ``nn.Module`` whose ``state_dict()`` becomes ``weights.pt``.
    model_settings:
        Fully resolved Pydantic model settings to freeze in ``model.toml``.
    model_family:
        Canonical family identifier (e.g. ``"tem-v1"``).
    model_type:
        Concrete type discriminator (e.g. ``"tem-v1"``).
    capabilities:
        Optional set of capability strings.
    selection_metric, selection_mode, selection_score:
        Checkpoint-selection provenance.
    training_epoch, training_global_step:
        Training-timing provenance.
    source_revision:
        Repository revision at training time.

    Returns
    -------
    ModelArtifact
        The newly created artifact, ready for evaluation consumption.
    """
    # Extract and deduplicate state dict from the training module.
    # LightningModules alias model parameters under bridge_adapter.model.*.
    state_dict = deduplicate_state_dict(module.state_dict())
    # Move tensors to CPU for portable storage.
    state_dict = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }

    return ModelArtifact.create(
        destination=destination,
        model_settings=model_settings,
        state_dict=state_dict,
        metadata=ArtifactMetadata(
            model_family=model_family,
            model_type=model_type,
            capabilities=capabilities or frozenset(),
            selection_metric=selection_metric,
            selection_mode=selection_mode,
            selection_score=selection_score,
            training_epoch=training_epoch,
            training_global_step=training_global_step,
            source_revision=source_revision,
        ),
    )


def publish_best_checkpoint_artifact(
    *,
    destination: Path,
    module: nn.Module,
    model_settings: BaseModel,
    model_family: str,
    model_type: str,
    capabilities: frozenset[str] | None = None,
    checkpoint_callback: object | None = None,
    trainer: object | None = None,
    source_revision: str | None = None,
) -> ModelArtifact | None:
    """Package the best checkpoint as a model artifact after training.

    Reads ``best_model_path``, ``best_model_score``, ``monitor``, and
    ``mode`` from the callback.  If no best checkpoint exists (e.g. no
    ``monitor`` was set), returns ``None``.

    Parameters
    ----------
    destination:
        Output directory.
    module:
        Trained module.
    model_settings:
        Resolved Pydantic model settings.
    model_family, model_type:
        Artifact identity.
    capabilities:
        Optional capability set.
    checkpoint_callback:
        A ``ModelCheckpoint``-like object with ``best_model_path``,
        ``best_model_score``, ``monitor``, and ``mode`` attributes.
    trainer:
        Optional ``Trainer`` with ``current_epoch`` and ``global_step``.
    source_revision:
        Optional Git revision.

    Returns
    -------
    ModelArtifact or None
        ``None`` when no ``best_model_path`` is available.
    """
    if checkpoint_callback is None:
        return None

    best_path: str | None = getattr(
        checkpoint_callback, "best_model_path", None
    )
    if not best_path:
        return None

    best_score_raw = getattr(checkpoint_callback, "best_model_score", None)
    best_score = float(best_score_raw) if best_score_raw is not None else None

    epoch: int | None = None
    global_step: int | None = None
    if trainer is not None:
        epoch = getattr(trainer, "current_epoch", None)
        global_step = getattr(trainer, "global_step", None)

    return publish_training_artifact(
        destination=destination,
        module=module,
        model_settings=model_settings,
        model_family=model_family,
        model_type=model_type,
        capabilities=capabilities,
        selection_metric=getattr(checkpoint_callback, "monitor", None),
        selection_mode=getattr(checkpoint_callback, "mode", None),
        selection_score=best_score,
        training_epoch=epoch,
        training_global_step=global_step,
        source_revision=source_revision,
    )


__all__ = [
    "publish_best_checkpoint_artifact",
    "publish_training_artifact",
]
