"""Shared infrastructure config types used by all experiment families.

Consolidates one canonical ``TrainerConfig`` and one canonical
``CheckpointingConfig`` from six per-family copies into a single
source of truth.

Also defines the configuration-boundary types :class:`ProviderConfig`,
:class:`RegimeConfig`, and the :class:`TrainerConfig` / checkpointing
configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# =============================================================================
class InvalidModelArtifactError(ValueError):
    """Raised when a model artifact is structurally incomplete."""


def resolve_core_model_config_path(
    *,
    model_artifact: object,  # ResolvedModelArtifact
) -> Path:
    """Return the core-model configuration path for evaluation.

    Extracts the ``core`` section from the artifact's embedded
    ``ModelAssembly`` and materialises it to a temporary TOML file.

    Parameters
    ----------
    model_artifact:
        The resolved model artifact.  Must have a ``model_config``
        attribute that is a ``ModelAssembly`` instance with a ``core``
        section.

    Returns
    -------
    Path
        Path to the core-model configuration TOML file.

    Raises
    ------
    InvalidModelArtifactError
        If the artifact has no ``model_config`` or the config is not
        a ``ModelAssembly`` with a valid ``core`` section.
    """
    if model_artifact is None:
        raise InvalidModelArtifactError(
            "No model artifact provided — a resolved artifact with an "
            "embedded model configuration is required."
        )

    source_kind = getattr(model_artifact, "source_kind", None)
    model_config = getattr(model_artifact, "model_config", None)
    resolved_source = getattr(model_artifact, "resolved_source", "")

    if source_kind != "model-artifact":
        raise InvalidModelArtifactError(
            f"Model artifact at {resolved_source} has source_kind "
            f"{source_kind!r}, expected 'model-artifact'."
        )

    if model_config is None:
        raise InvalidModelArtifactError(
            f"Model artifact at {resolved_source} does not contain "
            "an embedded model configuration."
        )

    import tempfile

    import tomli_w

    from ehc_sn.model_artifacts.assembly import ModelAssembly

    if not isinstance(model_config, ModelAssembly):
        raise InvalidModelArtifactError(
            f"Model artifact at {resolved_source} has a model_config "
            f"of type {type(model_config).__name__}, expected ModelAssembly."
        )

    snapshot_dir = Path(tempfile.mkdtemp(prefix="ehp-evaluation-config-"))
    snapshot = snapshot_dir / "model.toml"
    with snapshot.open("wb") as fh:
        tomli_w.dump(model_config.core, fh)
    return snapshot


# =============================================================================
class TrainerConfig(BaseModel, extra="forbid"):
    """Lightning Trainer configuration fields.

    Shared across all experiment families.  Fields that are only
    relevant to some families use ``None`` defaults.
    """

    accelerator: Literal["auto", "gpu", "cpu"] = Field(
        default="gpu",
        description="Trainer accelerator setting.",
    )
    strategy: Literal["auto", "ddp"] = Field(
        default="ddp",
        description="Trainer DDP strategy.",
    )
    devices: int = Field(
        default=1,
        ge=1,
        description="Number of devices per node.",
    )
    num_nodes: int = Field(
        default=1,
        ge=1,
        description="Number of nodes.",
    )
    precision: str = Field(
        default="16-mixed",
        description="Training precision.",
    )
    max_steps: int = Field(
        default=200000,
        ge=1,
        description="Maximum training steps.",
    )
    val_check_interval: int = Field(
        default=500,
        ge=1,
        description="Validation check interval in steps.",
    )
    log_every_n_steps: int = Field(
        default=10,
        ge=1,
        description="Log metrics every N steps.",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress bar.",
    )
    limit_val_batches: int | float = Field(
        default=1.0,
        description="Validation batches (int=N, float=fraction).",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="RNG seed for reproducibility.",
    )
    find_unused_parameters: bool = Field(
        default=False,
        description="Enable DDP find_unused_parameters.",
    )
    gradient_clip_val: float | None = Field(
        default=None,
        description="Gradient clipping value (None = disabled).  "
        "Used by seqmaze v1 training; ignored by families that "
        "route gradient clipping through their training config.",
    )


# =============================================================================
class PeriodicCheckpointConfig(BaseModel, extra="forbid"):
    """Recovery and history-snapshot checkpointing.

    Produces a ``ModelCheckpoint`` with ``save_top_k=-1`` (keep all)
    and no ``monitor`` — suitable for debugging and regression analysis.
    """

    enabled: bool = Field(
        default=True,
        description="Enable periodic checkpointing.",
    )
    dirpath: str | None = Field(
        default=None,
        description="Output directory for periodic checkpoints.",
    )
    filename: str = Field(
        default="{step:08d}",
        description="Checkpoint filename pattern.",
    )
    every_n_train_steps: int = Field(
        default=1000,
        ge=1,
        description="Save periodic checkpoint every N training steps.",
    )
    save_last: bool = Field(
        default=True,
        description="Always save last.ckpt for resume.",
    )
    save_on_exception: bool = Field(
        default=True,
        description="Save checkpoint on exception.",
    )


class SelectionCheckpointConfig(BaseModel, extra="forbid"):
    """Metric-based best-checkpoint selection for artifact publication.

    Produces a ``ModelCheckpoint`` with ``save_top_k=1``.
    """

    enabled: bool = Field(
        default=False,
        description="Enable metric-based best-checkpoint selection.",
    )
    dirpath: str | None = Field(
        default=None,
        description="Output directory for selected checkpoints.",
    )
    filename: str = Field(
        default="best-step={step:08d}",
        description="Checkpoint filename pattern.",
    )
    monitor: str | None = Field(
        default=None,
        description="Validation metric to monitor for best-checkpoint selection.",
    )
    mode: Literal["min", "max"] = Field(
        default="max",
        description="Optimization direction for the monitored metric.",
    )
    save_top_k: int = Field(
        default=1,
        ge=1,
        description="Number of best checkpoints to retain.  Must be 1 for "
        "canonical artifact selection.",
    )

    @model_validator(mode="after")
    def _validate_selection(self) -> "SelectionCheckpointConfig":
        if self.enabled and self.monitor is None:
            raise ValueError(
                "checkpointing.selection.monitor is required when selection "
                "checkpointing is enabled."
            )
        return self


class ArtifactPublicationConfig(BaseModel, extra="forbid"):
    """Policy controlling post-training model artifact publication."""

    enabled: bool = Field(
        default=False,
        description="Enable artifact publication after training.",
    )
    destination: str | None = Field(
        default=None,
        description="Output directory for the published artifact.",
    )
    state_source: Literal["best", "last", "final-state"] = Field(
        default="best",
        description="Which state to publish: best checkpoint, last checkpoint, "
        "or final in-memory state.",
    )


class CheckpointingConfig(BaseModel, extra="forbid"):
    """Checkpoint, weight-init, and artifact-publication settings.

    Shared across all experiment families.
    """

    periodic: PeriodicCheckpointConfig = Field(
        default_factory=PeriodicCheckpointConfig,
        description="Periodic recovery and history-snapshot checkpointing.",
    )
    selection: SelectionCheckpointConfig = Field(
        default_factory=SelectionCheckpointConfig,
        description="Metric-based best-checkpoint selection.",
    )
    artifact: ArtifactPublicationConfig = Field(
        default_factory=ArtifactPublicationConfig,
        description="Post-training artifact publication policy.",
    )

    resume_from: Optional[str] = Field(
        default=None,
        description="Checkpoint path to resume full trainer state.",
    )
    init_weights_from: Optional[str] = Field(
        default=None,
        description="Checkpoint path for model-weight initialization only.",
    )
    init_weights_groups: list[str] = Field(
        default_factory=lambda: ["all"],
        description="Named weight groups to hydrate from init_weights_from.",
    )
    diagnostic_level: Literal["minimal", "standard", "research"] = Field(
        default="standard",
        description="Instrumentation tier for diagnostic logging.",
    )
    non_finite_policy: Literal["drop", "raise"] = Field(
        default="raise",
        description="Policy for NaN/Inf diagnostics.",
    )


# =============================================================================
# Configuration-boundary types — Pydantic models for TOML/config validation
# =============================================================================


class ProviderConfig(BaseModel, extra="forbid", frozen=True):
    """Validated configuration-boundary representation of a provider spec.

    Use :meth:`resolve` to produce a runtime :class:`ProviderSpec`.
    """

    ref: str = Field(
        min_length=1,
        description="Dotted import path to an EvaluationSourceProvider class.",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Samples per evaluation batch.",
    )
    settings: dict[str, object] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the provider constructor. "
        "Must not contain 'batch_size'.",
    )

    @model_validator(mode="after")
    def _check_settings_no_batch_size(self) -> "ProviderConfig":
        if "batch_size" in self.settings:
            raise ValueError(
                "settings must not contain batch_size; "
                "use ProviderConfig.batch_size"
            )
        return self

    def resolve(self) -> "ProviderSpec":
        """Produce a runtime :class:`ProviderSpec` from this config."""
        from ehc_sn.evaluation.contracts import ProviderSpec

        return ProviderSpec(
            ref=self.ref,
            settings=dict(self.settings),
            batch_size=self.batch_size,
        )


class RegimeConfig(BaseModel, extra="forbid"):
    """Evaluation regime identity for one experiment."""

    id: str = Field(
        ...,
        min_length=1,
        description="Unique regime identifier (e.g. mazehard_diagnostic).",
    )
    kind: Literal["diagnostic", "benchmark"] = Field(
        default="diagnostic",
        description="Regime classification.",
    )


# =============================================================================
__all__ = [
    "ArtifactPublicationConfig",
    "CheckpointingConfig",
    "InvalidModelArtifactError",
    "PeriodicCheckpointConfig",
    "ProviderConfig",
    "RegimeConfig",
    "SelectionCheckpointConfig",
    "TrainerConfig",
    "resolve_core_model_config_path",
]
