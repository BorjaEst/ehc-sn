"""Shared infrastructure config types used by all experiment families.

Consolidates one canonical ``TrainerConfig`` and one canonical
``CheckpointingConfig`` from six per-family copies into a single
source of truth.

Also defines the resolved evaluation-experiment composition:
:class:`EvaluationExperiment`, :class:`EvaluationIdentity`,
:class:`EvaluationRunRequest`, and the experiment-level configuration
types :class:`ProviderConfig`, :class:`RegimeConfig`,
:class:`CaptureConfig`.

Public resolver
----------------
:func:`ehc_sn.eval.configuration.load_evaluation_experiment` is the
canonical public entry point for loading a TOML evaluation config
and resolving it to an :class:`EvaluationExperiment`.

The legacy :func:`resolve_evaluation_experiment` is deprecated and will
be removed.  New code should import from ``ehc_sn.eval.configuration``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ehc_sn.lightning.callbacks.checkpoint import CheckpointSettings
from ehc_sn.traces.observer import TraceSpec


# =============================================================================
@dataclass(frozen=True)
class ProviderSpec:
    """Provider specification — not a resolved provider instance.

    Providers may own file handles, datasets, or device-local resources.
    Instantiation is deferred to the runner, close to execution time.
    """

    ref: str
    settings: dict[str, Any]


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
class CheckpointingConfig(BaseModel, extra="forbid"):
    """Checkpoint and weight-init settings.

    Shared across all experiment families.  Fields that are only
    relevant to some families use ``None`` defaults.
    """

    checkpoint: Optional[CheckpointSettings] = Field(
        default_factory=CheckpointSettings,
        description="Model checkpoint settings. Defaults to a CheckpointSettings() "
        "instance so that a CheckpointCallback is always constructed. "
        "Set to None to disable checkpointing.",
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
# Evaluation experiment composition — resolved by experiment-specific builders
# =============================================================================


@dataclass(frozen=True)
class EvaluationIdentity:
    """Stable artifact-identity fields for one evaluated experiment.

    Every field is required — no optional fallbacks.  Downstream consumers
    (reporting, figure selectors, inspection) use this identity for
    discoverability and compatibility checks.
    """

    task: str
    model_family: str
    trace_paradigm: str


@dataclass(frozen=True)
class EvaluationExperiment:
    """Complete resolved evaluation experiment returned by a builder.

    The generic runner receives this as a single typed input.  All provider,
    regime, trace, and identity resolution has already been performed by the
    experiment-specific builder.
    """

    executor: Any
    provider_spec: ProviderSpec
    regime_id: str
    regime_kind: str
    trace_spec: TraceSpec | None = None
    capture_profile: str = "metrics_only"
    capture_include: tuple[str, ...] = ()
    capture_exclude: tuple[str, ...] = ()
    identity: EvaluationIdentity | None = None


@dataclass(frozen=True)
class EvaluationRunRequest:
    """Invocation-level parameters for one evaluation run.

    These are run-instance values — checkpoint identity, output location,
    device, and small bounded-execution overrides.  Experiment semantics
    belong in the configuration, not here.
    """

    checkpoint_path: Path
    output_dir: Path
    device: str = "cpu"
    max_batches: int = 0
    overwrite: bool = False


# =============================================================================
# Evaluation experiment configuration — TOML-serializable sections
# =============================================================================


class ProviderConfig(BaseModel, extra="forbid"):
    """Data/provider specification for one evaluation experiment.

    ``ref`` is a dotted import path to an :class:`EvaluationSourceProvider`
    class.  Retained as a string (interim mechanism) — a typed provider
    registry is a deferred improvement.
    """

    ref: str = Field(
        ...,
        min_length=1,
        description="Dotted import path to an EvaluationSourceProvider class.",
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the provider constructor.",
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


class CaptureConfig(BaseModel, extra="forbid"):
    """Trace capture policy for one evaluation experiment.

    The ``profile`` selects a named set of trace fields defined in
    :mod:`ehc_sn.traces.specs`.  ``include`` adds extra diagnostic fields;
    ``exclude`` removes optional profile fields.  Required profile fields
    cannot be excluded — doing so raises a configuration error at builder
    time.
    """

    profile: str = Field(
        default="metrics_only",
        description="Named capture profile (see ehc_sn.traces.specs).",
    )
    profile_version: int = Field(
        default=1,
        ge=1,
        description="Version of the named profile contract.",
    )
    include: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Extra trace-key paths beyond the profile.",
    )
    exclude: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Profile field paths to omit.  Must not intersect "
        "required profile fields.",
    )


# =============================================================================
# Public resolver — load TOML config → resolved EvaluationExperiment
# =============================================================================


def resolve_evaluation_experiment(
    config_path: Path,
) -> EvaluationExperiment:
    """Load a TOML config and resolve it to an :class:`EvaluationExperiment`.

    .. deprecated::
        Use :func:`ehc_sn.eval.configuration.load_evaluation_experiment`
        instead.  This wrapper exists for backward compatibility and calls
        the new function; it returns only ``.experiment`` and discards
        provenance.

    Parameters
    ----------
    config_path:
        Path to a TOML file with a top-level ``experiment_id`` string key.

    Returns
    -------
    EvaluationExperiment
        Fully resolved experiment composition (provenance discarded).

    Raises
    ------
    ehc_sn.eval.configuration.EvaluationConfigurationError
        On any configuration loading failure.
    """
    import warnings

    from ehc_sn.eval.configuration import load_evaluation_experiment

    warnings.warn(
        "resolve_evaluation_experiment is deprecated.  Use "
        "load_evaluation_experiment from ehc_sn.eval.configuration instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    loaded = load_evaluation_experiment(config_path)
    return loaded.experiment


__all__ = [
    "CheckpointingConfig",
    "TrainerConfig",
]
