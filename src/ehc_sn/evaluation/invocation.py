"""Generic invocation envelope for one evaluation run.

This module defines :class:`EvaluationInvocationConfig` and its ten
sub-config Pydantic models, as well as the narrow
:class:`EvaluationBuildRequest` that experiment-family builders receive.

It is a pure Layer-2 module owned by ``ehp_sn.evaluation`` and must not import
from ``experiments/``, ``models/``, ``tasks/``, ``adapters/``, or
``lightning/``.

The envelope carries user-supplied choices.  Recipe identity (which
controller, objective, provider, regime kind) is derived from the
``alias`` field by the recipe registry, not from fields in this module.

Usage::

    from ehc_sn.evaluation.invocation import EvaluationInvocationConfig

    config = EvaluationInvocationConfig.model_validate({
        "alias": "arena-tem-v1",
        "model": {"uri": "./checkpoints/best.ckpt"},
    })
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from ehc_sn.evaluation.model_ref import ModelRef
from ehc_sn.model_artifacts import ModelArtifact
from ehc_sn.model_artifacts.assembly import ModelAssembly

# =============================================================================
# Enums and literals
# =============================================================================


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class EvaluationPrecision(StrEnum):
    FLOAT32 = "32"
    FLOAT16_MIXED = "16-mixed"
    BF16_MIXED = "bf16-mixed"


# =============================================================================
# Sub-config models
# =============================================================================


class ModelRefConfig(BaseModel, frozen=True):
    """Reference to a model artifact.

    The ``uri`` is opaque to the invocation layer.  Interpretation
    belongs to ``ModelArtifactLoader`` (see ``ehp_sn.evaluation.model_ref``).

    A model reference may be omitted from TOML files when the CLI
    ``--model`` flag supplies it.
    """

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(
        default="", description="Model artifact URI or local path."
    )
    expected_digest: str | None = Field(
        default=None,
        description="Expected content digest (e.g. sha256:...).",
    )


class DatasetRefConfig(BaseModel, frozen=True):
    """Reference to a dataset artifact."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(
        ..., min_length=1, description="Dataset URI or local path."
    )
    expected_digest: str | None = Field(
        default=None,
        description="Expected content digest (e.g. sha256:...).",
    )


class CaseSelectionConfig(BaseModel, frozen=True):
    """Selection of evaluation cases from a dataset split.

    ``ids`` and ``count`` are mutually exclusive.  When neither is
    supplied the recipe/provider default applies.
    """

    model_config = ConfigDict(extra="forbid")

    split: Split = Field(default=Split.TEST, description="Dataset partition.")
    count: int | None = Field(
        default=None, ge=1, description="Maximum number of selected cases."
    )
    seed: int = Field(
        default=0, description="Deterministic sampling/order seed."
    )
    ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Explicit case IDs.  Mutually exclusive with count.",
    )
    dataset: DatasetRefConfig | None = Field(
        default=None,
        description="Optional override of the recipe-default dataset.",
    )

    @model_validator(mode="after")
    def _check_ids_or_count(self) -> Self:
        if self.ids and self.count is not None:
            raise ValueError(
                "cases.ids and cases.count are mutually exclusive; "
                f"got ids={self.ids!r} and count={self.count!r}."
            )
        return self


class EvaluationRuntimeConfig(BaseModel, frozen=True):
    """Operational execution parameters."""

    model_config = ConfigDict(extra="forbid")

    device: str = Field(default="auto", description="Torch device string.")
    batch_size: int = Field(
        default=1, ge=1, description="Samples per evaluation batch."
    )
    workers: int = Field(
        default=0, ge=0, description="DataLoader worker count."
    )
    precision: EvaluationPrecision = Field(
        default=EvaluationPrecision.FLOAT32,
        description="Evaluation-time precision.",
    )
    deterministic: bool = Field(
        default=True,
        description="Enable deterministic execution.",
    )


class EvaluationOptions(BaseModel, frozen=True):
    """Base class for pair-specific scientific evaluation overrides.

    Subclass this for recipes that need typed options (e.g.
    ``ArenaTEMV1EvaluationOptions``).  The base class is empty and
    accepts no fields beyond the defaults.
    """

    model_config = ConfigDict(extra="forbid")


class CaptureConfig(BaseModel, frozen=True):
    """Trace-capture policy for one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = Field(
        default=None,
        description="Override recipe default capture enablement.",
    )
    profile: str | None = Field(
        default=None,
        description="Named capture profile (see ehp_sn.traces.specs).",
    )
    profile_version: int | None = Field(
        default=None, ge=1, description="Version of the named profile contract."
    )
    include: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Extra trace-key paths beyond the profile.",
    )
    exclude: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Profile field paths to omit.",
    )
    max_cases: int | None = Field(
        default=None,
        ge=0,
        description="Max cases with persisted traces.  None=all.",
    )
    max_steps_per_case: int | None = Field(
        default=None, ge=1, description="Max steps captured per case."
    )
    max_units_per_field: int | None = Field(
        default=None, ge=1, description="Max units captured per trace field."
    )
    storage_budget_bytes: int | None = Field(
        default=None, ge=1, description="Hard storage budget for trace data."
    )


class InspectionConfig(BaseModel, frozen=True):
    """Inspection (figure rendering) policy for one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable figure rendering.")
    figures: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Requested figure identifiers.  Empty = recipe defaults.",
    )
    formats: tuple[Literal["png", "svg", "pdf"], ...] = Field(
        default=("png",),
        description="Output image formats.",
    )
    selected_cases: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Specific case IDs to render figures for.",
    )


class MetricThresholdConfig(BaseModel, frozen=True):
    """Acceptance threshold for one metric."""

    model_config = ConfigDict(extra="forbid")

    minimum: float | None = Field(
        default=None, description="Minimum acceptable value."
    )
    maximum: float | None = Field(
        default=None, description="Maximum acceptable value."
    )


class ValidationConfig(BaseModel, frozen=True):
    """Post-evaluation metric validation policy."""

    model_config = ConfigDict(extra="forbid")

    metrics: Mapping[str, MetricThresholdConfig] = Field(
        default_factory=dict,
        description="Per-metric acceptance thresholds.",
    )
    fail_on_violation: bool = Field(
        default=True,
        description="Raise / exit on threshold violation.",
    )


class OutputConfig(BaseModel, frozen=True):
    """Local artifact-store settings."""

    model_config = ConfigDict(extra="forbid")

    directory: Path = Field(
        default=Path("outputs/evaluation"),
        description="Local artifact output directory.",
    )
    overwrite: bool = Field(
        default=False, description="Replace existing output."
    )
    retain_traces: bool = Field(
        default=True, description="Persist per-step traces."
    )


class TrackingConfig(BaseModel, frozen=True):
    """MLflow tracking configuration — recording is always enabled.

    ``tracking_uri`` resolution follows this precedence::

        CLI ``--tracking-uri``
            > invocation ``[tracking].tracking_uri``
            > ``MLFLOW_TRACKING_URI`` env var
            > ``sqlite:///outputs/mlflow/mlflow.db``  (local fallback)
    """

    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(
        default="ehp-evaluation",
        description="MLflow experiment name.",
    )
    run_name: str | None = Field(
        default=None, description="Human-readable run name."
    )
    tags: Mapping[str, str] = Field(
        default_factory=dict,
        description="User-defined MLflow tags.",
    )
    tracking_uri: str | None = Field(
        default=None,
        description="MLflow tracking server URI.  Falls back to MLFLOW_TRACKING_URI, then local EHP SQLite store.",
    )
    nested: bool = Field(
        default=False,
        description="Create nested MLflow run.",
    )
    log_system_metrics: bool = Field(
        default=False,
        description="Log system metrics (CPU/GPU/memory).",
    )


# =============================================================================
# Top-level invocation envelope
# =============================================================================


class EvaluationInvocationConfig(BaseModel, frozen=True):
    """Versioned user-intent contract for one evaluation run.

    This is the single validated entry point for all evaluation
    configuration — TOML files and CLI overrides are merged and validated
    against this schema.

    Recipe identity is carried by ``alias`` alone.  All other fields are
    invocation choices with recipe-default fallbacks.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(
        default=1,
        description="Invocation artifact schema version.",
    )
    alias: str = Field(
        ...,
        min_length=1,
        description="Canonical evaluation recipe alias (e.g. arena-tem-v1).",
    )
    model: ModelRefConfig = Field(
        default_factory=ModelRefConfig,
        description="Model artifact reference.  May be supplied via --model CLI flag.",
    )

    cases: CaseSelectionConfig = Field(
        default_factory=CaseSelectionConfig,
        description="Evaluation case selection.",
    )
    runtime: EvaluationRuntimeConfig = Field(
        default_factory=EvaluationRuntimeConfig,
        description="Operational execution parameters.",
    )
    evaluation: Mapping[str, object] = Field(
        default_factory=dict,
        description="Pair-specific scientific options (validated by recipe).",
    )
    capture: CaptureConfig = Field(
        default_factory=CaptureConfig,
        description="Trace-capture policy.",
    )
    inspection: InspectionConfig = Field(
        default_factory=InspectionConfig,
        description="Figure rendering policy.",
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Metric threshold validation.",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Local artifact output.",
    )
    tracking: TrackingConfig = Field(
        default_factory=TrackingConfig,
        description="MLflow tracking configuration.",
    )


# =============================================================================
# Build request
# =============================================================================


@dataclass(frozen=True)
class ResolvedDatasetMetadata:
    """Resolved metadata for a dataset used in evaluation."""

    name: str
    version: str
    digest: str | None = None


@dataclass(frozen=True)
class ResolvedCaseSelection:
    """A fully resolved set of cases to evaluate.

    The ``case_ids`` tuple records the actual deterministic selection
    after the recipe's default and user overrides are applied.
    """

    dataset: ResolvedDatasetMetadata
    split: str
    case_ids: tuple[str, ...] = ()
    count: int | None = None


@dataclass(frozen=True)
class ResolvedCapturePlan:
    """A fully resolved capture plan for one evaluation run.

    Produced by applying user overrides to the recipe-default capture
    profile and validating against the model artifact's capabilities.
    """

    profile: str
    fields: tuple[str, ...]
    profile_version: int = 1
    exclude: tuple[str, ...] = ()
    max_cases: int | None = None
    max_steps_per_case: int | None = None
    unit_selection: dict[str, str] | None = None
    storage_budget_bytes: int | None = None


ModelArtifactSourceKind = Literal["model-artifact", "legacy-checkpoint"]


@dataclass(frozen=True)
class ResolvedModelArtifact:
    """Concrete model weights source after URI resolution.

    Attributes
    ----------
    requested_uri:
        The original user-supplied URI or path.
    resolved_source:
        Concrete local path or resolved MLflow model version URI.
    source_kind:
        ``"model-artifact"`` for artifact directories with manifest.json,
        ``"legacy-checkpoint"`` for raw checkpoint files.
    artifact:
        The opened ``ModelArtifact``, or ``None`` for legacy checkpoints.
    digest:
        Optional content digest for integrity verification.
    schema_version:
        Optional model artifact schema version.
    model_family:
        Canonical model-family identifier from the artifact manifest
        (e.g. ``"tem-v1"``).  ``None`` for legacy checkpoints.
    model_type:
        Concrete model type discriminator from the artifact manifest.
    model_config:
        The validated ``ModelAssembly`` from the artifact's ``assembly.toml``,
        the legacy ``model.toml`` content as a plain dict, or ``None`` for
        checkpoints with no embedded config.
    capabilities:
        Capability strings declared in the artifact manifest.
    """

    requested_uri: str
    resolved_source: str
    source_kind: ModelArtifactSourceKind = "legacy-checkpoint"
    artifact: object | None = None  # ModelArtifact | None
    digest: str | None = None
    schema_version: int | None = None
    model_family: str | None = None
    model_type: str | None = None
    model_config: ModelAssembly | dict[str, object] | None = None
    capabilities: frozenset[str] | None = None


@runtime_checkable
class ModelArtifactResolver(Protocol):
    """Resolve a ``ModelRef`` to a concrete ``ResolvedModelArtifact``."""

    def resolve(self, ref: ModelRef) -> ResolvedModelArtifact: ...


@dataclass(frozen=True)
class LocalModelArtifactResolver:
    """Resolve local-path ``ModelRef`` references.

    For the ``local`` scheme, the value is treated as a filesystem path.
    MLflow schemes raise ``NotImplementedError``.
    """

    def resolve(self, ref: ModelRef) -> ResolvedModelArtifact:
        if ref.scheme == "local":
            path = Path(ref.value).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Model path does not exist: {path}")

            # Check if the path is an artifact directory (has manifest.json).
            if path.is_dir() and (path / "manifest.json").is_file():
                return self._resolve_artifact_dir(path, ref.value)

            # Single checkpoint file.
            if not path.is_file():
                raise FileNotFoundError(f"Model path does not exist: {path}")
            return ResolvedModelArtifact(
                requested_uri=ref.value,
                resolved_source=str(path),
            )
        raise NotImplementedError(
            f"Model ref scheme {ref.scheme!r} is not yet supported. "
            f"Use a local file path for now."
        )

    @staticmethod
    def _resolve_artifact_dir(
        path: Path, requested_uri: str
    ) -> ResolvedModelArtifact:
        """Resolve a model artifact directory into a ``ResolvedModelArtifact``.

        Opens the artifact through ``ModelArtifact.open()`` — the single
        authoritative interpreter of the artifact format.
        """
        artifact = ModelArtifact.open(path)
        caps: frozenset[str] | None = (
            frozenset(artifact.manifest.capabilities)
            if artifact.manifest.capabilities
            else None
        )
        return ResolvedModelArtifact(
            requested_uri=requested_uri,
            resolved_source=str(path.resolve()),
            source_kind="model-artifact",
            artifact=artifact,
            schema_version=artifact.manifest.artifact_schema_version,
            model_family=artifact.manifest.model_family or None,
            model_type=artifact.manifest.model_type or None,
            # New-format artifacts carry a validated ModelAssembly;
            # legacy core-only artifacts carry the raw flat TOML dict.
            model_config=(
                artifact.assembly
                if artifact.assembly is not None
                else artifact.assembly_config
            ),
            capabilities=caps,
        )


@dataclass(frozen=True)
class EvaluationBuildRequest:
    """Narrow build request for an experiment-family pair builder.

    The builder receives only the inputs it needs to construct a resolved
    ``EvaluationExperiment``.  Model reference, inspection, validation,
    output, and tracking concerns stay with the application layer.

    When the model artifact provides a resolved configuration
    (``model_artifact.model_config``), the builder should prefer it over
    any hardcoded config path.
    """

    cases: ResolvedCaseSelection
    runtime: EvaluationRuntimeConfig
    options: EvaluationOptions
    capture: ResolvedCapturePlan
    model_artifact: ResolvedModelArtifact | None = None


@dataclass(frozen=True)
class EvaluationExecutionRequest:
    """Combines a resolved experiment with a resolved model artifact for execution.

    This is the input contract for ``run_offline_eval``.  The builder has
    already constructed the pair-specific architecture; this request carries
    the concrete weights source and run-instance parameters.
    """

    experiment: object  # EvaluationExperiment (avoid circular import)
    model_artifact: ResolvedModelArtifact
    output_dir: Path
    device: str = "cpu"
    max_batches: int = 0
    overwrite: bool = False
    no_reuse: bool = False
    precision: EvaluationPrecision = EvaluationPrecision.FLOAT32


@dataclass(frozen=True)
class EvaluationRunRequest:
    """Combines build inputs with post-execution policy.

    Separated from ``ResolvedEvaluationInvocation`` so that builders and
    the runtime never own validation thresholds, output persistence, or
    MLflow tracking configuration.
    """

    build: EvaluationBuildRequest
    model: ResolvedModelArtifact
    validation: ValidationConfig
    output: OutputConfig
    tracking: TrackingConfig


# =============================================================================
# Merge precedence — recipe/Pydantic defaults → file values → CLI overrides
# =============================================================================


def _remove_unset(
    mapping: Mapping[str, object],
) -> dict[str, object]:
    """Return a new dict with ``None`` values removed."""
    return {k: v for k, v in mapping.items() if v is not None}


def _deep_merge(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    """Deep-merge *override* into *base* and return a new dict.

    Nested ``Mapping`` values are merged recursively.  All other values
    (including tuples and lists) are replaced.  ``None`` values in
    *override* are skipped.
    """
    result = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def merge_evaluation_config(
    *,
    recipe_defaults: Mapping[str, object],
    file_values: Mapping[str, object] | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> EvaluationInvocationConfig:
    """Merge recipe/Pydantic defaults → file values → CLI overrides.

    Returns a validated ``EvaluationInvocationConfig``.  CLI values
    with ``None`` are treated as "not supplied" and do not override.

    Parameters
    ----------
    recipe_defaults:
        A mapping of default values derived from the recipe and its
        Pydantic sub-config field defaults.
    file_values:
        Optional dict parsed from a TOML evaluation config file.
    cli_overrides:
        Optional sparse dict of explicit CLI flag values.

    Returns
    -------
    EvaluationInvocationConfig
        Fully merged and validated invocation config.
    """
    merged = _deep_merge(recipe_defaults, file_values or {})
    merged = _deep_merge(merged, _remove_unset(cli_overrides or {}))
    return EvaluationInvocationConfig.model_validate(merged)


# =============================================================================
__all__ = [
    "CaptureConfig",
    "CaseSelectionConfig",
    "DatasetRefConfig",
    "EvaluationBuildRequest",
    "EvaluationExecutionRequest",
    "EvaluationInvocationConfig",
    "EvaluationOptions",
    "EvaluationPrecision",
    "EvaluationRunRequest",
    "EvaluationRuntimeConfig",
    "InspectionConfig",
    "LocalModelArtifactResolver",
    "MetricThresholdConfig",
    "ModelArtifactResolver",
    "ModelRefConfig",
    "OutputConfig",
    "ResolvedCapturePlan",
    "ResolvedCaseSelection",
    "ResolvedDatasetMetadata",
    "ResolvedModelArtifact",
    "Split",
    "TrackingConfig",
    "ValidationConfig",
    "merge_evaluation_config",
]
