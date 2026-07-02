"""Pydantic schemas for recipe TOML files.

Provides :class:`EvaluationRecipeConfig` for validating declarative
recipe definitions stored under ``config/evaluation/recipes/``.

Boundary rules:
    - Must not import from ``experiments/``, ``models/``, ``tasks/``,
      ``adapters/``, or ``lightning/``.
    - May reuse Pydantic sub-config models from ``ehp_sn.evaluation.invocation``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ehc_sn.evaluation.invocation import (
    CaptureConfig,
    CaseSelectionConfig,
    EvaluationPrecision,
    InspectionConfig,
)


class CapabilityId(StrEnum):
    """Registered capability identifiers for model-artifact compatibility checks.

    Each ID describes an executable interface that the evaluation protocol
    requires from the model artifact.  Unknown IDs are rejected at recipe
    parse time.
    """

    RECURRENT_STEP = "recurrent-step"
    SPATIAL_LATENTS = "spatial-latents"
    EPISODIC_MEMORY = "episodic-memory"
    DELIBERATIVE = "deliberative"
    ANALYTIC_FUTURE_STATE = "analytic-future-state"


class RecipeCatalogueError(ValueError):
    """Raised when a recipe TOML cannot be located, read, or validated."""


class ModelFamily(StrEnum):
    """Canonical model-family identifiers.

    Values use hyphenated form for TOML serialization.
    """

    TEM_V1 = "tem-v1"
    TEM_V2 = "tem-v2"
    HRM_V1 = "hrm-v1"
    HRM_V2 = "hrm-v2"


class RecipeContractError(ValueError):
    """Raised when the recipe TOML metadata disagrees with the Python binding
    or violates a cross-reference constraint (task identity, model family,
    primary metric, capabilities)."""


class EvaluationRecipeConfig(BaseModel):
    """Repository-owned declarative recipe loaded from TOML.

    This model validates one ``config/evaluation/recipes/<alias>.toml``
    file.  The ``alias`` field must match the filename stem and the
    requested :class:`EvaluationAlias`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Recipe schema version.",
    )
    alias: str = Field(
        ...,
        min_length=1,
        description="Canonical evaluation recipe alias.",
    )
    task: str = Field(
        ...,
        min_length=1,
        description="Task-family identifier (e.g. 'arena', 'mazehard').",
    )
    model_family: ModelFamily = Field(
        ...,
        description="Model-family identifier (e.g. 'tem-v1', 'hrm-v1').",
    )
    primary_metric: str | None = Field(
        default=None,
        description="Canonical primary metric name for this task--model pair.",
    )
    required_capabilities: tuple[CapabilityId, ...] = Field(
        ...,
        description="Model capabilities the loaded artifact must satisfy. "
        "Use an empty list when no additional constraints apply.",
    )
    cases: CaseSelectionConfig = Field(
        default_factory=CaseSelectionConfig,
        description="Default case selection (split, count, seed).",
    )
    evaluation: Mapping[str, object] = Field(
        default_factory=dict,
        description="Default pair-specific scientific options.",
    )
    capture: CaptureConfig = Field(
        default_factory=CaptureConfig,
        description="Default trace-capture policy.",
    )
    inspection: InspectionConfig = Field(
        default_factory=InspectionConfig,
        description="Default figure rendering policy.",
    )
    precision: EvaluationPrecision | None = Field(
        default=None,
        description="Default evaluation precision.  None = invocation default.",
    )


__all__ = [
    "CapabilityId",
    "EvaluationRecipeConfig",
    "ModelFamily",
    "RecipeCatalogueError",
    "RecipeContractError",
]
