"""Shared HRM runtime configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationRuntimeConfig(BaseModel, extra="forbid"):
    """Runner-owned safety limits for HRM validation rollouts."""

    max_rollout_steps: int | None = Field(
        default=None,
        ge=1,
        description="Normal runner-owned rollout bound for validation execution.",
    )
    hard_max_rollout_steps: int | None = Field(
        default=None,
        ge=1,
        description="Defensive runner cap for validation rollouts. Separate from semantic model max_steps.",
    )


class RuntimeConfig(BaseModel, extra="forbid"):
    """ """

    validation: ValidationRuntimeConfig = Field(
        default_factory=ValidationRuntimeConfig,
        description="Validation-only runner safety settings.",
    )


__all__ = ["RuntimeConfig", "ValidationRuntimeConfig"]
