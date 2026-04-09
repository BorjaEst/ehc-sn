"""Shared HRM runtime configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationRuntimeConfig(BaseModel, extra="forbid"):
    """Runner-owned safety limits for HRM validation rollouts."""

    hard_max_steps: int | None = Field(
        default=None,
        ge=1,
        description="Defensive runner cap for validation rollouts. Separate from semantic model max_steps.",
    )


class RuntimeConfig(BaseModel, extra="forbid"):
    """HRM runtime settings owned by the learner surface."""

    validation: ValidationRuntimeConfig = Field(
        default_factory=ValidationRuntimeConfig,
        description="Validation-only runner safety settings.",
    )


__all__ = ["RuntimeConfig", "ValidationRuntimeConfig"]
