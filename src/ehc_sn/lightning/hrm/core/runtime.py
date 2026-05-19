"""Shared HRM runtime configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ValidationRuntimeConfig(BaseModel, extra="forbid"):
    """Runner-owned safety limits for HRM validation rollouts."""

    max_rollout_steps: int | None = Field(
        default=10,
        ge=1,
        description="Normal runner-owned rollout bound for validation execution.",
    )
    hard_max_rollout_steps: int | None = Field(
        default=None,
        ge=1,
        description="Defensive runner cap for validation rollouts. Separate from the task-owned episode_horizon.",
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ValidationRuntimeConfig":
        max_steps = self.max_rollout_steps
        hard_steps = self.hard_max_rollout_steps
        if max_steps is not None and hard_steps is not None:
            if hard_steps <= max_steps:
                raise ValueError(
                    "hard_max_rollout_steps must be greater than "
                    "max_rollout_steps when both are set."
                )
        return self


class RuntimeConfig(BaseModel, extra="forbid"):
    """Shared runtime configuration for HRM rollouts, including validation
    safety limits. The runner should enforce these limits during rollout
    execution.
    """

    validation: ValidationRuntimeConfig = Field(
        default_factory=ValidationRuntimeConfig,
        description="Validation-only runner safety settings.",
    )


__all__ = ["RuntimeConfig", "ValidationRuntimeConfig"]
