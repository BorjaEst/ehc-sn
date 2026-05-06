"""Shared EHC training-runtime schedule configuration and resolution helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator


class MemoryRuntimeConfig(BaseModel, extra="forbid"):
    """Step-based runtime schedule for EHC memory dynamics."""

    eta: float = Field(
        default=0.5,
        description="Target Hebbian write rate reached after the eta ramp completes.",
    )
    eta_it: int = Field(
        default=16000,
        ge=1,
        description="Number of optimizer steps used to ramp eta to its target value.",
    )
    hebbian_decay: float = Field(
        default=0.9999,
        description="Target Hebbian decay reached after the decay ramp completes.",
    )
    lambda_it: int = Field(
        default=200,
        ge=1,
        description="Number of optimizer steps used to ramp Hebbian decay to its target value.",
    )


class UncertaintyRuntimeConfig(BaseModel, extra="forbid"):
    """Step-based runtime schedule for MEC uncertainty correction."""

    p2g_sig_half_it: int = Field(
        default=400,
        ge=0,
        description="Sigmoid midpoint for the p->g uncertainty offset schedule.",
    )
    p2g_sig_scale_it: int = Field(
        default=200,
        ge=1,
        description="Sigmoid scale for the p->g uncertainty offset schedule.",
    )
    offset_min: float = Field(
        default=0.0,
        description="Minimum additive uncertainty offset applied at convergence.",
    )
    offset_max: float = Field(
        default=10000.0,
        description="Maximum additive uncertainty offset applied at the start of training.",
    )

    @model_validator(mode="after")
    def validate_offset_range(self) -> "UncertaintyRuntimeConfig":
        if self.offset_max < self.offset_min:
            raise ValueError("offset_max must be greater than or equal to offset_min.")
        return self


class SequenceRuntimeConfig(BaseModel, extra="forbid"):
    """Sequence-level training settings for EHC chunked TBPTT."""

    tbptt_steps: int = Field(
        default=25,
        ge=1,
        description="Number of rollout steps accumulated before each optimizer update.",
    )


class ValidationRuntimeConfig(BaseModel, extra="forbid"):
    """Runner-owned safety limits for EHC validation rollouts."""

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
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Explicit evaluation seed used to make validation rollouts reproducible.",
    )


class RuntimeConfig(BaseModel, extra="forbid"):
    """Step-based runtime schedules for EHC training dynamics and validation safety."""

    memory: MemoryRuntimeConfig = Field(
        default_factory=MemoryRuntimeConfig,
        description="Runtime schedule for Hebbian plasticity parameters.",
    )
    uncertainty: UncertaintyRuntimeConfig = Field(
        default_factory=UncertaintyRuntimeConfig,
        description="Runtime schedule for MEC uncertainty parameters.",
    )
    sequence: SequenceRuntimeConfig = Field(
        default_factory=SequenceRuntimeConfig,
        description="Chunked-TBPTT sequence settings for EHC training.",
    )
    validation: ValidationRuntimeConfig = Field(
        default_factory=ValidationRuntimeConfig,
        description="Validation-only runner safety settings.",
    )


@dataclass(frozen=True)
class EHCRuntimeState:
    """Resolved EHC runtime values for the current optimizer step."""

    eta: float
    hebbian_decay: float
    p2g_uncertainty_offset: float
    p2g_trust: float


def resolve_ehc_runtime(step: int, config: RuntimeConfig) -> EHCRuntimeState:
    """Resolve EHC runtime values from the current global training step."""
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")

    memory = config.memory
    uncertainty = config.uncertainty
    progress_eta = min((step + 1) / float(memory.eta_it), 1.0)
    progress_decay = min((step + 1) / float(memory.lambda_it), 1.0)
    p2g_trust = 1.0 / (1.0 + math.exp(-(step - uncertainty.p2g_sig_half_it) / uncertainty.p2g_sig_scale_it))
    p2g_uncertainty_offset = uncertainty.offset_min + (1.0 - p2g_trust) * (uncertainty.offset_max - uncertainty.offset_min)

    return EHCRuntimeState(
        eta=progress_eta * memory.eta,
        hebbian_decay=progress_decay * memory.hebbian_decay,
        p2g_uncertainty_offset=p2g_uncertainty_offset,
        p2g_trust=p2g_trust,
    )


__all__ = [
    "MemoryRuntimeConfig",
    "RuntimeConfig",
    "SequenceRuntimeConfig",
    "EHCRuntimeState",
    "UncertaintyRuntimeConfig",
    "ValidationRuntimeConfig",
    "resolve_ehc_runtime",
]
