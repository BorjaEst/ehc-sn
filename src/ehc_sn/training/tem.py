"""TEM-specific training runtime configuration and weight-loading helpers.

TEM runtime: step-dependent schedules for memory dynamics and uncertainty.
TEM weight loading: hydrate named semantic-group weights from checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from pydantic import BaseModel, Field, model_validator
from torch import nn

# =============================================================================
# Runtime configuration
# =============================================================================


class MemoryRuntimeConfig(BaseModel, extra="forbid"):
    """Step-based runtime schedule for TEM memory dynamics."""

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
    """Step-based runtime schedule for the shared p->g gate and uncertainty correction."""

    p2g_sig_half_it: int = Field(
        default=400,
        ge=0,
        description="Sigmoid midpoint for the shared p->g gate schedule.",
    )
    p2g_sig_scale_it: int = Field(
        default=200,
        ge=1,
        description="Sigmoid scale for the shared p->g gate schedule.",
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
            raise ValueError(
                "offset_max must be greater than or equal to offset_min."
            )
        return self


class SequenceRuntimeConfig(BaseModel, extra="forbid"):
    """Sequence-level training settings for TEM chunked TBPTT."""

    tbptt_steps: int = Field(
        default=25,
        ge=1,
        description="Number of rollout steps accumulated before each optimizer update.",
    )


class ValidationRuntimeConfig(BaseModel, extra="forbid"):
    """Runner-owned safety limits for TEM validation rollouts."""

    max_rollout_steps: int | None = Field(
        default=None,
        ge=1,
        description="Normal runner-owned rollout bound for validation execution.",
    )
    hard_max_rollout_steps: int | None = Field(
        default=None,
        ge=1,
        description="Defensive runner cap for validation rollouts. Separate "
        "from semantic model max_steps.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Explicit evaluation seed used to make validation rollouts reproducible.",
    )


class RuntimeConfig(BaseModel, extra="forbid"):
    """TEM runtime configuration with configurable resolution semantics.

    - ``"scheduled"`` (default): step-dependent schedules — ``resolve_tem_runtime``
      applies ramps, sigmoid gates, and progress fractions via the current step.
    - ``"fixed"``: the schedule targets are applied directly as the runtime state —
      ``resolve_tem_runtime`` returns ``eta=config.memory.eta``,
      ``hebbian_decay=config.memory.hebbian_decay``, ``p2g_use=1.0`` (the sigmoid
      asymptote), ``p2g_uncertainty_offset=config.uncertainty.offset_min`` (the
      schedule floor).  The ``step`` argument is ignored.
    """

    kind: Literal["scheduled", "fixed"] = Field(
        default="scheduled",
        description="Runtime resolution mode.  ``'scheduled'`` applies "
        "step-dependent schedules; ``'fixed'`` applies schedule targets "
        "directly (step is ignored).",
    )

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
        description="Chunked-TBPTT sequence settings for TEM training.",
    )
    validation: ValidationRuntimeConfig = Field(
        default_factory=ValidationRuntimeConfig,
        description="Validation-only runner safety settings.",
    )


@dataclass(frozen=True)
class TEMRuntimeState:
    """Resolved TEM runtime values for the current optimizer step."""

    eta: float
    hebbian_decay: float
    p2g_use: float
    p2g_uncertainty_offset: float


def resolve_tem_runtime(  # ---------------------------------------------------
    step: int,
    config: RuntimeConfig,
) -> TEMRuntimeState:
    """Resolve TEM runtime values from the current global training step.

    When ``config.kind == "fixed"`` the ``step`` argument is ignored and the
    schedule target values in the config are applied directly — evaluation
    should not simulate training step progression.
    """
    if config.kind == "fixed":
        return TEMRuntimeState(
            eta=config.memory.eta,
            hebbian_decay=config.memory.hebbian_decay,
            p2g_use=1.0,
            p2g_uncertainty_offset=config.uncertainty.offset_min,
        )

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")

    memory = config.memory
    uncertainty = config.uncertainty
    progress_eta = min((step + 1) / float(memory.eta_it), 1.0)
    progress_decay = min((step + 1) / float(memory.lambda_it), 1.0)

    # One shared logistic gate drives both p->g loss weighting and uncertainty relaxation.
    p2g_inactive = 1.0 / (
        1.0
        + math.exp(
            (step - uncertainty.p2g_sig_half_it) / uncertainty.p2g_sig_scale_it
        )
    )
    p2g_use = 1.0 - p2g_inactive
    p2g_uncertainty_offset = (
        uncertainty.offset_min
        + (uncertainty.offset_max - uncertainty.offset_min) * p2g_inactive
    )

    return TEMRuntimeState(
        eta=progress_eta * memory.eta,
        hebbian_decay=progress_decay * memory.hebbian_decay,
        p2g_use=p2g_use,
        p2g_uncertainty_offset=p2g_uncertainty_offset,
    )


# =============================================================================
# Weight loading
# =============================================================================

# Mapping from public semantic group name to TEM model state-dict prefixes.
_GROUP_TO_PREFIXES: dict[str, tuple[str, ...]] = {
    "spatial_memory": ("hpc",),
    "path_integration": ("mec",),
    "sensory_binding": ("lec", "projections"),
    "all": (),
}

VALID_INIT_GROUPS: frozenset[str] = frozenset(_GROUP_TO_PREFIXES)
"""Set of valid semantic group names for TEM init-only hydration."""


def load_weights_from_checkpoint(  # ------------------------------------------
    model: nn.Module,
    checkpoint_path: str | Path,
    groups: Sequence[str],
) -> list[str]:
    """Hydrate named TEM semantic-group weights from a checkpoint into *model*.

    Loads only model parameter subsets for the specified groups. Optimizer,
    scheduler, and trainer-progress state in the checkpoint are not touched.
    This is distinct from ``Trainer.fit(ckpt_path=...)`` full-state resume.

    Lightning checkpoint keys prefixed with ``"model."`` are supported and
    stripped before matching.

    Args:
        model: Target TEM model to hydrate in place.
        checkpoint_path: Path to a Lightning or raw-model checkpoint.
        groups: Non-empty sequence of semantic group names.

    Returns:
        Sorted list of model state-dict keys loaded from the checkpoint.

    Raises:
        ValueError: If *groups* is empty, contains unknown names, or no keys
            matched the requested groups.
    """
    if not groups:
        raise ValueError("groups must be non-empty.")
    unknown = [g for g in groups if g not in _GROUP_TO_PREFIXES]
    if unknown:
        raise ValueError(
            f"Unknown semantic groups: {unknown!r}. "
            f"Valid groups: {sorted(_GROUP_TO_PREFIXES)!r}."
        )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_sd: dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    stripped: dict = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in raw_sd.items()
    }
    model_keys = set(model.state_dict().keys())

    use_all = "all" in groups
    prefixes = tuple(p for g in groups for p in _GROUP_TO_PREFIXES[g])
    filtered = {
        k: v
        for k, v in stripped.items()
        if k in model_keys
        and (use_all or any(k == p or k.startswith(f"{p}.") for p in prefixes))
    }
    if not filtered:
        raise ValueError(
            f"No model keys matched for groups {list(groups)!r} "
            f"in checkpoint {str(checkpoint_path)!r}."
        )

    model.load_state_dict(filtered, strict=False)
    return sorted(filtered)


# =============================================================================
__all__ = [
    "MemoryRuntimeConfig",
    "RuntimeConfig",
    "SequenceRuntimeConfig",
    "TEMRuntimeState",
    "UncertaintyRuntimeConfig",
    "ValidationRuntimeConfig",
    "resolve_tem_runtime",
    "VALID_INIT_GROUPS",
    "load_weights_from_checkpoint",
]
