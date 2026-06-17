"""Shared infrastructure config types used by all experiment families.

Consolidates one canonical ``TrainerConfig`` and one canonical
``CheckpointingConfig`` from six per-family copies into a single
source of truth.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ehc_sn.lightning.callbacks.checkpoint import CheckpointSettings


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
        default=None,
        description="Model checkpoint settings.",
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
    diagnostic_level: Literal["minimal", "standard", "research"] = Field(
        default="standard",
        description="Instrumentation tier for diagnostic logging.",
    )
    non_finite_policy: Literal["drop", "raise"] = Field(
        default="raise",
        description="Policy for NaN/Inf diagnostics.",
    )


__all__ = [
    "CheckpointingConfig",
    "TrainerConfig",
]
