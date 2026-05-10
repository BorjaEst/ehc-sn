"""Named evaluation regimes callback.

Drives diagnostic/benchmark evaluation regimes that run independently of the
fit-path ``validation_step``. Configured via ``EvaluationRegimesCallbackSettings``
in each training entry point.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from pydantic import BaseModel, Field


# =================================================================================================
class EvaluationScheduleSettings(BaseModel, extra="forbid"):
    """Schedule for a named evaluation regime."""

    every_n_epochs: int = Field(default=0, ge=0)
    every_n_steps: int = Field(default=0, ge=0)
    max_batches: int = Field(default=0, ge=0)


# =================================================================================================
class EvaluationTraceRequest(BaseModel, extra="forbid"):
    """Trace collection request for a named evaluation regime."""

    enabled: bool = Field(default=False)
    trace_keys: list[str] = Field(default_factory=list)


# =================================================================================================
class EvaluationRegimeSettings(BaseModel, extra="forbid"):
    """Configuration for a single named evaluation regime."""

    regime_id: str
    phase_kind: str = Field(default="diag")
    provider_ref: str
    provider_settings: dict[str, Any] = Field(default_factory=dict)
    schedule: EvaluationScheduleSettings = Field(default_factory=EvaluationScheduleSettings)
    trace_request: EvaluationTraceRequest = Field(default_factory=EvaluationTraceRequest)


# =================================================================================================
class EvaluationRegimesCallbackSettings(BaseModel, extra="forbid"):
    """Top-level settings for the named evaluation regimes callback."""

    regimes: list[EvaluationRegimeSettings] = Field(default_factory=list)


# =================================================================================================
class EvaluationRegimesCallback(pl.Callback):
    """Lightning callback that drives named evaluation regimes after each validation epoch."""

    def __init__(self, settings: EvaluationRegimesCallbackSettings) -> None:
        super().__init__()
        self.settings = settings

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run any due evaluation regimes after validation."""
        # Regime scheduling and execution is deferred to future implementation.


# =================================================================================================
__all__ = [
    "EvaluationRegimesCallbackSettings",
    "EvaluationRegimeSettings",
    "EvaluationRegimesCallback",
    "EvaluationScheduleSettings",
    "EvaluationTraceRequest",
]
