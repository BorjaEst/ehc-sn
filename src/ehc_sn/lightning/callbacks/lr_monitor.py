from typing import Literal

from lightning.pytorch import callbacks as lp_callbacks
from pydantic import BaseModel, Field


# =============================================================================
class LearningRateMonitorSettings(BaseModel, extra="forbid"):
    """Settings for Lightning's built-in LearningRateMonitor callback."""

    logging_interval: Literal["step", "epoch"] = Field(
        default="step",
        description="Cadence used by LearningRateMonitor when logging optimizer state.",
    )
    log_momentum: bool = Field(
        default=False,
        description="Whether to include momentum statistics in logged series.",
    )
    log_weight_decay: bool = Field(
        default=False,
        description="Whether to include weight-decay statistics in logged series.",
    )


# =============================================================================
class LearningRateMonitor(lp_callbacks.LearningRateMonitor):
    """Wrapper for Lightning's built-in LearningRateMonitor callback."""

    def __init__(self, settings: LearningRateMonitorSettings) -> None:
        super().__init__(**settings.model_dump())


# =============================================================================
__all__ = ["LearningRateMonitorSettings"]
