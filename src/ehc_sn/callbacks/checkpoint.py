from typing import Literal, Optional

from lightning.pytorch.callbacks import ModelCheckpoint
from pydantic import BaseModel, Field


class CheckpointSettings(BaseModel, extra="forbid"):
    """Settings for model checkpointing."""

    monitor: Optional[str] = Field(
        default=None,
        description="Optional validation metric name used to rank checkpoints.",
    )
    mode: Literal["min", "max"] = Field(
        default="max",
        description="Optimization direction for the monitored metric when monitor is set.",
    )
    save_top_k: int = Field(
        default=1,
        description="Number of validation-boundary checkpoints to retain.",
    )
    every_n_epochs: int = Field(
        default=1,
        ge=1,
        description="Checkpoint on validation boundaries for epochs divisible by this value.",
    )
    save_on_train_epoch_end: bool = Field(
        default=False,
        description="When false, save checkpoints after validation instead of at train epoch end.",
    )
    save_last: bool = Field(
        default=True,
        description="Whether to always save the last checkpoint.",
    )


class CheckpointCallback(ModelCheckpoint):
    """Custom ModelCheckpoint that accepts CheckpointSettings."""

    def __init__(self, settings: CheckpointSettings):
        super().__init__(**settings.model_dump())


__all__ = ["CheckpointSettings", "CheckpointCallback"]
