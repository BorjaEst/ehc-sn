from typing import Literal, Optional

from lightning.pytorch import callbacks as lp_callbacks
from pydantic import BaseModel, Field, model_validator


# =============================================================================
class CheckpointSettings(BaseModel, extra="forbid"):
    """Settings for model checkpointing."""

    dirpath: Optional[str] = Field(
        default=None,
        description="Optional directory where checkpoints are written.",
    )
    filename: Optional[str] = Field(
        default=None,
        description="Optional checkpoint filename pattern.",
    )

    monitor: Optional[str] = Field(
        default=None,
        description="Optional validation metric name used to rank checkpoints.",
    )
    mode: Literal["min", "max"] = Field(
        default="max",
        description="Optimization direction for the monitored metric when monitor is set.",
    )
    save_top_k: int = Field(
        default=0,
        ge=-1,
        description="Number of checkpoints to retain (-1 keeps all, 0 disables "
        "top-k saving).",
    )
    every_n_train_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Save every N training steps when set.",
    )
    save_last: bool = Field(
        default=True,
        description="Whether to always save the last checkpoint.",
    )
    save_on_exception: bool = Field(
        default=False,
        description="Whether to save a checkpoint when training exits due to "
        "an exception.",
    )
    save_weights_only: bool = Field(
        default=False,
        description="Whether to save only model weights without optimizer or "
        "scheduler state.",
    )

    @model_validator(mode="after")
    def _validate_policy(self) -> "CheckpointSettings":
        if self.monitor is None and self.save_top_k not in {0, -1}:
            raise ValueError(
                "When checkpoint.monitor is unset, checkpoint.save_top_k must "
                "be 0 (last-only via save_last) or -1 (keep-all)."
            )
        if self.every_n_train_steps is not None and self.save_top_k == 0:
            raise ValueError(
                "checkpoint.every_n_train_steps requires checkpoint.save_top_k "
                "!= 0. Set save_top_k = -1 to keep all periodic checkpoints, "
                "or a positive integer to keep only the top-K."
            )
        return self


# =============================================================================
class CheckpointCallback(lp_callbacks.ModelCheckpoint):
    """Custom ModelCheckpoint that accepts CheckpointSettings."""

    def __init__(self, settings: CheckpointSettings):
        super().__init__(**settings.model_dump())


# =============================================================================
__all__ = ["CheckpointSettings", "CheckpointCallback"]
