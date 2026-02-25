from lightning.pytorch.callbacks import ModelCheckpoint
from pydantic import BaseModel, Field


class CheckpointSettings(BaseModel, extra="forbid"):
    """Settings for model checkpointing."""

    every_n_train_steps: int = Field(
        default=1000,
        description="Save a checkpoint every N training steps.",
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
