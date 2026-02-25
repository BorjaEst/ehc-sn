from pathlib import Path
from typing import Optional

from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, Field


class LoggerSettings(BaseModel, extra="forbid"):
    """Logging settings for TensorBoard logger."""

    save_dir: Path = Field(
        default=Path("./logs"),
        description="Directory to save logs (default: ./logs).",
    )
    name: Optional[str] = Field(
        default=None,
        description="Experiment name for logger.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Version/run identifier (auto-increments if None).",
    )
    log_graph: bool = Field(
        default=False,
        description="Log model graph to TensorBoard.",
    )
    prefix: str = Field(
        default="",
        description="Prefix for all logged metrics.",
    )


class Logger(TensorBoardLogger):
    """Custom TensorBoard logger that accepts LoggerSettings."""

    def __init__(self, settings: LoggerSettings):
        super().__init__(**settings.model_dump())


__all__ = ["LoggerSettings", "Logger"]
