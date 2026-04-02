"""Shared orchestration for the B0 MazeHard bridge benchmark."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


class B0BenchmarkConfig(BaseModel, extra="forbid"):
    """Benchmark-owned metadata for one B0 variant."""

    @classmethod
    def from_config(cls, config_path: Path) -> "B0BenchmarkConfig":
        """Load the B0 benchmark configuration from the given path."""
        loaded = tomllib.load(config_path.open("rb"))
        return cls.model_validate(loaded)


class B0Benchmark:
    """Shared orchestration for the B0 MazeHard bridge benchmark."""

    def __init__(self, config: B0BenchmarkConfig) -> None:
        self.config = config


__all__ = ["B0Benchmark", "B0BenchmarkConfig"]
