"""Shared orchestration for the B0 MazeHard bridge benchmark."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# =================================================================================================
class B0BenchmarkConfig(BaseModel, extra="forbid"):
    """Benchmark-owned metadata for one B0 variant."""

    ...

    @classmethod
    def from_config(  # ---------------------------------------------------------------------------
        cls, config_path: Path,
    ) -> B0BenchmarkConfig:  # fmt: skip
        """Load the B0 benchmark configuration from the given path."""
        loaded = tomllib.load(config_path.open("rb"))
        return cls.model_validate(loaded)


# =================================================================================================
class B0Benchmark:
    """Shared orchestration for the B0 MazeHard bridge benchmark."""

    def __init__(self, config: B0BenchmarkConfig) -> None:
        self.config = config


# =================================================================================================
__all__ = [
    "B0BenchmarkConfig",  "B0Benchmark",
]  # fmt: skip
