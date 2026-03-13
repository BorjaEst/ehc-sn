from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.utils.data import Dataset

from ehc_sn.data.index import MazeIndexEntry


# =================================================================================================
class MazeMetadata(BaseModel, extra="allow"):
    """Dataset-level metadata for a maze dataset.

    This is typically read from a companion metadata file and is less strict
    than :class:`~ehc_sn.data.index.MazeIndexEntry` (``extra=allow``).
    """

    source: str = Field(..., description="Source dataset name")
    split: str = Field(..., description="Split name (e.g., 'train', 'val', 'test')")
    n_samples: int = Field(..., ge=0, description="Number of samples in the dataset")
    shape: list[int] = Field(..., description="Shape of the maze (height, width)")
    channels: list[str] = Field(..., description="List of channel names (e.g., ['observation', 'goal'])")


# =================================================================================================
class MazeDataset(Dataset):

    def __init__(  # ------------------------------------------------------------------------------
        self, entries: list[MazeIndexEntry], data_dir: Path, transform: Callable | None = None,
    ) -> None:  # fmt: skip
        """Dataset for maze data, backed by memory-mapped .npy files."""
        if not entries:
            raise ValueError("MazeDataset requires at least one index entry.")

        splits = {entry.split for entry in entries}
        if len(splits) != 1:
            raise ValueError(
                "MazeDataset requires entries from exactly one split; resolve a single split before loading arrays."
            )

        split = next(iter(splits))
        if (data_dir / split).is_dir():
            raise ValueError(
                f"MazeDataset expects a resolved split directory, got dataset root '{data_dir}'. "
                f"Use '{data_dir / split}' instead."
            )

        self._entries = entries
        self._transform = transform
        channels = entries[0].channels
        self._arrays = {ch: np.load(data_dir / f"{ch}.npy", mmap_mode="r") for ch in channels}

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        dirs = sorted(self._arrays)
        return f"{type(self).__name__}(" f"n={len(self)}, " f"dirs={dirs}, " f"transform={self._transform!r})"

    def __getitem__(  # ---------------------------------------------------------------------------
        self, idx: int,
    ) -> dict[str, Tensor]:  # fmt: skip
        sample = {k: v[idx] for k, v in self._arrays.items()}
        if self._transform:
            sample = self._transform(sample)
        return {k: torch.from_numpy(np.array(v)) for k, v in sample.items()}


# =================================================================================================
__all__ = ["MazeMetadata", "MazeDataset"]
