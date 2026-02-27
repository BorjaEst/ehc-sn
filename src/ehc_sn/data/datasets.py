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
    source: str = Field(..., description="Source dataset name")
    split: str = Field(..., description="Split name (e.g., 'train', 'val', 'test')")
    n_samples: int = Field(..., ge=0, description="Number of samples in the dataset")
    shape: list[int] = Field(..., description="Shape of the maze (height, width)")
    channels: list[str] = Field(..., description="List of channel names (e.g., ['observation', 'goal'])")


# =================================================================================================
class MazeDataset(Dataset):

    def __init__(  # ------------------------------------------------------------------------------
        self, entries: list[MazeIndexEntry], data_root: Path, transform: Callable | None = None,
    ) -> None:  # fmt: skip
        """Dataset for maze data, backed by memory-mapped .npy files."""
        self._entries = entries
        self._transform = transform
        self._arrays = {e.file: self.load_channels(entries, data_root, e.file) for e in entries}

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        dirs = sorted(self._arrays)
        return f"{type(self).__name__}(" f"n={len(self)}, " f"dirs={dirs}, " f"transform={self._transform!r})"

    @classmethod
    def load_channels(  # -------------------------------------------------------------------------
        cls, entries: list[MazeIndexEntry], data_root: Path, split_dir: str,
    ) -> dict[str, np.ndarray]:  # fmt: skip
        """Return the memory-mapped arrays for all split directories."""
        return {ch: cls.load_channel(data_root, split_dir, ch) for ch in entries[0].channels}

    @staticmethod
    def load_channel(  # --------------------------------------------------------------------------
        data_root: Path, split_dir: str, channel: str,
    ) -> np.ndarray:  # fmt: skip
        """Load a single channel array from a .npy file."""
        return np.load(data_root / split_dir / f"{channel}.npy", mmap_mode="r")

    def __getitem__(  # ---------------------------------------------------------------------------
        self, idx: int,
    ) -> dict[str, Tensor]:  # fmt: skip
        entry = self._entries[idx]
        arrays = self._arrays[entry.file]
        channels: dict[str, np.ndarray] = {k: v[entry.row] for k, v in arrays.items()}

        if self._transform is not None:
            channels = self._transform(channels)

        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in channels.items()}


# =================================================================================================
__all__ = ["MazeMetadata", "MazeDataset"]
