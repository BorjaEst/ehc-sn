"""PyTorch :class:`Dataset` wrappers for versioned processed split roots.

Public surface: :class:`DatasetMetadata`, :class:`ProcessedDataset`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.utils.data import Dataset

from ehc_sn.data.index import DatasetIndexEntry


# =================================================================================================
class DatasetMetadata(BaseModel, extra="allow"):
    """Dataset-level metadata for one resolved processed split root.

    This is typically read from a split-local metadata file and is less strict
    than :class:`~ehc_sn.data.index.DatasetIndexEntry` (``extra=allow``).
    """

    source: str = Field(..., description="Owning source or corpus name for the resolved split root")
    split: str = Field(..., description="Canonical split name for the resolved root (for example train/val/test)")
    n_samples: int = Field(..., ge=0, description="Number of samples materialized in the resolved split root")
    channels: list[str] = Field(..., description="Canonical processed channel names present in every sample")


# =================================================================================================
class ProcessedDataset(Dataset):

    def __init__(self, entries: list[DatasetIndexEntry], data_dir: Path, transform: (
        Callable | None
    ) = None,) -> None:  # fmt: skip  # ------------------------------------------------------------------------------
        """Dataset backed by memory-mapped arrays from one resolved split root."""
        if not entries:
            raise ValueError("ProcessedDataset requires at least one index entry.")

        splits = {entry.split for entry in entries}
        if len(splits) != 1:
            raise ValueError("ProcessedDataset requires entries from exactly one split; resolve a single split before loading arrays.")

        split = next(iter(splits))
        if (data_dir / split).is_dir():
            raise ValueError(
                f"ProcessedDataset expects a resolved split directory, got dataset root '{data_dir}'. " f"Use '{data_dir / split}' instead."
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

    def __getitem__(self, idx: int,) -> dict[str, Tensor]:  # fmt: skip  # ---------------------------------------------------------------------------
        sample = {k: v[idx] for k, v in self._arrays.items()}
        if self._transform:
            sample = self._transform(sample)
        return {k: torch.from_numpy(np.array(v)) for k, v in sample.items()}


# =================================================================================================
__all__ = ["DatasetMetadata", "ProcessedDataset"]
