"""PyTorch :class:`Dataset` wrappers for versioned processed split roots.

Public surface: :class:`DatasetMetadata`, :class:`ProcessedDataset`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.utils.data import Dataset

from ehc_sn.data.index import DatasetIndexEntry


# =============================================================================
class DatasetMetadata(BaseModel, extra="allow"):
    """Dataset-level metadata for one resolved processed split root.

    This is typically read from a split-local metadata file and is less strict
    than :class:`~ehc_sn.data.index.DatasetIndexEntry` (``extra=allow``).
    """

    source: str = Field(
        ...,
        description="Owning source or corpus name for the resolved split root",
    )
    split: str = Field(
        ...,
        description="Canonical split name for the resolved root (for example train/val/test)",
    )
    n_samples: int = Field(
        ...,
        ge=0,
        description="Number of samples materialized in the resolved split root",
    )
    channels: list[str] = Field(
        ...,
        description="Canonical processed channel names present in every sample",
    )


# =============================================================================
class ProcessedDataset(Dataset):

    def __init__(  # ----------------------------------------------------------
        self,
        entries: list[DatasetIndexEntry],
        data_dir: Path,
        transform: Callable | None = None,
    ) -> None:
        """Dataset backed by memory-mapped arrays from one resolved split root."""
        if not entries:
            raise ValueError(
                "ProcessedDataset requires at least one index entry."
            )

        splits = {entry.split for entry in entries}
        if len(splits) != 1:
            raise ValueError(
                "ProcessedDataset requires entries from exactly one split; "
                "resolve a single split before loading arrays."
            )

        split = next(iter(splits))
        if (data_dir / split).is_dir():
            raise ValueError(
                f"ProcessedDataset expects a resolved split directory, got dataset root '{data_dir}'. "
                f"Use '{data_dir / split}' instead."
            )

        self._entries = entries
        self._transform = transform
        channels = entries[0].channels
        self._arrays = {
            ch: np.load(data_dir / f"{ch}.npy", mmap_mode="r")
            for ch in channels
        }
        # Content-derived digest from the parsed index entries.
        # Computed once at construction; depends only on entry contents.
        self._dataset_digest: str = self._compute_digest(entries)

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def dataset_digest(self) -> str:
        """SHA-256 hex digest of the normalized index entries.

        Computed once at construction from the parsed entries.  Two
        ProcessedDataset instances with identical index contents always
        return the same digest regardless of path or construction time.
        """
        return self._dataset_digest

    @property
    def dataset_size(self) -> int:
        """Number of entries (identical to ``len(self)``)."""
        return len(self._entries)

    @staticmethod
    def _compute_digest(entries: list[DatasetIndexEntry]) -> str:
        """Return a SHA-256 hex digest of normalized index entries.

        Uses sorted ``model_dump()`` + ``json.dumps(sort_keys=True)`` for
        each entry so the hash is deterministic and order-independent
        within each entry.
        """
        hasher = hashlib.sha256()
        for entry in sorted(entries, key=lambda e: e.id):
            hasher.update(
                json.dumps(entry.model_dump(), sort_keys=True).encode()
            )
        return hasher.hexdigest()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        dirs = sorted(self._arrays)
        return (
            f"{type(self).__name__}("
            f"n={len(self)}, "
            f"dirs={dirs}, "
            f"transform={self._transform!r})"
        )

    def __getitem__(  # -------------------------------------------------------
        self,
        idx: int,
    ) -> dict[str, Tensor]:
        """Return a sample by index, applying the optional transform."""
        sample = {
            k: np.array(v[idx], copy=True) for k, v in self._arrays.items()
        }
        if self._transform:
            sample = self._transform(sample)
        result = {
            k: v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
            for k, v in sample.items()
        }
        # Stable fit-path identity derived from dataset index position.
        # Non-model-visible: consumed only by the replay controller at admission
        # to populate carry.trajectory_id; never passed to the model or objectives.
        result["__trajectory_id__"] = torch.tensor(idx, dtype=torch.int64)
        return result


# =============================================================================
__all__ = ["DatasetMetadata", "ProcessedDataset"]
