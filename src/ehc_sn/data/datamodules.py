"""Lightning :class:`~lightning.LightningDataModule` for processed datasets.

Public surface: :class:`Datamodule`, :class:`DatamoduleConfig`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import lightning as L
import numpy as np
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from ehc_sn.data.datasets import ProcessedDataset
from ehc_sn.data.index import filter_index, read_index
from ehc_sn.data.transforms import Compose, RandomDihedral


# =================================================================================================
class DatamoduleConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`Datamodule` (LightningDataModule)."""

    dataset_path: Path = Field(
        ...,
        description="Path to the processed dataset root (contains index.jsonl and per-split channel arrays).",
    )
    global_batch_size: int = Field(
        default=8,
        description="Global batch size across all devices.",  # TODO: Describe how per-device batch size is computed
    )
    num_workers: int = Field(
        default=4,
        description="Number of workers for DataLoader.",
    )
    prefetch_factor: int = Field(
        default=2,
        description="Number of batches to prefetch per worker.",
    )
    pin_memory: bool = Field(
        default=True,
        description="Whether to pin memory in DataLoader.",
    )
    persistent_workers: bool = Field(
        default=True,
        description="Whether to keep DataLoader workers alive between epochs.",
    )
    augment: bool = Field(
        default=True,
        description="Whether to apply augmentation to training data.",
    )
    seed: int = Field(
        default=42,
        description="RNG seed for training-split dihedral augmentation and reproducibility.",
    )


# =================================================================================================
class Datamodule(L.LightningDataModule):
    """Lightning DataModule for processed datasets.

    Loads a processed dataset directory containing an ``index.jsonl`` and
    per-split channel arrays. Optionally applies training-time augmentation.
    """

    def __init__(self, config: DatamoduleConfig, transform: Callable | None = None,) -> None:  # fmt: skip  # ------------------------------------------------------------------------------
        """Create the data module.

        Args:
            config: Datamodule configuration.
            transform: Optional adapter transform applied after built-in transforms.
        """
        super().__init__()
        self._config = config
        self._adapter = transform
        self._train: ProcessedDataset | None = None
        self._val: ProcessedDataset | None = None
        self._test: ProcessedDataset | None = None

    @property
    def config(self) -> DatamoduleConfig:
        """Return the parsed datamodule configuration."""
        return self._config

    @property
    def train_transform(self) -> list[Callable]:
        """Transforms applied to training samples."""
        return [
            RandomDihedral(rng=np.random.default_rng(self.config.seed)) if self.config.augment else None,
            self._adapter,
        ]

    @property
    def eval_transform(self) -> list[Callable]:
        """Transforms applied to validation/test samples."""
        return [
            self._adapter,
        ]

    def setup(self, stage: str,) -> None:  # fmt: skip  # ---------------------------------------------------------------------------------
        """Load datasets for the given stage(s)."""
        data_root = self.config.dataset_path
        all_entries = read_index(data_root / "index.jsonl")

        # Compose transforms here to avoid re-instantiating them on every worker in the DataLoader.
        train_transform = Compose(self.train_transform)
        eval_transform = Compose(self.eval_transform)

        # Load the datasets for the requested stage(s).
        # We rely on the DataLoader workers to apply the transforms
        if stage in ("fit", "validate"):
            entries = filter_index(all_entries, split="train")
            self._train = ProcessedDataset(entries, data_root / "train", transform=train_transform)
            entries = filter_index(all_entries, split="val")
            self._val = ProcessedDataset(entries, data_root / "val", transform=eval_transform)
        if stage == "test":
            entries = filter_index(all_entries, split="test")
            self._test = ProcessedDataset(entries, data_root / "test", transform=eval_transform)

    def _per_gpu_batch_size(self,) -> int:  # fmt: skip
        """Compute per-device batch size from the global value."""
        world_size = max(self.trainer.world_size if self.trainer is not None else 1, 1)
        return max(self.config.global_batch_size // world_size, 1)

    def _make_loader(self, dataset: ProcessedDataset, *, shuffle: bool,) -> DataLoader:  # fmt: skip  # --------------------------------------------------------------------------
        """Construct a DataLoader for the given dataset and settings."""
        return DataLoader(
            dataset,
            batch_size=self._per_gpu_batch_size(),
            shuffle=shuffle,  # Shuffle to ensure all data is seen during training (halt buffer)
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers and self.config.num_workers > 0,
            drop_last=True,  # Drop last batch to ensure consistent batch size
        )

    def train_dataloader(self,) -> DataLoader:  # fmt: skip  # ----------------------------------------------------------------------
        """Return the training DataLoader."""
        if self._train is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")
        return self._make_loader(self._train, shuffle=True)

    def val_dataloader(self,) -> DataLoader:  # fmt: skip  # ------------------------------------------------------------------------
        """Return the validation DataLoader."""
        if self._val is None:
            raise RuntimeError("Call setup('fit') or setup('validate') before val_dataloader()")
        return self._make_loader(self._val, shuffle=False)

    def test_dataloader(self,) -> DataLoader:  # fmt: skip  # -----------------------------------------------------------------------
        """Return the test DataLoader."""
        if self._test is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")
        return self._make_loader(self._test, shuffle=False)

    def val_sample_ids_for_batch(
        self,
        batch_idx: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> list[str]:
        """Return ordered sample IDs for a validation batch on the given process.

        Simulates ``DistributedSampler(shuffle=False, drop_last=False)`` plus
        sequential ``DataLoader(drop_last=True)`` batching — the same
        distribution that PyTorch Lightning applies to the val dataloader in
        DDP mode.  For ``world_size == 1`` this reduces to simple sequential
        batching.

        Args:
            batch_idx: Index of the batch within this process's local val stream.
            batch_size: Number of samples per batch on this process.
            rank: This process's global rank (0-based). Default ``0``.
            world_size: Total number of processes. Default ``1``.

        Returns:
            Ordered list of :attr:`~ehc_sn.data.index.DatasetIndexEntry.id`
            values for the requested batch.  Returns an empty list when
            the datamodule has not been set up or the batch falls outside the
            available samples (i.e. would have been dropped by ``drop_last``).
        """
        if self._val is None:
            return []
        entries = self._val._entries
        n = len(entries)
        if world_size <= 1:
            start = batch_idx * batch_size
            end = start + batch_size
            return [entries[i].id for i in range(start, min(end, n))]
        # Simulate DistributedSampler(shuffle=False, drop_last=False):
        # pad index list from the beginning so total is divisible by world_size.
        total_size = math.ceil(n / world_size) * world_size
        padding = total_size - n
        all_indices = list(range(n)) + list(range(padding))
        local_indices = all_indices[rank:total_size:world_size]
        start = batch_idx * batch_size
        end = start + batch_size
        if end > len(local_indices):
            return []  # incomplete batch; would be dropped by drop_last=True
        return [entries[i].id for i in local_indices[start:end]]


# =================================================================================================
__all__ = ["Datamodule", "DatamoduleConfig"]
