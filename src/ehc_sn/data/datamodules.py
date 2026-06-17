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
from torch.utils.data import DataLoader, IterableDataset

from ehc_sn.data.datasets import ProcessedDataset
from ehc_sn.data.index import filter_index, read_index
from ehc_sn.data.transforms import Compose, RandomDihedral
from ehc_sn.rollouts.sources import DemandDrivenReplaySource


# =============================================================================
class DemandDrivenTickIterable(IterableDataset):
    """Yields real episode batches from a ``DemandDrivenReplaySource``.

    Replaces the empty-sentinel ``_InfiniteTickIterable``.  The source
    is created by the experiment builder and shared with the module so
    that both the DataLoader and the module access the same carry state.

    ``num_workers`` MUST be 0 — multi-worker IterableDataset would replicate
    the source iterator and produce duplicate episodes.

    Termination is exclusively via ``Trainer.max_steps``.
    """

    def __init__(
        self, source_provider: Callable[[], DemandDrivenReplaySource]
    ) -> None:
        self._source_provider = source_provider
        self._source: DemandDrivenReplaySource | None = None

    def __iter__(self) -> "DemandDrivenTickIterable":
        return self

    def __next__(self) -> dict:
        if self._source is None:
            self._source = self._source_provider()
            if self._source is None:
                raise RuntimeError(
                    "DemandDrivenTickIterable: source provider returned None. "
                    "The LightningModule may not have called setup('fit') yet."
                )
        return next(self._source)


# =============================================================================
class DatamoduleConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`Datamodule` (LightningDataModule)."""

    dataset_path: Path = Field(
        ...,
        description="Path to the processed dataset root (contains index.jsonl and per-split channel arrays).",
    )
    num_slots: int = Field(
        default=8,
        ge=1,
        description="Pass-through for TOML parsing. Consumed by the experiment "
        "builder, not by the Datamodule itself.",
    )
    eval_batch_size: int = Field(
        default=8,
        ge=1,
        description="Per-rank batch size for validation and test DataLoaders. "
        "Does not affect training.",
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


# =============================================================================
class Datamodule(L.LightningDataModule):
    """Lightning DataModule for processed datasets.

    Loads a processed dataset directory containing an ``index.jsonl`` and
    per-split channel arrays. Optionally applies training-time augmentation.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: DatamoduleConfig,
        transform: Callable | None = None,
    ) -> None:
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
        self._source_provider: Callable[[], DemandDrivenReplaySource] | None = (
            None
        )

    def attach_source_provider(  # --------------------------------------------
        self,
        provider: Callable[[], DemandDrivenReplaySource],
    ) -> None:
        """Register a callable that provides the demand-driven replay source.

        The provider is called lazily when the iterable's first ``__next__``
        is invoked, guaranteeing the module's ``setup('fit')`` has already run
        and created the source.  The callable should capture a reference to
        ``module._train_source``.
        """
        self._source_provider = provider

    @property
    def config(self) -> DatamoduleConfig:
        """Return the parsed datamodule configuration."""
        return self._config

    @property
    def train_transform(self) -> list[Callable]:
        """Transforms applied to training samples."""
        return [
            (
                RandomDihedral(rng=np.random.default_rng(self.config.seed))
                if self.config.augment
                else None
            ),
            self._adapter,
        ]

    @property
    def eval_transform(self) -> list[Callable]:
        """Transforms applied to validation/test samples."""
        return [
            self._adapter,
        ]

    def setup(  # -------------------------------------------------------------
        self,
        stage: str,
    ) -> None:
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
            self._train = ProcessedDataset(
                entries, data_root / "train", transform=train_transform
            )
            entries = filter_index(all_entries, split="val")
            self._val = ProcessedDataset(
                entries, data_root / "val", transform=eval_transform
            )
        if stage == "test":
            entries = filter_index(all_entries, split="test")
            self._test = ProcessedDataset(
                entries, data_root / "test", transform=eval_transform
            )

    def _per_gpu_batch_size(  # -----------------------------------------------
        self,
    ) -> int:
        """Compute per-device batch size from the global value."""
        world_size = max(
            self.trainer.world_size if self.trainer is not None else 1, 1
        )
        return max(self.config.eval_batch_size // world_size, 1)

    def _make_loader(  # ------------------------------------------------------
        self,
        dataset: ProcessedDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        """Construct a DataLoader for the given dataset and settings."""
        return DataLoader(
            dataset,
            batch_size=self._per_gpu_batch_size(),
            shuffle=shuffle,  # Shuffle to ensure all data is seen during training (halt buffer)
            num_workers=self.config.num_workers,
            prefetch_factor=(
                self.config.prefetch_factor
                if self.config.num_workers > 0
                else None
            ),
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
            and self.config.num_workers > 0,
            drop_last=drop_last,  # Drop last batch only for training
        )

    def train_dataloader(  # --------------------------------------------------
        self,
    ) -> DataLoader:
        """Return a DataLoader wrapping the demand-driven replay source.

        Yields real episode batches from ``DemandDrivenReplaySource``.
        The source provider (a callable returning the module's
        ``_train_source``) must be registered via
        :meth:`attach_source_provider` before this method is called.
        """
        if self._train is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")
        if self._source_provider is None:
            raise RuntimeError(
                "train_dataloader() requires a source provider. "
                "Call datamodule.attach_source_provider(...) before training."
            )
        return DataLoader(
            DemandDrivenTickIterable(self._source_provider),
            batch_size=None,
            num_workers=0,
        )

    def val_dataloader(  # ----------------------------------------------------
        self,
    ) -> DataLoader:
        """Return the validation DataLoader."""
        if self._val is None:
            raise RuntimeError(
                "Call setup('fit') or setup('validate') before val_dataloader()"
            )
        return self._make_loader(self._val, shuffle=False, drop_last=False)

    def test_dataloader(  # ---------------------------------------------------
        self,
    ) -> DataLoader:
        """Return the test DataLoader."""
        if self._test is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")
        return self._make_loader(self._test, shuffle=False, drop_last=False)

    def val_sample_ids_for_batch(  # ------------------------------------------
        self,
        batch_idx: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> list[str]:
        """Return ordered sample IDs for a validation batch on the given process.

        Simulates ``DistributedSampler(shuffle=False, drop_last=False)`` plus
        sequential ``DataLoader(drop_last=False)`` batching — the same
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
            the datamodule has not been set up. Partial final batches are
            returned when present.
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
        sliced = local_indices[start : min(end, len(local_indices))]
        return [entries[i].id for i in sliced]


# =============================================================================
__all__ = ["Datamodule", "DatamoduleConfig"]
