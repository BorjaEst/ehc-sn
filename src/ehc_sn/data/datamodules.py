from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import lightning as L
import numpy as np
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from ehc_sn.data.datasets import MazeDataset
from ehc_sn.data.index import filter_index, read_index
from ehc_sn.data.transforms import Compose, RandomDihedral


# =================================================================================================
class DatamoduleConfig(BaseModel, extra="forbid"):

    dataset_path: Path = Field(
        ...,
        description="Path to the processed dataset directory (contains index.jsonl + NPZ files).",
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

    def __init__(  # ------------------------------------------------------------------------------
        self, config: DatamoduleConfig, transform: Callable | None = None,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config
        self._adapter = transform
        self._train: MazeDataset | None = None
        self._val: MazeDataset | None = None
        self._test: MazeDataset | None = None

    @property
    def config(self) -> DatamoduleConfig:
        """"""
        return self._config

    @property
    def train_transform(self) -> list[Callable]:
        """ """
        return [
            RandomDihedral(rng=np.random.default_rng(self.config.seed)) if self.config.augment else None,
            self._adapter,
        ]

    @property
    def eval_transform(self) -> list[Callable]:
        """ """
        return [
            self._adapter,
        ]

    def setup(  # ---------------------------------------------------------------------------------
        self, stage: str,
    ) -> None:  # fmt: skip
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
            self._train = MazeDataset(entries, data_root / "train", transform=train_transform)
            entries = filter_index(all_entries, split="val")
            self._val = MazeDataset(entries, data_root / "val", transform=eval_transform)
        if stage == "test":
            entries = filter_index(all_entries, split="test")
            self._test = MazeDataset(entries, data_root / "test", transform=eval_transform)

    def _per_gpu_batch_size(
        self,
    ) -> int:  # fmt: skip
        """Compute per-device batch size from the global value."""
        world_size = max(self.trainer.world_size if self.trainer is not None else 1, 1)
        return max(self.config.global_batch_size // world_size, 1)

    def _make_loader(  # --------------------------------------------------------------------------
        self, dataset: MazeDataset, *, shuffle: bool,
    ) -> DataLoader:  # fmt: skip
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

    def train_dataloader(  # ----------------------------------------------------------------------
        self,
    ) -> DataLoader:  # fmt: skip
        """ """
        if self._train is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")
        return self._make_loader(self._train, shuffle=True)

    def val_dataloader(  # ------------------------------------------------------------------------
        self,
    ) -> DataLoader:  # fmt: skip
        """ """
        if self._val is None:
            raise RuntimeError("Call setup('fit') or setup('validate') before val_dataloader()")
        return self._make_loader(self._val, shuffle=False)

    def test_dataloader(  # -----------------------------------------------------------------------
        self,
    ) -> DataLoader:  # fmt: skip
        """ """
        if self._test is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")
        return self._make_loader(self._test, shuffle=False)


# =================================================================================================
__all__ = ["Datamodule", "DatamoduleConfig"]
