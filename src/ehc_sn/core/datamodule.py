import collections.abc
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import lightning.pytorch as pl
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


# -------------------------------------------------------------------------------------------
class DataModuleParams(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Dataset Settings
    num_samples: int = Field(default=4000, ge=100, le=50000, description="Total number of samples")
    val_split: float = Field(default=0.1, ge=0.05, le=0.3, description="Validation split fraction")
    test_split: float = Field(default=0.1, ge=0.05, le=0.3, description="Test split fraction")
    n_predict: int = Field(default=10, ge=0, le=10000, description="Number of prediction samples")

    # DataLoader Settings
    batch_size: int = Field(default=16, ge=1, le=512, description="Training batch size")
    num_workers: int = Field(default=4, ge=0, le=16, description="Number of data loading workers")
    pin_memory: bool = Field(default=True, description="Pin memory for faster GPU transfer")
    drop_last: bool = Field(default=True, description="Drop last incomplete batch")

    @property
    def n_train(self) -> int:
        """Number of training samples derived from total samples and validation/test splits."""
        return self.num_samples - self.n_val - self.n_test

    @property
    def n_val(self) -> int:
        """Number of validation samples derived from total samples and validation split."""
        return int(self.val_split * self.num_samples)

    @property
    def n_test(self) -> int:
        """Number of test samples derived from total samples and test split."""
        return int(self.test_split * self.num_samples)


# -------------------------------------------------------------------------------------------
class GeneratedDataset(Dataset):
    """
    Dataset that generates data on-demand using a generator.

    This dataset creates samples dynamically using the provided generator,
    ensuring memory efficiency for large datasets.
    """

    def __init__(self, generator, num_samples: int, seed: Optional[int] = None):
        """
        Initialize dataset with generator.

        Args:
            generator: Data generator instance with generate() method
            num_samples: Number of samples in this dataset
            seed: Random seed for reproducible generation
        """
        self.generator = generator
        self.num_samples = num_samples
        self.seed = seed

        # Set seed if provided for reproducible generation
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Any:
        """
        Generate and return a single sample.

        Args:
            idx: Sample index

        Returns:
            Generated sample from the generator
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")

        # Generate sample using the generator
        # For reproducible results, set seed based on idx
        if self.seed is not None:
            sample_seed = self.seed + idx
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            random.seed(sample_seed)

        return self.generator.generate()


# -------------------------------------------------------------------------------------------
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, generator, params: Optional[DataModuleParams] = None):
        super().__init__()
        self.params = DataModuleParams() if params is None else params
        self.datasets = {"fit": None, "validate": None, "test": None, "predict": None}
        self.generator = generator

    def prepare_data(self):
        if hasattr(self.generator, "download"):
            self.generator.download()

    def gen_dataset(self, n_samples: int) -> Tuple[Dataset, Dict[str, Any]]:
        # create dataset
        dataset = GeneratedDataset(self.generator, n_samples)

        # count number of classes
        # build vocabulary
        # apply transforms (defined explicitly in your datamodule)
        pass  # TODO

        return dataset, {}

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.datasets["fit"] = self.gen_dataset(self.params.n_train)
        if stage in (None, "fit", "validate"):
            self.datasets["validate"] = self.gen_dataset(self.params.n_val)
        if stage in (None, "test"):
            self.datasets["test"] = self.gen_dataset(self.params.n_test)
        if stage in (None, "predict"):
            self.datasets["predict"] = self.gen_dataset(self.params.n_predict)

    def gen_dataloader(self, stage: str, **config: Any) -> DataLoader:
        return DataLoader(
            dataset=self.datasets[stage][0],
            batch_size=self.params.batch_size,
            shuffle=config.get("shuffle", False),
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            drop_last=config.get("drop_last", False),
        )

    def train_dataloader(self):
        return self.gen_dataloader("fit", shuffle=True, drop_last=self.params.drop_last)

    def val_dataloader(self):
        return self.gen_dataloader("validate", shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self.gen_dataloader("test", shuffle=False, drop_last=False)

    def predict_dataloader(self):
        return self.gen_dataloader("predict", shuffle=False, drop_last=False)

    def teardown(self, stage: str):
        pass  # No teardown needed on base datamodule


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage for testing
    class DummyGenerator:
        """Simple generator for testing purposes."""

        def generate(self):
            return torch.randn(3, 32, 32)  # Example tensor

    # Create generator and data module
    generator = DummyGenerator()
    params = DataModuleParams(num_samples=1000, batch_size=32)
    dm = BaseDataModule(generator, params)

    # Test setup and dataloader creation
    dm.setup("fit"), dm.setup("validate")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    print(f"Training batches: {len(train_dl)}")
    print(f"Validation batches: {len(val_dl)}")
    print(f"Training samples: {len(dm.datasets['fit'][0])}")
    print(f"Validation samples: {len(dm.datasets['validate'][0])}")
