import collections.abc
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

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
    num_samples: int = Field(default=400, ge=100, le=50000, description="Total number of samples")
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
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, generator: Callable[[int], Dataset], params: Optional[DataModuleParams] = None):
        super().__init__()
        self.params = DataModuleParams() if params is None else params
        self.datasets = {"fit": None, "validate": None, "test": None, "predict": None}
        self.generator = generator

    def prepare_data(self):
        if hasattr(self.generator, "download"):
            self.generator.download()

    def gen_dataset(self, n_samples: int) -> Tuple[Dataset, Dict[str, Any]]:
        # create dataset
        dataset = self.generator(n_samples)

        # apply transforms (defined explicitly in your datamodule)
        metadata = {}  # Can be extended for future use (class counts, vocabulary, etc.)

        return dataset, metadata

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

    # Create generator and data module
    generator = lambda n: TensorDataset(torch.randn(n, 10), torch.randn(n, 10))
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
