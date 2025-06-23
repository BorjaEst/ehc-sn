import collections.abc
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import lightning.pytorch as pl
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import DataLoader


class GeneratorParams(BaseModel, ABC):
    """Parameters abstract class for data generation parameters."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class Generator(ABC, collections.abc.Iterator):
    """Generator base class for generatig data."""

    def __init__(self, params: Optional[GeneratorParams] = None):
        params = params or GeneratorParams()

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)

    @abstractmethod
    def __next__(self) -> Tuple[Tensor, ...]:
        """Generate next data sample."""
        pass


class Dataset(torch.utils.data.IterableDataset):
    """Dataset for grid maps with obstacles and goal positions."""

    def __init__(self, num_samples: int, generator: Generator):
        """
        Initialize a dataset of grid maps with obstacles and goals.

        Args:
            num_samples: Number of grid map samples to generate per iteration
            generator: Generator class to use for creating grid maps
        """
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self) -> collections.abc.Iterator[Tuple[Tensor, ...]]:
        """Yield grid map samples on-the-fly."""
        for _ in range(self.num_samples):
            yield self.generator.__next__()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples


class DataModuleParams(BaseModel):
    """Parameters for the GridMapDataModule."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # Dataset generation parameters
    num_samples: int = Field(default=1000, description="Total number of samples to generate")
    val_split: float = Field(default=0.2, description="Fraction to use for validation (0.0-1.0)")
    test_split: float = Field(default=0.1, description="Fraction to use for testing (0.0-1.0)")
    # Dataloader parameters for validation and test datasets
    batch_size: int = Field(default=32, description="Batch size for dataloaders")
    num_workers: int = Field(default=0, description="Number of workers for dataloaders")
    pin_memory: bool = Field(default=True, description="Whether to keep dataset in memory for faster access")
    persistent_workers: bool = Field(default=False, description="Whether to use persistent workers for dataloaders")

    @field_validator("val_split", "test_split")
    @classmethod
    def validate_split(cls, v: float) -> float:
        """Ensure split values are between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Split values must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_splits(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure that the sum of validation and test splits does not exceed 1.0."""
        if values.val_split + values.test_split > 1.0:
            raise ValueError("Sum of validation and test splits cannot exceed 1.0")
        return values

    @property
    def train_split(self) -> float:
        """Calculate the training split as the remaining fraction."""
        return 1.0 - self.val_split - self.test_split

    def gen_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create a DataLoader with the specified parameters."""
        kwargs = self.model_dump(exclude={"val_split", "test_split", "num_samples"})
        return DataLoader(dataset, **kwargs)


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for grid maps with obstacles."""

    def __init__(self, generator: Generator, params: Optional[DataModuleParams] = None):
        """
        Initialize the data module.

        Args:
            num_samples: Total number of samples to generate
            params: Parameters for grid map generation
        """
        super().__init__()
        self.params = params or DataModuleParams()
        self.generator = generator

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset for the given stage."""

        if stage in (None, "fit", "validate"):
            num_samples = int(self.params.num_samples * self.params.train_split)
            self.train_dataset = Dataset(num_samples, self.generator)
        if stage in (None, "validate"):
            num_samples = int(self.params.num_samples * self.params.val_split)
            self.val_dataset = Dataset(num_samples, self.generator)
        if stage in (None, "test"):
            num_samples = int(self.params.num_samples * self.params.test_split)
            self.test_dataset = Dataset(num_samples, self.generator)

    def train_dataloader(self):
        """Return the training dataloader."""
        return self.params.gen_dataloader(self.train_dataset)

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self.params.gen_dataloader(self.val_dataset)

    def test_dataloader(self):
        """Return the test dataloader."""
        return self.params.gen_dataloader(self.test_dataset)


if __name__ == "__main__":
    # Example usage
    class MyGenerator(Generator):
        def __next__(self) -> Tuple[Tensor, ...]:
            return (torch.randint(0, 2, (16, 16)), torch.randint(0, 10, [1]))

    # Create the parameters for the data loader
    datamodule_params = DataModuleParams(num_samples=100, batch_size=16, val_split=0.2, test_split=0.1)
    data_module = DataModule(MyGenerator(), datamodule_params)
    data_module.setup()

    for batch in data_module.train_dataloader():
        print(batch)  # Process the batch as needed
