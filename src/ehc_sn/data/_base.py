import collections.abc
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import lightning.pytorch as pl
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import DataLoader


# -------------------------------------------------------------------------------------------
class GeneratorParams(BaseModel):
    """Parameters abstract class for data generation parameters."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    # -----------------------------------------------------------------------------------
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )


# -------------------------------------------------------------------------------------------
class Generator(ABC, collections.abc.Iterator):
    """Generator base class for generatig data."""

    # -----------------------------------------------------------------------------------
    def __init__(self, params: Optional[GeneratorParams] = None):
        params = params or GeneratorParams()

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)

    # -----------------------------------------------------------------------------------
    @abstractmethod
    def __next__(self) -> Tuple[Tensor, ...]:
        """Generate next data sample."""
        pass


# -------------------------------------------------------------------------------------------
class Dataset(torch.utils.data.IterableDataset):
    """Dataset for data with obstacles and goal positions."""

    # -----------------------------------------------------------------------------------
    def __init__(self, generator: Generator):
        """
        Initialize a dataset of data with obstacles and goals.

        Args:
            generator: Generator class to use for creating data
        """
        self.generator = generator

    # -----------------------------------------------------------------------------------
    def __iter__(self) -> collections.abc.Iterator[Tuple[Tensor, ...]]:
        """Yield individual data samples on-the-fly."""
        while True:
            yield self.generator.__next__()


# -------------------------------------------------------------------------------------------
class AugmenterParams(BaseModel):
    """Parameters for data augmentation of data."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # -----------------------------------------------------------------------------------
    augmentation_probability: float = Field(
        default=0.5,
        description="Probability of applying augmentation to a sample",
    )

    # -----------------------------------------------------------------------------------
    @field_validator("augmentation_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Ensure probability values are between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Probability values must be between 0.0 and 1.0")
        return v


# -------------------------------------------------------------------------------------------
class Augmenter(ABC):
    """Base class for data augmentation."""

    # -----------------------------------------------------------------------------------
    def __init__(self, params: Optional[AugmenterParams] = None):
        """Initialize the augmenter with parameters."""
        self.params = params or AugmenterParams()

    # -----------------------------------------------------------------------------------
    @abstractmethod
    def augment(self, data: Tensor) -> Tensor:
        """
        Apply augmentations to the data based on configured parameters.

        Args:
            data: Input data tensor

        Returns:
            Augmented data tensor
        """
        pass

    # -----------------------------------------------------------------------------------
    def __call__(self, data: Tensor) -> Tensor:
        """Apply augmentation if probability check passes."""
        if random.random() < self.params.augmentation_probability:
            return self.augment(data)
        return data


# -------------------------------------------------------------------------------------------
class AugmentedDataset(torch.utils.data.IterableDataset):
    """Dataset wrapper that applies augmentation to samples."""

    # -----------------------------------------------------------------------------------
    def __init__(self, dataset: Dataset, augmenters: List[Augmenter]):
        """
        Initialize the augmented dataset.

        Args:
            dataset: Base dataset to augment
            augmenters: List of augmenters to apply to samples
        """
        self.dataset = dataset
        self.augmenters = augmenters

    # -----------------------------------------------------------------------------------
    def __iter__(self) -> collections.abc.Iterator[Tuple[Tensor, ...]]:
        """Yield augmented samples."""
        for batch in self.dataset:

            # Assuming samples are in the first element of the batch
            augmenting_samples, *other_data = batch

            # Cycle through augmenters and apply them to the samples
            for augmenter in self.augmenters:
                augmenting_samples = augmenter(augmenting_samples)

            # Merge augmented samples with other data and yield
            yield (augmenting_samples, *other_data)


# -------------------------------------------------------------------------------------------
class DataGenParams(BaseModel):

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # -----------------------------------------------------------------------------------
    num_samples: int = Field(
        default=1000,
        description="Total number of samples to generate",
    )
    # -----------------------------------------------------------------------------------
    val_split: float = Field(
        default=0.2,
        description="Fraction to use for validation (0.0-1.0)",
    )
    # -----------------------------------------------------------------------------------
    test_split: float = Field(
        default=0.1,
        description="Fraction to use for testing (0.0-1.0)",
    )

    # -----------------------------------------------------------------------------------
    @field_validator("val_split", "test_split")
    @classmethod
    def validate_split(cls, v: float) -> float:
        """Ensure split values are between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Split values must be between 0.0 and 1.0")
        return v

    # -----------------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_splits(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure that the sum of validation and test splits does not exceed 1.0."""
        if values.val_split + values.test_split > 1.0:
            raise ValueError("Sum of validation and test splits cannot exceed 1.0")
        return values

    # -----------------------------------------------------------------------------------
    @property
    def train_split(self) -> float:
        """Calculate the training split as the remaining fraction."""
        return 1.0 - self.val_split - self.test_split


# -------------------------------------------------------------------------------------------
class DataLoaderParams(BaseModel):
    """Parameters for the DataLoader."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # -----------------------------------------------------------------------------------
    batch_size: int = Field(
        default=32,
        description="Batch size for dataloaders",
    )
    # -----------------------------------------------------------------------------------
    num_workers: int = Field(
        default=0,
        description="Number of workers for dataloaders",
    )
    # -----------------------------------------------------------------------------------
    pin_memory: bool = Field(
        default=True,
        description="Whether to keep dataset in memory for faster access",
    )
    # -----------------------------------------------------------------------------------
    persistent_workers: bool = Field(
        default=False,
        description="Whether to use persistent workers for dataloaders",
    )

    # -----------------------------------------------------------------------------------
    def gen_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create a DataLoader with the specified parameters."""
        keys = DataLoaderParams.model_fields.keys()
        kwargs = self.model_dump(include=keys, exclude_none=True)
        return DataLoader(dataset, **kwargs)


# -------------------------------------------------------------------------------------------
class DataModuleParams(DataGenParams, DataLoaderParams, AugmenterParams):
    """Parameters for the DataModule.

    Combines data generation, dataloader, and augmentation parameters.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # -----------------------------------------------------------------------------------
    augmenters: Optional[List[Augmenter]] = Field(
        default=None,
        description="List of augmenters to apply to samples",
    )

    # -----------------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_params(self) -> "DataModuleParams":
        """Validate that the total number of samples is sufficient."""
        if self.num_samples <= 0:
            raise ValueError("Number of samples must be greater than 0")
        return self


# -------------------------------------------------------------------------------------------
class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for simulation and training data."""

    # -----------------------------------------------------------------------------------
    def __init__(self, generator: Generator, params: Optional[DataModuleParams] = None):
        """
        Initialize the data module.

        Args:
            generator: Generator for data
            params: Parameters for data module configuration
        """
        super().__init__()
        self.params = params or DataModuleParams()
        self.generator = generator

    # -----------------------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        """Setup the dataset for the given stage."""
        # We'll continue to create datasets on-the-fly, no additional setup needed
        pass

    # -----------------------------------------------------------------------------------
    def _create_dataset(self, augment: bool = False) -> Dataset:
        """Create a dataset with optional augmentation."""
        dataset = Dataset(self.generator)

        if augment and self.params.augmenters:
            # Wrap the dataset to apply augmentations
            return AugmentedDataset(dataset, self.params.augmenters)
        return dataset

    # -----------------------------------------------------------------------------------
    def train_dataloader(self):
        """Return the training dataloader with augmentation."""
        dataset = self._create_dataset(augment=True)
        return self.params.gen_dataloader(dataset)

    # -----------------------------------------------------------------------------------
    def val_dataloader(self):
        """Return the validation dataloader without augmentation."""
        dataset = self._create_dataset(augment=False)
        return self.params.gen_dataloader(dataset)

    # -----------------------------------------------------------------------------------
    def test_dataloader(self):
        """Return the test dataloader without augmentation."""
        dataset = self._create_dataset(augment=False)
        return self.params.gen_dataloader(dataset)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This example demonstrates how to create a basic data generation pipeline using the base classes.
    It shows:
    1. Creating a simple generator that produces random tensors
    2. Defining a basic augmenter that adds noise to data
    3. Setting up a DataModule with the generator and augmenter
    4. Using the DataModule to create dataloaders
    """
    import random
    from typing import List, Optional, Tuple

    import torch

    # Create a simple concrete generator implementation
    class SimpleGeneratorParams(GeneratorParams):
        """Parameters for a simple random tensor generator."""

        tensor_size: Tuple[int, ...] = Field(
            default=(10, 10),
            description="Size of the tensors to generate",
        )
        value_range: Tuple[float, float] = Field(
            default=(0.0, 1.0),
            description="Range of values in the generated tensors",
        )

    class SimpleGenerator(Generator):
        """Generates random tensors of a specified size."""

        def __init__(self, params: Optional[SimpleGeneratorParams] = None):
            params = params or SimpleGeneratorParams()
            super().__init__(params)
            self.tensor_size = params.tensor_size
            self.value_range = params.value_range

        def __next__(self) -> Tuple[Tensor, ...]:
            """Generate a random tensor."""
            low, high = self.value_range
            tensor = torch.rand(self.tensor_size) * (high - low) + low
            # Return as a tuple to match the expected interface
            return (tensor,)

    # Create a simple augmenter that adds noise
    class NoiseAugmenterParams(AugmenterParams):
        """Parameters for noise augmentation."""

        noise_level: float = Field(
            default=0.1,
            description="Standard deviation of Gaussian noise to add",
        )

    class NoiseAugmenter(Augmenter):
        """Adds Gaussian noise to tensors."""

        def __init__(self, params: Optional[NoiseAugmenterParams] = None):
            params = params or NoiseAugmenterParams()
            super().__init__(params)
            self.noise_level = params.noise_level

        def augment(self, data: Tensor) -> Tensor:
            """Add Gaussian noise to the data."""
            noise = torch.randn_like(data) * self.noise_level
            return data + noise

    # Define parameters for the data module
    data_params = DataModuleParams(
        # Data generation parameters
        num_samples=1000,
        val_split=0.2,
        test_split=0.1,
        # DataLoader parameters
        batch_size=16,
        num_workers=0,
        pin_memory=True,
        # Augmentation parameters
        augmentation_probability=0.5,
    )

    # Create our generator and augmenter instances
    generator = SimpleGenerator(
        SimpleGeneratorParams(
            tensor_size=(10, 10),
            value_range=(0.0, 1.0),
            seed=42,
        )
    )

    augmenter = NoiseAugmenter(
        NoiseAugmenterParams(
            noise_level=0.1,
            augmentation_probability=0.7,
        )
    )

    # Create the data module
    data_module = DataModule(
        generator=generator,
        params=data_params,
    )

    # Manually set up the augmenters (in practice, this would be part of params)
    data_module.params.augmenters = [augmenter]

    # Create dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Demonstrate fetching a batch from each
    print("Fetching a training batch (with possible augmentation):")
    train_batch = next(iter(train_loader))
    print(f"Training batch shape: {train_batch[0].shape}")

    print("\nFetching a validation batch (no augmentation):")
    val_batch = next(iter(val_loader))
    print(f"Validation batch shape: {val_batch[0].shape}")

    print("\nDataModule pipeline setup successful!")
