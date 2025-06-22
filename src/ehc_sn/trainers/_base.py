import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Literal, Optional, Union

import lightning
import lightning.pytorch as pl
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.utilities.types import LRScheduler
from pydantic import BaseModel, Field
from torch import nn


# -------------------------------------------------------------------------------------------
class FabricConfig(BaseModel):
    """Configuration for Lightning Fabric."""

    # -----------------------------------------------------------------------------------
    model_config = {
        "extra": "forbid",  # Forbid extra fields not defined in the model
        "arbitrary_types_allowed": True,  # Forbid extra fields not defined in the model
    }

    # -----------------------------------------------------------------------------------
    accelerator: Union[str, Accelerator] = Field(
        description="Accelerator to use for training",
        default="auto",
    )

    # -----------------------------------------------------------------------------------
    strategy: Union[str, Strategy] = Field(
        description="Strategy for distributed training",
        default="auto",
    )

    # -----------------------------------------------------------------------------------
    devices: Union[list[int], str, int] = Field(
        description="Number of devices to use for training",
        default="auto",
    )

    # -----------------------------------------------------------------------------------
    num_nodes: int = Field(
        description="Number of nodes to use for distributed training",
        default=1,
    )

    # -----------------------------------------------------------------------------------
    precision: Optional[_PRECISION_INPUT] = Field(
        description="Numerical precision for training (e.g., '32-true', '16-mixed')",
        default=None,
    )

    # -----------------------------------------------------------------------------------
    plugins: Optional[Union[_PLUGIN_INPUT, list[_PLUGIN_INPUT]]] = Field(
        description="Plugins to use for training",
        default=None,
    )

    # -----------------------------------------------------------------------------------
    callbacks: Optional[Union[list[Any], Any]] = Field(
        description="Callbacks to use during training",
        default=None,
    )

    # -----------------------------------------------------------------------------------
    loggers: Optional[Union[Logger, list[Logger]]] = Field(
        description="Loggers to use for logging training progress",
        default=None,
    )

    # -----------------------------------------------------------------------------------
    def kwargs(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary of keyword arguments."""
        keys = FabricConfig.model_fields.keys()
        return self.model_dump(include=keys, exclude_none=True)


# -------------------------------------------------------------------------------------------
class BaseTrainer(ABC):
    """Abstract base class for trainers using Lightning Fabric."""

    # -----------------------------------------------------------------------------------
    def __init__(self, config: FabricConfig):
        """Initialize the trainer with the given configuration.
        Args:
            config: Configuration for Lightning Fabric
        """
        self.fabric = lightning.Fabric(**config.kwargs())
        self.global_step = 0
        self.current_epoch = 0

    # -----------------------------------------------------------------------------------
    @abstractmethod
    def fit(
        self: "BaseTrainer",
        model: nn.Module,
        data_module: pl.LightningDataModule,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
        scheduler: Optional[Union[LRScheduler, Dict[str, LRScheduler]]] = None,
    ) -> None:
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: Model to train with the trainer.
            data_module: Data module containing the training, validation, and test datasets.
            loss_function: Loss function or a dictionary of loss functions to use for training.
            optimizer: Optimizer or a dictionary of optimizers to use for training.
            scheduler: Learning rate scheduler or a dictionary of schedulers to use for training.
        """
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def callback_metrics(self) -> Dict[str, Any]:
        """Metrics collected by callbacks during training."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def logged_metrics(self) -> Dict[str, Any]:
        """Metrics sent to the loggers during training."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def progress_bar_metrics(self) -> Dict[str, Any]:
        """Metrics sent to the progress bar during training."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def current_epoch(self) -> int:
        """Current epoch during training."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def datamodule(self) -> Optional[pl.LightningDataModule]:
        """DataModule being used for training."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def is_last_batch(self) -> bool:
        """Whether the current batch is the last batch."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def global_step(self) -> int:
        """Total number of steps taken during training."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def logger(self) -> Optional[Logger]:
        """The main logger being used."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def loggers(self) -> list[Logger]:
        """All loggers being used."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def log_dir(self) -> Optional[str]:
        """Directory for logs."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def is_global_zero(self) -> bool:
        """Whether this process is the global zero process."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def estimated_stepping_batches(self) -> int:
        """Total number of expected stepping batches."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def state(self) -> Dict[str, Any]:
        """Current state of the trainer."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def should_stop(self) -> bool:
        """Whether training should stop."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def sanity_checking(self) -> bool:
        """Whether sanity checking is in progress."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def num_training_batches(self) -> int:
        """Number of batches in the training dataloader."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def num_sanity_val_batches(self) -> int:
        """Number of batches used for sanity checking."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def num_val_batches(self) -> Union[int, list[int]]:
        """Number of batches in the validation dataloader(s)."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def num_test_batches(self) -> Union[int, list[int]]:
        """Number of batches in the test dataloader(s)."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def num_predict_batches(self) -> Union[int, list[int]]:
        """Number of batches in the prediction dataloader(s)."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def train_dataloader(self) -> Optional[Iterable]:
        """Training dataloader."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def val_dataloaders(self) -> Union[Iterable, list[Iterable]]:
        """Validation dataloader(s)."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def test_dataloaders(self) -> Union[Iterable, list[Iterable]]:
        """Test dataloader(s)."""
        pass

    # -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def predict_dataloaders(self) -> Union[Iterable, list[Iterable]]:
        """Prediction dataloader(s)."""
        pass

    # -----------------------------------------------------------------------------------
    def sanity_check(
        self,
        model: nn.Module,
        data_module: pl.LightningDataModule,
    ) -> None:
        """Run a sanity check on the model before training.

        Args:
            model: Model to run the sanity check on.
            data_module: Data module containing the datasets.
        """
        raise NotImplementedError("sanity_check method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def train_batch(
        self: "BaseTrainer",
        model: nn.Module,
        batch: Any,
        batch_idx: int,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    ) -> Dict[str, Any]:
        """Process a single training batch.

        Args:
            model: Model being trained.
            batch: Current batch of data.
            batch_idx: Index of the current batch.
            loss_function: Loss function(s) to calculate training loss.
            optimizer: Optimizer(s) to update model parameters.

        Returns:
            Dict containing the batch results.
        """
        raise NotImplementedError("train_batch method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def train_epoch(
        self: "BaseTrainer",
        model: nn.Module,
        train_dataloader: Iterable,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    ) -> Dict[str, Any]:
        """Process a full training epoch.

        Args:
            model: Model being trained.
            train_dataloader: DataLoader for training data.
            loss_function: Loss function(s) to calculate training loss.
            optimizer: Optimizer(s) to update model parameters.

        Returns:
            Dict containing the epoch results.
        """
        raise NotImplementedError("train_epoch method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def validation_batch(
        self: "BaseTrainer",
        model: nn.Module,
        batch: Any,
        batch_idx: int,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
    ) -> Dict[str, Any]:
        """Process a single validation batch.

        Args:
            model: Model being validated.
            batch: Current batch of data.
            batch_idx: Index of the current batch.
            loss_function: Loss function(s) to calculate validation loss.

        Returns:
            Dict containing the batch results.
        """

        raise NotImplementedError("validation_batch method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def validation_epoch(
        self: "BaseTrainer",
        model: nn.Module,
        val_dataloader: Iterable,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
    ) -> Dict[str, Any]:
        """Process a full validation epoch.

        Args:
            model: Model being validated.
            val_dataloader: DataLoader for validation data.
            loss_function: Loss function(s) to calculate validation loss.

        Returns:
            Dict containing the epoch results.
        """

        raise NotImplementedError("validation_epoch method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def test_batch(
        self: "BaseTrainer",
        model: nn.Module,
        batch: Any,
        batch_idx: int,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
    ) -> Dict[str, Any]:
        """Process a single test batch.

        Args:
            model: Model being tested.
            batch: Current batch of data.
            batch_idx: Index of the current batch.
            loss_function: Loss function(s) to calculate test loss.

        Returns:
            Dict containing the batch results.
        """

        raise NotImplementedError("test_batch method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def test_epoch(
        self: "BaseTrainer",
        model: nn.Module,
        test_dataloader: Iterable,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
    ) -> Dict[str, Any]:
        """Process a full test epoch.

        Args:
            model: Model being tested.
            test_dataloader: DataLoader for test data.
            loss_function: Loss function(s) to calculate test loss.

        Returns:
            Dict containing the epoch results.
        """

        raise NotImplementedError("test_epoch method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def predict_batch(
        self: "BaseTrainer",
        model: nn.Module,
        batch: Any,
        batch_idx: int,
    ) -> Dict[str, Any]:
        """Process a single prediction batch.

        Args:
            model: Model for prediction.
            batch: Current batch of data.
            batch_idx: Index of the current batch.

        Returns:
            Dict containing the batch results.
        """

        raise NotImplementedError("predict_batch method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def predict_epoch(
        self: "BaseTrainer",
        model: nn.Module,
        predict_dataloader: Iterable,
    ) -> Dict[str, Any]:
        """Process a full prediction epoch.

        Args:
            model: Model for prediction.
            predict_dataloader: DataLoader for prediction data.

        Returns:
            Dict containing the epoch results.
        """

        raise NotImplementedError("predict_epoch method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def train(
        self: "BaseTrainer",
        model: nn.Module,
        train_dataloader: Iterable,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
        scheduler: Optional[Union[LRScheduler, Dict[str, LRScheduler]]] = None,
    ) -> Dict[str, Any]:
        """Run a full training process.

        Args:
            model: Model to train.
            train_dataloader: DataLoader for training data.
            loss_function: Loss function(s) to calculate training loss.
            optimizer: Optimizer(s) to update model parameters.
            scheduler: Optional learning rate scheduler(s).

        Returns:
            Dict containing the training results.
        """

        raise NotImplementedError("train method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def validation(
        self: "BaseTrainer",
        model: nn.Module,
        val_dataloader: Iterable,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
    ) -> Dict[str, Any]:
        """Run a full validation process.

        Args:
            model: Model to validate.
            val_dataloader: DataLoader for validation data.
            loss_function: Loss function(s) to calculate validation loss.

        Returns:
            Dict containing the validation results.
        """

        raise NotImplementedError("validation method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def test(
        self: "BaseTrainer",
        model: nn.Module,
        test_dataloader: Iterable,
        loss_function: Union[nn.Module, Dict[str, nn.Module]],
    ) -> Dict[str, Any]:
        """Run a full test process.

        Args:
            model: Model to test.
            test_dataloader: DataLoader for test data.
            loss_function: Loss function(s) to calculate test loss.

        Returns:
            Dict containing the test results.
        """

        raise NotImplementedError("test method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def predict(
        self: "BaseTrainer",
        model: nn.Module,
        predict_dataloader: Iterable,
    ) -> Dict[str, Any]:
        """Run a full prediction process.

        Args:
            model: Model for prediction.
            predict_dataloader: DataLoader for prediction data.

        Returns:
            Dict containing the prediction results.
        """

        raise NotImplementedError("predict method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def backward(
        self,
        model: nn.Module,
        loss: torch.Tensor,
    ) -> None:
        """Perform backward pass with the given loss.

        Args:
            model: The model being trained.
            loss: The loss tensor to backpropagate.
        """

        raise NotImplementedError("backward method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def optimizer_step(
        self: "BaseTrainer",
        model: nn.Module,
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    ) -> None:
        """Perform optimizer step.

        Args:
            model: The model being trained.
            optimizer: Optimizer or dictionary of optimizers to step.
        """

        raise NotImplementedError("optimizer_step method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def zero_grad(
        self: "BaseTrainer",
        model: nn.Module,
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    ) -> None:
        """Zero gradients for the given optimizer.

        Args:
            model: The model being trained.
            optimizer: Optimizer or dictionary of optimizers to zero gradients for.
        """

        raise NotImplementedError("zero_grad method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def save_checkpoint(
        self,
        path: str,
        model: nn.Module,
    ) -> None:
        """Save a checkpoint of the model state.

        Args:
            path: Path where to save the checkpoint.
            model: Model whose state to save.
        """

        raise NotImplementedError("save_checkpoint method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
    ) -> nn.Module:
        """Load a model from a checkpoint.

        Args:
            path: Path to the checkpoint file.
            model: Model to load the state into.

        Returns:
            The model with loaded state.
        """

        raise NotImplementedError("load_checkpoint method should be implemented in subclasses.")

    # -----------------------------------------------------------------------------------
    def catch_exception(
        self,
        exception: BaseException,
    ) -> None:
        """Handle an exception raised during training.

        Args:
            exception: The exception that was raised.
        """
        self.fabric.call("on_exception", self, exception)
        # Re-raise the exception after handling
        raise exception
