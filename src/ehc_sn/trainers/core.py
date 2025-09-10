"""Core trainer abstractions for different training strategies.

This module defines the base classes and interfaces for implementing different
training strategies for neural network models, particularly autoencoders.
The strategy pattern allows the same model to be trained with different
algorithms like backpropagation, DRTP, DFA, or reinforcement learning.
"""

import abc
from typing import Any, List, Optional

import lightning.pytorch as pl
from torch import Tensor
from torch.optim import Optimizer


class BaseTrainer(abc.ABC):
    """Abstract base class for training strategies.

    Defines the interface that all training strategies must implement.
    This allows models to be trained with different algorithms while
    maintaining a consistent interface and Lightning compatibility.

    Training strategies encapsulate the optimization logic, including:
    - Optimizer configuration
    - Forward/backward pass management
    - Gradient flow control
    - Loss computation and application

    Each concrete trainer implements a specific training algorithm
    (e.g., standard backprop, DRTP, DFA) while maintaining compatibility
    with PyTorch Lightning's training loop.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the trainer with strategy-specific parameters."""
        super().__init__()

    @abc.abstractmethod
    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Optional[Tensor]:
        """Execute one training step.

        Implements the core training logic for the specific strategy.
        This includes forward pass, loss computation, backward pass,
        and optimizer updates according to the strategy's requirements.

        Args:
            model: The Lightning module being trained.
            batch: Training batch data.
            batch_idx: Index of the current batch.

        Returns:
            Loss tensor if using automatic optimization, None for manual.
        """
        raise NotImplementedError("training_step method not implemented")

    @abc.abstractmethod
    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        """Execute one validation step.

        Implements validation logic that is consistent with the training
        strategy but without parameter updates.

        Args:
            model: The Lightning module being validated.
            batch: Validation batch data.
            batch_idx: Index of the current batch.

        Returns:
            Loss tensor for validation metrics.
        """
        raise NotImplementedError("validation_step method not implemented")

    def on_train_batch_start(self, model: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        """Hook called before training batch starts.

        Override this method to implement strategy-specific setup
        that needs to occur before each batch.

        Args:
            model: The Lightning module being trained.
            batch: Training batch data.
            batch_idx: Index of the current batch.
        """
        pass

    def on_train_batch_end(self, model: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Hook called after training batch ends.

        Override this method to implement strategy-specific cleanup
        that needs to occur after each batch.

        Args:
            model: The Lightning module being trained.
            outputs: Training step outputs.
            batch: Training batch data.
            batch_idx: Index of the current batch.
        """
        pass
