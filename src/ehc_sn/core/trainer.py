"""Core trainer abstractions for neural network training strategies.

This module provides the foundation for implementing various training algorithms
in the entorhinal-hippocampal circuit modeling library. It defines abstract base
classes that enable flexible training strategy selection while maintaining
compatibility with PyTorch Lightning's training infrastructure.

The trainer module implements the Strategy pattern, allowing the same neural
network model to be trained using different optimization algorithms such as:
- Backpropagation (BP)
- Direct Random Target Projection (DRTP)
- Direct Feedback Alignment (DFA)
- Reinforcement Learning approaches

Key Components:
    TrainerParams: Configuration parameters for training experiments
    BaseTrainer: Abstract base class defining the training strategy interface

The design ensures that training strategies can be swapped independently of
the model architecture, enabling systematic comparison of learning algorithms
on the same network structure.
"""

import abc
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import lightning.pytorch as pl
import torch
from lightning import pytorch as pl
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, Field
from torch import Tensor


# -------------------------------------------------------------------------------------------
class TrainerParams(BaseModel):
    """Configuration parameters for neural network training experiments.

    This class defines all configurable parameters for training neural networks
    in the entorhinal-hippocampal circuit modeling library. It uses Pydantic
    validation to ensure parameter correctness and provides sensible defaults
    for common training scenarios.

    The configuration covers essential training aspects including epoch limits,
    logging directories, experiment naming, and checkpoint management. All
    parameters include validation constraints to prevent invalid configurations
    that could lead to training failures.

    Attributes:
        max_epochs: Maximum number of training epochs (1-1000)
        log_dir: Directory path for storing experiment logs and outputs
        experiment_name: Unique identifier for the training experiment
        checkpoint_freq: Frequency of model checkpoint saving (epochs)

    Example:
        >>> params = TrainerParams(
        ...     max_epochs=100,
        ...     experiment_name="drtp_experiment",
        ...     checkpoint_freq=10
        ... )
        >>> print(params.max_epochs)
        100
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Training Settings
    max_epochs: int = Field(default=200, ge=1, le=1000, description="Maximum training epochs")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field(..., description="Experiment name")
    checkpoint_freq: int = Field(default=5, ge=1, le=50, description="Checkpoint frequency")


# -------------------------------------------------------------------------------------------
class BaseTrainer(abc.ABC):
    """Abstract base class for implementing neural network training strategies.

    This class defines the core interface that all training strategies must
    implement to be compatible with the entorhinal-hippocampal circuit modeling
    framework. It provides a standardized way to implement different learning
    algorithms while maintaining full compatibility with PyTorch Lightning's
    training infrastructure.

    The BaseTrainer encapsulates all aspects of the training process including:
    - Optimizer configuration and management
    - Forward and backward pass orchestration
    - Gradient flow control and manipulation
    - Loss computation and application strategies
    - Training loop hooks and callbacks

    Each concrete implementation represents a specific learning algorithm
    (e.g., standard backpropagation, DRTP, DFA) while providing a uniform
    interface for model training. This design enables systematic comparison
    of different learning strategies on identical network architectures.

    Attributes:
        params: Training configuration parameters

    Example:
        >>> class BackpropTrainer(BaseTrainer):
        ...     def training_step(self, model, batch, batch_idx):
        ...         # Implement backpropagation logic
        ...         return loss
        >>> trainer = BackpropTrainer()
        >>> trainer.fit(model, datamodule)
    """

    def __init__(self, params: Optional[TrainerParams] = None) -> None:
        """Initialize the trainer with configuration parameters.

        Args:
            params: Training configuration parameters. If None, default will be used.
        """
        super().__init__()
        self.params = TrainerParams() if params is None else params

    # -----------------------------------------------------------------------------------
    @property
    def callbacks(self) -> List[pl.Callback]:
        """Get the list of PyTorch Lightning callbacks for training.

        Returns a standard set of callbacks that provide essential training
        functionality including model checkpointing, progress monitoring,
        and model summary visualization.

        Returns:
            List of configured Lightning callbacks for the training process.
        """
        return [
            callbacks.ModelCheckpoint(every_n_epochs=self.params.checkpoint_freq, save_weights_only=True),
            callbacks.RichModelSummary(max_depth=2),
            # callbacks.RichProgressBar(),  # Currently impacts training with 48% of training time
        ]

    # -----------------------------------------------------------------------------------
    def fit(self, model: pl.LightningModule, datamodule: pl.LightningDataModule) -> None:
        """Execute the complete training process for the given model.

        This method sets up and runs the PyTorch Lightning training loop with
        the configured parameters and callbacks. It automatically detects
        available hardware acceleration and configures the trainer accordingly.

        Args:
            model: PyTorch Lightning module to train
            datamodule: Lightning data module providing training data

        Returns:
            None. Training results are logged and checkpoints are saved
            according to the configuration parameters.
        """

        return pl.Trainer(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            max_epochs=self.params.max_epochs,
            callbacks=self.callbacks,
            logger=TensorBoardLogger(self.params.log_dir, name=self.params.experiment_name),
            profiler="simple",
        ).fit(model, datamodule=datamodule)

    # -----------------------------------------------------------------------------------
    @abc.abstractmethod
    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Optional[Tensor]:
        """Execute one training step using the specific strategy.

        This abstract method must be implemented by each concrete training
        strategy to define the core training logic. The implementation should
        handle the complete training step including forward pass, loss computation,
        backward pass, and optimizer updates according to the specific algorithm.

        The method integrates with PyTorch Lightning's training loop and can
        use either automatic or manual optimization depending on the strategy
        requirements. For automatic optimization, return the loss tensor.
        For manual optimization, handle gradients manually and return None.

        Args:
            model: The PyTorch Lightning module being trained
            batch: Training batch data tensor
            batch_idx: Index of the current batch in the epoch

        Returns:
            Loss tensor for automatic optimization, None for manual optimization

        Raises:
            NotImplementedError: If the method is not implemented by subclass
        """
        raise NotImplementedError("training_step method not implemented")

    # -----------------------------------------------------------------------------------
    def on_train_batch_start(self, model: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        """Hook called before each training batch begins.

        This method provides an opportunity to perform strategy-specific
        setup operations that need to occur before processing each batch.
        Override this method in concrete implementations to add custom
        pre-batch logic such as parameter initialization, state preparation,
        or strategy-specific configurations.

        Args:
            model: The PyTorch Lightning module being trained
            batch: Training batch data
            batch_idx: Index of the current batch in the epoch

        Returns:
            None
        """
        pass

    # -----------------------------------------------------------------------------------
    def on_train_batch_end(self, model: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Hook called after each training batch completes.

        This method provides an opportunity to perform strategy-specific
        cleanup or post-processing operations after each batch has been
        processed. Override this method in concrete implementations to add
        custom post-batch logic such as gradient manipulation, parameter
        updates, or logging operations.

        Args:
            model: The PyTorch Lightning module being trained
            outputs: Training step outputs from the batch
            batch: Training batch data that was processed
            batch_idx: Index of the completed batch in the epoch

        Returns:
            None
        """
        pass
