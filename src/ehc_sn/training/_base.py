import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Literal, Optional, Union

import lightning as L
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from pydantic import BaseModel, Field
from tqdm import tqdm


class FabricConfig(BaseModel):
    """Configuration for Lightning Fabric."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    accelerator: Union[str, Accelerator] = Field(
        description="Accelerator to use for training",
        default="auto",
    )
    strategy: Union[str, Strategy] = Field(
        description="Strategy for distributed training",
        default="auto",
    )
    devices: Union[list[int], str, int] = Field(
        description="Number of devices to use for training",
        default="auto",
    )
    num_nodes: int = Field(
        description="Number of nodes to use for distributed training",
        default=1,
    )
    precision: Optional[_PRECISION_INPUT] = Field(
        description="Numerical precision for training (e.g., '32-true', '16-mixed')",
        default=None,
    )
    plugins: Optional[Union[_PLUGIN_INPUT, list[_PLUGIN_INPUT]]] = Field(
        description="Plugins to use for training",
        default=None,
    )
    callbacks: Optional[Union[list[Any], Any]] = Field(
        description="Callbacks to use during training",
        default=None,
    )
    loggers: Optional[Union[Logger, list[Logger]]] = Field(
        description="Loggers to use for logging training progress",
        default=None,
    )

    def kwargs(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary of keyword arguments."""
        return FabricConfig.model_dump(self, exclude={"model_config"})


class BaseTrainer(ABC):
    """Abstract base class for trainers using Lightning Fabric."""

    def __init__(self, config: FabricConfig):
        """Initialize the trainer with the given configuration.
        Args:
            config: Configuration for Lightning Fabric
        """
        self.fabric = L.Fabric(**config.kwargs())
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False

    @abstractmethod
    def fit(
        self,
        model: L.LightningModule,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.
        """

    @abstractmethod
    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.
        """

    @abstractmethod
    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.
        """

    @abstractmethod
    def training_step(
        self,
        model: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """

    @abstractmethod
    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping containing model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"), state)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])
