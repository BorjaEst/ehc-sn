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
from lightning_utilities import apply_to_collection
from pydantic import BaseModel, Field
from tqdm import tqdm


class FabricConfig(BaseModel):
    """Configuration for Lightning Fabric."""

    model_config = {
        "extra": "forbid",  # Forbid extra fields not defined in the model
        "arbitrary_types_allowed": True,  # Forbid extra fields not defined in the model
    }

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
        keys = FabricConfig.model_fields.keys()
        return self.model_dump(include=keys, exclude_none=True)


class BaseTrainer(ABC):
    """Abstract base class for trainers using Lightning Fabric."""

    def __init__(self, config: FabricConfig):
        """Initialize the trainer with the given configuration.
        Args:
            config: Configuration for Lightning Fabric
        """
        self.fabric = lightning.Fabric(**config.kwargs())
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False
        self.current_return = {}

    @abstractmethod
    def fit(self, train_module: pl.LightningModule, data_module: pl.LightningDataModule):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            train_module: LightningModule to train, can have the same hooks as :attr:`callbacks`.
        """

    @abstractmethod
    def train_loop(
        self,
        model: lightning.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        scheduler: LRScheduler,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            scheduler: The learning rate scheduler configuration.
        """

    @abstractmethod
    def val_loop(
        self,
        model: lightning.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
    ):
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
        """

    @abstractmethod
    def training_step(
        self,
        model: lightning.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """

    @abstractmethod
    def validation_step(
        self,
        model: lightning.LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        """A single validation step, running forward on the validation data.

        Args:
            model: the lightning module to evaluate
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """

    @abstractmethod
    def step_scheduler(
        self,
        model: lightning.LightningModule,
        scheduler: LRScheduler,
        level: Literal["step", "epoch"],
    ):
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level

        """

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

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any) -> Iterable:
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def format_iterable(self, prog_bar: Iterable, prefix: str):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and self.current_return is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(self.current_return, torch.Tensor, lambda x: x.item())
            if isinstance(self.current_return, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(self.current_return, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
