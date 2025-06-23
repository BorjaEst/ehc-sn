from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Union

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import Strategy
from pydantic import BaseModel, Field
from torch import nn


class TrainerParams(BaseModel):
    """
    Generic parameters for trainers in the EHC-SN framework.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Core training parameters
    accelerator: str = Field(
        default="auto",
        description="The accelerator to use for training. Can be 'cpu', 'gpu', 'tpu', 'hpu', 'mps', or 'auto'.",
    )
    strategy: Union[str, Strategy] = Field(
        default="auto",
        description="The strategy to use for distributed training. Can be 'ddp', 'dp', or custom strategies.",
    )
    devices: Union[list[int], str, int] = Field(
        default="auto",
        description="Devices to be used for training. Can be 'auto', a positive number, a sequence of device indices, or -1 for all devices.",
    )
    num_nodes: int = Field(
        default=1,
        description="Number of GPU nodes for distributed training.",
    )
    precision: Optional[str] = Field(
        default="32-true",
        description="Numerical precision for operations. Options include '64', '32-true', '16-mixed', 'bf16-mixed'.",
    )

    # Logging and callbacks
    logger: Optional[Union[Logger, Iterable[Logger], bool]] = Field(
        default=True,
        description="Logger (or iterable of loggers) for experiment tracking. True for default, False to disable.",
    )
    callbacks: Optional[Union[List[Callback], Callback]] = Field(
        default=None,
        description="Callback or list of callbacks to be used during training.",
    )

    # Training limits
    max_epochs: Optional[int] = Field(
        default=None,
        description="Maximum number of epochs for training. None defaults to 1000 if max_steps not set. -1 for infinite.",
    )
    min_epochs: Optional[int] = Field(
        default=None,
        description="Minimum number of epochs for training.",
    )
    max_steps: int = Field(
        default=-1,
        description="Maximum number of training steps. -1 means unlimited.",
    )
    min_steps: Optional[int] = Field(
        default=None,
        description="Minimum number of training steps.",
    )
    max_time: Optional[Union[str, timedelta, Dict[str, int]]] = Field(
        default=None,
        description="Maximum time for training in format DD:HH:MM:SS, as timedelta, or as dictionary.",
    )

    # Batch limits
    limit_train_batches: Optional[Union[int, float]] = Field(
        default=1.0,
        description="Limit number/percent of training batches per epoch.",
    )
    limit_val_batches: Optional[Union[int, float]] = Field(
        default=1.0,
        description="Limit number/percent of validation batches per epoch.",
    )
    limit_test_batches: Optional[Union[int, float]] = Field(
        default=1.0,
        description="Limit number/percent of test batches.",
    )
    limit_predict_batches: Optional[Union[int, float]] = Field(
        default=1.0,
        description="Limit number/percent of prediction batches.",
    )

    # Validation configuration
    val_check_interval: Optional[Union[int, float]] = Field(
        default=1.0,
        description="How often to check validation (float for fraction of epoch, int for specific batch interval).",
    )
    check_val_every_n_epoch: Optional[int] = Field(
        default=1,
        description="Run validation every n training epochs.",
    )
    num_sanity_val_steps: Optional[int] = Field(
        default=2,
        description="Number of validation batches to run before training. -1 runs all batches.",
    )

    # Optimization parameters
    accumulate_grad_batches: int = Field(
        default=1,
        description="Number of batches to accumulate gradients across.",
    )
    gradient_clip_val: Optional[Union[int, float]] = Field(
        default=None,
        description="Value to clip gradients at. None disables clipping.",
    )
    gradient_clip_algorithm: Optional[str] = Field(
        default="norm",
        description="Algorithm for gradient clipping ('value' or 'norm').",
    )

    # Output and logging
    log_every_n_steps: Optional[int] = Field(
        default=50,
        description="How often to log within steps.",
    )
    enable_checkpointing: Optional[bool] = Field(
        default=True,
        description="If True, enable checkpointing.",
    )
    enable_progress_bar: Optional[bool] = Field(
        default=True,
        description="If True, enable progress bar.",
    )
    enable_model_summary: Optional[bool] = Field(
        default=True,
        description="If True, display model summary at beginning of training.",
    )
    default_root_dir: Optional[_PATH] = Field(
        default=None,
        description="Default directory for saving logs and checkpoints. Can be remote paths like 's3://' or 'hdfs://'.",
    )

    # Debug/testing parameters
    fast_dev_run: Union[int, bool] = Field(
        default=False,
        description="Run a truncated version of training for debugging. Number of batches if int.",
    )
    overfit_batches: Union[int, float] = Field(
        default=0.0,
        description="Number/percent of batches to use for overfitting.",
    )
    detect_anomaly: bool = Field(
        default=False,
        description="If True, enable anomaly detection in PyTorch autograd engine.",
    )

    # Other settings
    inference_mode: bool = Field(
        default=True,
        description="If True, use inference mode for validation, testing, and prediction.",
    )
    deterministic: Optional[bool] = Field(
        default=None,
        description="If True or 'warn', sets PyTorch operations to use deterministic algorithms.",
    )
    benchmark: Optional[bool] = Field(
        default=None,
        description="Value to set torch.backends.cudnn.benchmark. Defaults to False if deterministic is True.",
    )
    reload_dataloaders_every_n_epochs: int = Field(
        default=0,
        description="Reload dataloaders every n epochs.",
    )

    # Completed fields with proper syntax
    use_distributed_sampler: bool = Field(
        default=True,
        description="Whether to wrap DataLoader's sampler with DistributedSampler. Automatically toggled for strategies that require it.",
    )
    profiler: Optional[Any] = Field(
        default=None,
        description="Profiler to use for identifying bottlenecks in training steps.",
    )
    barebones: bool = Field(
        default=False,
        description="Run in 'barebones mode' with all features that impact raw speed disabled, for trainer overhead analysis.",
    )
    plugins: Optional[Any] = Field(
        default=None,
        description="Plugins to modify core behavior like DDP and AMP, or to enable custom Lightning plugins.",
    )
    sync_batchnorm: bool = Field(
        default=False,
        description="Synchronize batch norm layers between process groups/whole world.",
    )
    model_registry: Optional[str] = Field(
        default=None,
        description="The name of the model being uploaded to Model hub.",
    )


class BaseTrainer(ABC, pl.LightningModule):
    """
    Abstract base class for all trainers in the EHC-SN framework.

    This class defines the interface for trainers and provides common functionality
    for hook calling, initialization, and Lightning integration.
    """

    def __init__(self, model: nn.Module, params: Optional[TrainerParams] = None):
        """
        Initialize the base trainer with optional parameters.

        Args:
            params: Configuration parameters for the trainer.
        """
        super().__init__()
        keys = TrainerParams.model_fields.keys()
        kwargs = params.model_dump(include=keys, exclude_none=True)
        self.trainer = pl.Trainer(**kwargs)
        self.model = model

    @abstractmethod
    def configure_optimizers(self):
        """Configure the optimizers for the model."""
        pass

    def fit(self, *args, **kwargs):
        """Forward the fit call to the Lightning trainer."""
        self.trainer.fit(self, *args, **kwargs)

    def validate(self, *args, **kwargs):
        """Forward the validate call to the Lightning trainer."""
        self.trainer.validate(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        """Forward the predict call to the Lightning trainer."""
        self.trainer.predict(self, *args, **kwargs)
