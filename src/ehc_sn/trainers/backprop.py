import os
from typing import Any, Literal, Optional, Union, cast

import lightning.pytorch as pl
import torch
from lightning.fabric import strategies
from lightning.fabric.utilities.types import LRScheduler
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from lightning_utilities import apply_to_collection
from pydantic import Field
from torch.utils.data import DataLoader

from ehc_sn.trainers import _base


class BPTrainerParams(_base.FabricConfig):
    """Configuration for the Backpropagation Trainer."""

    # Backpropagation-specific configuration
    max_epochs: Optional[int] = Field(
        description="Maximum number of training epochs",
        default=1000,
    )
    max_steps: Optional[int] = Field(
        description="Maximum number of training steps",
        default=None,
    )
    grad_accum_steps: int = Field(
        description="Number of gradient accumulation steps",
        default=1,
    )
    clip_grad_norm: Optional[float] = Field(
        description="Maximum norm for gradient clipping. If None, no clipping is applied.",
        default=None,
    )
    limit_train_batches: Union[int, float] = Field(
        description="Limit on the number of training batches per epoch",
        default=float("inf"),
    )
    limit_val_batches: Union[int, float] = Field(
        description="Limit on the number of validation batches per epoch",
        default=float("inf"),
    )
    validation_frequency: int = Field(
        description="Frequency of validation checks (in epochs)",
        default=1,
    )
    use_distributed_sampler: bool = Field(
        description="Whether to use distributed sampler for training data",
        default=True,
    )
    checkpoint_dir: str = Field(
        description="Directory to save the train module checkpoints",
        default="./checkpoints",
    )
    checkpoint_frequency: int = Field(
        description="Frequency of saving checkpoints (in epochs)",
        default=1,
    )


class BPTrainer(_base.BaseTrainer):
    """
    Trainer for backpropagation-based training of entorhinal-hippocampal circuit models.
    Uses Lightning Fabric for efficient device management and training acceleration.
    """

    def __init__(self, config: BPTrainerParams):
        """Initialize the trainer with the given configuration.

        Args:
            config: Configuration for back propagation Fabric trainer.
        """
        super().__init__(config)

        # Store configuration parameters as instance attributes
        self.max_epochs = config.max_epochs
        self.max_steps = config.max_steps
        self.grad_accum_steps = config.grad_accum_steps
        self.clip_grad_norm = config.clip_grad_norm
        self.limit_train_batches = config.limit_train_batches
        self.limit_val_batches = config.limit_val_batches
        self.validation_frequency = config.validation_frequency
        self.use_distributed_sampler = config.use_distributed_sampler
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_frequency = config.checkpoint_frequency

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def fit(self, model: pl.LightningModule, data_module: pl.LightningDataModule):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train
            data_module: the LightningDataModule providing train and validation dataloaders
            ckpt_path: Path to previous checkpoints to resume training from
        """
        self.fabric.launch()

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(
            data_module.train_dataloader(),
            use_distributed_sampler=self.use_distributed_sampler,
        )
        val_loader = self.fabric.setup_dataloaders(
            data_module.val_dataloader(),
            use_distributed_sampler=self.use_distributed_sampler,
        )

        # setup model and optimizer
        if isinstance(self.fabric.strategy, strategies.FSDPStrategy):
            raise NotImplementedError("FSDP strategy is not currently supported")

        optmsch_config: OptimizerLRSchedulerConfig = model.configure_optimizers()
        if "optimizer" not in optmsch_config:
            raise ValueError("Optimizer must be provided in model.configure_optimizers()")
        model, optimizer = self.fabric.setup(model, optmsch_config["optimizer"])
        scheduler_cfg: LRScheduler = optmsch_config.get("lr_scheduler")

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg["scheduler"]}

        # load last checkpoint if available
        latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint_path is not None:
            self.load(state, latest_checkpoint_path)

            # check if we even need to train here
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

        # Start the epoch loop
        while not self.should_stop:

            # Run training and validation loops
            self.train_loop(model, optimizer, train_loader, scheduler_cfg)
            if self._should_validate:
                self.val_loop(model, val_loader)

            # Step the scheduler on epoch level
            self.step_scheduler(model, scheduler_cfg, level="epoch")
            self.current_epoch += 1

            # Save checkpoint if it's time
            if self.current_epoch % self.checkpoint_frequency == 0:
                self.save(state)

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

        # Save final model state
        self.save(state)

        # reset for next fit call
        self.should_stop = False

    @property
    def _should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    def train_loop(
        self,
        model: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        scheduler_cfg: LRScheduler,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer
            train_loader: The dataloader yielding the training batches
            scheduler_cfg: The learning rate scheduler configuration
        """
        total = min(len(train_loader), self.limit_train_batches)
        iterable = self.progbar_wrapper(train_loader, total, desc=f"Epoch {self.current_epoch}")

        # Start training loop
        self.fabric.call("on_train_epoch_start")
        for batch_idx, batch in enumerate(iterable):

            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= self.limit_train_batches:
                break

            # check if optimizer should step in gradient accumulation
            self.fabric.call("on_train_batch_start", batch, batch_idx)
            should_optim_step = self.global_step % self.grad_accum_steps == 0

            if should_optim_step:
                # currently only supports a single optimizer
                self.training_step(model, batch, batch_idx)

                # gradient clipping if configured
                if self.clip_grad_norm is not None:
                    clip_val = self.clip_grad_norm
                    self.fabric.clip_gradients(model, optimizer, clip_val, error_if_nonfinite=True)

                # optimizer step
                self.fabric.call("on_before_optimizer_step", optimizer)
                optimizer.step()

                # optimizer step runs train step internally through closure
                self.fabric.call("on_before_zero_grad", optimizer)
                optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model, batch, batch_idx)

            # End the train batch
            self.fabric.call("on_train_batch_end", self.current_return, batch, batch_idx)

            # this guard ensures we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(model, scheduler_cfg, level="step")

            # add output values to progress bar
            self.format_iterable(iterable, "train")

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self.fabric.call("on_train_epoch_end")

    def val_loop(self, model: pl.LightningModule, val_loader: torch.utils.data.DataLoader):
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches
        """
        # Set model to evaluation mode or call custom hook
        if not is_overridden("on_validation_model_eval", _unwrap_objects(model)):
            model.eval()
        else:
            self.fabric.call("on_validation_model_eval")  # calls `model.eval()`
        torch.set_grad_enabled(False)

        total = min(len(val_loader), self.limit_train_batches)
        iterable = self.progbar_wrapper(val_loader, total, desc="Validation")

        # Start validation loop
        self.fabric.call("on_validation_epoch_start")
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= self.limit_train_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)
            out = self.validation_step(model, batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self.format_iterable(iterable, "val")

        # End the validation epoch
        self.fabric.call("on_validation_epoch_end")

        # Set model back to training mode or call custom hook
        if not is_overridden("on_validation_model_train", _unwrap_objects(model)):
            model.train()
        else:
            self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def training_step(self, model: pl.LightningModule, batch: Any, batch_idx: int):
        """A single training step, running forward and backward.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch
        """
        outputs = model.training_step(batch, batch_idx=batch_idx)
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        # store the current return value for later use
        self.current_return = apply_to_collection(
            outputs,
            dtype=torch.Tensor,
            function=lambda x: x.detach(),
        )

    def validation_step(self, model: pl.LightningModule, batch: Any, batch_idx: int):
        """A single validation step, running forward.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch
        """
        outputs = model.validation_step(batch, batch_idx=batch_idx)

        # store the current return value for later use
        self.current_return = apply_to_collection(
            outputs,
            dtype=torch.Tensor,
            function=lambda x: x.detach(),
        )

    def step_scheduler(
        self,
        model: pl.LightningModule,
        scheduler_cfg: LRScheduler,
        level: Literal["step", "epoch"],
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration
            level: whether we are trying to step on epoch- or step-level
        """
        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if self.global_step % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # Do not support ReduceLROnPlateau scheduler stepping here
        if scheduler_cfg["reduce_on_plateau"]:
            raise NotADirectoryError("ReduceLROnPlateau not supported.")

        # rely on model hook for actual step
        i = self.global_step if level == "step" else self.current_epoch
        scheduler_cfg["scheduler"].step(i)


# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset

    # Create a simple model using LightningModule
    class SimpleModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = nn.MSELoss()(y_hat, y)
            return {"loss": loss, "mse": loss}

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = nn.MSELoss()(y_hat, y)
            return {"val_loss": loss}

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=0.001)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "strict": True,
                },
            }

    # Create synthetic dataset
    x_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    x_val = torch.randn(20, 10)
    y_val = torch.randn(20, 1)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Create a custom DataModule
    class CustomDataModule(pl.LightningDataModule):
        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

    # Configure and initialize the trainer
    config = BPTrainerParams(
        max_epochs=10,
        checkpoint_dir="./example_checkpoints",
        limit_train_batches=5,  # Limit for demo purposes
        limit_val_batches=2,  # Limit for demo purposes
        accelerator="cpu",  # Use CPU for this example
    )

    trainer = BPTrainer(config)
    model = SimpleModel()
    data_module = CustomDataModule()

    # Train the model
    trainer.fit(model, data_module)

    print(f"Training completed successfully. Model saved to {config.checkpoint_dir}")
