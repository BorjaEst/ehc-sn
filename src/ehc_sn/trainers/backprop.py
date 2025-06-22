import os
from collections.abc import Iterable
from typing import Any, Dict, Literal, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.types import LRScheduler
from lightning.fabric.wrappers import _unwrap_objects
from pydantic import Field
from torch import nn
from torch.utils.data import DataLoader

from ehc_sn import utils
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

    # -----------------------------------------------------------------------------------
    def __init__(self, config: BPTrainerParams):  # -------------------------------------
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

        # Add metrics tracking
        self.logged_metrics = {}

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def fit(
        self: "BPTrainer",
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
        self.fabric.call("on_fit_start", self, model)

        # Setup dataloaders
        data_module.setup()
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader() if hasattr(data_module, "val_dataloader") else None

        # Apply fabric to dataloaders
        if self.use_distributed_sampler:
            train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
            if val_dataloader:
                val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        # Setup model, loss function, and optimizers with fabric
        model = self.fabric.setup(model)

        if isinstance(loss_function, Dict):
            loss_function = {name: self.fabric.setup(loss) for name, loss in loss_function.items()}
        else:
            loss_function = self.fabric.setup(loss_function)

        if isinstance(optimizer, Dict):
            optimizer = {name: self.fabric.setup_optimizers(opt) for name, opt in optimizer.items()}
        else:
            optimizer = self.fabric.setup_optimizers(optimizer)

        # Sanity check
        if val_dataloader:
            self.sanity_check(model, data_module)

        # Training loop
        epoch = 0
        global_step = 0

        while (self.max_epochs is None or epoch < self.max_epochs) and (
            self.max_steps is None or global_step < self.max_steps
        ):

            # Train for one epoch
            train_results = self.train(model, train_dataloader, loss_function, optimizer, scheduler)
            global_step += train_results.get("steps", 0)

            # Validate if needed
            if val_dataloader and epoch % self.validation_frequency == 0:
                self.validation(model, val_dataloader, loss_function)

            # Save checkpoint if needed
            if epoch % self.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path, model)

            epoch += 1

            # Check max_steps condition
            if self.max_steps is not None and global_step >= self.max_steps:
                break

        self.fabric.call("on_fit_end", self, model)

    def sanity_check(self, model: nn.Module, data_module: pl.LightningDataModule) -> None:
        """Run a sanity check on the model before training.

        Args:
            model: Model to run the sanity check on.
            data_module: Data module containing the datasets.
        """
        self.fabric.call("on_sanity_check_start", self, model)

        # Get the validation dataloader
        val_dataloader = data_module.val_dataloader()
        if val_dataloader is None:
            self.fabric.print("Skipping sanity check: No validation dataloader available")
        else:
            # Apply fabric to dataloaders
            if self.use_distributed_sampler:
                val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

            # Run a forward pass on a single batch to check for errors
            model.eval()
            try:
                batch = next(iter(val_dataloader))
                with torch.no_grad():
                    if isinstance(batch, (tuple, list)):
                        _ = model(*batch)
                    elif isinstance(batch, dict):
                        _ = model(**batch)
                    else:
                        _ = model(batch)
                self.fabric.print("Sanity check passed: Model forward pass successful")
            except Exception as e:
                self.fabric.print(f"Sanity check failed: {str(e)}")
                raise

        self.fabric.call("on_sanity_check_end", self, model)

    def train_batch(
        self: "BPTrainer",
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
        self.fabric.call("on_train_batch_start", self, model, batch, batch_idx)

        is_accumulating = (batch_idx + 1) % self.grad_accum_steps != 0

        # Set model to training mode
        model.train()

        # Forward pass
        if isinstance(batch, (tuple, list)):
            x, y, *_ = batch
            y_hat = model(*batch)
        elif isinstance(batch, dict):
            y = batch.pop("target", None)
            y_hat = model(**batch)
        else:
            y_hat = model(batch)
            y = None

        # Calculate loss
        if isinstance(loss_function, dict):
            # For multi-loss scenarios
            losses = {name: loss_fn(y_hat, y) for name, loss_fn in loss_function.items()}
            total_loss = sum(losses.values())
            batch_results = {f"train_{name}_loss": loss.detach() for name, loss in losses.items()}
            batch_results["train_loss"] = total_loss.detach()
        else:
            # Single loss function
            total_loss = loss_function(y_hat, y)
            batch_results = {"train_loss": total_loss.detach()}

        # Backward pass with accumulated gradients
        if is_accumulating:
            # Accumulate gradients but don't optimize yet
            self.backward(total_loss / self.grad_accum_steps)
        else:
            # Backward and optimize
            self.backward(total_loss / self.grad_accum_steps)
            self.optimizer_step(optimizer)
            self.zero_grad(optimizer)

        self.fabric.call("on_train_batch_end", self, model, batch_results, batch, batch_idx)
        return batch_results

    def train_epoch(
        self: "BPTrainer",
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
        self.fabric.call("on_train_epoch_start", self, model)

        # Initialize metrics
        epoch_results = {"train_loss": 0.0, "steps": 0}
        if isinstance(loss_function, dict):
            for name in loss_function.keys():
                epoch_results[f"train_{name}_loss"] = 0.0

        # Determine batch limit
        num_batches = utils.determine_batch_limit(self.limit_train_batches, len(train_dataloader))

        # Train for one epoch
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= num_batches:
                break

            batch_results = self.train_batch(model, batch, batch_idx, loss_function, optimizer)

            # Accumulate results
            for key in batch_results:
                if key in epoch_results:
                    epoch_results[key] += batch_results[key]
            epoch_results["steps"] += 1

        # Average metrics
        for key in epoch_results:
            if key != "steps":
                epoch_results[key] = epoch_results[key] / max(1, epoch_results["steps"])

        self.fabric.call("on_train_epoch_end", self, model, epoch_results)
        return epoch_results

    def validation_batch(
        self: "BPTrainer",
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
        self.fabric.call("on_validation_batch_start", self, model, batch, batch_idx)

        # Set model to evaluation mode
        model.eval()

        # Forward pass without gradients
        with torch.no_grad():
            if isinstance(batch, (tuple, list)):
                x, y = batch
                y_hat = model(x)
            elif isinstance(batch, dict):
                y = batch.pop("target", None)
                y_hat = model(**batch)
            else:
                y_hat = model(batch)
                y = None

            # Calculate loss
            if isinstance(loss_function, dict):
                # For multi-loss scenarios
                losses = {name: loss_fn(y_hat, y) for name, loss_fn in loss_function.items()}
                total_loss = sum(losses.values())
                batch_results = {f"val_{name}_loss": loss.detach() for name, loss in losses.items()}
                batch_results["val_loss"] = total_loss.detach()
            else:
                # Single loss function
                total_loss = loss_function(y_hat, y)
                batch_results = {"val_loss": total_loss.detach()}

        self.fabric.call("on_validation_batch_end", self, model, batch_results, batch, batch_idx)
        return batch_results

    def validation_epoch(
        self: "BPTrainer",
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
        self.fabric.call("on_validation_epoch_start", self, model)

        # Initialize metrics
        epoch_results = {"val_loss": 0.0, "steps": 0}
        if isinstance(loss_function, dict):
            for name in loss_function.keys():
                epoch_results[f"val_{name}_loss"] = 0.0

        # Determine batch limit
        num_batches = utils.determine_batch_limit(self.limit_val_batches, len(val_dataloader))

        # Validate for one epoch
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= num_batches:
                break

            batch_results = self.validation_batch(model, batch, batch_idx, loss_function)

            # Accumulate results
            for key in batch_results:
                if key in epoch_results:
                    epoch_results[key] += batch_results[key]
            epoch_results["steps"] += 1

        # Average metrics
        for key in epoch_results:
            if key != "steps":
                epoch_results[key] = epoch_results[key] / max(1, epoch_results["steps"])

        self.fabric.call("on_validation_epoch_end", self, model, epoch_results)
        return epoch_results

    def test_batch(
        self: "BPTrainer",
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
        self.fabric.call("on_test_batch_start", self, model, batch, batch_idx)

        # Set model to evaluation mode
        model.eval()

        # Forward pass without gradients
        with torch.no_grad():
            if isinstance(batch, (tuple, list)):
                x, y = batch
                y_hat = model(x)
            elif isinstance(batch, dict):
                y = batch.pop("target", None)
                y_hat = model(**batch)
            else:
                y_hat = model(batch)
                y = None

            # Calculate loss
            if isinstance(loss_function, dict):
                # For multi-loss scenarios
                losses = {name: loss_fn(y_hat, y) for name, loss_fn in loss_function.items()}
                total_loss = sum(losses.values())
                batch_results = {f"test_{name}_loss": loss.detach() for name, loss in losses.items()}
                batch_results["test_loss"] = total_loss.detach()
            else:
                # Single loss function
                total_loss = loss_function(y_hat, y)
                batch_results = {"test_loss": total_loss.detach()}

        self.fabric.call("on_test_batch_end", self, model, batch_results, batch, batch_idx)
        return batch_results

    def test_epoch(
        self: "BPTrainer",
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
        self.fabric.call("on_test_epoch_start", self, model)

        # Initialize metrics
        epoch_results = {"test_loss": 0.0, "steps": 0}
        if isinstance(loss_function, dict):
            for name in loss_function.keys():
                epoch_results[f"test_{name}_loss"] = 0.0

        # Test for one epoch
        for batch_idx, batch in enumerate(test_dataloader):
            batch_results = self.test_batch(model, batch, batch_idx, loss_function)

            # Accumulate results
            for key in batch_results:
                if key in epoch_results:
                    epoch_results[key] += batch_results[key]
            epoch_results["steps"] += 1

        # Average metrics
        for key in epoch_results:
            if key != "steps":
                epoch_results[key] = epoch_results[key] / max(1, epoch_results["steps"])

        self.fabric.call("on_test_epoch_end", self, model, epoch_results)
        return epoch_results

    def predict_batch(
        self: "BPTrainer",
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
        self.fabric.call("on_predict_batch_start", self, model, batch, batch_idx)

        # Set model to evaluation mode
        model.eval()

        # Forward pass without gradients
        with torch.no_grad():
            if isinstance(batch, (tuple, list)):
                x = batch[0]
                y_hat = model(x)
            elif isinstance(batch, dict):
                y_hat = model(**batch)
            else:
                y_hat = model(batch)

            batch_results = {"predictions": y_hat.detach()}

        self.fabric.call("on_predict_batch_end", self, model, batch_results, batch, batch_idx)
        return batch_results

    def predict_epoch(
        self: "BPTrainer",
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
        self.fabric.call("on_predict_epoch_start", self, model)

        # Initialize list to collect predictions
        all_predictions = []

        # Predict for the entire dataset
        for batch_idx, batch in enumerate(predict_dataloader):
            batch_results = self.predict_batch(model, batch, batch_idx)
            all_predictions.append(batch_results["predictions"])

        # Concatenate all predictions
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)

        epoch_results = {"predictions": all_predictions}

        self.fabric.call("on_predict_epoch_end", self, model, epoch_results)
        return epoch_results

    def train(
        self: "BPTrainer",
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
        self.fabric.call("on_train_start", self, model)

        epoch_results = self.train_epoch(model, train_dataloader, loss_function, optimizer)

        # Step scheduler if provided (on epoch level)
        if scheduler is not None:
            self.step_scheduler(model, scheduler, level="epoch")

        self.fabric.call("on_train_end", self, model, epoch_results)
        return epoch_results

    def validation(
        self: "BPTrainer",
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
        self.fabric.call("on_validation_start", self, model)

        epoch_results = self.validation_epoch(model, val_dataloader, loss_function)

        self.fabric.call("on_validation_end", self, model, epoch_results)
        return epoch_results

    def test(
        self: "BPTrainer",
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
        self.fabric.call("on_test_start", self, model)

        epoch_results = self.test_epoch(model, test_dataloader, loss_function)

        self.fabric.call("on_test_end", self, model, epoch_results)
        return epoch_results

    def predict(
        self: "BPTrainer",
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
        self.fabric.call("on_predict_start", self, model)

        epoch_results = self.predict_epoch(model, predict_dataloader)

        self.fabric.call("on_predict_end", self, model, epoch_results)
        return epoch_results

    def backward(
        self: "BPTrainer",
        loss: torch.Tensor,
    ) -> None:
        """Perform backward pass with the given loss.

        Args:
            loss: The loss tensor to backpropagate.
        """
        self.fabric.call("on_before_backward", self, loss)

        # Use fabric's backward
        self.fabric.backward(loss)

        # Apply gradient clipping if configured
        if self.clip_grad_norm is not None:
            self.fabric.clip_gradients(model=None, optimizer=None, max_norm=self.clip_grad_norm)

        self.fabric.call("on_after_backward", self)

    def optimizer_step(
        self: "BPTrainer",
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    ) -> None:
        """Perform optimizer step.

        Args:
            optimizer: Optimizer or dictionary of optimizers to step.
        """
        self.fabric.call("on_before_optimizer_step", self, optimizer)

        if isinstance(optimizer, dict):
            for opt in optimizer.values():
                opt.step()
        else:
            optimizer.step()

    def zero_grad(
        self: "BPTrainer",
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    ) -> None:
        """Zero gradients for the given optimizer.

        Args:
            optimizer: Optimizer or dictionary of optimizers to zero gradients for.
        """
        self.fabric.call("on_before_zero_grad", self, optimizer)

        if isinstance(optimizer, dict):
            for opt in optimizer.values():
                opt.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(self, path: str, model: nn.Module) -> None:
        """Save a checkpoint of the model state.

        Args:
            path: Path where to save the checkpoint.
            model: Model whose state to save.
        """
        import time

        checkpoint = {
            "model_state_dict": _unwrap_objects(model).state_dict(),
            "metadata": {
                "timestamp": time.time(),
                "version": "1.0",
            },
        }

        # Allow callback to modify the checkpoint
        self.fabric.call("on_save_checkpoint", self, checkpoint)

        # Save the checkpoint
        self.fabric.save(path, checkpoint)
        self.fabric.print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, model: nn.Module) -> nn.Module:
        """Load a model from a checkpoint.

        Args:
            path: Path to the checkpoint file.
            model: Model to load the state into.

        Returns:
            The model with loaded state.
        """
        checkpoint = self.fabric.load(path)

        # Allow callback to modify the checkpoint
        self.fabric.call("on_load_checkpoint", self, checkpoint)

        # Load model state
        _unwrap_objects(model).load_state_dict(checkpoint["model_state_dict"])
        self.fabric.print(f"Checkpoint loaded from {path}")

        return model

    def step_scheduler(
        self: "BPTrainer",
        model: nn.Module,
        scheduler: Union[LRScheduler, Dict[str, LRScheduler]],
        level: Literal["step", "epoch"],
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The model to train
            scheduler: The learning rate scheduler or dictionary of schedulers
            level: whether we are trying to step on epoch- or step-level
        """
        if scheduler is None:
            return

        if isinstance(scheduler, dict):
            for name, sched in scheduler.items():
                # For ReduceLROnPlateau
                if hasattr(sched, "step") and hasattr(sched, "mode"):
                    if level == "epoch":
                        # Get the appropriate metric for ReduceLROnPlateau
                        if hasattr(model, "validation_step"):
                            metric_value = self.logged_metrics.get("val_loss", None)
                        else:
                            metric_value = self.logged_metrics.get("train_loss", None)

                        if metric_value is not None:
                            sched.step(metric_value)
                # For other schedulers
                elif level == "epoch" and not hasattr(sched, "step_on_batch"):
                    sched.step()
                elif level == "step" and getattr(sched, "step_on_batch", False):
                    sched.step()
        else:
            # For ReduceLROnPlateau
            if hasattr(scheduler, "step") and hasattr(scheduler, "mode"):
                if level == "epoch":
                    # Get the appropriate metric
                    if hasattr(model, "validation_step"):
                        metric_value = self.logged_metrics.get("val_loss", None)
                    else:
                        metric_value = self.logged_metrics.get("train_loss", None)

                    if metric_value is not None:
                        scheduler.step(metric_value)
            # For other schedulers
            elif level == "epoch" and not hasattr(scheduler, "step_on_batch"):
                scheduler.step()
            elif level == "step" and getattr(scheduler, "step_on_batch", False):
                scheduler.step()


# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
            self.layer_2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
            self.output = nn.Linear(64, 1)

        def forward(self, x, *args):
            h = self.layer_1(x)
            h = self.layer_2(h)
            return self.output(h)

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
        def setup(self, *args, **kwargs):
            pass

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

    loss_function = nn.MSELoss()
    optimizers = optim.Adam(SimpleModel().parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizers, mode="min", factor=0.1, patience=2)
    trainer = BPTrainer(config)
    model = SimpleModel()
    data_module = CustomDataModule()

    # Train the model
    trainer.fit(model, data_module, loss_function, optimizers, scheduler)

    print(f"Training completed successfully. Model saved to {config.checkpoint_dir}")
