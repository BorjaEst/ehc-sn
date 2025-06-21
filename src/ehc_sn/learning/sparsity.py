import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader


class TrainParams(BaseModel):

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    learning_rate: float = Field(default=1e-3, description="Learning rate for the optimizer")
    weight_decay: float = Field(default=1e-5, description="Weight decay for the optimizer")
    sparsity_target: float = Field(default=0.05, description="Target activation rate for sparsity")
    sparsity_weight: float = Field(default=0.1, description="Weight for sparsity loss term (beta)")


@dataclasses.dataclass
class SparseLoss:
    """Loss dataclass for Kullback–Leibler (KL) Divergence for Sparse Activations."""

    sparsity: Tensor
    output: Tensor
    weight: float

    def __init__(self, y: Tensor, y_: Tensor, activations: List[Tensor], params: TrainParams):
        self.sparsity = sum(self.kl_divergence(x.mean(dim=0), params.sparsity_target) for x in activations)
        self.output = nn.functional.mse_loss(y, y_)
        self.weight = params.sparsity_weight

    @property
    def total(self) -> float:
        return self.output + self.weight * self.sparsity

    @staticmethod
    def kl_divergence(input: Tensor, target: float, eps=1e-10) -> Tensor:
        kl = input * torch.log((input + eps) / (target + eps))
        kl += (1 - input) * torch.log((1 - input + eps) / (1 - target + eps))
        return torch.sum(kl)


class TrainModule(pl.LightningModule):
    """Class for training sparse autoencoders with PyTorch Lightning"""

    def __init__(
        self,
        model: nn.Module,
        data_module: pl.LightningDataModule,
        params: Optional[TrainParams] = None,
    ):
        super().__init__()
        self.params = params or TrainParams()

        # Store the model and the data module
        self.model = model
        self.data_module = data_module

        # Setup activation hooks and storage
        self.activation_hooks = []
        self.activations = []
        self._register_activation_hooks()

        # Save hyperparameters automatically
        hyper_params = self.params.model_dump(exclude_none=True)
        self.save_hyperparameters(hyper_params)

    def _register_activation_hooks(self):
        """Register hooks to capture activations from ReLU layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):  # Warning: Assumes ReLU
                hook = module.register_forward_hook(self._activation_hook)
                self.activation_hooks.append(hook)

    def _activation_hook(self, module, input, output):
        """Store activations from forward pass"""
        self.activations.append(output.detach())

    def _clear_activations(self):
        """Clear stored activations"""
        self.activations = []

    def _remove_hooks(self):
        """Remove all registered activation hooks"""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model"""
        return self.model(x)

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer for training"""
        return optim.Adam(
            params=self.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )

    def _common_step(self, batch: list[Tensor]) -> Tuple[SparseLoss, Tensor, Tensor]:
        """Common computation for training and validation steps"""
        # Clear previous activations
        self._clear_activations()
        inputs = batch[0]  # Dataset.__getitem__ -> Tuple[Input, Target, ...]

        # Forward pass
        outputs = self(inputs)
        loss = SparseLoss(inputs, outputs, self.activations, self.params)
        return loss, inputs, outputs

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Lightning training step"""
        loss, _inputs, _outputs = self._common_step(batch)

        # Log metrics
        self.log("train_loss", loss.total, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss.output", loss.output, on_epoch=True)
        self.log("train_loss.sparsity", loss.sparsity, on_epoch=True)

        return loss.total

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Lightning validation step"""
        loss, _inputs, _outputs = self._common_step(batch)

        # Log metrics
        self.log("val_loss", loss.total, on_epoch=True, prog_bar=True)
        self.log("val_loss.output", loss.output, on_epoch=True)
        self.log("val_loss.sparsity", loss.sparsity, on_epoch=True)

        return loss.total

    def test_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Lightning test step"""
        loss, _inputs, _outputs = self._common_step(batch)

        # Log metrics
        self.log("test_loss", loss.total, on_epoch=True)
        self.log("test_loss.output", loss.output, on_epoch=True)
        self.log("test_loss.sparsity", loss.sparsity, on_epoch=True)

        return loss.total

    def train_dataloader(self):
        """Get train dataloader from the data module"""
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        """Get validation dataloader from the data module"""
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        """Get test dataloader from the data module"""
        return self.data_module.test_dataloader()

    def on_train_start(self):
        """Register hooks when training starts"""
        self._register_activation_hooks()

    def on_train_end(self):
        """Remove hooks when training ends"""
        self._remove_hooks()

    def on_validation_start(self):
        """Register hooks when validation starts"""
        self._register_activation_hooks()

    def on_validation_end(self):
        """Remove hooks when validation ends"""
        self._remove_hooks()

    def on_test_start(self):
        """Register hooks when test starts"""
        self._register_activation_hooks()

    def on_test_end(self):
        """Remove hooks when test ends"""
        self._remove_hooks()


if __name__ == "__main__":
    # Example usage
    import lightning
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    print("Initializing Sparse Training Example...")

    # Initialize Fabric
    fabric = lightning.Fabric(accelerator="cpu", devices=1)
    print(f"Using device: {fabric.device}")

    # Model configuration
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 15), nn.ReLU(), nn.Linear(15, 10))
    print(f"Model architecture:\n{model}")

    # Create a simple dataset
    print("Creating datasets...")
    train_dataset = torch.utils.data.TensorDataset(torch.randn(100, 10))
    val_dataset = torch.utils.data.TensorDataset(torch.randn(20, 10))
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Create a data module
    class SimpleDataModule(pl.LightningDataModule):
        def __init__(self, train_dataset, val_dataset):
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=16)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=16)

    data_module = SimpleDataModule(train_dataset, val_dataset)

    # Instantiate the LightningModule with sparsity parameters
    train_params = TrainParams(sparsity_target=0.05, sparsity_weight=0.1)
    print(f"Training parameters: {train_params.model_dump()}")

    model = TrainModule(model, data_module, train_params)

    # Get the optimizer(s) from the LightningModule
    optimizer = model.configure_optimizers()
    print(f"Optimizer: {optimizer}")

    # Get the dataloaders from the LightningModule
    train_dataloader = model.train_dataloader()
    val_dataloader = model.val_dataloader()

    # Set up objects
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    # Track metrics for plotting
    train_losses = []
    val_losses = []
    epochs = 5

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Training loop
        model.train()
        epoch_train_losses = []
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Call lifecycle methods manually
        model.on_train_start()

        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            optimizer.zero_grad()
            loss = model.training_step(batch, i)
            fabric.backward(loss)
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.on_train_end()

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        print(f"Avg train loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        epoch_val_losses = []

        # Call lifecycle methods manually
        model.on_validation_start()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
                val_loss = model.validation_step(batch, i)
                epoch_val_losses.append(val_loss.item())

        model.on_validation_end()

        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        print(f"Avg val loss: {avg_val_loss:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, "b-", label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("sparse_training_loss.png")
    print(f"Loss plot saved to sparse_training_loss.png")

    print("\nTraining complete!")
