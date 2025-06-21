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
        self.activations = []
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

    def on_train_end(self):
        """Remove hooks when training ends"""
        for hook in self.activation_hooks:
            hook.remove()


if __name__ == "__main__":
    # Example usage
    import lightning

    # Initialize Fabric
    fabric = lightning.Fabric(accelerator="cpu", devices=1)

    # Model configuration
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 10))

    # Create a simple dataset
    train_dataset = torch.utils.data.TensorDataset(torch.randn(100, 10))
    val_dataset = torch.utils.data.TensorDataset(torch.randn(20, 10))

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
    model = TrainModule(model, data_module, train_params)

    # Get the optimizer(s) from the LightningModule
    optimizer = model.configure_optimizers()

    # Get the training data loader from the LightningModule
    train_dataloader = model.train_dataloader()

    # Set up objects
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Training loop for a single epoch
    model.train()
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        fabric.backward(loss)
        optimizer.step()
