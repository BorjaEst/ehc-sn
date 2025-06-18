from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from ehc_sn.models.autoencoder import Autoencoder, SparseAutoencoder
from ehc_sn.training.loss import SparseLoss, sparse_loss


class TrainingParameters(BaseModel):
    learning_rate: float = Field(default=1e-3, description="Learning rate for the optimizer")
    weight_decay: float = Field(default=1e-5, description="Weight decay for the optimizer")


class TrainAutoencoder(pl.LightningModule):
    """Class for training autoencoders with PyTorch Lightning"""

    def __init__(
        self,
        model: Union[Autoencoder, SparseAutoencoder],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        parameters: Optional[TrainingParameters] = None,
    ):
        super().__init__()
        parameters = parameters or TrainingParameters()
        self.model = model

        # Store the dataloaders
        self.train_dataloader_obj = train_dataloader
        self.val_dataloader_obj = val_dataloader

        self.learning_rate = parameters.learning_rate
        self.weight_decay = parameters.weight_decay

        # Save hyperparameters automatically
        self.save_hyperparameters(ignore=["model", "train_dataloader_obj", "val_dataloader_obj"])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """Forward pass through the model"""
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer for training"""
        return optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train_dataloader(self):
        """Return the training dataloader"""
        return self.train_dataloader_obj

    def val_dataloader(self):
        """Return the validation dataloader"""
        if self.val_dataloader_obj:
            return self.val_dataloader_obj
        return None

    def _common_step(self, batch: Tensor) -> Tuple[SparseLoss, Tensor, Tensor]:
        """Common computation for training and validation steps"""
        x = batch  # Assuming batch is a single tensor for simplicity
        _, y_hat, activations = self(x)
        loss = sparse_loss(x, y_hat, activations, self.model.beta, self.model.sparsity)
        return loss, x, y_hat

    def training_step(self, batch: Tensor) -> Tensor:
        """Lightning training step"""
        loss, x, y_hat = self._common_step(batch)

        # Log metrics
        self.log("train_loss", loss.total, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_reconstruction", loss.reconstruction, on_epoch=True)
        self.log("train_sparsity", loss.sparsity, on_epoch=True)

        return loss.total

    def validation_step(self, batch: Tensor) -> Tensor:
        """Lightning validation step"""
        loss, x, y_hat = self._common_step(batch)

        # Log metrics
        self.log("val_loss", loss.total, on_epoch=True, prog_bar=True)
        self.log("val_reconstruction", loss.reconstruction, on_epoch=True)
        self.log("val_sparsity", loss.sparsity, on_epoch=True)

        return loss.total

    def test_step(self, batch: Tensor) -> Tensor:
        """Lightning test step"""
        loss, x, y_hat = self._common_step(batch)

        # Log metrics
        self.log("test_loss", loss.total, on_epoch=True)
        self.log("test_reconstruction", loss.reconstruction, on_epoch=True)
        self.log("test_sparsity", loss.sparsity, on_epoch=True)

        return loss.total
