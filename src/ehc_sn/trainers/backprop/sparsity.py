from typing import Any, Dict, List, Optional, Tuple

from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from pydantic import Field
from torch import Tensor, nn, optim
from torch.nn import functional

from ehc_sn.losses import autoencoders as loss_fns
from ehc_sn.trainers import _base as base


# -------------------------------------------------------------------------------------------
class SparsityBPTrainerParams(base.TrainerParams):
    """
    Parameters for the SparsityBPtrainer model.
    """

    sparsity_target: float = Field(default=0.05, description="Target activation rate for sparsity")
    sparsity_weight: float = Field(default=0.1, description="Weight for sparsity loss term (beta)")


# -------------------------------------------------------------------------------------------
class SparsityBPTrainer(base.BaseTrainer):
    """
    Trainer class for sparsity backpropagation-based training.
    """

    # -----------------------------------------------------------------------------------
    def __init__(self, model: nn.Module, params: Optional[SparsityBPTrainerParams] = None):
        params = SparsityBPTrainerParams() if params is None else params
        super().__init__(model, params)
        self.loss_reconsturction = loss_fns.ReconstructionLoss()
        self.loss_sparsity = loss_fns.SparsityLoss(params.sparsity_target)
        self.sparsity_weight = params.sparsity_weight

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer to use for training the MLP.

        Returns:
            OptimizerLRSchedulerConfig: A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler_config = {"mode": "min", "factor": 0.2, "patience": 20, "min_lr": 5e-5}
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # -----------------------------------------------------------------------------------
    def on_before_batch_transfer(self, batch: Tuple[Tensor], dataloader_idx: int) -> Tuple[Tensor, Tensor]:
        return batch[0], batch[0]  # Target is the reconstruction

    # -----------------------------------------------------------------------------------
    def _common_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Any]:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss_reconstruction = self.loss_reconsturction(outputs, targets)
        loss_sparsity = self.loss_sparsity(outputs, targets)
        return {
            "loss": loss_reconstruction + self.sparsity_weight * loss_sparsity,
            "reconstruction_loss": loss_reconstruction,
            "sparsity_loss": loss_sparsity,
        }

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_nb: int) -> Tensor:
        """Defines a single training step for the MLP.

        Args:
            batch: A tuple containing the input data and target labels.
            batch_idx: The index of the current batch.

        Returns:
            (torch.Tensor): The training loss.
        """
        return self._common_step(batch, batch_nb)["loss"]

    # -----------------------------------------------------------------------------------
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_nb: int) -> None:
        """Defines a single validation step for the MLP.

        Args:
            batch : A tuple containing the input data and target labels.
            batch_idx : The index of the current batch.
        """
        loss = self._common_step(batch, batch_nb)["loss"]

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)

    # -----------------------------------------------------------------------------------
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_nb: int) -> None:
        """Defines a single testing step for the MLP.

        Args:
            batch : A tuple containing the input data and target labels.
            batch_idx : The index of the current batch.
        """
        loss = self._common_step(batch, batch_nb)["loss"]

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Model example
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = functional.relu(self.fc1(x))
            return self.fc2(x), x

    # Trainer example
    model = Model()  # Example model
    trainer_params = SparsityBPTrainerParams(max_epochs=10, min_epochs=1)
    trainer = SparsityBPTrainer(model, trainer_params)

    # DataLoader example
    class DummyData(pl.LightningDataModule):
        def __init__(self, batch_size: int = 32):
            super().__init__()
            self.batch_size = batch_size
            self.xy = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))

        def train_dataloader(self):
            return DataLoader(self.xy, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.xy, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.xy, batch_size=self.batch_size)

    # Fit the model
    datalmodule = DummyData(batch_size=32)
    trainer.fit(datalmodule)
