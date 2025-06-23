from typing import Any, Dict, List, Optional, Tuple

from torch import nn, optim
from torch.nn import functional

from ehc_sn.trainers import _base as base


class BPTrainerParams(base.TrainerParams):
    """
    Parameters for the BPtrainer model.
    """


class BPTrainer(base.BaseTrainer):
    """
    Trainer class for backpropagation-based training.
    """

    def __init__(self, model: nn.Module, params: Optional[BPTrainerParams] = None):
        super().__init__(model, BPTrainerParams() if params is None else params)
        self.loss_fn = functional.mse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.loss_fn(self.model(x), y)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    # Example usage
    model = nn.Linear(10, 1)  # Example model
    trainer_params = BPTrainerParams(max_epochs=10, min_epochs=1)
    trainer = BPTrainer(model, trainer_params)

    # DataLoader example
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Create a dummy dataset
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)
    # Fit the model
    trainer.fit(dataloader)
