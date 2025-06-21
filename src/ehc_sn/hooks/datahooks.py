import torch
from pytorch_lightning.callbacks import Callback


class InvertProbCallback(Callback):
    """Callback to invert probabilities in the dataloader."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Invert probabilities in the batch before training."""
        if isinstance(batch, torch.Tensor):
            batch = 1.0 - batch
        return batch

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Invert probabilities in the batch before validation."""
        if isinstance(batch, torch.Tensor):
            batch = 1.0 - batch
        return batch

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Invert probabilities in the batch before testing."""
        if isinstance(batch, torch.Tensor):
            batch = 1.0 - batch
        return batch


class InputToLabel(Callback):
    """Callback to convert input tensors to label tensors."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Convert input tensors to label tensors before training."""
        if isinstance(batch, torch.Tensor):
            # Assuming the first element is the input and the second is the label
            batch = (batch[0], batch[0])  # Example conversion
        return batch

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Convert input tensors to label tensors before validation."""
        if isinstance(batch, torch.Tensor):
            batch = (batch[0], batch[0])  # Example conversion
        return batch

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Convert input tensors to label tensors before testing."""
        if isinstance(batch, torch.Tensor):
            batch = (batch[0], batch[0])  # Example conversion
        return batch


if __name__ == "__main__":
    import pytorch_lightning as pl

    # Example usage of the InvertProbCallback
    callback = InvertProbCallback()
    sample_batch = torch.tensor([0.1, 0.5, 0.9])  # Example probabilities

    # Just for demonstration (not how it would actually be called in Lightning)
    print("Original batch:", sample_batch)
    print("Would be inverted during training")

    # Load into a PyTorch Lightning Trainer
    trainer = pl.Trainer(callbacks=[callback])
    print("Trainer created with InvertProbCallback.")
