from typing import List, Optional, Tuple, Union

import torch
from pytorch_lightning.callbacks import Callback
from torch import Tensor, cat


class InvertProb(Callback):
    """Callback to invert probabilities in the dataloader."""

    def __init__(self, positions: Optional[List[int]] = None):
        """
        Initialize the InvertProb callback.

        Args:
            positions: List of indices in the batch to invert probabilities (default is [0]).
        """
        super().__init__()
        self.positions = [0] if positions is None else positions

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Invert probabilities in the batch before training."""
        for position in self.positions:
            self._invert(batch, position)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Invert probabilities in the batch before validation."""
        for position in self.positions:
            self._invert(batch, position)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Invert probabilities in the batch before testing."""
        for position in self.positions:
            self._invert(batch, position)

    @staticmethod
    def _invert(batch: Union[Tensor, Tuple[Tensor, ...], List[Tensor]], position: int):
        """Invert a tensor in-place."""
        if isinstance(batch, Tensor) and position == 0:
            # In-place operation
            batch.mul_(-1).add_(1.0)  # In-place operations with PyTorch
        elif isinstance(batch, (tuple, list)) and len(batch) > position:
            # Ensure position exists before modifying
            if not isinstance(batch[position], Tensor):
                raise TypeError(f"Expected Tensor at position {position}, got {type(batch[position])}")
            batch[position].data = 1.0 - batch[position]
        elif isinstance(batch, (tuple, list)) and position >= len(batch):
            # Log warning or raise error
            print(f"Warning: Position {position} out of range for batch of length {len(batch)}")
            return


class AppendToBatch(Callback):
    """Callback to append input tensors to end of batch iterable."""

    def __init__(self, positions: Optional[List[int]] = None):
        """
        Initialize the AppendToBatch callback.

        Args:
            positions: List of indices in the batch to append the input tensor to (default is [0]).
        """
        super().__init__()
        self.position = [0] if positions is None else positions

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Convert input tensors to label tensors before training."""
        for position in self.position:
            self._append(batch, position)
            trainer.current_hook_args = (batch, batch_idx)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Convert input tensors to label tensors before validation."""
        for position in self.position:
            self._append(batch, position)
            trainer.current_hook_args = (batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Convert input tensors to label tensors before testing."""
        for position in self.position:
            self._append(batch, position)
            trainer.current_hook_args = (batch, batch_idx, dataloader_idx)

    @staticmethod
    def _append(batch: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]], position: int) -> None:
        if isinstance(batch, torch.Tensor):
            if position == 0:
                # Concatenate the original tensor with a clone (unsqueezed) in an in-place manner
                appended = torch.cat((batch, batch.clone().unsqueeze(0)), dim=0)
                # Resize and copy to preserve in-place semantics without using .data
                batch.resize_(appended.size())
                batch.copy_(appended)
        elif isinstance(batch, list):
            if batch:
                batch.append(batch[0].clone())
            else:
                print("Warning: The batch list is empty; cannot append.")
        elif isinstance(batch, tuple):
            # Tuples are immutable, so warn the user instead of modifying
            print("Warning: Encountered a tuple batch; appending operation skipped as tuples are immutable.")
        else:
            raise TypeError(f"Unsupported type {type(batch)} provided to _append.")


if __name__ == "__main__":
    import pytorch_lightning as pl

    # Example usage of the InvertProbCallback
    callback = InvertProb(positions=[0])  # Invert probabilities at position 0
    sample_batch = torch.tensor([0.1, 0.5, 0.9])  # Example probabilities
    sample_list_batch = [torch.tensor([0.1, 0.5, 0.9]), torch.tensor([0.2, 0.3, 0.4])]

    # Just for demonstration (not how it would actually be called in Lightning)
    print("Original tensor batch:", sample_batch)
    print("Original list batch:", sample_list_batch)
    print("Would be inverted during training")

    # Load into a PyTorch Lightning Trainer
    trainer = pl.Trainer(callbacks=[callback])
    print("Trainer created with InvertProbCallback.")
