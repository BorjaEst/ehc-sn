"""Direct Feedback Alignment (DFA) training strategies.

This module implements DFA-based training where hidden layers receive
direct feedback from the output error via random feedback weights,
eliminating the need for symmetric weight transport.
"""

from typing import Any, Callable

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.hooks.registry import registry
from ehc_sn.trainers.core import BaseTrainer


class DFATrainer(BaseTrainer):
    """DFA trainer implementing Direct Feedback Alignment with centralized hook management.

    This trainer manages the complete DFA training process including:
    - Hook registration and cleanup for error signal capture
    - Manual optimization with proper gradient computation
    - Centralized management of DFA error signals via registry system

    The trainer follows the DFA algorithm where hidden layers receive direct
    feedback from the output error via random feedback weights, eliminating
    the need for symmetric weight transport and providing biologically
    plausible learning.

    Key features:
    - Centralized hook management (no hooks in model forward methods)
    - Manual optimization for precise control over DFA gradient flow
    - Automatic error signal cleanup between batches
    - Compatible with PyTorch Lightning training loops

    Args:
        optimizer_init: Function to initialize optimizers, typically a functools.partial
            of an optimizer class with preset hyperparameters.

    Example:
        >>> from functools import partial
        >>> import torch.optim as optim
        >>>
        >>> trainer = DFATrainer(
        ...     optimizer_init=partial(optim.Adam, lr=1e-3, weight_decay=1e-4)
        ... )
        >>> model = Autoencoder(params, trainer)
        >>> lightning_trainer = pl.Trainer(max_epochs=100)
        >>> lightning_trainer.fit(model, dataloader)

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, optimizer_init: Callable[[Any], Optimizer], *args: Any, **kwargs: Any) -> None:
        """Initialize DFA trainer with optimizer configuration.

        Args:
            optimizer_init: Function to create optimizers, should accept model parameters
                and return configured optimizer instance.
            *args: Additional positional arguments (currently unused).
            **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.optimizer_init = optimizer_init

    # -----------------------------------------------------------------------------------
    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        """Execute one DFA training step with manual optimization.

        This method implements the complete DFA training procedure:
        1. Clear any previous error signals from registry
        2. Perform forward pass through the model
        3. Register DFA hook on model output to capture error signal
        4. Compute loss and perform backward pass (automatically uses DFA gradients)
        5. Update model parameters via optimizer step

        Args:
            model: The PyTorch Lightning module being trained. Should contain
                encoder and decoder components with DFA layers.
            batch: Training batch data. First element should be input tensor;
                additional elements are ignored.
            batch_idx: Index of the current batch within the epoch.

        Note:
            This method uses manual optimization to maintain precise control over
            the DFA gradient computation and error signal management. The model's
            automatic_optimization should be set to False when using this trainer.
        """
        x, *_ = batch

        # Get optimizers configured by Lightning
        optimizers = model.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Clear any previous DFA error signals from registry
        registry.clear("batch")

        # Single forward pass to get reconstruction and embedding
        # Note: Autoencoder forward returns [reconstruction, embedding]
        outputs = model(x)
        reconstruction, embedding = outputs[0], outputs[1]

        # Register DFA hook on the reconstruction tensor to capture error signal
        hook_remover = None
        if reconstruction.requires_grad:
            hook_remover = registry.register_output_error_hook(reconstruction, key="global")

        # Manually compute losses using the reconstruction and embedding we just obtained
        # This ensures DFA hook is connected to the same tensors used in loss computation

        # Encoder-side losses (same as in model.compute_loss)
        gramian_loss = model.gramian_loss(embedding)
        model.log("train/gramian_loss", gramian_loss, on_epoch=True)
        homeo_loss = model.homeo_loss(embedding)
        model.log("train/homeostatic_loss", homeo_loss, on_epoch=True)
        encoder_loss = model.gramian_weight * gramian_loss + model.homeo_weight * homeo_loss
        model.log("train/encoder_loss", encoder_loss, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        sparsity_rate = (embedding > 0.01).float().mean()
        model.log("train/sparsity_rate", sparsity_rate, on_epoch=True)

        # Decoder-side losses (same as in model.compute_loss)
        reconstruction_loss = model.reconstruction_loss(reconstruction, x)
        model.log("train/reconstruction_loss", reconstruction_loss, on_epoch=True)
        decoder_loss = reconstruction_loss
        model.log("train/decoder_loss", decoder_loss, on_epoch=True, prog_bar=True)

        # Combine losses (same as in training step)
        loss_components = [decoder_loss, encoder_loss]
        total_loss = sum(loss_components)

        # Zero gradients before backward pass
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Backward pass - DFA gradients are computed automatically via registered hooks
        model.manual_backward(total_loss)

        # Update parameters
        for optimizer in optimizers:
            optimizer.step()

        # Clean up: remove the hook and clear error signals
        if hook_remover is not None:
            hook_remover()
        registry.clear("batch")

        # Log training metrics
        model.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for i, loss_component in enumerate(loss_components):
            model.log(f"train_loss_component_{i}", loss_component, on_step=False, on_epoch=True)

    # -----------------------------------------------------------------------------------
    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        """Execute validation step with standard gradient flow for loss computation.

        Performs model evaluation on validation data using the same forward pass
        and loss computation as training, but without parameter updates or DFA
        hook registration. This ensures validation metrics are computed consistently.

        Args:
            model: The PyTorch Lightning module being validated. Should be the
                same model used in training with encoder and decoder components.
            batch: Validation batch containing input data. First element should
                be the input tensor; additional elements are ignored.
            batch_idx: Index of the current validation batch. Used for logging
                and potential batch-specific operations.

        Returns:
            Combined validation loss as a single tensor. This represents the
            sum of all loss components (reconstruction, regularization, etc.)
            that would be used for training optimization.

        Note:
            No gradient computation, hook registration, or parameter updates occur
            during validation. The method maintains the same loss computation logic
            as training to ensure consistent evaluation metrics.
        """
        x, *_ = batch

        # Forward pass without DFA hooks (evaluation mode)
        with torch.no_grad():
            reconstruction = model(x)

        # Compute validation loss (no gradients, no DFA hooks needed)
        loss_components = model.compute_loss(x, "val", detach_grad=False)
        total_loss = sum(loss_components)

        # Log validation metrics
        model.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, loss_component in enumerate(loss_components):
            model.log(f"val_loss_component_{i}", loss_component, on_step=False, on_epoch=True)

        return total_loss

    # -----------------------------------------------------------------------------------
    def on_train_batch_start(self, model: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        """Hook called before training batch starts.

        Ensures clean state at the beginning of each training batch by clearing
        any residual error signals or activations from the registry. This prevents
        cross-batch contamination of DFA error signals.

        Args:
            model: The Lightning module being trained.
            batch: Training batch data.
            batch_idx: Index of the current batch.
        """
        # Clear any residual error signals and activations from previous batches
        registry.clear("batch")

    # -----------------------------------------------------------------------------------
    def on_train_batch_end(self, model: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Hook called after training batch ends.

        Performs cleanup operations to ensure no residual state remains after
        batch processing. This is important for memory management and preventing
        cross-batch interference in DFA training.

        Args:
            model: The Lightning module being trained.
            outputs: Training step outputs.
            batch: Training batch data.
            batch_idx: Index of the current batch.
        """
        # Clean up any remaining error signals and activations
        registry.clear("batch")
