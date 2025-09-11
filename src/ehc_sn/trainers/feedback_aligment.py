"""Direct Feedback Alignment (DFA) training strategies for biologically plausible learning.

This module implements DFA-based training strategies where hidden layers receive
direct feedback from the output error via fixed random feedback weights, eliminating
the need for symmetric weight transport. This provides a biologically plausible
alternative to standard backpropagation for neural network training.

The DFA approach addresses the weight transport problem in biological neural networks
by using random feedback connections that don't require knowledge of forward weights.
Despite using random projections, DFA can achieve effective learning through the
feedback alignment principle.

Key Features:
    - Centralized hook management for error signal capture and cleanup
    - Manual optimization with precise control over gradient computation
    - Registry-based error signal management for multi-layer coordination
    - Biologically plausible learning without symmetric weight constraints

Classes:
    DFATrainer: Complete DFA training strategy with hook management and optimization.

Examples:
    >>> from functools import partial
    >>> import torch.optim as optim
    >>>
    >>> # Create DFA trainer with Adam optimization
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
            optimizer_init: Factory function for creating optimizer instances.
                Should be a callable that takes model parameters as input and
                returns an optimizer. Typically a partial function like
                `partial(torch.optim.Adam, lr=1e-3)` or a lambda function.
            *args: Additional positional arguments passed to parent class.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
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

        # Clear any previous DFA error signals from registry
        registry.clear("batch")

        # Autoencoder-style loss computation with full gradient flow
        outputs = model(x, detach_grad=False)  # Forward pass
        loss_components = model.compute_loss(outputs, batch, "train")
        total_loss = sum(loss_components)  # Combine losses
        optm_list = model.optimizers()

        # Check if optimizers need to step based on parameter gradients
        do_step_list = [
            any(p.requires_grad for p in component.parameters()) and loss.requires_grad
            for component, loss in zip([model.decoder, model.encoder], loss_components)
        ]

        # Register DFA hook on the reconstruction tensor to capture error signal
        reconstruction, _ = outputs  # Get reconstruction from model outputs
        errors = model.get_output_error(outputs, batch)
        hook_remover = registry.register_output_error_hook(reconstruction, key="global")

        # Zero gradients for active optimizers
        for optm, do_step in zip(optm_list, do_step_list):
            if do_step:
                optm.zero_grad()

        # Backward pass - DFA gradients are computed automatically via registered hooks
        model.manual_backward(total_loss)

        # Step active optimizers
        for optm, do_step in zip(optm_list, do_step_list):
            if do_step:
                optm.step()

        # Clean up: remove the hook and clear error signals
        if hook_remover is not None:
            hook_remover()
        registry.clear("batch")

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

        # Forward pass without DFA hooks
        outputs = model(x, detach_grad=False)

        # Compute validation loss (no gradients, no DFA hooks needed)
        loss_components = model.compute_loss(outputs, batch, "val")
        total_loss = sum(loss_components)

        # Return total validation loss for logging
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
