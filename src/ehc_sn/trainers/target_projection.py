"""Direct Random Target Projection (DRTP) training strategies for biologically plausible learning.

This module implements DRTP-based training strategies where hidden layers receive
direct error signals via fixed random projection matrices from target outputs,
eliminating the need for symmetric weight transport. This provides a biologically
plausible alternative to standard backpropagation for neural network training.

The DRTP approach addresses the weight transport problem in biological neural networks
by using fixed random projection matrices to propagate target signals directly to
hidden layers, enabling local learning without requiring knowledge of forward weights.

Key Features:
    - Direct target projection using fixed random matrices
    - Manual optimization with precise control over gradient computation
    - Frozen encoder support for decoder-only training scenarios
    - Biologically plausible learning without symmetric weight constraints

Classes:
    DRTPTrainer: Complete DRTP training strategy with target projection management.

Examples:
    >>> from functools import partial
    >>> import torch.optim as optim
    >>>
    >>> # Create DRTP trainer with Adam optimization
    >>> trainer = DRTPTrainer(
    ...     optimizer_init=partial(optim.Adam, lr=1e-3, weight_decay=1e-4)
    ... )
    >>> model = Autoencoder(params, trainer)
    >>> lightning_trainer = pl.Trainer(max_epochs=100)
    >>> lightning_trainer.fit(model, dataloader)

References:
    Nøkland, A. (2016). Direct feedback alignment provides learning in deep neural
    networks without loss gradients. arXiv preprint arXiv:1609.01596.
"""

from typing import Any, Callable, Optional

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.trainers.core import BaseTrainer


class DRTPTrainer(BaseTrainer):
    """DRTP trainer implementing Direct Random Target Projection with manual optimization.

    This trainer manages the complete DRTP training process including:
    - Target signal propagation via fixed random projection matrices
    - Manual optimization with proper gradient computation control
    - Support for frozen encoder training scenarios
    - Selective optimizer stepping based on parameter gradients

    The trainer follows the DRTP algorithm where target signals are projected
    directly to hidden layers via fixed random matrices, eliminating the need
    for symmetric weight transport and providing biologically plausible learning.

    Key features:
    - Direct target projection using random feedback matrices
    - Manual optimization for precise control over DRTP gradient flow
    - Selective component training (supports frozen encoders)
    - Compatible with PyTorch Lightning training loops

    Args:
        optimizer_init: Factory function for creating optimizers. Should be a partial
            function that takes model parameters and returns an optimizer instance.

    Example:
        >>> from functools import partial
        >>> import torch.optim as optim
        >>>
        >>> trainer = DRTPTrainer(
        ...     optimizer_init=partial(optim.Adam, lr=1e-3, weight_decay=1e-4)
        ... )
        >>> model = Autoencoder(params, trainer)
        >>> lightning_trainer = pl.Trainer(max_epochs=100)
        >>> lightning_trainer.fit(model, dataloader)

    References:
        Nøkland, A. (2016). Direct feedback alignment provides learning in deep neural
        networks without loss gradients. arXiv preprint arXiv:1609.01596.
    """

    def __init__(self, optimizer_init: Callable[[Any], Optimizer], *args, **kwds) -> None:
        """Initialize DRTP trainer with optimizer configuration.

        Args:
            optimizer_init: Factory function for creating optimizers.
            *args: Additional positional arguments passed to base trainer.
            **kwds: Additional keyword arguments passed to base trainer.
        """
        super().__init__(*args, **kwds)
        self.optimizer_init = optimizer_init

    # -----------------------------------------------------------------------------------
    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        """Execute one DRTP training step with target projection.

        Performs forward pass with target signal propagation, computes losses,
        and applies manual optimization with selective component updates based
        on parameter gradients and training requirements.

        The training step uses the input as the target signal, which is projected
        through fixed random matrices to hidden layers during the backward pass.
        This eliminates the need for symmetric weight transport.

        Args:
            model: The Lightning module being trained (should be an Autoencoder).
            batch: Training batch data, expected to be a tuple with input tensor first.
            batch_idx: Index of the current batch (unused but required by interface).

        Note:
            This method uses manual optimization and will handle gradient zeroing,
            backward pass, and optimizer stepping internally. The method supports
            scenarios where some components (e.g., encoder) may be frozen.
        """
        x, *_ = batch

        # Forward pass with frozen encoder and DRTP decoder
        outputs = model(x, detach_grad=False)  # Use input as target
        loss_components = model.compute_loss(outputs, batch, "train")
        total_loss = sum(loss_components)  # Combine losses
        optm_list = model.optimizers()

        # Check if optimizers need to step based on parameter gradients
        do_step_list = [
            any(p.requires_grad for p in component.parameters()) and loss.requires_grad
            for component, loss in zip([model.decoder, model.encoder], loss_components)
        ]

        # Check if optimizers need to step based on parameter gradients
        do_step_list = [
            any(p.requires_grad for p in component.parameters()) and loss.requires_grad
            for component, loss in zip([model.decoder, model.encoder], loss_components)
        ]

        # Zero gradients for active optimizers
        for optm, do_step in zip(optm_list, do_step_list):
            if do_step:
                optm.zero_grad()

        # Backward pass
        model.manual_backward(total_loss)

        # Step active optimizers
        for optm, do_step in zip(optm_list, do_step_list):
            if do_step:
                optm.step()

    # -----------------------------------------------------------------------------------
    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        """Execute one DRTP validation step.

        Performs forward pass without target signals and computes validation
        losses for monitoring training progress. No parameter updates are
        performed during validation.

        Args:
            model: The Lightning module being validated (should be an Autoencoder).
            batch: Validation batch data, expected to be a tuple with input tensor first.
            batch_idx: Index of the current batch (unused but required by interface).

        Returns:
            Total validation loss tensor for logging and monitoring purposes.

        Note:
            During validation, target=None is used to avoid unnecessary DRTP
            computations since no gradient updates will be performed.
        """
        x, *_ = batch

        # Forward pass without DFA hooks
        outputs = model(x, detach_grad=False)

        # Compute validation loss (no gradients, no DFA hooks needed)
        loss_components = model.compute_loss(outputs, batch, "val")
        total_loss = sum(loss_components)

        # Return total validation loss for logging
        return total_loss
