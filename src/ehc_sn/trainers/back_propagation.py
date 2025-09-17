"""Backpropagation-based training strategies for entorhinal-hippocampal circuit models.

This module provides training strategies that implement backpropagation variants
specifically designed for autoencoder architectures in spatial navigation modeling.
The strategies support both standard full gradient flow and detached gradient
training for independent component optimization.

Classes:
    ClassicTrainer: Standard backpropagation with full gradient flow between components.
    DetachedTrainer: Split training with detached gradients for independent optimization.

The training strategies are designed to work with PyTorch Lightning modules and
support the dual-optimizer pattern used in encoder-decoder architectures.

Example:
    >>> from functools import partial
    >>> import torch.optim as optim
    >>>
    >>> # Standard backpropagation training
    >>> classic_trainer = ClassicTrainer(
    ...     optimizer_init=partial(optim.Adam, lr=1e-3)
    ... )
    >>>
    >>> # Detached gradient training for independent component optimization
    >>> detached_trainer = DetachedTrainer(
    ...     optimizer_init=partial(optim.Adam, lr=1e-3)
    ... )
"""

from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.core.trainer import BaseTrainer


# -------------------------------------------------------------------------------------------
class BackwardTrainer(BaseTrainer):

    # -----------------------------------------------------------------------------------
    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:

        sensors, *_ = batch

        # Clear any previous gradients from model parameters
        optimizers = model.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for optm in optimizers:
            optm.zero_grad()

        # Forward pass and loss computation
        outputs = model(sensors)
        loss_components = model.compute_loss(outputs, batch)

        # Calculate gradients using full backpropagation
        total_loss = sum(loss_components)
        model.manual_backward(total_loss)

        # Optimizer step
        for optm in optimizers:
            optm.step()
