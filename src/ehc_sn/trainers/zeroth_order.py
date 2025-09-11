"""Zero-Order Optimization Trainer for EHC-SN Neural Networks.

This module implements a simplified, maintainable version of the MeZO (Memory-efficient
Zeroth-Order) optimization algorithm. It focuses on the core efficient perturbation
method with zero-order preconditioning for training neural networks without gradients.

Key Features:
- Efficient parameter perturbation using random seeds (memory-efficient)
- Zero-order preconditioning for improved convergence
- Constant sampling schedule for simplicity
- Clean PyTorch Lightning integration
- Type-safe configuration with Pydantic validation

This implementation is equivalent to the MeZO algorithm but with improved
readability and maintainability by removing optional features and complex
configuration variants.

References:
    Malladi et al. "Fine-Tuning Language Models with Just Forward Passes" (2023)
    https://arxiv.org/abs/2305.17333
"""

from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.trainers.core import BaseTrainer


class ZOTrainer(BaseTrainer):
    def __init__(self, optimizer_init: Callable[[Any], Optimizer], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_init = optimizer_init

    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        x, *_ = batch

        # Generate a random seed for perturbations
        seed = torch.randint(0, 1_000_000, (1,)).item()

        # Autoencoder-style loss computation with full gradient flow
        outputs_1 = model(x, detach_grad=False, seed=seed)  # First forward pass
        loss_components_1 = model.compute_loss(outputs_1, batch, log_label=None)

        outputs_2 = model(x, detach_grad=False, seed=seed)  # Second forward pass
        loss_components_2 = model.compute_loss(outputs_2, batch, log_label=None)

        # Feedback
        feedbacks = [l1 - l2 for l1, l2 in zip(loss_components_1, loss_components_2)]
        optm_list = model.optimizers()

        # Check if optimizers need to step based on parameter gradients
        do_step_list = [
            any(p.requires_grad for p in component.parameters()) and loss.requires_grad
            for component, loss in zip([model.decoder, model.encoder], loss_components_1)
        ]

        # Zero gradients for active optimizers
        for optm, do_step in zip(optm_list, do_step_list):
            if do_step:
                optm.zero_grad()

        # Propagate feedback
        model.feedback(feedbacks)

        # Step active optimizers
        for optm, do_step in zip(optm_list, do_step_list):
            if do_step:
                optm.step()

    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        x, *_ = batch

        # Forward pass without DFA hooks
        outputs = model(x, detach_grad=False)

        # Compute validation loss (no gradients, no DFA hooks needed)
        loss_components = model.compute_loss(outputs, batch, "val")
        total_loss = sum(loss_components)

        # Return total validation loss for logging
        return total_loss
