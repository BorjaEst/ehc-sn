from typing import Any, Callable, List

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.hooks.registry import registry


# -------------------------------------------------------------------------------------------
class DFATrainer(BaseTrainer):

    def __init__(self, optimizer_init: Callable[[Any], Optimizer], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_init = optimizer_init

    # -----------------------------------------------------------------------------------
    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
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
        # Clear any residual error signals and activations from previous batches
        registry.clear("batch")

    # -----------------------------------------------------------------------------------
    def on_train_batch_end(self, model: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        # Clean up any remaining error signals and activations
        registry.clear("batch")


# -------------------------------------------------------------------------------------------
class FeedbackTainer(BaseTrainer):

    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        sensors, *_ = batch

        # Clear any previous gradients from model parameters
        optimizers = model.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        
        for optm in optimizers:
            optm.zero_grad()

        # Forward pass and compute feedback signals
        outputs = model(sensors)
        feedback = model.compute_feedback(outputs, batch)

        # Calculate gradients using feedback signals
        model.apply_feedback(feedback)

        # Optimizer step
        for optm in optimizers:
            optm.step()
