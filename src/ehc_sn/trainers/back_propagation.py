"""Backpropagation-based training strategies.

This module implements different variants of backpropagation training,
including standard full backpropagation and detached gradient training
for autoencoder architectures.
"""

from typing import List, Optional

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.trainers.core import BaseTrainer


class ClassicTrainer(BaseTrainer):
    """Standard backpropagation trainer with full gradient flow.

    Implements classic backpropagation where gradients flow through
    the entire model architecture. This is the standard training
    approach for most neural networks.

    Features:
    - Full gradient flow between all model components
    - Single or dual optimizer support
    - Standard PyTorch Lightning optimization pattern
    """

    def __init__(self, optimizer_init, *args, **kwargs):
        """Initialize classic backpropagation trainer.

        Args:
            optimizer_init: Function to initialize optimizers.
                Should be a partial function or lambda that takes model
                parameters and returns an optimizer instance.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.optimizer_init = optimizer_init

    def configure_optimizers(self, model: pl.LightningModule) -> List[Optimizer]:
        """Configure optimizers for standard backpropagation.

        Creates optimizers based on model architecture. For autoencoders
        with separate encoder/decoder components, creates dual optimizers.
        For unified models, creates a single optimizer.

        Args:
            model: The Lightning module to configure optimizers for.

        Returns:
            List of optimizers (1-2 optimizers depending on model structure).
        """
        if hasattr(model, "encoder") and hasattr(model, "decoder"):
            # Dual optimizers for encoder/decoder architecture
            enc_opt = self.optimizer_init(model.encoder.parameters())
            dec_opt = self.optimizer_init(model.decoder.parameters())
            return [enc_opt, dec_opt]
        else:
            # Single optimizer for unified model
            return [self.optimizer_init(model.parameters())]

    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        """Standard backpropagation training step with full gradient flow.

        Performs forward pass, computes losses, and applies gradients
        with full gradient flow between all model components.

        Args:
            model: The Lightning module being trained.
            batch: Training batch data.
            batch_idx: Index of the current batch.
        """
        x, *_ = batch

        if hasattr(model, "compute_loss"):
            # Autoencoder-style loss computation with full gradient flow
            dec_loss, enc_loss = model.compute_loss(x, "train", detach_gradients=False)
            self._train_full_flow(model, enc_loss, dec_loss)
        else:
            # Standard model loss computation
            loss = model(x)
            if hasattr(loss, "loss"):
                loss = loss.loss
            opt = model.optimizers()[0]
            opt.zero_grad()
            model.manual_backward(loss)
            opt.step()

    def _train_full_flow(self, model: pl.LightningModule, enc_loss: Tensor, dec_loss: Tensor) -> None:
        """Train with full gradient flow between encoder and decoder.

        Combines encoder and decoder losses and performs a single backward
        pass, allowing gradients to flow through the entire model.

        Args:
            model: The Lightning module being trained.
            enc_loss: Encoder loss (sparsity/regularization).
            dec_loss: Decoder loss (reconstruction).
        """
        enc_opt, dec_opt = model.optimizers()

        # Check if components need training
        enc_do_step = any(p.requires_grad for p in model.encoder.parameters()) and enc_loss.requires_grad
        dec_do_step = any(p.requires_grad for p in model.decoder.parameters()) and dec_loss.requires_grad

        if not (enc_do_step or dec_do_step):
            return

        # Zero gradients for active optimizers
        if enc_do_step:
            enc_opt.zero_grad()
        if dec_do_step:
            dec_opt.zero_grad()

        # Combined loss with full gradient flow
        total_loss = enc_loss + dec_loss
        model.manual_backward(total_loss)

        # Step active optimizers
        if enc_do_step:
            enc_opt.step()
        if dec_do_step:
            dec_opt.step()

    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        """Standard validation step.

        Performs forward pass and loss computation without parameter updates.

        Args:
            model: The Lightning module being validated.
            batch: Validation batch data.
            batch_idx: Index of the current batch.

        Returns:
            Combined validation loss.
        """
        x, *_ = batch
        if hasattr(model, "compute_loss"):
            dec_loss, enc_loss = model.compute_loss(x, "val", detach_gradients=False)
            return dec_loss + enc_loss
        else:
            return model(x)


class DetachedTrainer(BaseTrainer):
    """Detached gradient trainer for split training strategies.

    Implements training where gradients between model components are
    detached, allowing independent optimization of different parts.
    This is particularly useful for autoencoder architectures where
    encoder and decoder can be trained with different objectives.

    Features:
    - Detached gradients between encoder and decoder
    - Independent optimization of model components
    - Support for different loss functions per component
    """

    def __init__(self, optimizer_init, *args, **kwargs):
        """Initialize detached gradient trainer.

        Args:
            optimizer_init: Function to initialize optimizers.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.optimizer_init = optimizer_init

    def configure_optimizers(self, model: pl.LightningModule) -> List[Optimizer]:
        """Configure optimizers for detached training.

        Creates separate optimizers for encoder and decoder components
        to enable independent optimization.

        Args:
            model: The Lightning module to configure optimizers for.

        Returns:
            List containing encoder and decoder optimizers.
        """
        if hasattr(model, "encoder") and hasattr(model, "decoder"):
            enc_opt = self.optimizer_init(model.encoder.parameters())
            dec_opt = self.optimizer_init(model.decoder.parameters())
            return [enc_opt, dec_opt]
        else:
            # Fallback for models without explicit encoder/decoder
            return [self.optimizer_init(model.parameters())]

    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        """Detached gradient training step.

        Performs forward pass with detached gradients and independent
        optimization of encoder and decoder components.

        Args:
            model: The Lightning module being trained.
            batch: Training batch data.
            batch_idx: Index of the current batch.
        """
        x, *_ = batch

        if hasattr(model, "compute_loss"):
            # For autoencoder models, compute losses with detached gradients
            # Use the forward method with detach_gradients=True for proper separation
            reconstruction, embedding = model.forward(x, detach_gradients=True)

            # Compute losses separately
            reconstruction_loss = model.reconstruction_loss(reconstruction, x)
            gramian_loss = model.gramian_loss(embedding)
            homeo_loss = model.homeo_loss(embedding)

            # Decoder loss (reconstruction only)
            dec_loss = reconstruction_loss

            # Encoder loss (sparsity constraints only)
            enc_loss = model.gramian_weight * gramian_loss + model.homeo_weight * homeo_loss

            # Log losses
            model.log("train/reconstruction_loss", reconstruction_loss, on_epoch=True)
            model.log("train/gramian_loss", gramian_loss, on_epoch=True)
            model.log("train/homeostatic_loss", homeo_loss, on_epoch=True)
            model.log("train/decoder_loss", dec_loss, on_epoch=True, prog_bar=True)
            model.log("train/encoder_loss", enc_loss, on_epoch=True, prog_bar=True)

            # Log sparsity metrics
            sparsity_rate = (embedding > 0.01).float().mean()
            model.log("train/sparsity_rate", sparsity_rate, on_epoch=True)

            self._train_detached(model, enc_loss, dec_loss)
        else:
            # Fallback for models without explicit loss computation
            loss = model(x)
            if hasattr(loss, "loss"):
                loss = loss.loss
            opt = model.optimizers()[0]
            opt.zero_grad()
            model.manual_backward(loss)
            opt.step()

    def _train_detached(self, model: pl.LightningModule, enc_loss: Tensor, dec_loss: Tensor) -> None:
        """Train encoder and decoder independently with detached gradients.

        Optimizes encoder and decoder separately, preventing gradients
        from flowing between components. This allows each component to
        focus on its specific objective.

        Args:
            model: The Lightning module being trained.
            enc_loss: Encoder loss (sparsity/regularization).
            dec_loss: Decoder loss (reconstruction).
        """
        enc_opt, dec_opt = model.optimizers()

        # Check if components need training
        enc_do_step = any(p.requires_grad for p in model.encoder.parameters()) and enc_loss.requires_grad
        dec_do_step = any(p.requires_grad for p in model.decoder.parameters()) and dec_loss.requires_grad

        # Train encoder independently
        if enc_do_step:
            enc_opt.zero_grad()
            model.manual_backward(enc_loss)
            enc_opt.step()

        # Train decoder independently
        if dec_do_step:
            dec_opt.zero_grad()
            model.manual_backward(dec_loss)
            dec_opt.step()

    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        """Detached validation step.

        Performs validation with the same detached logic as training
        but without parameter updates.

        Args:
            model: The Lightning module being validated.
            batch: Validation batch data.
            batch_idx: Index of the current batch.

        Returns:
            Combined validation loss.
        """
        x, *_ = batch
        if hasattr(model, "compute_loss"):
            dec_loss, enc_loss = model.compute_loss(x, "val", detach_gradients=True)
            return dec_loss + enc_loss
        else:
            reconstruction, embedding = model.forward(x, detach_gradients=True)
            return model.reconstruction_loss(reconstruction, x)
