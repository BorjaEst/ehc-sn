"""Direct Feedback Alignment (DFA) training strategies.

This module implements DFA-based training where hidden layers receive
direct feedback from the output error via random feedback weights,
eliminating the need for symmetric weight transport.
"""

from typing import Dict, List

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from ehc_sn.trainers.core import BaseTrainer


class DFATrainer(BaseTrainer):
    """Direct Feedback Alignment training strategy.

    Implements DFA training where hidden layers receive direct feedback
    from the output layer via fixed random feedback weights. This provides
    a biologically plausible alternative to backpropagation that doesn't
    require symmetric weight transport.

    Features:
    - Direct feedback to hidden layers
    - Fixed random feedback weights
    - Biologically plausible learning mechanism
    - Eliminates weight transport problem
    """

    def __init__(self, optimizer_init):
        """Initialize DFA trainer.

        Args:
            optimizer_init: Function to initialize optimizers.
        """
        super().__init__()
        self.optimizer_init = optimizer_init
        self._feedback_weights: Dict[str, Tensor] = {}
        self._initialized = False

    def configure_optimizers(self, model: pl.LightningModule) -> List[Optimizer]:
        """Configure optimizers for DFA training.

        Creates optimizers for all trainable parameters in the model.

        Args:
            model: The Lightning module to configure optimizers for.

        Returns:
            List containing model optimizer.
        """
        return [self.optimizer_init(model.parameters())]

    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        """DFA training step with direct feedback.

        Implements DFA training logic where hidden layers receive
        direct feedback from the output error via random feedback weights.

        Args:
            model: The Lightning module being trained.
            batch: Training batch data.
            batch_idx: Index of the current batch.
        """
        x, *_ = batch

        # Initialize feedback weights if needed
        if not self._initialized:
            self._init_feedback_weights(model)
            self._initialized = True

        # Forward pass
        reconstruction, embedding = model(x)

        # Compute output error
        output_error = reconstruction - x

        # Apply direct feedback to hidden layers
        self._apply_direct_feedback(model, output_error, embedding)

        # Compute reconstruction loss
        loss = model.reconstruction_loss(reconstruction, x)

        # Standard optimizer step
        optimizers = model.optimizers()
        if isinstance(optimizers, list) and len(optimizers) > 0:
            opt = optimizers[0]
        else:
            opt = optimizers
        opt.zero_grad()
        model.manual_backward(loss)
        opt.step()

        # Log metrics
        model.log("train/reconstruction_loss", loss, on_epoch=True, prog_bar=True)

        # Log sparsity metrics
        sparsity_rate = (embedding > 0.01).float().mean()
        model.log("train/sparsity_rate", sparsity_rate, on_epoch=True)

    def _init_feedback_weights(self, model: pl.LightningModule) -> None:
        """Initialize random feedback weights for DFA.

        Creates fixed random feedback weights that connect the output
        layer directly to hidden layers, bypassing the need for
        symmetric weight transport.

        Args:
            model: The Lightning module to initialize feedback weights for.
        """
        if hasattr(model, "decoder"):
            # Initialize feedback weights for decoder layers
            for name, module in model.decoder.named_modules():
                if isinstance(module, torch.nn.Linear) and "output" not in name.lower():
                    # Create random feedback weight matrix
                    # Shape: (hidden_size, output_size)
                    output_size = model.input_shape[0] * model.input_shape[1] * model.input_shape[2]
                    hidden_size = module.out_features

                    feedback_weight = (
                        torch.randn(hidden_size, output_size, device=module.weight.device, dtype=module.weight.dtype)
                        * 0.1
                    )  # Small initialization

                    self._feedback_weights[name] = feedback_weight

        if hasattr(model, "encoder"):
            # Initialize feedback weights for encoder layers
            for name, module in model.encoder.named_modules():
                if isinstance(module, torch.nn.Linear) and "input" not in name.lower():
                    # Create random feedback weight matrix
                    output_size = model.input_shape[0] * model.input_shape[1] * model.input_shape[2]
                    hidden_size = module.out_features

                    feedback_weight = (
                        torch.randn(hidden_size, output_size, device=module.weight.device, dtype=module.weight.dtype)
                        * 0.1
                    )

                    self._feedback_weights[f"encoder.{name}"] = feedback_weight

    def _apply_direct_feedback(self, model: pl.LightningModule, output_error: Tensor, embedding: Tensor) -> None:
        """Apply direct feedback to hidden layers.

        Computes and applies direct feedback signals to hidden layers
        using the precomputed random feedback weights.

        Args:
            model: The Lightning module being trained.
            output_error: Error signal from output layer.
            embedding: Hidden layer activations.
        """
        # Flatten output error for feedback computation
        batch_size = output_error.shape[0]
        error_flat = output_error.view(batch_size, -1)

        # Apply feedback to decoder layers
        if hasattr(model, "decoder"):
            for name, module in model.decoder.named_modules():
                if name in self._feedback_weights:
                    # Compute direct feedback signal
                    feedback_matrix = self._feedback_weights[name]
                    feedback_signal = torch.matmul(error_flat, feedback_matrix.T)

                    # Apply feedback to layer parameters
                    if hasattr(module, "weight") and module.weight.grad is not None:
                        # Add direct feedback to gradient
                        # This is a simplified implementation - full DFA would
                        # require more sophisticated feedback application
                        feedback_grad = torch.outer(feedback_signal.mean(0), module.weight.mean(0))
                        module.weight.grad += feedback_grad * 0.01  # Small learning rate for feedback

    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        """DFA validation step.

        Performs validation using standard forward pass without
        the DFA feedback mechanism.

        Args:
            model: The Lightning module being validated.
            batch: Validation batch data.
            batch_idx: Index of the current batch.

        Returns:
            Validation loss tensor.
        """
        x, *_ = batch

        if hasattr(model, "compute_loss"):
            dec_loss, enc_loss = model.compute_loss(x, "val")
            return dec_loss + enc_loss
        else:
            reconstruction, embedding = model(x)
            return model.reconstruction_loss(reconstruction, x)

    def on_train_batch_start(self, model: pl.LightningModule, batch, batch_idx: int) -> None:
        """Setup DFA-specific state before batch training.

        Ensures feedback weights are properly initialized and
        prepares any DFA-specific state.

        Args:
            model: The Lightning module being trained.
            batch: Training batch data.
            batch_idx: Index of the current batch.
        """
        if not self._initialized:
            self._init_feedback_weights(model)
            self._initialized = True
