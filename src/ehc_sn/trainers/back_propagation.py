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

from ehc_sn.trainers.core import BaseTrainer


class ClassicTrainer(BaseTrainer):
    """Standard backpropagation trainer with full gradient flow.

    Implements classic backpropagation where gradients flow through the entire
    model architecture without interruption. This trainer combines all loss
    components into a single optimization step, allowing gradients to propagate
    from the decoder back through the encoder in autoencoder architectures.

    This is the conventional training approach for most neural networks and
    provides the most direct optimization path. However, it may lead to
    suboptimal encoder representations when the decoder loss dominates.

    Key Characteristics:
        - Full gradient flow between all model components
        - Single combined loss for joint optimization
        - Standard PyTorch Lightning optimization pattern
        - Suitable for traditional end-to-end training

    Use Cases:
        - Standard autoencoder training where encoder and decoder should be
          jointly optimized for reconstruction accuracy
        - Models where component interdependence is desired
        - Baseline training for comparison with alternative strategies

    Note:
        This trainer expects models with dual optimizers (encoder and decoder)
        but trains them jointly using a combined loss function.

    Example:
        >>> from functools import partial
        >>> import torch.optim as optim
        >>>
        >>> trainer = ClassicTrainer(
        ...     optimizer_init=partial(optim.Adam, lr=1e-3, weight_decay=1e-4)
        ... )
        >>> model = Autoencoder(params, trainer)
        >>> lightning_trainer = pl.Trainer(max_epochs=100)
        >>> lightning_trainer.fit(model, dataloader)
    """

    def __init__(self, optimizer_init: Callable[[Any], Optimizer], *args: Any, **kwargs: Any) -> None:
        """Initialize classic backpropagation trainer with optimizer configuration.

        Sets up the trainer with a factory function for creating optimizers.
        The same optimizer configuration will be used for both encoder and
        decoder components, though they will be trained jointly.

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

    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        """Execute standard backpropagation training step with full gradient flow.

        Performs a complete training iteration using classic backpropagation:
        1. Forward pass through the model with full gradient flow
        2. Compute all loss components (reconstruction, regularization, etc.)
        3. Combine losses into a single total loss
        4. Backward pass with gradients flowing through all components
        5. Simultaneous optimizer step for all active components

        This approach ensures that encoder representations are optimized not only
        for their specific objectives but also for reconstruction quality through
        the decoder pathway.

        Args:
            model: The PyTorch Lightning module being trained. Expected to have
                encoder and decoder components with separate optimizers.
            batch: Training batch containing input data. First element should be
                the input tensor; additional elements are ignored.
            batch_idx: Index of the current batch within the epoch. Used for
                logging and debugging purposes.

        Note:
            The method uses manual optimization to support the dual-optimizer
            pattern while maintaining joint gradient flow. Optimizers are only
            stepped if their corresponding components have trainable parameters
            and valid gradients.

        Side Effects:
            - Modifies model parameters through optimizer steps
            - Logs training metrics through the model's logging system
            - Clears and computes gradients for all active components
        """
        x, *_ = batch

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

    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        """Execute validation step with full gradient flow for loss computation.

        Performs model evaluation on validation data using the same forward pass
        and loss computation as training, but without parameter updates. This
        ensures validation metrics are computed with the same gradient flow
        configuration as training for consistency.

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
            No gradient computation or parameter updates occur during validation.
            The method maintains the same loss computation logic as training to
            ensure consistent evaluation metrics.
        """
        x, *_ = batch
        loss_components = model.compute_loss(x, "val", detach_grad=False)
        return sum(loss_components)


class DetachedTrainer(BaseTrainer):
    """Detached gradient trainer for independent component optimization.

    Implements a split training strategy where gradients between model components
    are detached, enabling independent optimization of encoder and decoder with
    their respective loss functions. This approach prevents the decoder's
    reconstruction loss from overwhelming encoder-specific objectives like
    sparsity or orthogonality constraints.

    The detached training strategy is particularly beneficial for autoencoder
    architectures in spatial navigation modeling, where the encoder should learn
    structured sparse representations (similar to hippocampal place cells) while
    the decoder focuses purely on reconstruction accuracy.

    Key Characteristics:
        - Detached gradients between encoder and decoder components
        - Independent optimization steps for each component
        - Component-specific loss functions and optimization schedules
        - Prevention of gradient interference between objectives

    Advantages:
        - Encoder can learn meaningful sparse representations without being
          dominated by reconstruction objectives
        - Decoder can focus purely on reconstruction quality
        - More stable training for models with conflicting objectives
        - Better control over individual component behavior

    Trade-offs:
        - Components may not learn optimal joint representations
        - Requires careful tuning of component-specific loss weights
        - May lead to suboptimal overall reconstruction if components diverge

    Use Cases:
        - Sparse autoencoder training where sparsity is critical
        - Models with multiple competing objectives
        - Research scenarios requiring controlled component behavior
        - Biological plausibility experiments where independent learning is desired

    Example:
        >>> from functools import partial
        >>> import torch.optim as optim
        >>>
        >>> trainer = DetachedTrainer(
        ...     optimizer_init=partial(optim.Adam, lr=1e-3)
        ... )
        >>> # Encoder will be optimized with sparsity + orthogonality losses
        >>> # Decoder will be optimized with reconstruction loss only
        >>> model = Autoencoder(params, trainer)
    """

    def __init__(self, optimizer_init: Callable[[Any], Optimizer], *args: Any, **kwargs: Any) -> None:
        """Initialize detached gradient trainer with optimizer configuration.

        Sets up the trainer with a factory function for creating optimizers.
        The same optimizer configuration will be used for both encoder and
        decoder components, but they will be trained independently with
        detached gradients.

        Args:
            optimizer_init: Factory function for creating optimizer instances.
                Should be a callable that takes model parameters as input and
                returns an optimizer. The function will be called twice to create
                separate optimizers for encoder and decoder components.
                Typically a partial function like `partial(torch.optim.Adam, lr=1e-3)`.
            *args: Additional positional arguments passed to parent class.
            **kwargs: Additional keyword arguments passed to parent class.

        Note:
            While the same optimizer configuration is used for both components,
            they maintain separate optimization states and are stepped independently.
            This allows for different effective learning rates if components have
            different gradient magnitudes.

        Example:
            >>> from functools import partial
            >>> import torch.optim as optim
            >>>
            >>> # Both encoder and decoder use Adam with same hyperparameters
            >>> trainer = DetachedTrainer(
            ...     optimizer_init=partial(optim.Adam, lr=1e-3, weight_decay=1e-4)
            ... )
        """
        super().__init__(*args, **kwargs)
        self.optimizer_init = optimizer_init

    def training_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> None:
        """Execute detached gradient training step with independent component optimization.

        Performs a split training iteration where encoder and decoder are optimized
        independently with detached gradients:
        1. Forward pass through the model with gradient detachment at latent layer
        2. Compute component-specific losses (encoder: sparsity/orthogonality, decoder: reconstruction)
        3. Independent backward passes for each component
        4. Separate optimizer steps without gradient interference

        This approach allows the encoder to learn meaningful sparse representations
        optimized for biological plausibility (sparsity, orthogonality) while the
        decoder focuses purely on reconstruction accuracy without conflicting gradients.

        Args:
            model: The PyTorch Lightning module being trained. Must have encoder
                and decoder components with separate optimizers and a compute_loss
                method that supports detached gradient computation.
            batch: Training batch containing input data. First element should be
                the input tensor; additional elements are ignored.
            batch_idx: Index of the current batch within the epoch. Used for
                logging and debugging purposes.

        Note:
            The detached training prevents gradients from flowing between components,
            meaning the encoder's representation learning is not directly influenced
            by reconstruction error. This can lead to more biologically plausible
            representations but may reduce overall reconstruction quality.

        Side Effects:
            - Modifies encoder parameters based on sparsity and orthogonality losses
            - Modifies decoder parameters based on reconstruction loss only
            - Logs component-specific training metrics
            - Maintains separate gradient computation graphs for each component
        """
        x, *_ = batch

        # Autoencoder-style loss computation with detached gradients
        loss_components = model.compute_loss(x, "train", detach_grad=True)
        optm_list = model.optimizers()

        # Check if optimizers need to step based on parameter gradients
        do_step_list = [
            any(p.requires_grad for p in component.parameters()) and loss.requires_grad
            for component, loss in zip([model.decoder, model.encoder], loss_components)
        ]

        # Train components independently with detached gradients
        for optm, loss, do_step in zip(optm_list, loss_components, do_step_list):
            if do_step:
                optm.zero_grad()
                model.manual_backward(loss)
                optm.step()

    def validation_step(self, model: pl.LightningModule, batch: Tensor, batch_idx: int) -> Tensor:
        """Execute validation step with detached gradient computation for consistency.

        Performs model evaluation on validation data using the same detached
        gradient configuration as training. This ensures validation metrics
        accurately reflect the training behavior where encoder and decoder
        operate with independent gradient flows.

        The validation uses detached gradients to maintain consistency with the
        training regime, providing more accurate estimates of model performance
        under the split training strategy.

        Args:
            model: The PyTorch Lightning module being validated. Should be the
                same model used in detached training with encoder and decoder
                components.
            batch: Validation batch containing input data. First element should
                be the input tensor; additional elements are ignored.
            batch_idx: Index of the current validation batch. Used for logging
                and potential batch-specific operations.

        Returns:
            Combined validation loss as a single tensor. Represents the sum of
            all component losses (encoder-specific and decoder reconstruction)
            computed with the same detached gradient configuration as training.

        Note:
            While gradients are computed in a detached manner for consistency,
            no parameter updates occur during validation. The detached computation
            ensures that validation metrics accurately reflect the model's behavior
            under the split training regime.
        """
        x, *_ = batch
        loss_components = model.compute_loss(x, "val", detach_grad=True)
        return sum(loss_components)
