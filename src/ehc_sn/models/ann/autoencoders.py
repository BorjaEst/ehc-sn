"""Autoencoder models for entorhinal-hippocampal circuit spatial navigation modeling.

This module implements autoencoder architectures specifically designed for modeling
the entorhinal-hippocampal circuit's spatial navigation and memory functions. The
autoencoders combine configurable encoder and decoder components with biologically-
inspired sparsity constraints to mimic hippocampal place cell firing patterns.

The autoencoders support multiple training strategies through a pluggable trainer
interface, enabling experimentation with standard backpropagation, biologically
plausible alternatives (DFA, DRTP), and custom optimization schemes.

Key Features:
    - Configurable encoder-decoder architectures with dimension validation
    - Biologically-inspired sparsity constraints (Gramian orthogonality, homeostatic activity)
    - Pluggable training strategies via BaseTrainer interface
    - Manual optimization with configurable gradient flow control
    - PyTorch Lightning integration for efficient training and logging

Classes:
    AutoencoderParams: Configuration parameters with Pydantic validation
    Autoencoder: Main autoencoder model with trainer strategy pattern

Functions:
    validate_dimensions: Ensures encoder-decoder compatibility validation

Architecture:
    Input → Encoder → Sparse Latent Representation → Decoder → Reconstruction

    The sparse latent representation mimics hippocampal place cell activity patterns,
    incorporating biological constraints like firing rate homeostasis and decorrelated
    representations essential for effective spatial memory encoding.

Examples:
    >>> from functools import partial
    >>> from ehc_sn.trainers.back_propagation import DetachedTrainer
    >>>
    >>> # Configure autoencoder with detached gradient training
    >>> trainer = DetachedTrainer(optimizer_init=partial(torch.optim.Adam, lr=1e-3))
    >>> params = AutoencoderParams(
    ...     encoder=Linear(EncoderParams(...)),
    ...     decoder=Linear(DecoderParams(...)),
    ...     gramian_weight=1.0,
    ...     homeo_weight=1.0,
    ...     rate_target=0.10,
    ... )
    >>> model = Autoencoder(params, trainer)
    >>> lightning_trainer = pl.Trainer(max_epochs=100)
    >>> lightning_trainer.fit(model, dataloader)

Biological Motivation:
    The autoencoder architecture models the information flow in the entorhinal-
    hippocampal circuit where sensory inputs are compressed into sparse neural
    codes (similar to place cells) and then reconstructed for spatial navigation
    and memory retrieval tasks.

References:
    - O'Keefe, J., & Nadel, L. (1978). The hippocampus as a cognitive map.
    - Hafting, T., et al. (2005). Microstructure of a spatial map in the entorhinal cortex.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import lightning.pytorch as pl
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.hooks import registry
from ehc_sn.models.ann import encoders
from ehc_sn.models.ann.decoders import BaseDecoder, DecoderParams
from ehc_sn.models.ann.encoders import BaseEncoder, EncoderParams
from ehc_sn.modules.loss import GramianOrthogonalityLoss, HomeostaticActivityLoss
from ehc_sn.trainers.core import BaseTrainer


class AutoencoderParams(BaseModel):
    """Configuration parameters for the Autoencoder model.

    Defines parameters for constructing an autoencoder model for entorhinal-hippocampal
    circuit modeling, ensuring encoder and decoder compatibility.

    Attributes:
        encoder: Encoder component for transforming input to latent representations.
        decoder: Decoder component for reconstructing output from latent representations.
        gramian_center: Whether to center activations before computing Gramian matrix.
        gramian_weight: Weight coefficient for Gramian orthogonality loss.
        rate_target: Target mean firing rate for homeostatic regulation (0.0-1.0).
        min_active: Minimum number of active neurons per sample.
        homeo_weight: Weight coefficient for homeostatic activity loss.

    Example:
        >>> params = AutoencoderParams(
        ...     encoder=Linear(EncoderParams(...)),
        ...     decoder=Linear(DecoderParams(...)),
        ...     gramian_weight=1.0,
        ...     homeo_weight=1.0,
        ...     rate_target=0.10,
        ... )
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Encoder and decoder components
    encoder: BaseEncoder = Field(..., description="Parameters for the encoder component.")
    decoder: BaseDecoder = Field(..., description="Parameters for the decoder component.")

    # Gramian orthogonality loss parameters
    gramian_center: bool = Field(True, description="Center activations before normalization.")
    gramian_weight: float = Field(1.0, description="Weight for Gramian orthogonality term.")

    # Homeostatic activity loss parameters
    rate_target: float = Field(0.10, description="Target mean firing rate for codes.")
    min_active: int = Field(8, description="Min active neurons per-sample.")
    homeo_weight: float = Field(1.0, description="Weight for homeostatic term.")


def validate_dimensions(params: AutoencoderParams) -> None:
    """Validate encoder and decoder dimension compatibility.

    Ensures encoder and decoder can work together by verifying dimensional
    consistency between components. This validation prevents runtime errors
    and ensures proper data flow through the autoencoder architecture.

    The function performs two critical checks:
    1. Encoder input shape matches decoder output shape (end-to-end consistency)
    2. Encoder latent dimension matches decoder latent dimension (bottleneck compatibility)

    Args:
        params: AutoencoderParams containing encoder and decoder configurations
            to be validated for compatibility.

    Raises:
        ValueError: If encoder input shape does not match decoder output shape,
            or if encoder latent dimension does not match decoder latent dimension.
            The error message includes the specific dimensional mismatch details.

    Example:
        >>> encoder_params = EncoderParams(input_shape=(1, 64, 64), latent_dim=128)
        >>> decoder_params = DecoderParams(output_shape=(1, 64, 64), latent_dim=128)
        >>> params = AutoencoderParams(encoder=encoder, decoder=decoder)
        >>> validate_dimensions(params)  # Should pass without error
        >>>
        >>> # This would raise ValueError due to shape mismatch:
        >>> bad_decoder = DecoderParams(output_shape=(1, 32, 32), latent_dim=128)
        >>> bad_params = AutoencoderParams(encoder=encoder, decoder=bad_decoder)
        >>> validate_dimensions(bad_params)  # Raises ValueError

    Note:
        This function should be called before creating an Autoencoder instance
        to catch configuration errors early in the model setup process.
    """
    if params.encoder.input_shape != params.decoder.output_shape:
        raise ValueError(
            f"Input shape of encoder ({params.encoder.input_shape}) "
            f"does not match output shape of decoder ({params.decoder.output_shape})."
        )
    if params.encoder.latent_dim != params.decoder.latent_dim:
        raise ValueError(
            f"Latent dimension of encoder ({params.encoder.latent_dim}) "
            f"does not match embedding dimension of decoder ({params.decoder.latent_dim})."
        )


class Autoencoder(pl.LightningModule):
    """Autoencoder for entorhinal-hippocampal circuit spatial navigation modeling.

    Combines an encoder that transforms spatial input into sparse latent representations
    and a decoder that reconstructs the original input. Uses manual optimization with
    configurable gradient flow to support different training strategies.

    Incorporates biologically-inspired sparsity constraints mimicking hippocampal
    place cell firing patterns for spatial memory encoding and retrieval.

    Architecture:
        Input → Encoder → Sparse Latent → Decoder → Reconstruction

    Args:
        params: Model configuration parameters.
        trainer: Training strategy (required for proper separation of concerns).

    Attributes:
        encoder: Encoder neural network component.
        decoder: Decoder neural network component.
        trainer_strategy: Training strategy handling optimization logic.
        reconstruction_loss: BCE loss for reconstruction accuracy.
        gramian_loss: Gramian orthogonality loss for decorrelated representations.
        homeo_loss: Homeostatic activity loss for firing rate regulation.

    Example:
        >>> trainer = DetachedTrainer(optimizer_init=partial(torch.optim.Adam, lr=1e-3))
        >>> model = Autoencoder(params, trainer)
        >>> lightning_trainer = pl.Trainer(max_epochs=100)
        >>> lightning_trainer.fit(model, dataloader)
    """

    def __init__(self, params: AutoencoderParams, trainer: BaseTrainer) -> None:
        """Initialize autoencoder with parameters and training strategy.

        Sets up model architecture, loss functions, and training configuration.
        Delegates training logic to the provided trainer strategy.

        Args:
            params: Model configuration including encoder/decoder and loss weights.
            trainer: Training strategy for optimization logic.

        Raises:
            ValueError: If encoder and decoder dimensions are incompatible.

        Note:
            Manual optimization is enabled to support custom training strategies.
        """
        validate_dimensions(params)
        super(Autoencoder, self).__init__()

        # Enable manual optimization for custom training strategies
        self.automatic_optimization = False

        # Set training strategy
        # If trainer is None, model will not be able to train
        self.trainer_strategy = trainer

        # Model components
        self.encoder = params.encoder
        self.decoder = params.decoder

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.reconstruction_weight = 1.0
        self.gramian_loss = GramianOrthogonalityLoss(params.gramian_center)
        self.gramian_weight = params.gramian_weight
        self.homeo_loss = HomeostaticActivityLoss(params.rate_target, params.min_active)
        self.homeo_weight = params.homeo_weight

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["encoder", "decoder", "trainer"])

    # -----------------------------------------------------------------------------------
    # Optimizer configuration
    # -----------------------------------------------------------------------------------

    def configure_optimizers(self) -> List[Optimizer]:
        """Configure dual optimizers for independent encoder-decoder training.

        Creates separate optimizers for encoder and decoder components to enable
        independent optimization strategies essential for biological plausibility
        in spatial navigation modeling. This dual-optimizer architecture allows
        the encoder to focus on learning sparse, orthogonal representations
        (mimicking hippocampal place cells) while the decoder optimizes purely
        for reconstruction accuracy.

        The optimizer configuration delegates to the training strategy to ensure
        consistency with the chosen training approach (e.g., DetachedTrainer or
        ClassicTrainer). Both optimizers use identical hyperparameters but
        maintain separate optimization states and parameter groups.

        Returns:
            List[Optimizer]: Two-element list containing optimizers in fixed order:
                - Index 0: Encoder optimizer for representation learning objectives
                  (sparsity, orthogonality, homeostatic regulation)
                - Index 1: Decoder optimizer for reconstruction objectives
                  (binary cross-entropy, output quality)

        Raises:
            ValueError: If no training strategy is provided during model initialization.
                The training strategy is required to supply the optimizer factory function.

        Note:
            The optimizer order is critical for training step implementation. The
            training strategy relies on this fixed indexing to properly assign
            loss functions to their corresponding optimizers during split training.

        Example:
            >>> # Optimizers are accessed in training via:
            >>> enc_opt, dec_opt = self.optimizers()
            >>> # enc_opt handles encoder-specific losses (sparsity, orthogonality)
            >>> # dec_opt handles decoder-specific losses (reconstruction)

        See Also:
            - BaseTrainer.optimizer_init: Factory function for creating optimizers
            - DetachedTrainer: Training strategy using independent optimization
            - ClassicTrainer: Training strategy using joint optimization
        """
        if self.trainer_strategy is None:
            raise ValueError("Trainer strategy must be provided for optimizer configuration.")
        enc_opt = self.trainer_strategy.optimizer_init(self.encoder.parameters())
        dec_opt = self.trainer_strategy.optimizer_init(self.decoder.parameters())
        return [enc_opt, dec_opt]

    # -----------------------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------------------

    def forward(self, x: Tensor, detach_grad: bool = False, *args: Any, **kwds: Any) -> List[Tensor]:
        """Execute forward pass through encoder-decoder architecture.

        Performs complete forward propagation from input through encoder to latent
        representation, then through decoder to reconstruction. Supports optional
        gradient detachment for implementing split training strategies essential
        for biologically plausible representation learning.

        The forward pass follows this sequence:
        1. Encode input to sparse latent representation (mimicking place cell activity)
        2. Optionally detach gradients at the latent bottleneck
        3. Decode latent representation back to input space
        4. Return both reconstruction and original (non-detached) embedding

        Args:
            x: Input tensor with shape matching encoder's expected dimensions.
                Typically spatial maps with shape (batch_size, channels, height, width).
            detach_grad: Whether to detach gradients between encoder and decoder.
                When True, prevents gradient flow from decoder losses to encoder,
                enabling independent optimization of each component. Default: False.
            *args: Additional positional arguments passed to encoder and decoder
                for interface compatibility with different network architectures.
            **kwds: Additional keyword arguments passed to encoder and decoder
                for interface compatibility and extensibility.

        Returns:
            List[Tensor]: Two-element list containing:
                - Index 0: Reconstructed output tensor with same shape as input
                - Index 1: Original latent embedding tensor (never detached)
                  with shape (batch_size, latent_dim)

        Note:
            When detach_grad=True, gradients cannot flow from decoder losses back
            to the encoder, implementing the split training strategy. However, the
            returned embedding is always the original (non-detached) tensor to
            preserve gradient information for encoder-specific losses.

        Example:
            >>> # Standard forward pass with full gradient flow
            >>> reconstruction, embedding = model(input_tensor)
            >>>
            >>> # Forward pass with detached gradients for split training
            >>> reconstruction, embedding = model(input_tensor, detach_grad=True)
            >>> # Decoder receives detached embedding, encoder losses use original
        """
        embedding = self.encoder(x, *args, target=None, **kwds)  # No sparse target

        # Optionally detach gradients between encoder and decoder
        z = embedding.detach() if detach_grad else embedding
        reconstruction = self.decoder(z, *args, target=x, **kwds)

        # Return reconstruction and embedding (always return original embedding)
        return [reconstruction, embedding]

    def feedback(self, feedbacks: Union[float, List[float]]) -> None:
        """Propagate feedback signals to components supporting gradient-free optimization.

        Distributes feedback signals (loss differences or gradients) to model components
        that implement feedback-based learning mechanisms such as Zero-Order optimization,
        Direct Feedback Alignment (DFA), or Direct Random Target Projection (DRTP).
        This method enables biologically plausible learning without traditional backpropagation.

        The feedback mechanism works as follows:
        1. Receive feedback signals from training strategy (e.g., loss differences in ZO)
        2. Distribute signals to decoder and encoder components in that order
        3. Components implement their own feedback processing (gradient estimation, etc.)
        4. Only components with feedback capability receive signals (checked via hasattr)

        Args:
            feedbacks: Feedback signals to propagate to model components. Can be:
                - Single float: Same feedback applied to all components with feedback capability
                - List[float]: Individual feedback per component in [decoder, encoder] order
                  Must have exactly 2 elements matching the expected component ordering

        Raises:
            ValueError: If feedbacks is a list but doesn't have exactly 2 elements
                to match the [decoder, encoder] component ordering.

        Note:
            The component ordering [decoder, encoder] matches the loss ordering returned
            by compute_loss() and expected by training strategies. Only components that
            implement a feedback() method will receive signals - standard backpropagation
            components are safely ignored.

        Example:
            >>> # Zero-Order optimization with loss differences
            >>> loss_diff = loss_1 - loss_2  # From ZO trainer
            >>> model.feedback([loss_diff, loss_diff])  # Same feedback to both components
            >>>
            >>> # Component-specific feedback
            >>> model.feedback([decoder_feedback, encoder_feedback])
            >>>
            >>> # Single feedback for all components
            >>> model.feedback(overall_feedback)

        See Also:
            - ZOLinear.feedback(): Zero-Order gradient estimation
            - DFALinear.feedback(): Direct Feedback Alignment
            - DRTPLinear.feedback(): Direct Random Target Projection
        """
        # Handle single feedback for all components
        if isinstance(feedbacks, (int, float)):
            feedbacks = [float(feedbacks)] * 2

        # Validate feedback list length
        if len(feedbacks) != 2:
            raise ValueError(f"Expected 2 feedback values for [decoder, encoder], got {len(feedbacks)}")

        # Distribute feedback to components with feedback capability
        for component, projected_grad in zip([self.decoder, self.encoder], feedbacks):
            if hasattr(component, "feedback"):
                component.feedback(projected_grad)

    # -----------------------------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------------------------

    def compute_loss(self, output: Any, batch: Any, log_label: str) -> List[Tensor]:
        """Compute and log component-specific losses for autoencoder training.

        Executes forward pass and computes all loss components with comprehensive
        logging for monitoring training progress. Implements biologically-inspired
        loss functions that encourage sparse, orthogonal representations similar
        to hippocampal place cell firing patterns.

        The method computes three categories of losses:
        1. Encoder losses: Gramian orthogonality and homeostatic activity regulation
        2. Decoder losses: Binary cross-entropy reconstruction error
        3. Auxiliary metrics: Sparsity rate for monitoring representation quality

        Loss Components:
            - Gramian Loss: Encourages orthogonal, decorrelated representations
            - Homeostatic Loss: Regulates firing rates to maintain target sparsity
            - Reconstruction Loss: Measures decoder's reconstruction accuracy

        Args:
            TODO

        Returns:
            List[Tensor]: Two-element list containing computed losses:
                - Index 0: Decoder loss (reconstruction error only)
                - Index 1: Encoder loss (weighted combination of Gramian and homeostatic losses)

        Logged Metrics:
            - {log_label}/reconstruction_loss: BCE loss between input and reconstruction
            - {log_label}/gramian_loss: Orthogonality constraint on latent representations
            - {log_label}/homeostatic_loss: Firing rate regulation penalty
            - {log_label}/decoder_loss: Total decoder optimization target (with progress bar)
            - {log_label}/encoder_loss: Total encoder optimization target (with progress bar)
            - {log_label}/sparsity_rate: Fraction of active neurons (> 0.01 threshold)

        Example:
            >>> # During training with detached gradients
            >>> dec_loss, enc_loss = model.compute_loss(batch, "train", detach_grad=True)
            >>> # Logged: train/reconstruction_loss, train/gramian_loss, etc.
            >>>
            >>> # During validation
            >>> dec_loss, enc_loss = model.compute_loss(batch, "val", detach_grad=False)
            >>> # Logged: val/reconstruction_loss, val/gramian_loss, etc.

        Note:
            Loss weights are configured during model initialization via AutoencoderParams.
            The sparsity threshold (0.01) is chosen to identify meaningfully active neurons
            while filtering out near-zero activations from numerical precision issues.
        """
        # Recover reconstruction and embedding from forward output
        target, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        reconstruction, embedding = output  # Unpack forward output

        # Encoder-side losses
        gramian_loss = self.gramian_loss(embedding)
        self.log(f"{log_label}/gramian_loss", gramian_loss, on_epoch=True)
        homeo_loss = self.homeo_loss(embedding)
        self.log(f"{log_label}/homeostatic_loss", homeo_loss, on_epoch=True)
        encoder_loss = self.gramian_weight * gramian_loss + self.homeo_weight * homeo_loss
        self.log(f"{log_label}/encoder_loss", encoder_loss, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        sparsity_rate = (embedding > 0.01).float().mean()
        self.log(f"{log_label}/sparsity_rate", sparsity_rate, on_epoch=True)

        # Decoder-side losses
        reconstruction_loss = self.reconstruction_loss(reconstruction, target)
        self.log(f"{log_label}/reconstruction_loss", reconstruction_loss, on_epoch=True)
        decoder_loss = reconstruction_loss
        self.log(f"{log_label}/decoder_loss", decoder_loss, on_epoch=True, prog_bar=True)

        # Return the decoder and the encoder loss
        return [decoder_loss, encoder_loss]

    def get_output_error(self, outputs: Any, batch: Any) -> List[Tensor]:
        """Compute output error for feedback alignment training.

        Args:
            outputs: Model outputs from forward pass.
            batch: Input batch containing target data.

        Returns:
            List[Tensor]: List containing the output error tensor.
        """
        target, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        reconstruction, _ = outputs  # Unpack forward output
        error = reconstruction - target  # Compute output error
        return [error]

    # -----------------------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------------------

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Initialize registry and delegate to training strategy.

        Args:
            batch: Training batch.
            batch_idx: Batch index.

        Raises:
            ValueError: If no trainer strategy is provided.
        """
        if self.trainer_strategy is None:
            raise ValueError("Trainer strategy must be provided for training.")

        # Clear registry at the start of each batch
        registry.clear("batch")

        # Delegate to training strategy
        self.trainer_strategy.on_train_batch_start(self, batch, batch_idx)

    def training_step(self, batch: Tensor, batch_idx: int) -> Optional[Any]:
        """Delegate training step to the training strategy.

        Args:
            batch: Training batch.
            batch_idx: Batch index within the epoch.

        Note:
            All training logic is handled by the training strategy.
        """
        return self.trainer_strategy.training_step(self, batch, batch_idx)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Delegate batch end cleanup to training strategy.

        Args:
            outputs: Training step outputs.
            batch: Training batch.
            batch_idx: Batch index.
        """
        self.trainer_strategy.on_train_batch_end(self, outputs, batch, batch_idx)

    # -----------------------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------------------

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Delegate validation step to the training strategy.

        Args:
            batch: Validation batch.
            batch_idx: Batch index (unused).

        Returns:
            Validation loss tensor.
        """
        return self.trainer_strategy.validation_step(self, batch, batch_idx)

    # -----------------------------------------------------------------------------------
    # Test step
    # -----------------------------------------------------------------------------------

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Evaluate model on test batch and return total loss.

        Args:
            batch: Test batch with input tensor as first element.
            batch_idx: Batch index (unused).

        Returns:
            Sum of decoder and encoder losses for this batch.

        Note:
            Logs metrics under 'test/' namespace for clear separation from validation.
        """
        x, *_ = batch
        dec_loss, enc_loss = self.compute_loss(x, log_label="test")
        return dec_loss + enc_loss

    # -----------------------------------------------------------------------------------
    # Prediction step
    # -----------------------------------------------------------------------------------

    def predict_step(self, batch: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor]:  # noqa: ARG002
        """Return reconstructions and embeddings for inference.

        Args:
            batch: Input batch for prediction.
            batch_idx: Batch index (unused).

        Returns:
            Tuple of (reconstruction, embedding) tensors.

        Note:
            Useful for inference and analysis of learned representations.
        """
        x, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        return self(x)

    # -----------------------------------------------------------------------------------
    # Model properties
    # -----------------------------------------------------------------------------------

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Input feature map shape as (channels, height, width)."""
        return self.encoder.input_shape

    @property
    def input_channels(self) -> int:
        """Number of input channels."""
        return self.encoder.input_channels

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """Spatial dimensions as (height, width)."""
        return self.encoder.spatial_dimensions

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent representation."""
        return self.encoder.latent_dim

    # -----------------------------------------------------------------------------------
    # Additional utility methods
    # -----------------------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation.

        Args:
            x: Input tensor matching encoder's expected dimensions.

        Returns:
            Latent representation tensor with shape (batch_size, latent_dim).

        Note:
            Efficient for tasks requiring only encoded representations.
        """
        return self.encoder(x, target=None)

    def decode(self, embedding: Tensor) -> Tensor:
        """Decode latent representation to reconstruction.

        Args:
            embedding: Latent representation tensor.

        Returns:
            Reconstructed output tensor matching original input dimensions.

        Note:
            Useful for generative tasks or analyzing decoder capabilities.
        """
        return self.decoder(embedding)

    def reconstruct(self, x: Tensor) -> Tensor:
        """Perform full encode-decode reconstruction.

        Args:
            x: Input tensor.

        Returns:
            Reconstructed output tensor with same shape as input.

        Note:
            Equivalent to forward() but returns only reconstruction.
        """
        embedding = self.encode(x)
        return self.decode(embedding)


# -----------------------------------------------------------------------------------
# Example usage and testing
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    """Example: autoencoder training with configurable gradient flow.

    Demonstrates:
    1. Creating random obstacle map dataset
    2. Configuring autoencoder with training strategy
    3. Training with PyTorch Lightning
    4. Testing reconstruction performance and sparsity
    """
    from functools import partial

    import lightning.pytorch as pl

    from ehc_sn.models.ann import decoders
    from ehc_sn.trainers.back_propagation import DetachedTrainer

    # Simple dataset for testing
    class SimpleDataset(torch.utils.data.Dataset):
        """Dataset generating random binary obstacle maps."""

        def __init__(self, size: int = 100) -> None:
            """Initialize with specified dataset size."""
            self.size = size

        def __len__(self):
            """Return dataset size."""
            return self.size

        def __getitem__(self, idx: int) -> Tuple[Tensor]:
            """Generate random obstacle map sample.

            Args:
                idx: Sample index (unused, generates random data).

            Returns:
                Tuple containing tensor with random obstacle placement.
            """
            # Create random obstacle maps with random obstacle placement
            x = torch.zeros(1, 16, 32)
            num_obstacles = torch.randint(3, 8, (1,)).item()
            x[0, torch.randint(0, 16, (num_obstacles,)), torch.randint(0, 32, (num_obstacles,))] = 1.0
            return (x,)  # Return as tuple for batch unpacking

    # Create autoencoder with configurable training strategy
    trainer = DetachedTrainer(optimizer_init=partial(torch.optim.Adam, lr=1e-3))
    encoder_param = encoders.EncoderParams(input_shape=(1, 16, 32), latent_dim=32, activation_fn=torch.nn.GELU)
    decoder_param = decoders.DecoderParams(output_shape=(1, 16, 32), latent_dim=32, activation_fn=torch.nn.GELU)
    autoencoder_params = AutoencoderParams(
        encoder=encoders.Linear(encoder_param),
        decoder=decoders.Linear(decoder_param),
        gramian_weight=1.0,
        homeo_weight=1.0,
        rate_target=0.10,
    )
    model = Autoencoder(autoencoder_params, trainer)

    # Create training data
    dataset = SimpleDataset(200)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Train model with Lightning
    trainer = pl.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    trainer.fit(model, loader)

    # Test reconstruction performance
    test_input = torch.zeros(1, 1, 16, 32)
    test_input[0, 0, 5:10, 10:20] = 1.0  # Create test pattern

    with torch.no_grad():
        reconstruction, embedding = model(test_input)
        mse_loss = nn.MSELoss()(reconstruction, test_input)

    print(f"Test MSE loss: {mse_loss.item():.4f}")
    print(f"Embedding sparsity: {(embedding > 0.01).float().mean().item():.2%}")
    print(f"Reconstruction range: [{reconstruction.min().item():.3f}, {reconstruction.max().item():.3f}]")
