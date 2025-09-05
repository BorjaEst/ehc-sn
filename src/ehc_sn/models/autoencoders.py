from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import lightning.pytorch as pl
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.hooks import registry
from ehc_sn.models.decoders import BaseDecoder, DecoderParams
from ehc_sn.models.encoders import BaseEncoder, EncoderParams
from ehc_sn.modules.loss import GramianOrthogonalityLoss, HomeostaticActivityLoss, TargetL1SparsityLoss


class AutoencoderParams(BaseModel):
    """Parameters for configuring the Autoencoder model.

    This configuration class defines all the parameters needed to construct
    and train an autoencoder for the entorhinal-hippocampal circuit modeling.
    It ensures that the encoder and decoder components are properly configured
    and compatible with each other.

    The parameters are organized into several groups:
    - Core components: encoder, decoder, optimizer initialization
    - Legacy sparsity: backward compatibility parameters
    - Gramian orthogonality: decorrelation loss parameters
    - Homeostatic activity: firing rate regulation parameters
    - L1 sparsity: simple sparsity regularization parameters

    Attributes:
        encoder: The encoder component that transforms input into latent representations.
            Must be an instance of BaseEncoder with appropriate input dimensions.
        decoder: The decoder component that reconstructs outputs from latent representations.
            Must be an instance of BaseDecoder with output dimensions matching encoder input.
        gramian_center: Whether to center activations before computing Gramian matrix.
            Centering removes mean activation, improving correlation stability. Default: True.
        gramian_weight: Weight coefficient for Gramian orthogonality loss term.
            Controls strength of decorrelation constraint. Default: 1.0.
        rate_target: Target mean firing rate for homeostatic regulation.
            Should be between 0.0 and 1.0, typically 0.05-0.20. Default: 0.10.
        min_active: Minimum number of neurons that should be active per sample.
            Ensures robust distributed representations. Default: 8.
        homeo_weight: Weight coefficient for homeostatic activity loss term.
            Controls strength of firing rate regulation. Default: 1.0.
        detach_gradients: Whether to detach gradients between decoder and encoder.
            When True, prevents gradients from flowing from decoder to encoder during training.
            This allows for split training behavior while using manual optimization. Default: True.
        optimizer_init: Callable function to initialize optimizers.
            Should be a partial function or lambda that takes model parameters as input
            and returns an optimizer instance (e.g., functools.partial(torch.optim.Adam, lr=1e-3)).

    Example:
        >>> from functools import partial
        >>> params = AutoencoderParams(
        ...     encoder=Linear(EncoderParams(...)),
        ...     decoder=Linear(DecoderParams(...)),
        ...     gramian_weight=1.0,
        ...     homeo_weight=1.0,
        ...     rate_target=0.10,
        ...     detach_gradients=True,  # Use gradient detachment for split training
        ...     optimizer_init=partial(torch.optim.Adam, lr=1e-3)
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

    # Gradient flow control
    detach_gradients: bool = Field(True, description="Whether to detach gradients between decoder and encoder.")

    # Optimizer initialization
    optimizer_init: Callable = Field(..., description="Callable to initialize the optimizer.")


def validate_dimensions(params: AutoencoderParams) -> None:
    """Validate that encoder and decoder dimensions are compatible.

    This function performs essential compatibility checks between the encoder
    and decoder components to ensure they can work together properly in the
    autoencoder architecture.

    Args:
        params: AutoencoderParams instance containing encoder and decoder configurations.

    Raises:
        ValueError: If encoder input shape doesn't match decoder output shape.
        ValueError: If encoder latent dimension doesn't match decoder latent dimension.

    Note:
        This validation ensures that:
        1. The decoder can reconstruct the same shape as the encoder input
        2. The latent representations have matching dimensions for proper data flow
    """
    if params.encoder.input_shape != params.decoder.output_shape:
        raise ValueError(
            f"Input shape of encoder ({params.encoder.input_shape}) "
            f"does not match input shape of decoder ({params.decoder.input_shape})."
        )
    if params.encoder.latent_dim != params.decoder.latent_dim:
        raise ValueError(
            f"Latent dimension of encoder ({params.encoder.latent_dim}) "
            f"does not match embedding dimension of decoder ({params.decoder.latent_dim})."
        )


class Autoencoder(pl.LightningModule):
    """Neural network autoencoder for the entorhinal-hippocampal circuit with configurable gradient flow.

    This autoencoder combines an encoder that transforms spatial input into a compact
    embedding and a decoder that reconstructs the original input from the embedding.
    It uses manual optimization with configurable gradient flow between encoder and decoder,
    allowing for both standard training and split training strategies.

    The autoencoder serves as a computational model for how the entorhinal-hippocampal
    circuit might encode, store, and retrieve spatial information. It incorporates
    biologically-inspired sparsity constraints that encourage the model to learn
    efficient representations similar to those found in hippocampal place cells.

    Key Features:
        - Manual optimization with configurable gradient flow between components
        - Sparse latent representations mimicking hippocampal neuron firing patterns
        - Configurable encoder/decoder architectures for different input types
        - Multiple sparsity regularization methods (Gramian, homeostatic)
        - Lightning-based training with comprehensive logging and metrics
        - Support for both training and inference modes with proper checkpointing

    Architecture:
        Input → Encoder → Sparse Latent Representation → Decoder → Reconstruction

    Training Strategy:
        When detach_gradients=True: Split training with encoder focused on sparsity losses
        and decoder focused on reconstruction, with no gradient flow between them.
        When detach_gradients=False: Standard autoencoder with full gradient flow.

    Args:
        params: AutoencoderParams instance containing all configuration parameters.
            Must include compatible encoder/decoder pairs and training hyperparameters.

    Attributes:
        encoder: The encoder neural network component.
        decoder: The decoder neural network component.
        detach_gradients: Whether to detach gradients between decoder and encoder.
        optimizer_init: Function to initialize optimizers for both components.
        gramian_loss: Gramian orthogonality loss for decorrelated representations.
        homeo_loss: Homeostatic activity loss for firing rate regulation.
        reconstruction_loss: BCE loss for reconstruction accuracy.

    Example:
        >>> # Create and train an autoencoder with configurable gradient flow
        >>> params = AutoencoderParams(..., detach_gradients=True)
        >>> model = Autoencoder(params)
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(model, dataloader)
    """

    def __init__(self, params: Optional[AutoencoderParams] = None):
        """Initialize the Autoencoder with given parameters.

        Sets up the complete autoencoder architecture including encoder, decoder,
        loss functions, and training configuration. The model uses manual optimization
        with configurable gradient flow between encoder and decoder components.

        Args:
            params: AutoencoderParams instance containing model configuration.
                Must include compatible encoder/decoder pairs and all required
                hyperparameters. If None, default parameters will be used (not
                recommended for production use).

        Raises:
            ValueError: If encoder and decoder dimensions are incompatible, as
                validated by validate_dimensions().

        Sets up:
            - Model components: encoder, decoder
            - Loss functions: reconstruction (BCE), Gramian orthogonality, homeostatic
            - Training configuration: manual optimization with configurable gradient flow
            - Hyperparameter weights for loss combination

        Note:
            Manual optimization is enabled to support dual optimizer training.
            The detach_gradients parameter controls whether gradients flow from
            decoder to encoder during training.
        """
        validate_dimensions(params)
        super(Autoencoder, self).__init__()

        # Enable manual optimization for dual optimizer support
        # Required by Lightning when using multiple optimizers in configure_optimizers()
        self.automatic_optimization = False

        # Model components
        self.encoder = params.encoder
        self.decoder = params.decoder

        # Gradient flow control
        self.detach_gradients = params.detach_gradients

        # Training hyperparameters
        self.optimizer_init = params.optimizer_init

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.reconstruction_weight = 1.0
        self.gramian_loss = GramianOrthogonalityLoss(params.gramian_center)
        self.gramian_weight = params.gramian_weight
        self.homeo_loss = HomeostaticActivityLoss(params.rate_target, params.min_active)
        self.homeo_weight = params.homeo_weight

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    # -----------------------------------------------------------------------------------
    # Optimizer configuration
    # -----------------------------------------------------------------------------------

    def configure_optimizers(self) -> List[Optimizer]:
        """Configure two optimizers for split training architecture.

        Creates separate optimizers for encoder and decoder components to enable
        independent optimization with different loss functions. This split training
        approach allows the encoder to focus on learning structured sparse
        representations while the decoder focuses solely on reconstruction accuracy.

        The optimizer configuration uses the same initialization function for both
        components, ensuring consistent hyperparameters (learning rate, momentum, etc.)
        while maintaining separate optimization states.

        Returns:
            List containing two optimizers:
                - Index 0: Encoder optimizer for representation learning losses
                - Index 1: Decoder optimizer for reconstruction losses

        Note:
            The order of optimizers in the returned list is important as it determines
            how they are accessed in training_step() via self.optimizers(). The encoder
            optimizer must be first, followed by the decoder optimizer.

        Example:
            >>> # In training_step:
            >>> enc_opt, dec_opt = self.optimizers()
            >>> # enc_opt trains encoder with composite loss
            >>> # dec_opt trains decoder with reconstruction loss only
        """
        enc_opt = self.optimizer_init(self.encoder.parameters())
        dec_opt = self.optimizer_init(self.decoder.parameters())
        return [enc_opt, dec_opt]

    # -----------------------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------------------

    def forward(self, x: Tensor, *args: Any) -> Tuple[Tensor, Tensor]:
        """Forward pass through the autoencoder.

        Processes input through the encoder to obtain a latent representation,
        then through the decoder to reconstruct the original input. This method
        defines the complete forward computation graph for the autoencoder.

        Args:
            x: Input tensor with shape matching the encoder's expected input shape.
                For cognitive maps, typically (batch_size, channels, height, width).
            *args: Additional arguments (unused, maintained for interface compatibility).

        Returns:
            Tuple containing:
                - reconstruction: Decoded output tensor with same shape as input.
                - embedding: Latent representation tensor from the encoder.

        Note:
            The encoder is called with target=None since no sparse target is
            available during the forward pass. The decoder uses the original
            input as target for potential supervised learning scenarios.
        """
        embedding = self.encoder(x, target=None)  # No sparse target available
        reconstruction = self.decoder(embedding, target=x)

        # Return reconstruction and embedding
        return reconstruction, embedding

    # -----------------------------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------------------------

    def encoder_loss(self, x: Tensor, embedding: Tensor) -> Dict[str, Tensor]:
        """Compute encoder loss components for representation learning.

        Calculates individual loss components that guide the encoder to learn
        structured sparse representations. These losses work together to encourage
        orthogonal, homeostatic, and sparse latent representations that mimic
        the firing patterns observed in hippocampal place cells.

        Args:
            x: Input tensor (unused but kept for interface consistency).
                Shape: (batch_size, channels, height, width).
            embedding: Latent representation tensor from the encoder.
                Shape: (batch_size, latent_dim).

        Returns:
            Dictionary containing individual encoder loss components:
                - "gramian_orthogonality": Promotes decorrelated representations
                - "homeostatic_activity": Regulates firing rates and minimum activity
                - "l1": Encourages sparse activations with L1 penalty

        Note:
            These loss components are combined with their respective weights
            in the training and validation steps to create the final encoder loss.
        """
        return {
            "gramian_orthogonality": self.gramian_loss(embedding),
            "homeostatic_activity": self.homeo_loss(embedding),
        }

    def decoder_loss(self, x: Tensor, reconstruction: Tensor) -> Dict[str, Tensor]:
        """Compute decoder loss components for reconstruction learning.

        Calculates loss components that guide the decoder to accurately reconstruct
        the original input from the latent representations. Currently contains only
        reconstruction loss, but structured as a dictionary for future extensibility.

        Args:
            x: Original input tensor used as the reconstruction target.
                Shape: (batch_size, channels, height, width).
            reconstruction: Decoded output tensor from the decoder.
                Shape: (batch_size, channels, height, width).

        Returns:
            Dictionary containing decoder loss components:
                - "reconstruction": BCE loss measuring reconstruction accuracy

        Note:
            The dictionary structure allows for easy addition of future decoder-specific
            losses such as perceptual loss, adversarial loss, or spatial consistency loss.
        """
        return {
            "reconstruction": self.reconstruction_loss(reconstruction, x),
        }

    # -----------------------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------------------

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        """Initialize registry and hooks before training batch starts.

        Clears the registry at the start of each batch to ensure clean state
        for hook management and potential debugging/monitoring systems.

        Args:
            batch: Training batch (unused in this implementation).
            batch_idx: Index of the current batch (unused in this implementation).

        Note:
            This hook is called automatically by PyTorch Lightning before each
            training batch is processed. The registry clearing ensures that any
            batch-specific state from previous iterations doesn't interfere with
            the current batch.
        """
        # Clear registry at the start of each batch
        registry.clear("batch")

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        """Training step with manual optimization and configurable gradient flow.

        Implements the training strategy where encoder and decoder are trained
        with different loss functions. The detach_gradients parameter
        controls whether gradients flow from decoder to encoder:

        When detach_gradients=True:
        1. Encoder step: Optimized using combined Gramian orthogonality and homeostatic
           activity losses to learn structured sparse representations.
        2. Decoder step: Optimized using reconstruction loss only with detached embedding.

        When detach_gradients=False:
        Uses both optimizers but computes combined loss to allow full gradient flow.

        Args:
            batch: Training batch containing input tensors. Expected format is
                (input_tensor, ...) where additional elements are ignored.
            batch_idx: Index of the current batch in the training epoch. Used
                implicitly by Lightning for scheduling and logging.

        Logs:
            Encoder metrics:
                - train/gramian_loss: Orthogonality constraint loss
                - train/homeostatic_loss: Combined firing rate regulation
                - train/encoder_loss: Combined encoder losses
            Decoder metrics:
                - train/reconstruction_loss: BCE reconstruction error
                - train/decoder_loss: Total decoder loss

        Note:
            Uses manual optimization with two optimizers accessed via self.optimizers().
            The detach_gradients parameter controls the training strategy.
        """
        # Common setup
        x, *_ = batch
        encoder_optimizer, decoder_optimizer = self.optimizers()

        # Common encoder forward pass and loss computation
        embedding = self.encoder(x, target=None)
        encoder_losses = self.encoder_loss(x, embedding)
        encoder_loss = sum(
            [
                self.gramian_weight * encoder_losses["gramian_orthogonality"],
                self.homeo_weight * encoder_losses["homeostatic_activity"],
            ]
        )

        # Common encoder logging
        self.log("train/gramian_loss", encoder_losses["gramian_orthogonality"], on_step=True, on_epoch=True)
        self.log("train/homeostatic_loss", encoder_losses["homeostatic_activity"], on_step=True, on_epoch=True)
        self.log("train/encoder_loss", encoder_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Common decoder forward pass and loss computation
        # Use detached embedding for split training, original for combined training
        z = embedding.detach() if self.detach_gradients else embedding
        reconstruction = self.decoder(z, target=x)
        decoder_losses = self.decoder_loss(x, reconstruction)
        decoder_loss = decoder_losses["reconstruction"]

        if self.detach_gradients:
            # Split training: encoder and decoder trained separately
            encoder_optimizer.zero_grad()
            self.manual_backward(encoder_loss)
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            self.manual_backward(decoder_loss)
            decoder_optimizer.step()

        else:
            # Combined training: both optimizers step with full gradient flow
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            total_loss = encoder_loss + decoder_loss
            self.manual_backward(total_loss)
            encoder_optimizer.step()
            decoder_optimizer.step()

        # Common decoder logging
        self.log("train/reconstruction_loss", decoder_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/decoder_loss", decoder_loss, on_step=True, on_epoch=True, prog_bar=True)

    # -----------------------------------------------------------------------------------
    # Training batch lifecycle hooks
    # -----------------------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Clean up registry after training batch completion.

        Provides a hook for potential cleanup operations after each training batch.
        Currently, registry cleanup is handled by the next batch's on_train_batch_start(),
        and hook cleanup is managed by individual encoder/decoder models that register them.

        Args:
            outputs: Training step outputs (unused in this implementation).
            batch: Training batch (unused in this implementation).
            batch_idx: Index of the current batch (unused in this implementation).

        Note:
            This method complements on_train_batch_start() to provide complete
            lifecycle management for the registry during training. The actual cleanup
            operations are deferred to other parts of the system for efficiency.
        """
        # Registry cleanup is handled by on_train_batch_start of next batch
        # Hook cleanup is handled by individual models that register them
        pass

    # -----------------------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------------------

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Validation step for the autoencoder using split loss evaluation.

        Executes one validation iteration by processing a batch through the model
        and computing validation metrics consistent with the split training approach.
        This method evaluates both encoder and decoder components separately to
        provide detailed performance insights.

        Args:
            batch: Validation batch containing input tensors and optional additional data.
                Expected format: (input_tensor, ...) where input_tensor has shape
                (batch_size, channels, height, width).
            batch_idx: Index of the current batch in the validation epoch (unused).

        Returns:
            Total validation loss tensor (encoder_loss + decoder_loss).

        Logs:
            Encoder metrics:
                - val/gramian_loss: Gramian orthogonality loss
                - val/homeostatic_loss: Combined homeostatic activity loss
                - val/l1_loss: L1 sparsity penalty
                - val/encoder_loss: Weighted total encoder loss
            Decoder metrics:
                - val/reconstruction_loss: BCE reconstruction loss
                - val/decoder_loss: Total decoder loss (same as reconstruction)
            Combined metrics:
                - val/total_loss: Combined encoder and decoder loss
                - val/sparsity_rate: Percentage of active units (> 0.01 threshold)

        Note:
            Validation metrics are logged per-epoch only to reduce logging overhead
            and focus on epoch-level performance tracking. This follows the same
            split loss approach as the training_step for consistency.
        """
        x, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        reconstruction, embedding = self(x)

        # Compute encoder loss components using the same method as training
        encoder_losses = self.encoder_loss(x, embedding)
        encoder_loss = sum(
            [
                self.gramian_weight * encoder_losses["gramian_orthogonality"],
                self.homeo_weight * encoder_losses["homeostatic_activity"],
            ]
        )

        # Compute decoder loss components using the same method as training
        decoder_losses = self.decoder_loss(x, reconstruction)
        decoder_loss = decoder_losses["reconstruction"]

        # Calculate total loss
        total_loss = encoder_loss + decoder_loss

        # Log encoder metrics
        self.log("val/gramian_loss", encoder_losses["gramian_orthogonality"], on_step=False, on_epoch=True)
        self.log("val/homeostatic_loss", encoder_losses["homeostatic_activity"], on_step=False, on_epoch=True)
        self.log("val/encoder_loss", encoder_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log decoder metrics
        self.log("val/reconstruction_loss", decoder_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/decoder_loss", decoder_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log combined metrics
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        sparsity_rate = (embedding > 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, on_step=False, on_epoch=True)

        return total_loss

    # -----------------------------------------------------------------------------------
    # Test step
    # -----------------------------------------------------------------------------------

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:  # noqa: ARG002
        """Test step for the autoencoder using split loss evaluation.

        Executes one test iteration by processing a batch through the model and
        computing comprehensive test metrics. This method provides detailed
        evaluation metrics beyond the standard training/validation losses,
        following the same split training approach for consistency.

        Args:
            batch: Test batch containing input tensors and optional additional data.
                Expected format: (input_tensor, ...) where input_tensor has shape
                (batch_size, channels, height, width).
            batch_idx: Index of the current batch in the test epoch (unused).

        Returns:
            Dictionary containing all computed test metrics.

        Logs:
            Encoder metrics:
                - test/gramian_loss: Gramian orthogonality loss
                - test/homeostatic_loss: Combined homeostatic activity loss
                - test/l1_loss: L1 sparsity penalty
                - test/encoder_loss: Weighted total encoder loss
            Decoder metrics:
                - test/reconstruction_loss: BCE reconstruction loss
                - test/decoder_loss: Total decoder loss (same as reconstruction)
            Combined metrics:
                - test/total_loss: Combined encoder and decoder loss
                - test/mse: Mean squared error between input and reconstruction
                - test/mae: Mean absolute error between input and reconstruction
                - test/sparsity_rate: Percentage of active units (> 0.01 threshold)

        Note:
            Test metrics are logged per-epoch only and include additional
            evaluation metrics (MSE, MAE) for comprehensive model assessment.
            The split loss approach ensures consistency with training/validation.
        """
        x, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        reconstruction, embedding = self(x)

        # Compute encoder loss components using the same method as training
        encoder_losses = self.encoder_loss(x, embedding)
        encoder_loss = sum(
            [
                self.gramian_weight * encoder_losses["gramian_orthogonality"],
                self.homeo_weight * encoder_losses["homeostatic_activity"],
            ]
        )

        # Compute decoder loss components using the same method as training
        decoder_losses = self.decoder_loss(x, reconstruction)
        decoder_loss = decoder_losses["reconstruction"]

        # Calculate total loss and additional metrics
        total_loss = encoder_loss + decoder_loss
        mse = nn.MSELoss()(reconstruction, x)
        mae = nn.L1Loss()(reconstruction, x)
        sparsity_rate = (embedding > 0.01).float().mean()

        # Create metrics dictionary
        metrics = {
            "test/gramian_loss": encoder_losses["gramian_orthogonality"],
            "test/homeostatic_loss": encoder_losses["homeostatic_activity"],
            "test/encoder_loss": encoder_loss,
            "test/reconstruction_loss": decoder_loss,
            "test/decoder_loss": decoder_loss,
            "test/total_loss": total_loss,
            "test/mse": mse,
            "test/mae": mae,
            "test/sparsity_rate": sparsity_rate,
        }

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    # -----------------------------------------------------------------------------------
    # Prediction step
    # -----------------------------------------------------------------------------------

    def predict_step(self, batch: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor]:  # noqa: ARG002
        """Prediction step returns reconstructions and embeddings.

        Executes one prediction iteration by processing a batch through the model
        and returning both the reconstructed outputs and the latent embeddings.
        This method is useful for inference and analysis of learned representations.

        Args:
            batch: Input batch containing tensors for prediction.
                Expected format: (input_tensor, ...) where input_tensor has shape
                (batch_size, channels, height, width).
            batch_idx: Index of the current batch in prediction (unused).

        Returns:
            Tuple containing:
                - reconstruction: Reconstructed outputs with same shape as input.
                - embedding: Latent representations from the encoder.

        Note:
            This method is typically used during inference phases or when
            analyzing the learned latent representations for research purposes.
        """
        x, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        return self(x)

    # -----------------------------------------------------------------------------------
    # Model properties
    # -----------------------------------------------------------------------------------

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of the input feature map.

        Returns:
            3D tuple representing (channels, height, width) of expected input tensors.

        Note:
            This property provides convenient access to the encoder's expected
            input dimensions for validation and preprocessing purposes.
        """
        return self.encoder.input_shape

    @property
    def input_channels(self) -> int:
        """Returns the number of input channels.

        Returns:
            Integer representing the number of channels in the input tensor.
            For cognitive maps, this is typically 1 (single-channel obstacle grid).

        Note:
            This property extracts the channel dimension from the input shape
            for convenient access during model configuration and debugging.
        """
        return self.encoder.input_channels

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """Returns the spatial dimensions as (height, width).

        Returns:
            2D tuple representing (height, width) of the spatial input dimensions.
            For cognitive maps, this might be (32, 16) representing the grid size.

        Note:
            This property extracts the spatial dimensions from the input shape,
            excluding the channel dimension, for convenient access during
            spatial processing and visualization.
        """
        return self.encoder.spatial_dimensions

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent representation.

        Returns:
            Integer representing the size of the compressed latent space.
            This dimension determines the bottleneck capacity of the autoencoder.

        Note:
            The latent dimension is crucial for controlling the representational
            capacity and the degree of compression achieved by the autoencoder.
            Smaller values force more compression but may lose information.
        """
        return self.encoder.latent_dim

    # -----------------------------------------------------------------------------------
    # Additional utility methods
    # -----------------------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation.

        Processes input through the encoder component only, returning the
        compressed latent representation without reconstruction. Useful for
        analyzing learned representations or using the encoder independently.

        Args:
            x: Input tensor with shape matching the encoder's expected dimensions.
                For cognitive maps, typically (batch_size, channels, height, width).

        Returns:
            Latent representation tensor with shape (batch_size, latent_dim).

        Note:
            This method bypasses the decoder, making it efficient for tasks
            that only require the encoded representation, such as clustering
            or dimensionality reduction analysis. The encoder is called with
            target=None since no supervised target is available.
        """
        return self.encoder(x, target=None)

    def decode(self, embedding: Tensor) -> Tensor:
        """Decode latent representation to reconstruction.

        Processes latent representations through the decoder component only,
        returning reconstructed outputs. Useful for generating new samples
        from latent space or using the decoder independently.

        Args:
            embedding: Latent representation tensor with shape (batch_size, latent_dim).

        Returns:
            Reconstructed output tensor with shape matching the original input dimensions.

        Note:
            This method can be used for generative tasks by providing custom
            latent representations, or for analyzing the decoder's reconstruction
            capabilities independently of the encoder.
        """
        return self.decoder(embedding)

    def reconstruct(self, x: Tensor) -> Tensor:
        """Full reconstruction from input.

        Performs complete encode-decode cycle by processing input through both
        the encoder and decoder components. This is equivalent to the forward
        method but returns only the reconstruction without the embedding.

        Args:
            x: Input tensor with shape matching the encoder's expected dimensions.

        Returns:
            Reconstructed output tensor with same shape as input.

        Note:
            This method is convenient for evaluation and testing scenarios
            where only the final reconstruction is needed, without access
            to the intermediate latent representation.
        """
        embedding = self.encode(x)
        return self.decode(embedding)


# -----------------------------------------------------------------------------------
# Example usage and testing
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    """Example demonstrating autoencoder training with configurable gradient flow.

    This example shows how to:
    1. Create a simple dataset of random obstacle maps
    2. Configure autoencoder parameters with gradient flow control
    3. Train the model using PyTorch Lightning with manual optimization
    4. Test reconstruction performance and sparsity metrics

    The example uses Linear encoder/decoder components with GELU activation
    and demonstrates the gradient detachment approach using manual optimization
    with a simple conditional for controlling gradient flow.
    """
    from functools import partial

    import lightning.pytorch as pl

    from ehc_sn.models import decoders, encoders

    # Simple dataset for testing
    class SimpleDataset(torch.utils.data.Dataset):
        """Simple dataset generating random binary obstacle maps."""

        def __init__(self, size: int = 100):
            """Initialize dataset with specified size.

            Args:
                size: Number of samples in the dataset.
            """
            self.size = size

        def __len__(self):
            """Return dataset size."""
            return self.size

        def __getitem__(self, idx):
            """Generate a random obstacle map sample.

            Args:
                idx: Sample index (unused, generates random data).

            Returns:
                Tuple containing a single tensor with random obstacle placement.
            """
            # Create random obstacle maps with random obstacle placement
            x = torch.zeros(1, 16, 32)
            num_obstacles = torch.randint(3, 8, (1,)).item()
            x[0, torch.randint(0, 16, (num_obstacles,)), torch.randint(0, 32, (num_obstacles,))] = 1.0
            return (x,)  # Return as tuple for batch unpacking

    # Create autoencoder with configurable gradient flow
    autoencoder_params = AutoencoderParams(
        encoder=encoders.Linear(
            encoders.EncoderParams(input_shape=(1, 16, 32), latent_dim=32, activation_fn=torch.nn.GELU)
        ),
        decoder=decoders.Linear(
            decoders.DecoderParams(output_shape=(1, 16, 32), latent_dim=32, activation_fn=torch.nn.GELU)
        ),
        gramian_weight=1.0,
        homeo_weight=1.0,
        rate_target=0.10,
        detach_gradients=True,  # Use gradient detachment for split training
        optimizer_init=partial(torch.optim.Adam, lr=1e-3),
    )
    model = Autoencoder(autoencoder_params)

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
