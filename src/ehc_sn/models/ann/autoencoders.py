from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import lightning.pytorch as pl
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.hooks import registry
from ehc_sn.models.ann import encoders
from ehc_sn.models.ann.decoders import BaseDecoder, DecoderParams
from ehc_sn.models.ann.encoders import BaseEncoder, EncoderParams
from ehc_sn.modules.loss import GramianOrthogonalityLoss, HomeostaticActivityLoss, TargetL1SparsityLoss
from ehc_sn.trainers.core import BaseTrainer


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

    Example:
        >>> from functools import partial
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
            f"does not match output shape of decoder ({params.decoder.output_shape})."
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

    def __init__(self, params: AutoencoderParams, trainer: Optional[BaseTrainer] = None):
        """Initialize the Autoencoder with given parameters and training strategy.

        Sets up the complete autoencoder architecture including encoder, decoder,
        loss functions, and training configuration. The model delegates training
        logic to the provided trainer strategy, maintaining clean separation
        between model architecture and training algorithms.

        Args:
            params: AutoencoderParams instance containing model configuration.
                Must include compatible encoder/decoder pairs and all required
                hyperparameters.
            trainer: Training strategy to use. If None, a DetachedTrainer will
                be created using the optimizer_init from params.

        Raises:
            ValueError: If encoder and decoder dimensions are incompatible, as
                validated by validate_dimensions().

        Sets up:
            - Model components: encoder, decoder
            - Loss functions: reconstruction (BCE), Gramian orthogonality, homeostatic
            - Training strategy: delegates to provided trainer
            - Hyperparameter weights for loss combination

        Note:
            Manual optimization is enabled to support custom training strategies.
            The trainer controls gradient flow and optimization behavior.
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
        """Configure optimizers using the training strategy.

        Delegates optimizer configuration to the training strategy,
        allowing different training algorithms to configure optimizers
        according to their specific requirements.

        Returns:
            List of optimizers configured by the training strategy.

        Note:
            The training strategy determines the number and configuration
            of optimizers based on the training algorithm requirements.
        """
        if self.trainer_strategy is None:
            raise ValueError("Trainer strategy must be provided to configure optimizers.")
        return self.trainer_strategy.configure_optimizers(self)

    # -----------------------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------------------

    def forward(self, x: Tensor, *args: Any, **kwds: dict) -> Tuple[Tensor, Tensor]:
        """Forward pass through the autoencoder.

        Processes input through the encoder to obtain a latent representation,
        then through the decoder to reconstruct the original input. The gradient
        flow behavior depends on the training strategy being used.

        Args:
            x: Input tensor with shape matching the encoder's expected input shape.
                For cognitive maps, typically (batch_size, channels, height, width).
            *args: Additional arguments (unused, maintained for interface compatibility).
            **kwds: Additional keyword arguments (unused, maintained for interface compatibility).

        Returns:
            Tuple containing:
                - reconstruction: Decoded output tensor with same shape as input.
                - embedding: Latent representation tensor from the encoder.

        Note:
            The encoder is called with target=None since no sparse target is
            available during the forward pass. The decoder uses the original
            input as target for potential supervised learning scenarios.
        """
        embedding = self.encoder(x, *args, target=None, **kwds)  # No sparse target
        reconstruction = self.decoder(embedding, *args, target=x, **kwds)

        # Return reconstruction and embedding
        return reconstruction, embedding

    # -----------------------------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------------------------

    def compute_loss(self, x: Tensor, log_label: str) -> Tuple[Tensor, Tensor]:
        """Compute and log decoder + encoder losses for a batch.

        Encapsulates the full loss decomposition used across training,
        validation, and test loops. Performs a forward pass, derives
        regularization (Gramian + homeostatic) and reconstruction losses,
        then logs them with a namespace prefix.

        Args:
            x: Input batch tensor (first element of dataloader batch) whose
               shape excluding batch dimension matches ``self.input_shape``.
            log_label: Namespace prefix for metric logging (``train``,
               ``validation`` or ``test``).

        Returns:
            (decoder_loss, encoder_loss) where:
              decoder_loss: Reconstruction criterion (BCE mean).
              encoder_loss: Weighted sum of Gramian + homeostatic penalties.

        Logging:
            Emits (step + epoch):
              ``{log_label}/reconstruction_loss``
              ``{log_label}/gramian_loss``
              ``{log_label}/homeostatic_loss``
              ``{log_label}/decoder_loss`` (prog bar)
              ``{log_label}/encoder_loss`` (prog bar)

        Notes:
            Keeps gradient behaviour consistent with current mode; Lightning
            disables grads automatically during validation/test evaluation.
        """
        # Forward pass to get reconstruction and embedding
        reconstruction, embedding = self(x)

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
        reconstruction_loss = self.reconstruction_loss(reconstruction, x)
        self.log(f"{log_label}/reconstruction_loss", reconstruction_loss, on_epoch=True)
        decoder_loss = reconstruction_loss
        self.log(f"{log_label}/decoder_loss", decoder_loss, on_epoch=True, prog_bar=True)

        return decoder_loss, encoder_loss

    # -----------------------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------------------

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        """Initialize registry and delegate to training strategy.

        Clears the registry at the start of each batch and delegates
        to the training strategy for any additional setup.

        Args:
            batch: Training batch.
            batch_idx: Index of the current batch.
        """
        if self.trainer_strategy is None:
            raise ValueError("Trainer strategy must be provided for training.")

        # Clear registry at the start of each batch
        registry.clear("batch")

        # Delegate to training strategy
        self.trainer_strategy.on_train_batch_start(self, batch, batch_idx)

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        """Delegate training step to the training strategy.

        Uses the training strategy to perform the training step,
        allowing different training algorithms to implement their
        specific optimization logic.

        Args:
            batch: Batch tuple whose first element is the input tensor.
            batch_idx: Index within the epoch.

        Note:
            All training logic is handled by the training strategy,
            including gradient computation, optimizer steps, and
            any special handling required by the algorithm.
        """
        return self.trainer_strategy.training_step(self, batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Delegate batch end cleanup to training strategy.

        Provides a hook for training strategy-specific cleanup operations
        after each training batch.

        Args:
            outputs: Training step outputs.
            batch: Training batch.
            batch_idx: Index of the current batch.
        """
        self.trainer_strategy.on_train_batch_end(self, outputs, batch, batch_idx)

    # -----------------------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------------------

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Delegate validation step to the training strategy.

        Uses the training strategy to perform the validation step,
        ensuring consistency between training and validation logic.

        Args:
            batch: Validation batch.
            batch_idx: Index of the batch (unused).

        Returns:
            Validation loss tensor.
        """
        return self.trainer_strategy.validation_step(self, batch, batch_idx)

    # -----------------------------------------------------------------------------------
    # Test step
    # -----------------------------------------------------------------------------------

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Evaluate the model on a held-out test batch and return total loss.

        Mirrors `validation_step` but logs metrics under the `test/` namespace
        to clearly separate final evaluation from validation monitoring.

        Args:
            batch: Test batch structured as `(x, *_)` where `x` conforms to
                the expected input tensor shape.
            batch_idx: Sequential index of the test batch (unused).

        Returns:
            A scalar tensor equal to the sum of decoder reconstruction loss
            and encoder regularization loss for this batch.

        Notes:
            - No gradients are produced (evaluation mode).
            - Individual component losses are already logged by `compute_loss`.
            - Keeping symmetry with `training_step` and `validation_step`
              simplifies downstream aggregation and comparisons.
        """
        x, *_ = batch
        dec_loss, enc_loss = self.compute_loss(x, log_label="test")
        return dec_loss + enc_loss

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

    from ehc_sn.models.ann import decoders
    from ehc_sn.trainers.back_propagation import DetachedTrainer

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
