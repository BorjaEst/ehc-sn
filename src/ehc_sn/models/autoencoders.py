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
        z = embedding.detach() if self.detach_gradients else embedding
        reconstruction = self.decoder(z, target=x)

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
        self.log(f"{log_label}/gramian_loss", gramian_loss, on_step=True, on_epoch=True)
        homeo_loss = self.homeo_loss(embedding)
        self.log(f"{log_label}/homeostatic_loss", homeo_loss, on_step=True, on_epoch=True)
        encoder_loss = self.gramian_weight * gramian_loss + self.homeo_weight * homeo_loss
        self.log(f"{log_label}/encoder_loss", encoder_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Decoder-side losses
        reconstruction_loss = self.reconstruction_loss(reconstruction, x)
        self.log(f"{log_label}/reconstruction_loss", reconstruction_loss, on_step=True, on_epoch=True)
        decoder_loss = reconstruction_loss
        self.log(f"{log_label}/decoder_loss", decoder_loss, on_step=True, on_epoch=True, prog_bar=True)

        return decoder_loss, encoder_loss

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
        """Perform one training iteration (manual optimization).

        Uses :meth:`compute_loss` to obtain the two loss components then
        dispatches either split or joint optimization depending on
        ``self.detach_gradients``.

        Args:
            batch: Batch tuple whose first element is the input tensor.
            batch_idx: Index within the epoch (unused).

        Notes:
            Skips backward/step if a module is frozen or its loss does not
            require gradients.
        """
        x, *_ = batch
        dec_loss, enc_loss = self.compute_loss(x, log_label="train")
        if self.detach_gradients:
            self.train_detached(enc_loss, dec_loss)
        else:
            self.train_full(enc_loss, dec_loss)

    def train_detached(self, enc_loss: Tensor, dec_loss: Tensor) -> None:
        """Optimize encoder and decoder independently (split mode).

        The latent embedding passed to the decoder was detached during the
        forward pass, so reconstruction gradients cannot influence encoder
        weights. Each component receives a separate backward pass and optimizer
        step if (a) it has trainable parameters and (b) its loss requires
        gradients.

        Args:
            enc_loss: Encoder composite sparsity / regularization loss.
            dec_loss: Decoder reconstruction loss.

        Notes:
            - Ensures zeroed gradients per component before its backward pass.
            - Safe-guards avoid unnecessary backward calls when modules are
              frozen (supports curriculum or staged training).
        """
        enc_opt, dec_opt = self.optimizers()
        enc_do_step = any(p.requires_grad for p in self.encoder.parameters()) and enc_loss.requires_grad
        dec_do_step = any(p.requires_grad for p in self.decoder.parameters()) and dec_loss.requires_grad
        if enc_do_step:
            enc_opt.zero_grad()
            self.manual_backward(enc_loss)
            enc_opt.step()
        if dec_do_step:
            dec_opt.zero_grad()
            self.manual_backward(dec_loss)
            dec_opt.step()

    def train_full(self, enc_loss: Tensor, dec_loss: Tensor) -> None:
        """Optimize encoder + decoder jointly with full gradient flow.

        Sums losses and performs a single backward pass so decoder (reconstruction)
        gradients propagate through the encoder. Each optimizer steps if its
        parameter set is trainable.

        Args:
            enc_loss: Encoder sparsity / regularization term.
            dec_loss: Decoder reconstruction term.

        Notes:
            - Skips early if both modules are effectively frozen.
            - Keeps separate optimizers to allow potential divergence of
              schedulers or hyperparameters later while still sharing a
              unified backward graph.
        """
        enc_opt, dec_opt = self.optimizers()
        enc_do_step = any(p.requires_grad for p in self.encoder.parameters()) and enc_loss.requires_grad
        dec_do_step = any(p.requires_grad for p in self.decoder.parameters()) and dec_loss.requires_grad
        if not (enc_do_step or dec_do_step):
            return
        if enc_do_step:
            enc_opt.zero_grad()
        if dec_do_step:
            dec_opt.zero_grad()
        total_loss = enc_loss + dec_loss
        self.manual_backward(total_loss)
        if enc_do_step:
            enc_opt.step()
        if dec_do_step:
            dec_opt.step()

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
        """Run one validation iteration and return combined loss.

        Performs a forward pass on a validation batch, computing both decoder
        (reconstruction) and encoder (regularization / sparsity) losses via
        :meth:`compute_loss`. Returns their sum so Lightning can aggregate it
        for epoch metrics. Individual component losses are logged inside
        :meth:`compute_loss` under the ``validation/`` namespace
        (e.g. ``validation/reconstruction_loss``).

        Args:
            batch: Validation batch. Expected structure ``(x, *_)`` where ``x``
                matches the model's ``input_shape``.
            batch_idx: Index of the batch inside the validation epoch (unused).

        Returns:
            Scalar tensor: ``decoder_loss + encoder_loss`` for the batch.

        Notes:
            - Gradients are disabled automatically in eval mode.
            - Wrapper keeps loop minimal; decomposition lives in
              :meth:`compute_loss`.
        """
        x, *_ = batch
        dec_loss, enc_loss = self.compute_loss(x, log_label="val")
        return dec_loss + enc_loss

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
