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

    Ensures encoder and decoder can work together by checking:
    1. Encoder input shape matches decoder output shape
    2. Encoder latent dimension matches decoder latent dimension

    Args:
        params: AutoencoderParams containing encoder and decoder configurations.

    Raises:
        ValueError: If dimensions are incompatible.
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

    def __init__(self, params: AutoencoderParams, trainer: BaseTrainer):
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
        """Configure optimizers using the training strategy.

        Returns:
            List of optimizers configured by the training strategy.

        Raises:
            ValueError: If no trainer strategy is provided.
        """
        if self.trainer_strategy is None:
            raise ValueError("Trainer strategy must be provided to configure optimizers.")
        return self.trainer_strategy.configure_optimizers(self)

    # -----------------------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------------------

    def forward(self, x: Tensor, detach_gradients: bool = False, *args: Any, **kwds: dict) -> Tuple[Tensor, Tensor]:
        """Forward pass through encoder and decoder.

        Args:
            x: Input tensor matching encoder's expected shape.
            detach_gradients: Whether to detach gradients between encoder and decoder.
            *args: Additional arguments for interface compatibility.
            **kwds: Additional keyword arguments for interface compatibility.

        Returns:
            Tuple of (reconstruction, embedding) tensors.

        Note:
            When detach_gradients=True, prevents gradient flow from decoder to encoder.
        """
        embedding = self.encoder(x, *args, target=None, **kwds)  # No sparse target

        # Optionally detach gradients between encoder and decoder
        z = embedding.detach() if detach_gradients else embedding

        reconstruction = self.decoder(z, *args, target=x, **kwds)

        # Return reconstruction and embedding (always return original embedding)
        return reconstruction, embedding

    # -----------------------------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------------------------

    def compute_loss(self, x: Tensor, log_label: str, detach_gradients: bool = False) -> Tuple[Tensor, Tensor]:
        """Compute and log decoder and encoder losses for a batch.

        Performs forward pass and computes reconstruction, Gramian orthogonality,
        and homeostatic activity losses with logging.

        Args:
            x: Input batch tensor.
            log_label: Namespace prefix for metric logging (e.g., 'train', 'val').
            detach_gradients: Whether to detach gradients between encoder and decoder.

        Returns:
            Tuple of (decoder_loss, encoder_loss).

        Logs:
            - reconstruction_loss, gramian_loss, homeostatic_loss
            - decoder_loss, encoder_loss (with progress bar)
            - sparsity_rate
        """
        # Forward pass to get reconstruction and embedding
        reconstruction, embedding = self.forward(x, detach_gradients)

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

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        """Delegate training step to the training strategy.

        Args:
            batch: Training batch.
            batch_idx: Batch index within the epoch.

        Note:
            All training logic is handled by the training strategy.
        """
        return self.trainer_strategy.training_step(self, batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
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

        def __init__(self, size: int = 100):
            """Initialize with specified dataset size."""
            self.size = size

        def __len__(self):
            """Return dataset size."""
            return self.size

        def __getitem__(self, idx):
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
