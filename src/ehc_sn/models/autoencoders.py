from typing import Any, Callable, Dict, Optional, Tuple, Type

import lightning.pytorch as pl
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.models.decoders import BaseDecoder, DecoderParams
from ehc_sn.models.encoders import BaseEncoder, EncoderParams


class AutoencoderParams(BaseModel):
    """Parameters for configuring the Autoencoder model.

    This configuration class defines all the parameters needed to construct
    and train an autoencoder for the entorhinal-hippocampal circuit modeling.
    It ensures that the encoder and decoder components are properly configured
    and compatible with each other.

    Attributes:
        encoder: The encoder component that transforms input into latent representations.
            Must be an instance of BaseEncoder with appropriate input dimensions.
        decoder: The decoder component that reconstructs outputs from latent representations.
            Must be an instance of BaseDecoder with output dimensions matching encoder input.
        sparsity_weight: Weight coefficient for the sparsity regularization term.
            Higher values encourage sparser latent representations, mimicking the sparse
            firing patterns observed in hippocampal neurons. Typical range: 0.01-1.0.
        sparsity_target: Target average activation level for latent units.
            Lower values encourage sparser representations. Typical values: 1e-4 to 1e-2.
        optimizer_init: Callable function to initialize the optimizer.
            Should be a partial function or lambda that takes model parameters as input
            and returns an optimizer instance (e.g., functools.partial(torch.optim.Adam, lr=1e-3)).

    Example:
        >>> from functools import partial
        >>> params = AutoencoderParams(
        ...     encoder=Linear(EncoderParams(...)),
        ...     decoder=Linear(DecoderParams(...)),
        ...     sparsity_weight=0.1,
        ...     sparsity_target=1e-3,
        ...     optimizer_init=partial(torch.optim.Adam, lr=1e-3)
        ... )
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    encoder: BaseEncoder = Field(..., description="Parameters for the encoder component.")
    decoder: BaseDecoder = Field(..., description="Parameters for the decoder component.")
    sparsity_weight: float = Field(0.1, description="Weight for the sparsity loss term.")
    sparsity_target: float = Field(-1e-3, description="Target sparsity level for the embedding.")
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
    """Neural network autoencoder for the entorhinal-hippocampal circuit.

    This autoencoder combines an encoder that transforms spatial input into a compact
    embedding and a decoder that reconstructs the original input from the embedding.
    It supports sparse activations to model the sparse firing patterns observed in
    the hippocampus, which is critical for pattern separation and completion.

    The autoencoder serves as a computational model for how the entorhinal-hippocampal
    circuit might encode, store, and retrieve spatial information. It incorporates
    biologically-inspired sparsity constraints that encourage the model to learn
    efficient representations similar to those found in hippocampal place cells.

    Key Features:
        - Sparse latent representations mimicking hippocampal neuron firing patterns
        - Configurable encoder/decoder architectures for different input types
        - Integrated sparsity regularization with adjustable target levels
        - Lightning-based training with comprehensive logging and metrics
        - Support for both training and inference modes with proper checkpointing

    Architecture:
        Input → Encoder → Sparse Latent Representation → Decoder → Reconstruction

    Loss Function:
        Total Loss = Reconstruction Loss + λ × Sparsity Loss
        where λ (sparsity_weight) controls the trade-off between reconstruction
        accuracy and representation sparsity.

    Args:
        params: AutoencoderParams instance containing all configuration parameters.
            Must include compatible encoder/decoder pairs and training hyperparameters.

    Attributes:
        encoder: The encoder neural network component.
        decoder: The decoder neural network component.
        optimizer_init: Function to initialize the optimizer.
        reconstruction_loss: Binary cross-entropy loss for reconstruction.
        sparsity_loss: L1 loss for enforcing sparsity in latent representations.
        sparsity_target: Target activation level for sparsity regularization.
        sparsity_weight: Weight coefficient for sparsity loss term.

    Example:
        >>> # Create and train an autoencoder
        >>> params = AutoencoderParams(...)
        >>> model = Autoencoder(params)
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(model, dataloader)
    """

    def __init__(self, params: Optional[AutoencoderParams] = None):
        """Initialize the Autoencoder with given parameters.

        Args:
            params: AutoencoderParams instance containing model configuration.
                If None, default parameters will be used (not recommended for
                production use).

        Raises:
            ValueError: If encoder and decoder dimensions are incompatible.
        """
        validate_dimensions(params)
        super(Autoencoder, self).__init__()

        # Model components
        self.encoder = params.encoder
        self.decoder = params.decoder

        # Training hyperparameters
        self.optimizer_init = params.optimizer_init

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.sparsity_loss = nn.L1Loss()
        self.sparsity_target = params.sparsity_target
        self.sparsity_weight = params.sparsity_weight

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    # -----------------------------------------------------------------------------------
    # Optimizer configuration
    # -----------------------------------------------------------------------------------

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer for training.

        This method is called by PyTorch Lightning to set up the optimizer
        for training the autoencoder. It uses the optimizer initialization
        function provided in the model parameters.

        Returns:
            Optimizer: Configured optimizer instance ready for training.

        Note:
            The optimizer is initialized with the model's parameters using
            the optimizer_init function from the configuration.
        """
        optimizer = self.optimizer_init(self.parameters())  # functools.partial or similar
        return optimizer

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
        return reconstruction, embedding

    # -----------------------------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------------------------

    def compute_loss(self, x: Tensor, reconstruction: Tensor, embedding: Tensor) -> Dict[str, Tensor]:
        """Compute all losses for a given input.

        Calculates the total loss as a combination of reconstruction loss and
        sparsity regularization loss. The reconstruction loss measures how well
        the autoencoder reproduces the input, while the sparsity loss encourages
        sparse latent representations similar to biological neural activity.

        Args:
            x: Original input tensor used as reconstruction target.
            reconstruction: Reconstructed output from the decoder.
            embedding: Latent representation from the encoder.

        Returns:
            Dictionary containing:
                - 'reconstruction': Binary cross-entropy loss between input and reconstruction.
                - 'sparsity': L1 loss between embedding and sparsity target.
                - 'total': Weighted sum of reconstruction and sparsity losses.

        Note:
            The sparsity loss compares the embedding activations to a tensor filled
            with the target sparsity value, encouraging most units to remain near
            the target activation level (typically close to zero).
        """

        # Calculate reconstruction and sparsity losses
        recon_loss = self.reconstruction_loss(reconstruction, x)
        sparsity_target = torch.full_like(embedding, self.sparsity_target)
        sparsity_loss = self.sparsity_loss(embedding, sparsity_target)
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Return a dictionary of losses
        return {"reconstruction": recon_loss, "sparsity": sparsity_loss, "total": total_loss}

    # -----------------------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------------------

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Training step for the autoencoder.

        Executes one training iteration by processing a batch through the model,
        computing losses, and logging training metrics. This method is called
        automatically by PyTorch Lightning during training.

        Args:
            batch: Training batch containing input tensors and optional additional data.
                Expected format: (input_tensor, ...) where input_tensor has shape
                (batch_size, channels, height, width).
            batch_idx: Index of the current batch in the training epoch.

        Returns:
            Total loss tensor for backpropagation.

        Logs:
            - train/reconstruction_loss: BCE loss for reconstruction accuracy
            - train/sparsity_loss: L1 loss for sparsity regularization
            - train/total_loss: Combined weighted loss
            - train/sparsity_rate: Percentage of active units (> 0.01 threshold)

        Note:
            Metrics are logged both per-step and per-epoch for comprehensive
            monitoring during training.
        """

        x, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        reconstruction, embedding = self(x)
        loss = self.compute_loss(x, reconstruction, embedding)  # Calculate losses

        # Log metrics
        self.log("train/reconstruction_loss", loss["reconstruction"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/sparsity_loss", loss["sparsity"], on_step=True, on_epoch=True)
        self.log("train/total_loss", loss["total"], on_step=True, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        sparsity_rate = (embedding > 0.01).float().mean()
        self.log("train/sparsity_rate", sparsity_rate, on_step=True, on_epoch=True)

        return loss["total"]

    # -----------------------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------------------

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Validation step for the autoencoder.

        Executes one validation iteration by processing a batch through the model
        and computing validation metrics. This method is called automatically by
        PyTorch Lightning during validation phases.

        Args:
            batch: Validation batch containing input tensors and optional additional data.
                Expected format: (input_tensor, ...) where input_tensor has shape
                (batch_size, channels, height, width).
            batch_idx: Index of the current batch in the validation epoch (unused).

        Returns:
            Total validation loss tensor.

        Logs:
            - val/reconstruction_loss: BCE loss for reconstruction accuracy
            - val/sparsity_loss: L1 loss for sparsity regularization
            - val/total_loss: Combined weighted loss
            - val/sparsity_rate: Percentage of active units (> 0.01 threshold)

        Note:
            Validation metrics are logged per-epoch only to reduce logging overhead
            and focus on epoch-level performance tracking.
        """

        x, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        reconstruction, embedding = self(x)
        loss = self.compute_loss(x, reconstruction, embedding)  # Calculate losses

        # Log metrics
        self.log("val/reconstruction_loss", loss["reconstruction"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/sparsity_loss", loss["sparsity"], on_step=False, on_epoch=True)
        self.log("val/total_loss", loss["total"], on_step=False, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        sparsity_rate = (embedding > 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, on_step=False, on_epoch=True)

        return loss["total"]

    # -----------------------------------------------------------------------------------
    # Test step
    # -----------------------------------------------------------------------------------

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:  # noqa: ARG002
        """Test step for the autoencoder.

        Executes one test iteration by processing a batch through the model and
        computing comprehensive test metrics. This method provides detailed
        evaluation metrics beyond the standard training/validation losses.

        Args:
            batch: Test batch containing input tensors and optional additional data.
                Expected format: (input_tensor, ...) where input_tensor has shape
                (batch_size, channels, height, width).
            batch_idx: Index of the current batch in the test epoch (unused).

        Returns:
            Dictionary containing all computed test metrics.

        Logs:
            - test/reconstruction_loss: BCE loss for reconstruction accuracy
            - test/sparsity_loss: L1 loss for sparsity regularization
            - test/total_loss: Combined weighted loss
            - test/mse: Mean squared error between input and reconstruction
            - test/mae: Mean absolute error between input and reconstruction
            - test/sparsity_rate: Percentage of active units (> 0.01 threshold)

        Note:
            Test metrics are logged per-epoch only and include additional
            evaluation metrics (MSE, MAE) for comprehensive model assessment.
        """

        x, *_ = batch  # Unpack batch, assuming first element is the cognitive map tensor
        reconstruction, embedding = self(x)
        loss = self.compute_loss(x, reconstruction, embedding)  # Calculate losses

        # Calculate additional metrics
        metrics = {
            "test/reconstruction_loss": loss["reconstruction"],
            "test/sparsity_loss": loss["sparsity"],
            "test/total_loss": loss["total"],
            "test/mse": nn.MSELoss()(reconstruction, x),
            "test/mae": nn.L1Loss()(reconstruction, x),
            "test/sparsity_rate": (embedding > 0.01).float().mean(),
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

        Returns:
            Latent representation tensor with shape (batch_size, latent_dim).

        Note:
            This method bypasses the decoder, making it efficient for tasks
            that only require the encoded representation, such as clustering
            or dimensionality reduction analysis.
        """
        return self.encoder(x)

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


# Example usage with training
if __name__ == "__main__":
    from functools import partial

    import lightning.pytorch as pl

    from ehc_sn.models import decoders, encoders

    # Simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size: int = 100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Random obstacle maps - make sure to return the right tensor format
            x = torch.zeros(1, 16, 32)
            x[0, torch.randint(0, 16, (5,)), torch.randint(0, 32, (5,))] = 1.0
            return (x,)  # Return as tuple for batch unpacking

    # Create model
    autoencoder_params = AutoencoderParams(
        encoder=encoders.Linear(
            encoders.EncoderParams(input_shape=(1, 16, 32), latent_dim=32, activation_fn=torch.nn.GELU)
        ),
        decoder=decoders.Linear(
            decoders.DecoderParams(output_shape=(1, 16, 32), latent_dim=32, activation_fn=torch.nn.GELU)
        ),
        sparsity_weight=0.1,
        optimizer_init=partial(torch.optim.Adam, lr=1e-3),
    )
    model = Autoencoder(autoencoder_params)

    # Create data
    dataset = SimpleDataset(200)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Train
    trainer = pl.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    trainer.fit(model, loader)

    # Test
    test_input = torch.zeros(1, 1, 16, 32)
    test_input[0, 0, 5:10, 10:20] = 1.0

    with torch.no_grad():
        reconstruction, embedding = model(test_input)
        loss = nn.MSELoss()(reconstruction, test_input)

    print(f"Test loss: {loss.item():.4f}")
    print(f"Sparsity: {(embedding > 0.01).float().mean().item():.2%}")
