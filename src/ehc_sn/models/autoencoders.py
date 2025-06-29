from typing import Any, Callable, Dict, Optional, Tuple, Type

import lightning.pytorch as pl
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.models.decoders import BaseDecoder, DecoderParams
from ehc_sn.models.encoders import BaseEncoder, EncoderParams


class AutoencoderParams(BaseModel):
    """Parameters for configuring the Autoencoder model."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    encoder: BaseEncoder = Field(..., description="Parameters for the encoder component.")
    decoder: BaseDecoder = Field(..., description="Parameters for the decoder component.")
    sparsity_weight: float = Field(0.1, description="Weight for the sparsity loss term.")
    sparsity_target: float = Field(0.05, description="Target sparsity level for the embeddings.")
    optimizer_init: Callable = Field(..., description="Callable to initialize the optimizer.")


def validate_dimensions(params: AutoencoderParams) -> None:
    """Validate that encoder and decoder dimensions are compatible."""
    if params.encoder.input_shape != params.decoder.input_shape:
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
    circuit might encode, store, and retrieve spatial information.
    """

    def __init__(self, params: Optional[AutoencoderParams] = None):
        validate_dimensions(params)
        super(Autoencoder, self).__init__()

        # Model components
        self.encoder = params.encoder
        self.decoder = params.decoder

        # Training hyperparameters
        self.optimizer_init = params.optimizer_init

        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.sparsity_weight = params.sparsity_weight
        self.sparsity_target = params.sparsity_target

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    # -----------------------------------------------------------------------------------
    # Optimizer configuration
    # -----------------------------------------------------------------------------------

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer for training."""
        optimizer = self.optimizer_init(self.parameters())  # functools.partial or similar
        return optimizer

    # -----------------------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------------------

    def forward(self, x: Tensor, *args: Any) -> Tuple[Tensor, Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    # -----------------------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------------------

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Training step for the autoencoder."""
        x = batch
        reconstruction, embedding = self(x)

        # Calculate losses
        recon_loss = self.reconstruction_loss(reconstruction, x)
        sparsity_loss = torch.mean(torch.abs(embedding))  # L1 norm
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Log metrics
        self.log("train/reconstruction_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/sparsity_loss", sparsity_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        sparsity_rate = (embedding > 0.01).float().mean()
        self.log("train/sparsity_rate", sparsity_rate, on_step=True, on_epoch=True)

        return total_loss

    # -----------------------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------------------

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Validation step for the autoencoder."""
        x = batch
        reconstruction, embedding = self(x)

        # Calculate losses
        recon_loss = self.reconstruction_loss(reconstruction, x)
        sparsity_loss = torch.mean(torch.abs(embedding))
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Log metrics
        self.log("val/reconstruction_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/sparsity_loss", sparsity_loss, on_step=False, on_epoch=True)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        sparsity_rate = (embedding > 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, on_step=False, on_epoch=True)

        return total_loss

    # -----------------------------------------------------------------------------------
    # Test step
    # -----------------------------------------------------------------------------------

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:  # noqa: ARG002
        """Test step for the autoencoder."""
        x = batch
        reconstruction, embedding = self(x)

        # Calculate losses
        recon_loss = self.reconstruction_loss(reconstruction, x)
        sparsity_loss = torch.mean(torch.abs(embedding))
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Calculate additional metrics
        mse = nn.MSELoss()(reconstruction, x)
        mae = nn.L1Loss()(reconstruction, x)
        sparsity_rate = (embedding > 0.01).float().mean()

        # Log metrics
        metrics = {
            "test/reconstruction_loss": recon_loss,
            "test/sparsity_loss": sparsity_loss,
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
        """Prediction step returns reconstructions and embeddings."""
        return self(batch)

    # -----------------------------------------------------------------------------------
    # Model properties
    # -----------------------------------------------------------------------------------

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of the input feature map."""
        return self.encoder.input_shape

    @property
    def input_channels(self) -> int:
        """Returns the number of input channels."""
        return self.encoder.input_channels

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """Returns the output shape as (height, width)."""
        return self.encoder.spatial_dimensions

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent representation."""
        return self.encoder.latent_dim

    # -----------------------------------------------------------------------------------
    # Additional utility methods
    # -----------------------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, embedding: Tensor) -> Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(embedding)

    def reconstruct(self, x: Tensor) -> Tensor:
        """Full reconstruction from input."""
        embedding = self.encode(x)
        return self.decode(embedding)

    def compute_loss(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute all losses for a given input."""
        reconstruction, embedding = self(x)
        recon_loss = self.reconstruction_loss(reconstruction, x)
        sparsity_loss = torch.mean(torch.abs(embedding))
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        return {
            "reconstruction_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "total_loss": total_loss,
        }


# Example usage with training
if __name__ == "__main__":
    from functools import partial

    import lightning.pytorch as pl

    from ehc_sn.models.decoders import LinearDecoder
    from ehc_sn.models.encoders import LinearEncoder

    # Simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size: int = 100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Random obstacle maps
            x = torch.zeros(1, 16, 32)
            x[0, torch.randint(0, 16, (5,)), torch.randint(0, 32, (5,))] = 1.0
            return x

    # Create model
    autoencoder_params = AutoencoderParams(
        encoder=LinearEncoder(EncoderParams(input_shape=(1, 16, 32), latent_dim=32)),
        decoder=LinearDecoder(DecoderParams(input_shape=(1, 16, 32), latent_dim=32)),
        sparsity_weight=0.1,
        sparsity_target=0.05,
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
