from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ehc_sn.losses.autoencoders import ReconstructionLoss, SparsityLoss
from ehc_sn.models.decoders import BaseDecoder, DecoderParams
from ehc_sn.models.encoders import BaseEncoder, EncoderParams


def validate_dimensions(encoder: BaseEncoder, decoder: BaseDecoder) -> None:
    """Validate that encoder and decoder dimensions are compatible."""
    if encoder.input_shape != decoder.input_shape:
        raise ValueError(
            f"Input shape of encoder ({encoder.input_shape}) "
            f"does not match input shape of decoder ({decoder.input_shape})."
        )
    if encoder.latent_dim != decoder.latent_dim:
        raise ValueError(
            f"Latent dimension of encoder ({encoder.latent_dim}) "
            f"does not match embedding dimension of decoder ({decoder.latent_dim})."
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

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        learning_rate: float = 1e-3,
        sparsity_weight: float = 0.1,
        sparsity_target: float = 0.05,
        optimizer: str = "adam",
        **optimizer_kwargs: Any,
    ):
        validate_dimensions(encoder, decoder)
        super(Autoencoder, self).__init__()

        # Model components
        self.encoder = encoder
        self.decoder = decoder

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.sparsity_weight = sparsity_weight
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        # Loss functions
        self.reconstruction_loss = ReconstructionLoss()
        self.sparsity_loss = SparsityLoss(sparsity_target=sparsity_target)

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["encoder", "decoder"])

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
        outputs = self(x)

        # Calculate losses
        recon_loss = self.reconstruction_loss(outputs, x)
        sparsity_loss = self.sparsity_loss(outputs, x)
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Log metrics
        self.log("train/reconstruction_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/sparsity_loss", sparsity_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        _, embeddings = outputs
        sparsity_rate = (embeddings > 0.01).float().mean()
        self.log("train/sparsity_rate", sparsity_rate, on_step=True, on_epoch=True)

        return total_loss

    # -----------------------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------------------

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        """Validation step for the autoencoder."""
        x = batch
        outputs = self(x)

        # Calculate losses
        recon_loss = self.reconstruction_loss(outputs, x)
        sparsity_loss = self.sparsity_loss(outputs, x)
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Log metrics
        self.log("val/reconstruction_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/sparsity_loss", sparsity_loss, on_step=False, on_epoch=True)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Calculate and log sparsity metrics
        _, embeddings = outputs
        sparsity_rate = (embeddings > 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, on_step=False, on_epoch=True)

        return total_loss

    # -----------------------------------------------------------------------------------
    # Test step
    # -----------------------------------------------------------------------------------

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:  # noqa: ARG002
        """Test step for the autoencoder."""
        x = batch
        outputs = self(x)
        reconstruction, embeddings = outputs

        # Calculate losses
        recon_loss = self.reconstruction_loss(outputs, x)
        sparsity_loss = self.sparsity_loss(outputs, x)
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Calculate additional metrics
        mse = nn.MSELoss()(reconstruction, x)
        mae = nn.L1Loss()(reconstruction, x)
        sparsity_rate = (embeddings > 0.01).float().mean()

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
    # Optimizer configuration
    # -----------------------------------------------------------------------------------

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer for training."""
        if self.optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

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
        outputs = self(x)
        recon_loss = self.reconstruction_loss(outputs, x)
        sparsity_loss = self.sparsity_loss(outputs, x)
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        return {
            "reconstruction_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "total_loss": total_loss,
        }


# Example usage of the Autoencoder
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from ehc_sn.models.decoders import LinearDecoder
    from ehc_sn.models.encoders import LinearEncoder

    # For structured obstacle map processing (16x32 grid)
    # Encoder: 16x32 grid -> embedding of 64
    encoder_params = EncoderParams(
        input_shape=(1, 16, 32),  # 1 channel, 16x32 grid
        latent_dim=64,
    )
    encoder = LinearEncoder(encoder_params)

    # Decoder: embedding 64 -> reconstruct to 16x132 grid
    decoder_params = DecoderParams(
        input_shape=(1, 16, 32),  # 1 channel, 16x32 grid
        latent_dim=64,
    )
    decoder = LinearDecoder(decoder_params)

    # Create sparse autoencoder
    autoencoder = Autoencoder(encoder, decoder)

    # Create a sample batch of 4 obstacle maps (1s and 0s)
    sample_maps = torch.zeros(4, 1, 16, 32)

    # Add some block obstacles
    sample_maps[0, 0, 5:9, 5:9] = 1.0  # Block in first map
    sample_maps[1, 0, 3:5, 10:15] = 1.0  # Block in second map
    sample_maps[2, 0, 8:12, 2:7] = 1.0  # Block in third map
    sample_maps[3, 0, 1:15, 8:10] = 1.0  # Wall-like structure in fourth map

    # Forward pass through the autoencoder
    reconstructions, embeddings = autoencoder(sample_maps)

    # Calculate reconstruction loss (mean squared error)
    mse_loss = nn.MSELoss()(reconstructions, sample_maps)

    # Calculate actual sparsity (% of neurons active)
    active_neurons = (embeddings > 0.01).float().mean().item()

    # Print model information
    print("Sparse Autoencoder architecture:")
    print(f"  - Input shape: {sample_maps.shape}")
    print(f"  - Embedding shape: {embeddings.shape}")
    print(f"  - Reconstruction shape: {reconstructions.shape}")
    print(f"  - Reconstruction loss (MSE): {mse_loss.item():.6f}")
    print(f"  - Actual activation rate: {active_neurons:.2%}")

    # Visualize some reconstructions
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))

    for i in range(4):
        # Original
        axes[i, 0].imshow(sample_maps[i, 0].detach().numpy(), cmap="binary")
        axes[i, 0].set_title(f"Original Map {i+1}")
        axes[i, 0].axis("off")

        # Reconstruction
        axes[i, 1].imshow(reconstructions[i, 0].detach().numpy(), cmap="binary")
        axes[i, 1].set_title(f"Reconstruction {i+1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------------------
    # Dummy training example
    # -----------------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("DUMMY TRAINING EXAMPLE")
    print("=" * 60)

    # Create dummy dataset for training
    class DummyObstacleDataset(torch.utils.data.Dataset):
        """Simple dataset that generates random obstacle maps."""

        def __init__(self, num_samples: int = 1000, height: int = 16, width: int = 32):
            self.num_samples = num_samples
            self.height = height
            self.width = width

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> Tensor:
            # Generate random obstacle map
            obstacle_map = torch.zeros(1, self.height, self.width)

            # Add random rectangular obstacles
            num_obstacles = torch.randint(1, 5, (1,)).item()
            for _ in range(num_obstacles):
                h_start = torch.randint(0, self.height - 4, (1,)).item()
                w_start = torch.randint(0, self.width - 4, (1,)).item()
                h_size = torch.randint(2, min(6, self.height - h_start), (1,)).item()
                w_size = torch.randint(2, min(6, self.width - w_start), (1,)).item()

                obstacle_map[0, h_start : h_start + h_size, w_start : w_start + w_size] = 1.0

            return obstacle_map

    # Create datasets
    train_dataset = DummyObstacleDataset(num_samples=800)
    val_dataset = DummyObstacleDataset(num_samples=200)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Create a new autoencoder for training
    training_autoencoder = Autoencoder(
        encoder=LinearEncoder(encoder_params),
        decoder=LinearDecoder(decoder_params),
        learning_rate=1e-3,
        sparsity_weight=0.1,
        sparsity_target=0.05,
    )

    print(f"Training autoencoder with {len(train_dataset)} training samples")
    print(f"Validation set has {len(val_dataset)} samples")
    print(f"Architecture: {encoder_params.input_shape} -> {encoder_params.latent_dim} -> {decoder_params.input_shape}")

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=5,  # Short training for demo
        accelerator="auto",
        devices=1,
        logger=False,  # Disable logging for demo
        enable_checkpointing=False,  # Disable checkpointing for demo
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    print("\nStarting training...")
    trainer.fit(training_autoencoder, train_loader, val_loader)

    print("\nTraining completed!")

    # Test the trained model
    print("\nTesting trained model...")

    # Get a test batch
    test_batch = next(iter(val_loader))

    # Before and after training comparison
    with torch.no_grad():
        # Trained model
        trained_reconstructions, trained_embeddings = training_autoencoder(test_batch)
        trained_loss = nn.MSELoss()(trained_reconstructions, test_batch)
        trained_sparsity = (trained_embeddings > 0.01).float().mean().item()

        # Untrained model (for comparison)
        untrained_autoencoder = Autoencoder(
            encoder=LinearEncoder(encoder_params),
            decoder=LinearDecoder(decoder_params),
        )
        untrained_reconstructions, untrained_embeddings = untrained_autoencoder(test_batch)
        untrained_loss = nn.MSELoss()(untrained_reconstructions, test_batch)
        untrained_sparsity = (untrained_embeddings > 0.01).float().mean().item()

    print(f"\nReconstruction Loss Comparison:")
    print(f"  Untrained model: {untrained_loss.item():.6f}")
    print(f"  Trained model:   {trained_loss.item():.6f}")
    print(f"  Improvement:     {(untrained_loss.item() - trained_loss.item()):.6f}")

    print(f"\nSparsity Comparison:")
    print(f"  Untrained model: {untrained_sparsity:.2%}")
    print(f"  Trained model:   {trained_sparsity:.2%}")
    print(f"  Target sparsity: {training_autoencoder.sparsity_loss.sparsity_target:.2%}")

    # Visualize training results
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # Show first 4 samples from test batch
    for i in range(4):
        # Original
        axes[0, i].imshow(test_batch[i, 0].detach().numpy(), cmap="binary")
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis("off")

        # Untrained reconstruction
        axes[1, i].imshow(untrained_reconstructions[i, 0].detach().numpy(), cmap="binary", vmin=0, vmax=1)
        axes[1, i].set_title(f"Untrained {i+1}")
        axes[1, i].axis("off")

        # Trained reconstruction
        axes[2, i].imshow(trained_reconstructions[i, 0].detach().numpy(), cmap="binary", vmin=0, vmax=1)
        axes[2, i].set_title(f"Trained {i+1}")
        axes[2, i].axis("off")

    plt.suptitle("Training Results: Original vs Untrained vs Trained", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\nDummy training example completed!")
    print("The trained model should show better reconstruction quality and sparsity.")
