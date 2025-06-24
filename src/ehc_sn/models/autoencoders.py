from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

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


class Autoencoder(nn.Module):
    """Neural network autoencoder for the entorhinal-hippocampal circuit.

    This autoencoder combines an encoder that transforms spatial input into a compact
    embedding and a decoder that reconstructs the original input from the embedding.
    It supports sparse activations to model the sparse firing patterns observed in
    the hippocampus, which is critical for pattern separation and completion.

    The autoencoder serves as a computational model for how the entorhinal-hippocampal
    circuit might encode, store, and retrieve spatial information.
    """

    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder):
        validate_dimensions(encoder, decoder)
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor, *args: Any) -> Tuple[Tensor, Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

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
    print(f"Sparse Autoencoder architecture:")
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
