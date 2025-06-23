from typing import Any, List, Tuple

import torch
from torch import Tensor, nn

from ehc_sn.models.decoders import Decoder, DecoderParams
from ehc_sn.models.encoders import Encoder, EncoderParams


def validate_dimensions(encoder: Encoder, decoder: Decoder) -> None:
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

    def __init__(self, encoder: Encoder, decoder: Decoder):
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

    # Encoder: 16x16 grid -> embedding of to 32
    encoder_params = EncoderParams(
        input_shape=(1, 16, 32),
        latent_dim=32,
    )
    encoder = Encoder(encoder_params)

    # Decoder: embedding 32 -> reconstruct to 16x16 grid
    decoder_params = DecoderParams(
        input_shape=(1, 16, 32),
        latent_dim=32,
    )
    decoder = Decoder(decoder_params)

    # Create autoencoder with custom regularization parameters
    autoencoder = Autoencoder(encoder, decoder)

    # Create a sample batch of 4 grid maps
    sample_maps = torch.rand(4, *autoencoder.input_shape)

    # Forward pass through the autoencoder
    reconstructions, embeddings = autoencoder(sample_maps)

    # Calculate reconstruction loss (mean squared error)
    mse_loss = nn.MSELoss()(reconstructions, sample_maps)

    # Print model information
    print(f"Autoencoder architecture:")
    print(f"  - Input shape: {sample_maps.shape}")
    print(f"  - Embedding shape: {embeddings.shape}")
    print(f"  - Reconstruction shape: {reconstructions.shape}")
    print(f"  - Reconstruction loss: {mse_loss.item():.6f}")

    # Verify shapes match expected
    assert embeddings.shape == (4, autoencoder.latent_dim)
    assert reconstructions.shape == (4, *autoencoder.input_shape)
    print("Autoencoder works as expected!")
