from abc import ABC
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn

from ehc_sn.models.decoders import Decoder, DecoderParams
from ehc_sn.models.encoders import Encoder, EncoderParams


class AutoencoderParams(BaseModel):
    """Parameters for configuring the autoencoder's regularization behavior."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    sparsity: float = Field(0.05, description="Target sparsity level for the activations")
    beta: float = Field(0.01, description="Sparsity regularization coefficient for the autoencoder")

    @field_validator("sparsity_target", "sparsity_weight")
    def validate_sparsity(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError(f"Value must be between 0 and 1, got {v}.")
        return v


def validate_dimensions(encoder: Encoder, decoder: Decoder) -> None:
    """Validate that encoder and decoder dimensions are compatible."""
    if encoder.embedding_dim != decoder.embedding_dim:
        raise ValueError(
            f"Output dimension of encoder ({encoder.embedding_dim}) "
            f"does not match input dimension of decoder ({decoder.embedding_dim})."
        )
    if encoder.feature_dims != decoder.feature_dims:
        raise ValueError(
            f"Input feature dimensions of encoder ({encoder.feature_dims}) "
            f"do not match output feature dimensions of decoder ({decoder.feature_dims})."
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

    def __init__(self, encoder: Encoder, decoder: Decoder, params: Optional[AutoencoderParams] = None):
        super(Autoencoder, self).__init__()
        self._params = params or AutoencoderParams()
        self.encoder = encoder
        self.decoder = decoder
        validate_dimensions(self.encoder, self.decoder)

    @property
    def params(self) -> AutoencoderParams:
        return self._params

    @property
    def sparsity_target(self) -> float:
        return self._params.sparsity_target

    @property
    def sparsity_weight(self) -> float:
        return self._params.sparsity_weight

    @property
    def feature_dims(self) -> List[int]:
        return self.encoder.feature_dims

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        # Forward pass through the autoencoder.
        embedding, encoder_activations = self.encoder(x)
        reconstruction, decoder_activations = self.decoder(embedding)
        return embedding, reconstruction, encoder_activations + decoder_activations


# Example usage of the Autoencoder
if __name__ == "__main__":
    # Define parameters for a simple model that works with 10x10 spatial maps
    feature_dims = [10, 10]  # 10x10 grid maps
    embedding_dim = 32  # 32-dimensional latent space

    # Encoder: 10x10 grid -> flatten to 100 -> hidden layer 64 -> embedding 32
    encoder_params = EncoderParams(
        feature_dims=feature_dims,
        embedding_dim=embedding_dim,
        dims=[100, 64, embedding_dim],
    )
    encoder = Encoder(encoder_params)

    # Decoder: embedding 32 -> hidden layer 64 -> flattened output 100 -> reshape to 10x10
    decoder_params = DecoderParams(
        feature_dims=feature_dims,
        embedding_dim=embedding_dim,
        dims=[embedding_dim, 64, 100],
    )
    decoder = Decoder(decoder_params)

    # Create autoencoder with custom regularization parameters
    autoencoder_params = AutoencoderParams()
    autoencoder = Autoencoder(encoder, decoder, autoencoder_params)

    # Create a sample batch of 4 grid maps
    sample_maps = torch.rand(4, *feature_dims)

    # Forward pass through the autoencoder
    embeddings, reconstructions, activations = autoencoder(sample_maps)

    # Calculate reconstruction loss (mean squared error)
    mse_loss = nn.MSELoss()(reconstructions, sample_maps)

    # Print model information
    print(f"Autoencoder architecture:")
    print(f"  - Input shape: {sample_maps.shape}")
    print(f"  - Embedding shape: {embeddings.shape}")
    print(f"  - Reconstruction shape: {reconstructions.shape}")
    print(f"  - Number of activations: {len(activations)}")
    print(f"  - Regularization: sparsity={autoencoder.sparsity_target}")
    print(f"  - Regularization: weight={autoencoder.sparsity_weight}")
    print(f"  - Reconstruction loss: {mse_loss.item():.6f}")

    # Verify shapes match expected
    assert embeddings.shape == (4, embedding_dim)
    assert reconstructions.shape == sample_maps.shape
    print("Autoencoder works as expected!")
