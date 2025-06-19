from abc import ABC
from math import prod
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor, nn


class DecoderParams(BaseModel):
    """Configuration parameters for the neural network decoder."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    feature_dims: List[int] = Field(..., description="Output feature dimension for the decoder")
    embedding_dim: int = Field(..., description="Input embedding dimension for the decoder")
    dims: List[int] = Field(..., description="List of layer dimensions for the sequence")

    @field_validator("dims")
    def validate_dims(cls, v: List[int]) -> List[int]:
        if len(v) < 2:
            raise ValueError("Sequence must have at least 2 dimensions (input and output)")
        for i, dim in enumerate(v):
            if dim <= 0:
                raise ValueError(f"Dimension at position {i} must be positive, got {dim}")
        return v

    @model_validator(mode="after")
    def validate_dims_after(self) -> "DecoderParams":
        if self.embedding_dim != self.dims[0]:
            raise ValueError(
                f"Input embedding dimension {self.embedding_dim} "
                f"does not match the first dimension {self.dims[0]} in dims."
            )
        if prod(self.feature_dims) != self.dims[-1]:
            raise ValueError(
                f"Output feature dimensions product {prod(self.feature_dims)} "
                f"does not match the last dimension {self.dims[-1]} in dims."
            )
        return self

    def layers(self) -> List[nn.Linear]:
        return [nn.Linear(*x) for x in zip(self.dims[:-1], self.dims[1:])]


class Decoder(nn.Module):
    """Neural network decoder that transforms embeddings into reconstructed features.

    This decoder takes embedding vectors and passes them through a sequence of
    linear layers with non-linear activations to produce a reconstructed feature output
    which is then reshaped to the original feature dimensions.

    Attributes:
        _params: Configuration parameters for the decoder.
        layers: Sequence of linear layers.
        hidden_activation: Activation function for hidden layers.
        output_activation: Activation function for the output layer.
    """

    def __init__(self, params: DecoderParams):
        super(Decoder, self).__init__()
        self._params = params
        self.layers = nn.ModuleList(params.layers())
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        activations = []
        h = x

        for i, layer in enumerate(self.layers[:-1]):
            h = self.hidden_activation(layer(h))
            activations.append(h)

        # Final layer output
        h = self.output_activation(self.layers[-1](h))
        activations.append(h)

        # Reshape to original feature dimensions
        batch_size = x.size(0)
        output = h.view(batch_size, *self._params.feature_dims)

        return output, activations

    @property
    def feature_dims(self) -> List[int]:
        return self._params.feature_dims

    @property
    def embedding_dim(self) -> int:
        return self._params.embedding_dim


# Example usage of the Decoder class
if __name__ == "__main__":
    # Create decoder parameters:
    # embedding 32 -> hidden layer 64 -> flattened output 100 -> reshape to 10x10
    params = DecoderParams(
        feature_dims=[10, 10],  # 10x10 grid output
        embedding_dim=32,  # input embedding dimension
        dims=[32, 64, 100],  # network architecture (embedding -> hidden -> flattened output)
    )

    # Create the decoder
    decoder = Decoder(params)

    # Create sample embeddings (batch size 4, embedding dim 32)
    sample_embeddings = torch.rand(4, 32)

    # Forward pass
    reconstructed_features, activations = decoder(sample_embeddings)

    # Print decoder details
    print(f"Decoder architecture: {decoder}")
    print(f"Input embedding shape: {sample_embeddings.shape}")
    print(f"Reconstructed features shape: {reconstructed_features.shape}")
    print(f"Number of intermediate activations: {len(activations)}")

    # Verify output shape matches expected
    assert reconstructed_features.shape == (4, 10, 10)
    print("Decoder works as expected!")
