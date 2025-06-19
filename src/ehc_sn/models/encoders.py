from abc import ABC
from math import prod
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor, nn


class EncoderParams(BaseModel):
    """Configuration parameters for the neural network encoder."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    feature_dims: List[int] = Field(..., description="Input feature dimension for the encoder")
    embedding_dim: int = Field(..., description="Output embedding dimension for the encoder")
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
    def validate_feature_dims(self) -> "EncoderParams":
        if prod(self.feature_dims) != self.dims[0]:
            raise ValueError(
                f"Input feature dimension {self.feature_dims} "
                f"does not match the first dimension {self.dims[0]} in dims."
            )
        return self

    def layers(self) -> Iterable[Tuple[int, int]]:
        return [nn.Linear(*x) for x in zip(self.dims[:-1], self.dims[1:])]


class Encoder(nn.Module):
    """Neural network encoder that transforms multi-dimensional inputs into fixed-size embeddings.

    This encoder flattens multi-dimensional inputs and passes them through a sequence of
    linear layers with non-linear activations to produce an embedding vector.

    Attributes:
        _params: Configuration parameters for the encoder.
        flatten: Layer to flatten multi-dimensional inputs.
        layers: Sequence of linear layers.
        hidden_activation: Activation function for hidden layers.
        output_activation: Activation function for the output layer.
    """

    def __init__(self, params: EncoderParams):
        super(Encoder, self).__init__()
        self._params = params
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList(params.layers())
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        activations = []

        # Flatten the input if it is multi-dimensional
        h = self.flatten(x)

        for i, layer in enumerate(self.layers[:-1]):
            h = self.hidden_activation(layer(h))
            activations.append(h)

        output = self.output_activation(self.layers[-1](h))
        activations.append(output)

        return output, activations

    @property
    def feature_dim(self) -> List[int]:
        return self._params.feature_dims

    @property
    def embedding_dim(self) -> int:
        return self._params.embedding_dim


# Example usage of the Encoder class
if __name__ == "__main__":
    # Create encoder parameters for a simple case:
    # 2D input (10x10) -> flatten to 100 -> hidden layer 64 -> embedding 32
    params = EncoderParams(
        feature_dims=[10, 10],  # 10x10 grid input
        embedding_dim=32,  # output embedding dimension
        dims=[100, 64, 32],  # network architecture (flattened input -> hidden -> output)
    )

    # Create the encoder
    encoder = Encoder(params)

    # Create a sample input (batch size 4, 10x10 grid)
    sample_input = torch.rand(4, 10, 10)

    # Forward pass
    embedding, activations = encoder(sample_input)

    # Print encoder details
    print(f"Encoder architecture: {encoder}")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Number of intermediate activations: {len(activations)}")

    # Verify embedding dimension matches expected
    assert embedding.shape == (4, 32), f"Expected shape (4, 32), got {embedding.shape}"
    print("Encoder works as expected!")
