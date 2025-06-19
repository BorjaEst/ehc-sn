from abc import ABC
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn


class SequenceParams(BaseModel):
    dims: List[int] = Field(..., description="List of layer dimensions for the sequence")

    @field_validator("dims")
    def validate_dims(cls, v: List[int]) -> List[int]:
        if len(v) < 2:
            raise ValueError("Sequence must have at least 2 dimensions (input and output)")
        for i, dim in enumerate(v):
            if dim <= 0:
                raise ValueError(f"Dimension at position {i} must be positive, got {dim}")
        return v

    def args_iter(self) -> Iterable[Tuple[int, int]]:
        return zip(self.dims[:-1], self.dims[1:])


class Sequence(nn.Module, ABC):
    def __init__(self, parameters: SequenceParams):
        super(Sequence, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(*x) for x in parameters.args_iter()])
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.ReLU()

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        activations = []

        for i, layer in enumerate(self.layers[:-1]):
            h = self.hidden_activation(layer(h))
            activations.append(h)

        output = self.output_activation(self.layers[-1](h))
        activations.append(output)

        return output, activations


class Encoder(Sequence):
    def __init__(self, parameters: SequenceParams):
        super(Encoder, self).__init__(parameters)
        # Assuming the output is in [0, inf) range
        self.output_activation = nn.ReLU()

    @property
    def feature_dim(self) -> int:
        return self.layers[0].in_features

    @property
    def embedding_dim(self) -> int:
        return self.layers[-1].out_features


class Decoder(Encoder):
    def __init__(self, parameters: SequenceParams):
        super(Decoder, self).__init__(parameters)
        # Assuming the output is in [0, 1] range
        self.output_activation = nn.Sigmoid()

    @property
    def feature_dim(self) -> int:
        return self.layers[-1].out_features

    @property
    def embedding_dim(self) -> int:
        return self.layers[0].in_features


class AutoencoderParams(BaseModel):
    sparsity: float = Field(0.05, description="Target sparsity level for the activations")
    beta: float = Field(0.01, description="Sparsity regularization coefficient for the autoencoder")


class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, params: Optional[AutoencoderParams] = None):
        super(Autoencoder, self).__init__()
        self.params = params or AutoencoderParams()
        self.encoder = encoder
        self.decoder = decoder
        self.val_dimensions()

    def val_dimensions(self):
        if self.encoder.embedding_dim != self.decoder.embedding_dim:
            raise ValueError(
                f"Output dimension of encoder ({self.encoder.embedding_dim}) "
                f"does not match input dimension of decoder ({self.decoder.embedding_dim})."
            )

    def forward(self, y: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        x, encoder_activations = self.encoder(y)
        y, decoder_activations = self.decoder(x)
        return x, y, encoder_activations + decoder_activations

    @property
    def sparsity(self) -> float:
        return self.params.sparsity

    @property
    def beta(self) -> float:
        return self.params.beta
