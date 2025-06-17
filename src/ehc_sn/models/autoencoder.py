from abc import ABC
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn


class AutoencoderParameters(BaseModel):
    encoder: "SeqParameters" = Field(..., description="Parameters for the encoder")
    decoder: "SeqParameters" = Field(..., description="Parameters for the decoder")

    def __init__(self, **data):
        super(AutoencoderParameters, self).__init__(**data)
        self._val_dimensions()

    def _val_dimensions(self):
        if self.encoder.dims[-1] != self.decoder.dims[0]:
            raise ValueError(
                f"Output dimension of encoder ({self.encoder.dims[-1]}) "
                f"does not match input dimension of decoder ({self.decoder.dims[0]})."
            )


class Autoencoder(nn.Module):
    def __init__(self, parameters: AutoencoderParameters):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(parameters.encoder)
        self.decoder = Decoder(parameters.decoder)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(y)
        return x, self.decoder(x)


class SeqParameters(BaseModel):
    dims: List[int] = Field(..., description="List of layer dimensions for the sequence")

    def args_iter(self) -> Iterable[Tuple[int, int]]:
        return zip(self.dims[1:], self.dims[:-1])


class Sequence(nn.Module, ABC):
    def __init__(self, parameters: SeqParameters):
        super(Sequence, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(*x) for x in parameters.args_iter()])
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.ReLU()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = self.hidden_activation(self.layers[0](y))
        for layer in self.layers[1:-1]:
            h = self.hidden_activation(layer(h))
        return self.output_activation(self.layers[-1](h))


class Encoder(Sequence):
    def __init__(self, parameters: SeqParameters):
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
    def __init__(self, parameters: SeqParameters):
        super(Decoder, self).__init__(parameters)
        # Assuming the output is in [0, 1] range
        self.output_activation = nn.Sigmoid()

    @property
    def feature_dim(self) -> int:
        return self.layers[-1].in_features

    @property
    def embedding_dim(self) -> int:
        return self.layers[0].out_features


class SparseEncoderParameters(SeqParameters):
    sparsity: float = Field(0.05, description="Sparsity parameter for the encoder")
    beta: float = Field(0.1, description="Regularization parameter for sparsity loss")


class SparseEncoder(Encoder):
    def __init__(self, p: SparseEncoderParameters):
        super(SparseEncoder, self).__init__(p)
        self.sparsity = p.sparsity
        self.beta = p.beta

    def sparsity_loss(self, activations: List[Tensor]) -> Tensor:
        total_loss = sum(self.kl_divergence(x.mean(dim=0)) for x in activations)
        return self.beta * total_loss

    def kl_divergence(self, rho_hat: Tensor, eps=1e-10) -> Tensor:
        kl = self.sparsity * torch.log((self.sparsity + eps) / (rho_hat + eps))
        kl += (1 - self.sparsity) * torch.log((1 - self.sparsity + eps) / (1 - rho_hat + eps))
        return torch.sum(kl)
