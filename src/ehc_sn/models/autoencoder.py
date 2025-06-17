from abc import ABC
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn


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

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        activations = []

        for i, layer in enumerate(self.layers[:-1]):
            h = self.hidden_activation(layer(h))
            activations.append(h)

        output = self.output_activation(self.layers[-1](h))
        activations.append(output)

        return output, activations


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


class AutoencoderParameters(BaseModel):
    encoder: SeqParameters = Field(..., description="Parameters for the encoder")
    decoder: SeqParameters = Field(..., description="Parameters for the decoder")

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


class SparseAutoencoderParameters(AutoencoderParameters):
    sparsity: float = Field(0.05, description="Target sparsity level for the activations")
    beta: float = Field(0.01, description="Sparsity regularization coefficient for the autoencoder")


class SparseAutoencoder(Autoencoder):
    def __init__(self, parameters: SparseAutoencoderParameters):
        super(SparseAutoencoder, self).__init__(parameters)
        self.sparsity = parameters.sparsity
        self.beta = parameters.beta

    def forward(self, y: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        x, encoder_activations = self.encoder(y)
        y, decoder_activations = self.decoder(x)
        return x, y, encoder_activations + decoder_activations

    def calculate_loss(self, y: Tensor, y_: Tensor, activations: List[Tensor]) -> Tuple[Tensor, Tensor]:
        reconstruction_loss = torch.nn.functional.mse_loss(y, y_)
        sparsity_loss = sparsity_loss(activations, self.beta, self.sparsity)
        return reconstruction_loss, sparsity_loss


def sparsity_loss(activations: List[Tensor], beta: float, sparsity: float) -> Tensor:
    """Calculate sparsity loss using KL divergence"""
    return beta * sum(kl_divergence(x.mean(dim=0), sparsity) for x in activations)


def kl_divergence(rho_hat: Tensor, sparsity: float, eps=1e-10) -> Tensor:
    """Calculate KL divergence between target sparsity and actual activations"""
    kl = sparsity * torch.log((sparsity + eps) / (rho_hat + eps))
    kl += (1 - sparsity) * torch.log((1 - sparsity + eps) / (1 - rho_hat + eps))
    return torch.sum(kl)
