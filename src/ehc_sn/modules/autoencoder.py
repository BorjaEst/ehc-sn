from __future__ import annotations

from typing import List, Literal, Optional, Sequence

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.mlp import MLP


# =================================================================================================
class AutoencoderSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for autoencoder modules."""

    encode_mode: Literal["two_hot"] = Field(
        "two_hot",
        description="Compression mode for observations (e.g., 'two_hot')",
    )
    decode_mode: Literal["mlp"] = Field(
        "mlp",
        description="Decompression mode for observations (e.g., 'nnet')",
    )


# =================================================================================================
class Autoencoder(nn.Module):
    """Encode observations into features and decode them back.

    This module is a thin wrapper around a selected encoder/decoder pair based
    on :class:`AutoencoderSettings`.
    """

    def __init__(self, n_o: int, n_c: int, settings: AutoencoderSettings):
        super().__init__()
        self.encoder = _select_encoder(settings, n_o, n_c)
        self.decoder = _select_decoder(settings, n_c, n_o)
        self.settings = settings

    @property
    def n_observations(self) -> int:
        return self.encoder.n_o

    @property
    def n_features(self) -> int:
        return self.encoder.n_c

    def forward(self, *args, **kwds) -> Tensor:
        raise NotImplementedError("Use encoder and decoder methods separately.")

    def encode(self, x_o: Tensor) -> Tensor:
        return self.encoder(x_o)

    def decode(self, c: Tensor) -> Tensor:
        return self.decoder(c)


# =================================================================================================
class TwoHotEncoder(nn.Module):
    """Two-hot encoder.

    Converts a one-hot observation vector into a fixed two-hot representation
    using a precomputed lookup table.
    """

    def __init__(self, n_o: int, n_c: int):
        super(TwoHotEncoder, self).__init__()
        self.n_o = n_o
        self.n_c = n_c
        # Create two-hot encoding table and stack into tensor
        two_hot_tensor = utils.create_encoding_table(n_o, n_c, n_hot=2)
        self.register_buffer("two_hot_table", two_hot_tensor)

    def forward(self, o: Tensor) -> Tensor:
        # Extract indices from one-hot and lookup two-hot encoding
        indices = torch.argmax(o, dim=1)
        return torch.stack([self.two_hot_table[i] for i in indices], dim=0)


# =================================================================================================
class MLPDecoder(nn.Module):
    """MLP-based decoder from features to observation logits."""

    def __init__(self, n_c: int, n_o: int):
        super(MLPDecoder, self).__init__()
        self.n_c = n_c
        self.n_o = n_o
        # MLP for decompressing features to observations
        self.MLP_c_star = MLP(self.n_c, self.n_o, hidden_dim=20 * self.n_c)

    def forward(self, c: Tensor) -> Tensor:
        return self.MLP_c_star(c)


# =================================================================================================
def _select_encoder(settings: AutoencoderSettings, n_o: int, n_c: int) -> nn.Module:
    if settings.encode_mode == "two_hot":
        return TwoHotEncoder(n_o, n_c)
    raise ValueError(f"Unsupported encode_mode: {settings.encode_mode}")


# =================================================================================================
def _select_decoder(settings: AutoencoderSettings, n_c: int, n_o: int) -> nn.Module:
    if settings.decode_mode == "mlp":
        return MLPDecoder(n_c, n_o)
    raise ValueError(f"Unsupported decode_mode: {settings.decode_mode}")
