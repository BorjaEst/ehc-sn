"""Module for the model class."""

from abc import ABC, abstractmethod

import torch
from ehc_sn import config
from torch import nn


class BaseWindowDecoder(ABC, nn.Module):
    """The base class for the decoders."""

    def __init__(self, window_length: int, *args, **kwargs):
        super().__init__()
        self._window_length = window_length
        self._index = 0
        self._buffer: torch.Tensor | None = None
        self._kernel = self.kernel(window_length, *args, **kwargs)
        self._kernel /= self._kernel.sum()  # Normalize the kernel
        self._kernel = self._kernel.to(config.device).view(window_length, 1)

    def _initialize_buffer(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the rolling buffer based on the input tensor shape."""
        buffer_shape = (self._window_length, *x.shape)
        return torch.zeros(buffer_shape, device=x.device)

    @staticmethod
    @abstractmethod
    def kernel(window_length: int, *args, **kwds) -> torch.Tensor:
        """Decode the input current."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the input current."""
        if self._buffer is None:  # Initialize the buffer
            self._buffer = self._initialize_buffer(x)
        self._buffer[self._index] = x  # Store the input in buffer
        self._index = (self._index + 1) % self._window_length
        return (self._buffer * self._kernel).sum(dim=0)


class RectangularDecoder(BaseWindowDecoder):
    """Rectangular window decoder for the output spikes."""

    @staticmethod
    def kernel(window_length: int, *args, **kwds) -> torch.Tensor:
        """Return the Hann window kernel."""
        return torch.ones(window_length)


class HannDecoder(BaseWindowDecoder):
    """Hann window decoder for the output spikes."""

    @staticmethod
    def kernel(window_length: int, *args, **kwds) -> torch.Tensor:
        """Return the Hann window kernel."""
        return torch.hann_window(window_length)


class HammingDecoder(BaseWindowDecoder):
    """Hamming window decoder for the output spikes."""

    @staticmethod
    def kernel(window_length: int, *args, **kwds) -> torch.Tensor:
        """Return the Hamming window kernel."""
        return torch.hamming_window(window_length)


class BlackmanDecoder(BaseWindowDecoder):
    """Blackman window decoder for the output spikes."""

    @staticmethod
    def kernel(window_length: int, *args, **kwds) -> torch.Tensor:
        """Return the Blackman window kernel."""
        return torch.blackman_window(window_length)


class BarlettDecoder(BaseWindowDecoder):
    """Barlett window decoder for the output spikes."""

    @staticmethod
    def kernel(window_length: int, *args, **kwds) -> torch.Tensor:
        """Return the Barlett window kernel."""
        return torch.bartlett_window(window_length)
