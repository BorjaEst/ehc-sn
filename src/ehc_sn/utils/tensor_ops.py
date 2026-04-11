"""Small tensor-shape and multiscale helpers shared by model code."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor, nn


def as_batch_column(
    value: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    """Return one batch-aligned column tensor with shape ``(B, 1)``."""
    if value.ndim == 1:
        if int(value.shape[0]) != batch_size:
            raise ValueError(f"{name} must have batch size {batch_size}, got shape {tuple(value.shape)}.")
        value = value.unsqueeze(-1)
    elif value.ndim != 2 or int(value.shape[0]) != batch_size or int(value.shape[1]) != 1:
        raise ValueError(f"{name} must have shape ({batch_size},) or ({batch_size}, 1), got {tuple(value.shape)}.")
    return value.to(device=device, dtype=dtype)


def as_optional_batch_column(
    value: Optional[Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    fill_value: float = 0.0,
) -> Tensor:
    """Return an optional batch-aligned column tensor with shape ``(B, 1)``."""
    if value is None:
        return torch.full((batch_size, 1), fill_value, dtype=dtype, device=device)
    return as_batch_column(value, batch_size=batch_size, device=device, dtype=dtype, name=name)


def as_optional_feature_matrix(
    value: Optional[Tensor],
    *,
    batch_size: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    """Return an optional batch-aligned feature matrix with shape ``(B, D)``."""
    if value is None:
        return torch.zeros((batch_size, width), dtype=dtype, device=device)

    if value.ndim == 1:
        if batch_size == 1 and int(value.shape[0]) == width:
            value = value.unsqueeze(0)
        elif width == 1 and int(value.shape[0]) == batch_size:
            value = value.unsqueeze(-1)
        else:
            raise ValueError(f"{name} must have shape ({batch_size}, {width}), got {tuple(value.shape)}.")
    elif value.ndim != 2 or int(value.shape[0]) != batch_size or int(value.shape[1]) != width:
        raise ValueError(f"{name} must have shape ({batch_size}, {width}), got {tuple(value.shape)}.")

    return value.to(device=device, dtype=dtype)


def apply_per_band(
    codes: Sequence[Tensor],
    modules: Sequence[nn.Module],
    *,
    dtype: Optional[torch.dtype] = None,
) -> list[Tensor]:
    """Apply one module per aligned multiscale band."""
    outputs: list[Tensor] = []
    for code, module in zip(codes, modules, strict=True):
        if dtype is not None:
            code = code.to(dtype)
        outputs.append(module(code))
    return outputs


def multiscale_mean_abs(codes: Sequence[Tensor]) -> Tensor:
    """Return the per-row mean absolute activation across multiscale bands."""
    if len(codes) < 1:
        raise ValueError("codes must contain at least one band.")
    return torch.stack([code.to(torch.float32).abs().mean(dim=1) for code in codes], dim=0).mean(dim=0)


def multiscale_row_mse(left: Sequence[Tensor], right: Sequence[Tensor]) -> Tensor:
    """Return the per-row mean squared error across aligned multiscale bands."""
    if len(left) < 1 or len(right) < 1:
        raise ValueError("left and right must contain at least one band.")
    return torch.stack(
        [(lhs.to(torch.float32) - rhs.to(torch.float32)).pow(2).mean(dim=1) for lhs, rhs in zip(left, right, strict=True)],
        dim=0,
    ).mean(dim=0)


__all__ = [
    "apply_per_band",
    "as_batch_column",
    "as_optional_batch_column",
    "as_optional_feature_matrix",
    "multiscale_mean_abs",
    "multiscale_row_mse",
]
