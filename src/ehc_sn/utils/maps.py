"""Map tensor utilities for spatial processing and metrics."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


# -------------------------------------------------------------------------------------------
def pad_or_crop(t: Tensor, target_hw: Tuple[int, int]) -> Tensor:
    """Pad (bottom/right) or center-crop a (C,H,W) tensor to target size.

    Args:
        t: Input tensor (C,H,W)
        target_hw: (target_h, target_w)
    Returns:
        Tensor with shape (C,target_h,target_w)
    """
    if t.dim() != 3:
        raise ValueError("pad_or_crop expects tensor of shape (C,H,W)")
    c, h, w = t.shape
    th, tw = target_hw
    # Fast path
    if h == th and w == tw:
        return t
    # Pad if smaller
    if h <= th and w <= tw:
        out = torch.zeros(c, th, tw, dtype=t.dtype)
        out[:, :h, :w] = t
        return out
    # Need cropping on at least one dimension – center crop
    top = max(0, (h - th) // 2)
    left = max(0, (w - tw) // 2)
    return t[:, top : top + th, left : left + tw]


# -------------------------------------------------------------------------------------------
def wall_density(t: Tensor) -> float:
    """Compute fraction of wall cells assuming channel 0 are walls (binary)."""
    if t.dim() != 3:
        raise ValueError("wall_density expects tensor of shape (C,H,W)")
    h, w = t.shape[1:]
    total = h * w
    if total == 0:
        return 0.0
    return float(t[0].sum().item() / total)


__all__ = ["pad_or_crop", "wall_density"]
