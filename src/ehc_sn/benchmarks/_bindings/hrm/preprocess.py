"""Shared HRM benchmark preprocessing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ehc_sn.data.schema import CHANNEL_SOLUTION, O_ID
from ehc_sn.data.transforms import channels_to_grid


def _coerce_numpy_channels(batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return numpy channel arrays for one raw benchmark batch."""
    channels: dict[str, np.ndarray] = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            channels[key] = value
            continue
        if isinstance(value, Tensor):
            channels[key] = value.detach().cpu().numpy()
            continue
        raise TypeError(f"Unsupported MazeHard channel type for key {key!r}: {type(value).__name__}.")
    return channels


def supervised_maze_tokenize(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Convert raw MazeHard channels into flattened input and label tensors."""
    channels = _coerce_numpy_channels(batch)
    grid = channels_to_grid(channels)["grid"]
    input_ids = torch.from_numpy(grid.reshape(-1).astype(np.int64, copy=False))
    labels = input_ids.clone()
    if CHANNEL_SOLUTION in channels:
        solution_mask = torch.from_numpy((channels[CHANNEL_SOLUTION].reshape(-1) > 0).astype(np.bool_, copy=False))
        labels[solution_mask] = O_ID
    return {"input_ids": input_ids, "labels": labels}


__all__ = ["supervised_maze_tokenize"]
