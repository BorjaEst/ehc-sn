from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor

from hrm_sn.training.buffers import FifoBuffer


class PartialResetBatchAssembler:
    def __init__(self, *, buffer: FifoBuffer, keys: Sequence[str]):
        self.buffer = buffer
        self.keys = list(keys)

    def make_step_batch(self, *, incoming: Dict[str, Tensor], reset_mask: Tensor) -> Dict[str, Tensor]:
        return self.ingest_and_make_step_batch(incoming=incoming, reset_mask=reset_mask)

    def ingest_and_make_step_batch(self, *, incoming: Dict[str, Tensor], reset_mask: Tensor) -> Dict[str, Tensor]:
        """
        incoming: tensors on GPU (Lightning already moved them).
        reset_mask: bool tensor on GPU, shape (B,). True => this slot will load new data now.

        Returns: step_batch on GPU, same shapes as incoming.
        Also: pushes unused incoming rows to CPU buffer.
        """
        device = incoming[self.keys[0]].device
        B = incoming[self.keys[0]].shape[0]
        assert reset_mask.shape == (B,)

        reset_idx = reset_mask.nonzero(as_tuple=False).flatten()  # slots that WILL consume fresh data
        keep_idx = (~reset_mask).nonzero(as_tuple=False).flatten()  # slots that will ignore incoming data

        # Always buffer the incoming rows that won't be used (keep_idx)
        self.buffer.push_rows(incoming, keep_idx)

        # Optionally overwrite some reset slots with buffered rows (buffer-first refill)
        n_reset = int(reset_idx.numel())
        if n_reset == 0:
            return incoming

        n_from_buf = min(n_reset, len(self.buffer))
        if n_from_buf == 0:
            return incoming

        buf_rows_cpu = self.buffer.pop(n_from_buf)  # CPU pinned tensors, leading dim n_from_buf
        buf_rows_gpu = {k: v.to(device, non_blocking=True) for k, v in buf_rows_cpu.items()}

        # Choose which reset slots to fill from buffer: first n_from_buf reset indices
        fill_idx = reset_idx[:n_from_buf]

        # Any incoming rows at those fill_idx are now unused (since overwritten) => buffer them too
        self.buffer.push_rows(incoming, fill_idx)

        # Build step_batch by shallow-copying incoming and overwriting filled indices
        step_batch = dict(incoming)
        for k in self.keys:
            x = step_batch[k]
            x = x.clone()  # avoid in-place on incoming (safer)
            x.index_copy_(0, fill_idx, buf_rows_gpu[k])
            step_batch[k] = x

        return step_batch

    def refill_only(self, *, template: Dict[str, Tensor], reset_mask: Tensor) -> Dict[str, Tensor] | None:
        """
        Refill reset slots from the buffer only.

        Returns None if the buffer cannot satisfy the requested refill.
        """
        device = template[self.keys[0]].device
        B = template[self.keys[0]].shape[0]
        assert reset_mask.shape == (B,)

        reset_idx = reset_mask.nonzero(as_tuple=False).flatten()
        n_reset = int(reset_idx.numel())
        if n_reset == 0:
            return template
        if n_reset > len(self.buffer):
            return None

        buf_rows_cpu = self.buffer.pop(n_reset)
        buf_rows_gpu = {k: v.to(device, non_blocking=True) for k, v in buf_rows_cpu.items()}

        step_batch = dict(template)
        for k in self.keys:
            x = step_batch[k]
            x = x.clone()
            x.index_copy_(0, reset_idx, buf_rows_gpu[k])
            step_batch[k] = x

        return step_batch
