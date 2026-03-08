"""Small FIFO buffer utilities for partial-reset training.

The main consumer is :class:`~ehc_sn.training.partial_reset.PartialResetBatchAssembler`.
Rows are pushed from (typically GPU) batches and stored as CPU tensors (optionally
pinned) so they can be used to refill halted slots later.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence

import torch
from torch import Tensor


# =================================================================================================
@dataclass
class _Chunk:
    """Internal storage unit for :class:`FifoBuffer`.

    A chunk is a dict of CPU tensors (same number of rows per key) plus a cursor
    tracking how many rows have already been consumed.
    """

    rows: Dict[str, Tensor]  # CPU tensors, leading dim = n_rows
    start: int = 0  # how many rows already consumed


# =================================================================================================
class FifoBuffer:
    """FIFO buffer storing complete batch rows on CPU.

    Designed for partial-reset batching: when some slots halt, their rows can be
    pushed into the buffer and later popped to refill a step batch.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, *, capacity_rows: int, keys: Sequence[str], pin_memory: bool = True,
    ) -> None:  # fmt: skip
        """Create a FIFO buffer.

        Args:
            capacity_rows: Maximum number of rows stored across all chunks.
            keys: Keys to store from incoming batches.
            pin_memory: If True, pin CPU tensors for faster H2D transfers.
        """
        self.capacity_rows = capacity_rows
        self.keys = list(keys)
        self.pin_memory = pin_memory
        self._chunks: Deque[_Chunk] = deque()
        self._size_rows = 0

    def __len__(  # -------------------------------------------------------------------------------
        self,
    ) -> int:  # fmt: skip
        """Number of rows currently stored in the buffer."""
        return self._size_rows

    def clear(  # ---------------------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        self._chunks.clear()
        self._size_rows = 0

    def push_rows(  # -----------------------------------------------------------------------------
        self, batch: Dict[str, Tensor], row_indices: Tensor,
    ) -> None:  # fmt: skip
        """Takes rows from (likely GPU) batch, stores them on CPU as one chunk."""
        if row_indices.numel() == 0:
            return

        # select + detach
        rows = {k: batch[k].index_select(0, row_indices).detach() for k in self.keys}

        # move to CPU (whole rows per key)
        rows = {k: v.to("cpu") for k, v in rows.items()}
        if self.pin_memory:
            rows = {k: v.pin_memory() for k, v in rows.items()}

        # enforce capacity (drop newest or oldest; pick one policy)
        n = next(iter(rows.values())).shape[0]
        self._chunks.append(_Chunk(rows=rows))
        self._size_rows += n
        self._trim_to_capacity()

    def pop(  # -----------------------------------------------------------------------------------
        self, n: int,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Pop up to n rows (CPU tensors)."""
        n = min(n, self._size_rows)
        out: Dict[str, list[Tensor]] = {k: [] for k in self.keys}

        remaining = n
        while remaining > 0:
            ch = self._chunks[0]
            avail = ch.rows[self.keys[0]].shape[0] - ch.start
            take = min(avail, remaining)
            sl = slice(ch.start, ch.start + take)

            for k in self.keys:
                out[k].append(ch.rows[k][sl])

            ch.start += take
            self._size_rows -= take
            remaining -= take

            if ch.start >= ch.rows[self.keys[0]].shape[0]:
                self._chunks.popleft()

        # Return properly shaped empty tensors if no data
        if n == 0:
            return {k: torch.empty((0,), dtype=torch.int32) for k in self.keys}
        return {k: torch.cat(v, dim=0) for k, v in out.items()}

    def _trim_to_capacity(  # ---------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Simple policy: drop oldest chunks until within capacity."""
        while self._size_rows > self.capacity_rows and self._chunks:
            ch = self._chunks.popleft()
            n = ch.rows[self.keys[0]].shape[0] - ch.start
            self._size_rows -= n


# =================================================================================================
__all__ = ["FifoBuffer"]
