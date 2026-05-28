"""Bounded diagnostic reducers (torchmetrics.Metric subclasses).

These reducers accumulate bounded sufficient statistics over validation
batches and produce summary tensors for epoch-end visualization.  They
follow the ``update() -> compute() -> reset()`` lifecycle established by
:class:`~torchmetrics.Metric` and are DDP-safe via ``dist_reduce_fx="sum"``.

They must not store full trajectories, raw TraceTree objects, or unbounded
per-step metadata.  Total state size is O(n_locations) for occupancy and
O(n_bins) for histograms.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric


# =============================================================================
class OccupancyHistogram(Metric):
    """Bounded occupancy histogram over discrete locations.

    Accumulates location visit counts across validation batches and
    normalizes into a probability distribution on ``compute()``.
    """

    full_state_update = False

    def __init__(  # ----------------------------------------------------------
        self,
        n_locations: int,
        max_batches: int = 0,
    ) -> None:
        """Initialise the occupancy histogram.

        Args:
            n_locations: Number of discrete locations (environment size).
            max_batches: Maximum number of ``update()`` calls to accumulate.
                ``0`` means no limit.

        Raises:
            ValueError: If ``n_locations < 1`` or ``max_batches < 0``.
        """
        if n_locations < 1:
            raise ValueError(f"n_locations must be >= 1, got {n_locations}.")
        if max_batches < 0:
            raise ValueError(f"max_batches must be >= 0, got {max_batches}.")
        super().__init__(sync_on_compute=True)
        self._n_locations = n_locations
        self._max_batches = max_batches
        self.add_state("grid", default=torch.zeros(n_locations), dist_reduce_fx="sum")  # type: ignore[arg-type]
        self.add_state("batch_count", default=torch.tensor(0), dist_reduce_fx="sum")  # type: ignore[arg-type]

    def update(  # ------------------------------------------------------------
        self,
        location_ids: Tensor,
    ) -> None:
        """Accumulate visit counts from a batch of location indices.

        Args:
            location_ids: ``LongTensor`` of shape ``(B,)`` or ``(B, T)``.
                Values must be in ``[0, n_locations)``.

        Raises:
            IndexError: If any index is outside ``[0, n_locations)``.
        """
        if self._max_batches > 0 and self.batch_count >= self._max_batches:
            return
        flat = location_ids.reshape(-1).long()
        self.grid.scatter_add_(  # type: ignore[union-attr]
            0,
            flat.to(self.grid.device),
            torch.ones_like(flat, dtype=torch.float32),
        )
        self.batch_count += 1  # type: ignore[operator]

    def compute(
        self,
    ) -> Tensor:  # -------------------------------------------------
        """Return the normalized occupancy distribution.

        Returns:
            Tensor of shape ``(n_locations,)`` of type ``float32``,
            summing to 1.0.  Returns uniform distribution when no data
            has been accumulated.
        """
        total = self.grid.sum().clamp_min(1.0)  # type: ignore[union-attr]
        return self.grid / total  # type: ignore[return-value]

    def reset(self) -> None:  # ----------------------------------------------
        """Zero the grid and batch counter."""
        self.grid.zero_()  # type: ignore[union-attr]
        self.batch_count.zero_()  # type: ignore[union-attr]


# =============================================================================
class HiddenNormHistogram(Metric):
    """Bounded histogram of hidden-state L2 norms.

    Uses fixed bin edges; accumulates per-bin counts.  Bins are immutable
    after construction.
    """

    full_state_update = False

    def __init__(  # ----------------------------------------------------------
        self,
        bin_edges: Tensor,
        max_batches: int = 0,
    ) -> None:
        """Initialise the norm histogram.

        Args:
            bin_edges: 1-D ``float32`` tensor of length ``>= 2``.
            max_batches: Maximum number of ``update()`` calls to accumulate.
                ``0`` means no limit.

        Raises:
            ValueError: If ``bin_edges`` is not 1-D, has fewer than 2 elements,
                or ``max_batches < 0``.
        """
        if bin_edges.dim() != 1:
            raise ValueError(
                f"bin_edges must be 1-D, got shape {bin_edges.shape}."
            )
        if bin_edges.shape[0] < 2:
            raise ValueError(
                f"bin_edges must have at least 2 elements, got {bin_edges.shape[0]}."
            )
        if max_batches < 0:
            raise ValueError(f"max_batches must be >= 0, got {max_batches}.")
        super().__init__(sync_on_compute=True)
        self.register_buffer("_edges", bin_edges, persistent=False)
        n_bins = bin_edges.shape[0] - 1
        self._max_batches = max_batches
        self.add_state("counts", default=torch.zeros(n_bins), dist_reduce_fx="sum")  # type: ignore[arg-type]
        self.add_state("batch_count", default=torch.tensor(0), dist_reduce_fx="sum")  # type: ignore[arg-type]

    def update(  # ------------------------------------------------------------
        self,
        norms: Tensor,
    ) -> None:
        """Accumulate norm values into the fixed-bin histogram.

        Args:
            norms: ``float32`` tensor of shape ``(B,)`` or ``(B, D)``.
        """
        if self._max_batches > 0 and self.batch_count >= self._max_batches:
            return
        flat = norms.detach().reshape(-1).to(self._edges.device)
        bin_indices = torch.bucketize(flat, self._edges) - 1
        bin_indices = bin_indices.clamp(0, self.counts.shape[0] - 1)
        self.counts.scatter_add_(  # type: ignore[union-attr]
            0,
            bin_indices,
            torch.ones_like(bin_indices, dtype=torch.float32),
        )
        self.batch_count += 1  # type: ignore[operator]

    def compute(  # -----------------------------------------------------------
        self,
    ) -> tuple[Tensor, Tensor]:
        """Return the bin centres and normalized density.

        Returns:
            A tuple ``(bin_centers, density)``, each of shape ``(n_bins,)``.
            ``density`` sums to 1.0.
        """
        centers = (self._edges[:-1] + self._edges[1:]) / 2.0  # type: ignore[operator]
        total = self.counts.sum().clamp_min(1.0)  # type: ignore[union-attr]
        return centers, self.counts / total  # type: ignore[return-value]

    def reset(  # -------------------------------------------------------------
        self,
    ) -> None:
        """Zero the bin counts and batch counter."""
        self.counts.zero_()  # type: ignore[union-attr]
        self.batch_count.zero_()  # type: ignore[union-attr]


# =============================================================================
__all__ = [
    "OccupancyHistogram",
    "HiddenNormHistogram",
]
