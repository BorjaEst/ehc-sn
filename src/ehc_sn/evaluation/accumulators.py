"""Statistical accumulator implementations for evaluation consumers.

This module provides concrete ``EvaluationConsumer`` implementations that
accumulate model activity statistics on the evaluation device and materialize
canonical CPU payloads once on finalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ehc_sn.contracts.dependencies import Dependency, model_view
from ehc_sn.evaluation.contracts import (
    ArtifactKey,
    ArtifactKind,
    EvaluationConsumer,
    EvaluationRunContext,
    ProducedArtifact,
)
from ehc_sn.traces.observer import StepContext
from ehc_sn.types import (
    MultiScaleView,
    ScaleMetadata,
    SpatialPopulationStatisticsPayload,
)


# =============================================================================
class SpatialPopulationAccumulator(EvaluationConsumer):
    """On-device spatial sufficient-statistics accumulator.

    Bins neural activity by spatial location using on-device scatter-add
    and index-add operations.  Accumulators stay on the model device
    throughout evaluation; ``finalize()`` performs one CPU transfer and
    returns a ``ProducedArtifact``.

    Usage::

        from ehc_sn.evaluation.accumulators import SpatialPopulationAccumulator
        from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry

        acc = SpatialPopulationAccumulator(
            name="mec",
            population_view="mec.cells",
            n_locations=64,
            environment_id="arena-grid-01",
            geometry=SpatialBinGeometry(bin_size_x=1.0, bin_size_y=1.0),
            scale_metadata={
                "freq_0": ScaleMetadata(band_name="freq_0", feature_dim=32, index=0),
            },
        )
        # Use as consumer in execute_replay_evaluation_batch(consumers=[acc])
    """

    def __init__(  # ----------------------------------------------------------
        self,
        *,
        name: str,
        population_view: str,
        n_locations: int,
        environment_id: str,
        geometry: object,
        scale_metadata: dict[str, ScaleMetadata],
        moments: int = 1,
        device_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Configure one spatial sufficient-statistics accumulator.

        Args:
            name: Consumer name used as the ``ProducedArtifact`` key name.
            population_view: Semantic view key for the population activity
                (e.g. ``"mec.cells"``).  Must correspond to a
                ``MultiScaleView``-valued key in ``StepContext.views``.
            n_locations: Number of discrete location bins.
            environment_id: Stable string identifier for the environment.
            geometry: A ``SpatialBinGeometry`` instance describing the bin
                geometry for downstream analysis.
            scale_metadata: Mapping of band name to ``ScaleMetadata`` for
                each frequency band.  Keys must match the bands in the
                ``MultiScaleView`` returned by ``trace_views()``.
            moments: Number of statistical moments to accumulate.
                ``1`` = occupancy + activation sums.
                ``2`` = occupancy + activation sums + squared sums.
            device_dtype: Device dtype for accumulation.  ``float32`` is
                typical for GPU; ``float64`` provides higher precision.

        Raises:
            ValueError: If ``moments`` is not 1 or 2.
            ValueError: If ``n_locations < 1``.
            ValueError: If ``population_view`` is empty.
            ValueError: If ``scale_metadata`` is empty.
        """
        if moments not in (1, 2):
            raise ValueError(f"moments must be 1 or 2, got {moments}.")
        if n_locations < 1:
            raise ValueError(f"n_locations must be >= 1, got {n_locations}.")
        if not population_view:
            raise ValueError("population_view must be a non-empty string.")
        if not scale_metadata:
            raise ValueError("scale_metadata must be non-empty.")

        self._name = name
        self._population_view = population_view
        self._n_locations = n_locations
        self._environment_id = environment_id
        self._geometry = geometry
        self._scale_metadata = scale_metadata
        self._moments = moments
        self._device_dtype = device_dtype

        super().__init__(name=name)

        # Lazily allocated on first update() — we don't know the device
        # until the first ctx arrives.
        self._device: torch.device | None = None
        self._occupancy: Tensor | None = None
        self._activation_sum: dict[str, Tensor] | None = None
        self._activation_sq_sum: dict[str, Tensor] | None = None
        self._valid_step_count: int = 0

    # ── EvaluationConsumer (ABC) contract ────────────────────────────────────

    @property
    def dependencies(self) -> frozenset[Dependency]:
        """Typed dependencies: model view for the population, record field for location."""
        return frozenset({model_view(self._population_view)})

    def update(self, ctx: StepContext) -> None:
        """Bin one step's activity by spatial location.

        Silently skips when the population view is absent from
        ``ctx.views`` or location_id is ``None`` in snapshot data.
        """
        cells: MultiScaleView | None = ctx.views.get(self._population_view)
        if cells is None:
            return

        location_id: Tensor | None = ctx.record.snapshot.data.get("location_id")
        if location_id is None:
            return

        # Lazy allocation on first step.
        if self._occupancy is None:
            self._allocate(cells)

        # (B, 1) or (B,) -> (B,) long.
        flat_idx = location_id.squeeze(-1).long()
        batch_size = flat_idx.shape[0]
        valid = torch.ones(batch_size, dtype=torch.bool, device=self._device)

        # Occupancy: scatter_add ones at visited locations.
        self._occupancy.scatter_add_(
            0,
            flat_idx,
            valid.to(torch.int64),
        )

        # Activation sums per band.
        for band_name, tensor in cells.values.items():
            accum = self._activation_sum[band_name]
            cells_band = tensor.to(dtype=self._device_dtype)
            # cells_band: (B, U), flat_idx: (B,)
            accum.index_add_(0, flat_idx, cells_band)

            if self._activation_sq_sum is not None:
                sq_accum = self._activation_sq_sum[band_name]
                sq_accum.index_add_(0, flat_idx, cells_band.square())

        self._valid_step_count += batch_size

    def finalize(self) -> tuple[ProducedArtifact, ...]:
        """Transfer accumulated state to CPU and return a typed produced artifact.

        Returns:
            Tuple with one ``ProducedArtifact`` containing the
            ``SpatialPopulationStatisticsPayload`` on CPU.  Returns empty
            tuple if no data was accumulated.
        """
        if self._occupancy is None:
            return ()

        occupancy_np = self._occupancy.detach().cpu().numpy()
        activation_sum_np: dict[str, Tensor] = {
            band: t.detach().cpu().numpy()
            for band, t in self._activation_sum.items()
        }
        activation_sq_sum_np: dict[str, Tensor] | None = None
        if self._activation_sq_sum is not None:
            activation_sq_sum_np = {
                band: t.detach().cpu().numpy()
                for band, t in self._activation_sq_sum.items()
            }

        payload = SpatialPopulationStatisticsPayload(
            occupancy=occupancy_np,
            activation_sum=activation_sum_np,
            activation_sq_sum=activation_sq_sum_np,
            geometry=self._geometry,
            environment_ids=(self._environment_id,),
            scale_metadata=dict(self._scale_metadata),
            valid_step_count=self._valid_step_count,
            accumulation_dtype=str(self._device_dtype),
        )

        artifact = ProducedArtifact(
            key=ArtifactKey(ArtifactKind.AGGREGATE, self._name),
            schema_version=1,
            path=Path(f"aggregates/{self._name}.zarr"),
            media_type="application/vnd+zarr",
            producer_digest="",
            content_digest="",
            metadata={"payload_type": "SpatialPopulationStatisticsPayload"},
        )
        return (artifact,)

    def close(self) -> None:
        """Release resources.  No-op for device-resident accumulators."""
        return

    # ── MergeableAccumulator / distributed evaluation hooks ───────────────────────

    def merge(self, other: SpatialPopulationAccumulator) -> None:
        """Combine *other*'s state into this accumulator.

        Both accumulators must have identical configuration (geometry,
        environment IDs, scale metadata, and band structure).  Occupancy
        and activation sums are added element-wise.

        Args:
            other: Another accumulator of the same configuration.

        Raises:
            ValueError: If accumulator configurations are incompatible.
        """
        _assert_merge_compatible(self, other)

        if other._occupancy is None:
            return

        if self._occupancy is None:
            self._copy_from(other)
            return

        self._occupancy += other._occupancy.to(device=self._occupancy.device)
        for band in self._activation_sum:
            self._activation_sum[band] += other._activation_sum[band].to(
                device=self._activation_sum[band].device
            )
        if self._activation_sq_sum is not None:
            for band in self._activation_sq_sum:
                self._activation_sq_sum[band] += other._activation_sq_sum[
                    band
                ].to(device=self._activation_sq_sum[band].device)
        self._valid_step_count += other._valid_step_count

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        return {
            "occupancy": self._occupancy,
            "activation_sum": self._activation_sum,
            "activation_sq_sum": self._activation_sq_sum,
            "valid_step_count": self._valid_step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from a previous ``state_dict()`` call."""
        self._occupancy = state["occupancy"]
        self._activation_sum = state["activation_sum"]
        self._activation_sq_sum = state["activation_sq_sum"]
        self._valid_step_count = state["valid_step_count"]

    def reset(self) -> None:
        """Zero all accumulated state."""
        self._device = None
        self._occupancy = None
        self._activation_sum = None
        self._activation_sq_sum = None
        self._valid_step_count = 0

    # ── internal helpers ────────────────────────────────────────────────────

    def _allocate(self, cells: MultiScaleView) -> None:
        """Lazily allocate device-side accumulators on first step."""
        sample_tensor = next(iter(cells.values.values()))
        self._device = sample_tensor.device

        self._occupancy = torch.zeros(
            self._n_locations,
            dtype=torch.int64,
            device=self._device,
        )

        self._activation_sum = {}
        self._activation_sq_sum = {} if self._moments >= 2 else None
        for band_name in cells.values:
            feat_dim = self._scale_metadata[band_name].feature_dim
            self._activation_sum[band_name] = torch.zeros(
                (self._n_locations, feat_dim),
                dtype=self._device_dtype,
                device=self._device,
            )
            if self._activation_sq_sum is not None:
                self._activation_sq_sum[band_name] = torch.zeros(
                    (self._n_locations, feat_dim),
                    dtype=self._device_dtype,
                    device=self._device,
                )

    def _copy_from(self, other: SpatialPopulationAccumulator) -> None:
        """Copy accumulated state from *other* when self is empty."""
        self._device = other._device
        self._occupancy = other._occupancy.clone()
        self._activation_sum = {
            k: v.clone() for k, v in other._activation_sum.items()
        }
        if other._activation_sq_sum is not None:
            self._activation_sq_sum = {
                k: v.clone() for k, v in other._activation_sq_sum.items()
            }
        self._valid_step_count = other._valid_step_count


# =============================================================================
def _assert_merge_compatible(
    a: SpatialPopulationAccumulator,
    b: SpatialPopulationAccumulator,
) -> None:
    """Verify that two accumulators can be merged."""
    mismatches: list[str] = []
    if a._n_locations != b._n_locations:
        mismatches.append(f"n_locations: {a._n_locations} != {b._n_locations}")
    if a._environment_id != b._environment_id:
        mismatches.append(
            f"environment_id: {a._environment_id} != {b._environment_id}"
        )
    if a._scale_metadata.keys() != b._scale_metadata.keys():
        mismatches.append(
            f"scale_metadata keys: {set(a._scale_metadata.keys())} !="
            f" {set(b._scale_metadata.keys())}"
        )
    if mismatches:
        raise ValueError(
            "Accumulator merge failed — incompatible configuration:\n"
            + "\n".join(mismatches)
        )


# =============================================================================
__all__ = [
    "SpatialPopulationAccumulator",
]
