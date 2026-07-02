"""Canonical trace field vocabulary — shared semantic field specs.

Defines the global vocabulary of trace field names that capture profiles
may reference.  Each entry is a :class:`TraceFieldSpec` with metadata
(name, dtype, shape axes, storage category) but **no getter** — extraction
is implemented by task-specific ``trace_task_fields()`` in task packages.

Boundary rules:
    - Must not import from ``tasks/``.
    - Must not import ``specs.py`` (specs may import vocabulary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceFieldSpec:
    """Specification for one semantic trace field.

    This is the **vocabulary** entry — it declares that a field with
    this name exists and what its properties are.  Extraction logic
    lives in task-specific ``TraceField`` objects (see ``TraceField``
    in ``traces.observer``).
    """

    name: str
    description: str = ""
    storage: str = "dense"
    dtype: str = "unknown"
    """Approximate NumPy dtype string for documentation / validation."""
    axes: tuple[str, ...] = ()
    """Semantic axis names (e.g. ``("reasoning_step", "batch", "slot")``)."""


# =============================================================================
# Prediction fields — shared across tasks that produce decoded outputs
# =============================================================================

PRED_PREDICTION_OVERLAY = TraceFieldSpec(
    name="pred/prediction_overlay",
    description="Decoded path overlay over maze-grid spatial slots, per reasoning step.",
    storage="dense",
    dtype="uint8",
    axes=("reasoning_step", "batch", "spatial_slot"),
)

TARGET_PREDICTION_OVERLAY = TraceFieldSpec(
    name="target/prediction_overlay",
    description="Oracle/ground-truth path overlay over maze-grid spatial slots.",
    storage="meta",
    dtype="uint8",
    axes=("batch", "spatial_slot"),
)

PRED_OBSERVATION_ID_POST = TraceFieldSpec(
    name="pred/observation_id/post",
    description="Post-update predicted observation id from TEM.",
    storage="dense",
    dtype="int64",
    axes=("reasoning_step", "batch"),
)

PRED_OBSERVATION_ID_RECALL = TraceFieldSpec(
    name="pred/observation_id/recall",
    description="Recalled observation id from TEM retrieval.",
    storage="dense",
    dtype="int64",
    axes=("reasoning_step", "batch"),
)

PRED_OBSERVATION_ID_PATH = TraceFieldSpec(
    name="pred/observation_id/path",
    description="Bayesian path posterior over observation ids from TEM.",
    storage="dense",
    dtype="float32",
    axes=("reasoning_step", "batch", "observation"),
)

TARGET_OBSERVATION_ID = TraceFieldSpec(
    name="target/observation_id",
    description="Oracle/ground-truth observation id for the current step.",
    storage="meta",
    dtype="int64",
    axes=("batch",),
)

ROUTEBIND_TRAJECTORY_FIELD = TraceFieldSpec(
    name="routebind/trajectory_field",
    description="Predicted trajectory field over spatial grid.",
    storage="dense",
    dtype="float32",
    axes=("reasoning_step", "batch", "spatial_x", "spatial_y"),
)

# =============================================================================
# Registry
# =============================================================================

TRACE_VOCABULARY: dict[str, TraceFieldSpec] = {
    spec.name: spec
    for spec in [
        PRED_PREDICTION_OVERLAY,
        TARGET_PREDICTION_OVERLAY,
        PRED_OBSERVATION_ID_POST,
        PRED_OBSERVATION_ID_RECALL,
        PRED_OBSERVATION_ID_PATH,
        TARGET_OBSERVATION_ID,
        ROUTEBIND_TRAJECTORY_FIELD,
    ]
}


def list_vocabulary_names() -> list[str]:
    """Return sorted list of all vocabulary field names."""
    return sorted(TRACE_VOCABULARY)


__all__ = [
    "PRED_PREDICTION_OVERLAY",
    "TARGET_PREDICTION_OVERLAY",
    "PRED_OBSERVATION_ID_POST",
    "PRED_OBSERVATION_ID_RECALL",
    "PRED_OBSERVATION_ID_PATH",
    "TARGET_OBSERVATION_ID",
    "ROUTEBIND_TRAJECTORY_FIELD",
    "TRACE_VOCABULARY",
    "TraceFieldSpec",
    "list_vocabulary_names",
]
