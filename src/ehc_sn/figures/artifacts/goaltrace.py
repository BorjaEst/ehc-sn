"""Offline goaltrace task-overview artifact builder.

Produces a ``load_artifact_run_cases()``-compatible offline artifact from a
curated goaltrace corpus sample.  The artifact deserialises into a normal
``TraceTree`` containing goaltrace metadata (no dense prediction trace).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.evaluation.artifacts import (
    _SUCCESS_FILENAME,
    _write_dense_npz,
    build_evaluation_artifact_manifest,
)
from ehc_sn.traces.keys import (
    GOALTRACE_META_KEY_CURRENT_FLAG,
    GOALTRACE_META_KEY_GOAL_FLAG,
    GOALTRACE_META_KEY_NODE_MASK,
    GOALTRACE_META_KEY_OBSERVATION_ID,
    GOALTRACE_META_KEY_SUCCESSOR_INDICES,
    GOALTRACE_META_KEY_SUCCESSOR_MASK,
    GOALTRACE_META_KEY_TARGET_FIELD,
    GOALTRACE_META_KEY_WEIGHT,
)

# A representative goaltrace sample selected for diversity:
# - multiple viable branches
# - at least one non-viable branch
# - distinct current and goal nodes
# - nontrivial continuous weights
# - enough depth to show decay
# - not excessively dense arrows
_CURATED_SAMPLE: dict[str, Any] = {
    GOALTRACE_META_KEY_OBSERVATION_ID: np.array(
        [0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32
    ),
    GOALTRACE_META_KEY_WEIGHT: np.array(
        [0.0, 0.75, 0.90, 0.00, 0.60, 0.00, 0.00, 0.00],
        dtype=np.float32,
    ),
    GOALTRACE_META_KEY_CURRENT_FLAG: np.array(
        [True, False, False, False, False, False, False, False],
        dtype=bool,
    ),
    GOALTRACE_META_KEY_GOAL_FLAG: np.array(
        [False, False, False, False, False, True, False, False],
        dtype=bool,
    ),
    GOALTRACE_META_KEY_NODE_MASK: np.array(
        [True, True, True, True, True, True, True, True],
        dtype=bool,
    ),
    GOALTRACE_META_KEY_TARGET_FIELD: np.array(
        [1.0, 0.80, 0.64, 0.50, 0.40, 0.32, 0.0, 0.0],
        dtype=np.float32,
    ),
    GOALTRACE_META_KEY_SUCCESSOR_INDICES: np.array(
        [
            [1, 2, 3, 0],
            [4, 5, 0, 0],
            [4, 0, 0, 0],
            [5, 0, 0, 0],
            [5, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    ),
    GOALTRACE_META_KEY_SUCCESSOR_MASK: np.array(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    ),
}


def build_goaltrace_offline_artifact(output_dir: Path) -> Path:
    """Build a curated goaltrace task-overview artifact at *output_dir*.

    The produced artifact is compatible with ``load_artifact_run_cases()``
    and contains a single case with goaltrace metadata (no dense trace).

    Args:
        output_dir: Target directory for the artifact.  Created if missing.

    Returns:
        ``output_dir`` resolved to an absolute path.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    # Write empty dense NPZ (no prediction trace; metadata-only case)
    _write_dense_npz(cases_dir / "0000-sample.dense.npz", {})

    # Write metadata JSON — flat slash-delimited paths
    meta = {
        GOALTRACE_META_KEY_OBSERVATION_ID: _CURATED_SAMPLE[
            GOALTRACE_META_KEY_OBSERVATION_ID
        ].tolist(),
        GOALTRACE_META_KEY_WEIGHT: _CURATED_SAMPLE[
            GOALTRACE_META_KEY_WEIGHT
        ].tolist(),
        GOALTRACE_META_KEY_CURRENT_FLAG: _CURATED_SAMPLE[
            GOALTRACE_META_KEY_CURRENT_FLAG
        ].tolist(),
        GOALTRACE_META_KEY_GOAL_FLAG: _CURATED_SAMPLE[
            GOALTRACE_META_KEY_GOAL_FLAG
        ].tolist(),
        GOALTRACE_META_KEY_NODE_MASK: _CURATED_SAMPLE[
            GOALTRACE_META_KEY_NODE_MASK
        ].tolist(),
        GOALTRACE_META_KEY_TARGET_FIELD: _CURATED_SAMPLE[
            GOALTRACE_META_KEY_TARGET_FIELD
        ].tolist(),
        GOALTRACE_META_KEY_SUCCESSOR_INDICES: _CURATED_SAMPLE[
            GOALTRACE_META_KEY_SUCCESSOR_INDICES
        ].tolist(),
        GOALTRACE_META_KEY_SUCCESSOR_MASK: _CURATED_SAMPLE[
            GOALTRACE_META_KEY_SUCCESSOR_MASK
        ].tolist(),
    }
    (cases_dir / "0000-sample.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write manifest via shared constructor
    manifest = build_evaluation_artifact_manifest(
        task="goaltrace",
        regime_id="offline_overview",
        regime_kind="diagnostic",
        phase_kind="diag",
        trigger_kind="manual",
        epoch=0,
        step=0,
        evaluation={
            "model_family": "none",
            "trace_paradigm": "none",
            "global_step": 0,
            "epoch": 0,
        },
        summary={"n_cases": 1},
        cases=[
            {
                "case_id": "goaltrace-overview-sample",
                "source_context": None,
                "loss": None,
                "has_trace": True,
                "dense_artifact": "cases/0000-sample.dense.npz",
                "meta_artifact": "cases/0000-sample.meta.json",
            }
        ],
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write success sentinel
    (output_dir / _SUCCESS_FILENAME).write_text("", encoding="utf-8")

    return output_dir


# =============================================================================
__all__ = [
    "build_goaltrace_offline_artifact",
]
