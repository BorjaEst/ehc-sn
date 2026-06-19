"""SeqMaze task supervision — target coercion for path prediction.

This module owns the conversion of SeqMaze batch tensors into
learning-ready supervision structs.  It is task-owned and reusable
across model families (HRM v1, HRM v2, EHP).

Loss-label construction (PAD → ignore index) moved here from
:mod:`ehc_sn.adapters.hrm.objectives`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ehc_sn.tasks.seqmaze.contracts import SEQMAZE_IGNORE_LABEL_ID
from ehc_sn.types import Batch

# =============================================================================
SEQMAZE_META_KEY_TARGET_PATH: str = "target_path"
SEQMAZE_META_KEY_PATH_MASK: str = "path_mask"
SEQMAZE_META_KEY_PATH_LENGTH: str = "path_length"


# =============================================================================
@dataclass(frozen=True)
class SeqMazeTokenSupervision:
    """Typed supervision struct for SeqMaze path prediction.

    Attributes:
        labels: Integer labels with ignore positions set to
            :data:`SEQMAZE_IGNORE_LABEL_ID`, shape ``(B, T)``.
            PAD positions (``path_mask=False``) are replaced with the
            ignore label.
        weights: Per-token loss weights, shape ``(B, T)`` float32, or
            ``None`` for uniform weighting.
    """

    labels: Tensor
    weights: Tensor | None = None


# =============================================================================
def build_seqmaze_supervision(executed_batch: Batch) -> SeqMazeTokenSupervision:
    """Build SeqMaze path supervision from an executed batch frame.

    Extracts the target path, path mask, and path length from the batch,
    then constructs loss-ready labels with PAD positions replaced by
    :data:`SEQMAZE_IGNORE_LABEL_ID`.

    Args:
        executed_batch: Must contain ``"target_path"``, ``"path_mask"``,
            and ``"path_length"``.

    Returns:
        Typed supervision struct.  ``weights`` is ``None`` (uniform).

    Raises:
        KeyError: If any required key is missing from *executed_batch*.
    """
    for key in (
        SEQMAZE_META_KEY_TARGET_PATH,
        SEQMAZE_META_KEY_PATH_MASK,
        SEQMAZE_META_KEY_PATH_LENGTH,
    ):
        if key not in executed_batch:
            raise KeyError(
                f"build_seqmaze_supervision: '{key}' missing from "
                f"executed_batch."
            )

    path_index = executed_batch[SEQMAZE_META_KEY_TARGET_PATH].to(
        dtype=torch.int64
    )
    path_mask = executed_batch[SEQMAZE_META_KEY_PATH_MASK].to(dtype=torch.bool)

    labels = torch.where(
        path_mask,
        path_index,
        torch.full_like(path_index, SEQMAZE_IGNORE_LABEL_ID, dtype=torch.int64),
    )
    return SeqMazeTokenSupervision(labels=labels, weights=None)


# =============================================================================
__all__ = [
    "SEQMAZE_META_KEY_TARGET_PATH",
    "SEQMAZE_META_KEY_PATH_MASK",
    "SEQMAZE_META_KEY_PATH_LENGTH",
    "SeqMazeTokenSupervision",
    "build_seqmaze_supervision",
]
