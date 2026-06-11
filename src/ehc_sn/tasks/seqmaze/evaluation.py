"""SeqMaze probe evaluation — binary edge-prediction score report.

Phase 0 metrics: accuracy, precision, recall, F1 over edge predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


# =============================================================================
@dataclass
class SeqMazeProbeScoreReport:
    """Aggregate edge-prediction scores for one evaluation pass.

    Attributes:
        accuracy: Fraction of correctly predicted edges (binary).
        precision: Fraction of positive predictions that are correct.
        recall: Fraction of true positive edges that were predicted.
        f1: Harmonic mean of precision and recall.
        total_valid_pairs: Number of (i,j) pairs that were unmasked.
    """

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    total_valid_pairs: int = 0

    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        parts = [
            f"accuracy={self.accuracy:.4f}",
            f"precision={self.precision:.4f}",
            f"recall={self.recall:.4f}",
            f"f1={self.f1:.4f}",
            f"pairs={self.total_valid_pairs}",
        ]
        return "SeqMazeProbeScoreReport(" + ", ".join(parts) + ")"

    # ------------------------------------------------------------------ #
    def merge(self, other: SeqMazeProbeScoreReport) -> SeqMazeProbeScoreReport:
        """Merge (average) another report into this one.

        Used for aggregating per-batch scores across an epoch.
        """
        total = self.total_valid_pairs + other.total_valid_pairs
        if total == 0:
            return self
        w_self = self.total_valid_pairs / total
        w_other = other.total_valid_pairs / total
        return SeqMazeProbeScoreReport(
            accuracy=self.accuracy * w_self + other.accuracy * w_other,
            precision=self.precision * w_self + other.precision * w_other,
            recall=self.recall * w_self + other.recall * w_other,
            f1=self.f1 * w_self + other.f1 * w_other,
            total_valid_pairs=total,
        )


# =============================================================================
def compute_edge_score_report(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> SeqMazeProbeScoreReport:
    """Compute binary edge-prediction metrics from logits, labels, and masks.

    Args:
        logits: (B, N, N, 2) — raw edge logits for each (i, j) pair.
        labels: (B, N, N) int64 — ground-truth 0/1 edge labels.
        mask: (B, N, N) bool — valid pairs.

    Returns:
        Aggregated score report.
    """
    preds = logits.argmax(dim=-1)  # (B, N, N)
    valid = mask

    # Flatten all valid pairs
    p = preds[valid]
    l = labels[valid]

    if p.numel() == 0:
        return SeqMazeProbeScoreReport()

    correct = (p == l).sum().item()
    total = p.numel()

    # True positives: pred=1 and label=1
    tp = ((p == 1) & (l == 1)).sum().item()
    # False positives: pred=1 and label=0
    fp = ((p == 1) & (l == 0)).sum().item()
    # False negatives: pred=0 and label=1
    fn = ((p == 0) & (l == 1)).sum().item()

    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return SeqMazeProbeScoreReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        total_valid_pairs=total,
    )
