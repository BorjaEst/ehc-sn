"""SeqMaze probe and v1 evaluation — edge-prediction and path-prediction score reports.

Phase 0 metrics: accuracy, precision, recall, F1 over edge predictions.
Phase 1 metrics: sequence_exact, path_position_accuracy, valid_transition_rate,
    reaches_goal, path_length_regret, eos_accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ehc_sn.eval.contracts import EvaluationCaseResult
from ehc_sn.metrics.spec import MetricSpec, TaskScoringSpec
from ehc_sn.types import Batch

from .contracts import SeqMazeTargets


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


# =============================================================================
# Phase 1 — Path prediction (v1) evaluation
# =============================================================================

SEQMAZE_V1_METRIC_SPECS: list[MetricSpec] = [
    MetricSpec(
        name="sequence_exact",
        label="Exact sequence accuracy",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=True,
        description=(
            "Fraction of sequences that exactly match the target after "
            "EOS canonicalization."
        ),
    ),
    MetricSpec(
        name="path_position_accuracy",
        label="Path position accuracy",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=True,
        description=(
            "Fraction of non-PAD target positions predicted correctly "
            "(including EOS)."
        ),
    ),
    MetricSpec(
        name="valid_transition_rate",
        label="Valid transition rate",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=True,
        description=(
            "Proportion of adjacent candidate-token pairs (before first EOS) "
            "that are valid graph edges."
        ),
    ),
    MetricSpec(
        name="reaches_goal",
        label="Reaches goal rate",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=True,
        description=(
            "Proportion of sequences where the goal token appears "
            "before the first EOS."
        ),
    ),
    MetricSpec(
        name="path_length_regret",
        label="Path length regret",
        higher_is_better=False,
        unit="count",
        scope="task",
        benchmark_eligible=True,
        description=(
            "Extra tokens beyond the shortest-path length in the "
            "candidate-token prefix before first EOS."
        ),
    ),
    MetricSpec(
        name="eos_accuracy",
        label="EOS position accuracy",
        higher_is_better=True,
        unit="proportion",
        scope="task",
        benchmark_eligible=True,
        description=(
            "Fraction of samples where the first predicted EOS position "
            "matches the target EOS position."
        ),
    ),
]

SEQMAZE_V1_SCORING_SPEC: TaskScoringSpec = TaskScoringSpec(
    task_name="seqmaze",
    metrics={spec.name: spec for spec in SEQMAZE_V1_METRIC_SPECS},
    default_score="sequence_exact",
)


# =============================================================================
class SeqMazeValidationScorer:
    """Stateful online validation scorer for SeqMaze path-prediction metrics.

    Accumulates counts across validation batches and computes epoch-level
    rates.  Reuses the canonical evaluation primitives for decoding but
    uses count accumulation rather than ratio averaging.

    Computed metrics::

        seqmaze/eos_present_rate
        seqmaze/goal_reached_rate
        seqmaze/valid_transition_rate
    """

    def __init__(self, eos_id: int, pad_id: int, n_max: int) -> None:
        self._eos_id = eos_id
        self._pad_id = pad_id
        self._n_max = n_max
        self.reset()

    def update_from_evaluation(
        self, result: EvaluationCaseResult, batch: Batch
    ) -> None:
        """Update accumulated counters from one evaluation batch.

        Uses the last step's task output (constrained path logits) and
        the batch metadata.
        """
        from ehc_sn.tasks.seqmaze.runtime import extract_seqmaze_targets

        last_step = result.evaluated.last_step
        objective_step = last_step.outputs  # ACTObjectiveStep
        step_output = objective_step.outputs  # ACTStepOutput | None
        if step_output is None:
            raise RuntimeError(
                "SeqMaze validation requires the retained ACT step output, "
                "but this evaluation result contains metrics-only objective data."
            )
        logits = step_output.task.path_logits  # (B, T, V)

        targets = extract_seqmaze_targets(batch)
        adj = _build_adjacency(
            batch["successor_indices"],
            batch["successor_mask"],
            self._n_max,
        )
        goal_idx = batch["node_goal_flag"].to(torch.int64).argmax(dim=-1)
        pred = logits.argmax(dim=-1)
        canonical_pred = _canonicalize(pred, self._eos_id, self._pad_id)

        B = int(canonical_pred.shape[0])
        for b in range(B):
            self._sequence_count += 1

            pred_eos_pos = int(
                _extract_eos_position(canonical_pred[b : b + 1], self._eos_id)[
                    0
                ]
            )
            # EOS presence
            if pred_eos_pos < canonical_pred.shape[1]:
                self._eos_present_count += 1

            # Goal reached
            prefix = canonical_pred[b, :pred_eos_pos]
            if (prefix == goal_idx[b]).any():
                self._goal_reached_count += 1

            # Valid transitions
            cand_tokens = prefix[prefix < self._n_max]
            if cand_tokens.shape[0] >= 2:
                total = cand_tokens.shape[0] - 1
                valid = 0
                for t in range(total):
                    src = int(cand_tokens[t].item())
                    dst = int(cand_tokens[t + 1].item())
                    if (
                        src < self._n_max
                        and dst < self._n_max
                        and adj[b, src, dst]
                    ):
                        valid += 1
                self._valid_transition_count += valid
                self._total_transition_count += total

    def compute(self) -> dict[str, Tensor]:
        """Compute epoch-level rates from accumulated counters."""
        metrics: dict[str, Tensor] = {}
        if self._sequence_count > 0:
            metrics["seqmaze/eos_present_rate"] = torch.tensor(
                self._eos_present_count / self._sequence_count
            )
            metrics["seqmaze/goal_reached_rate"] = torch.tensor(
                self._goal_reached_count / self._sequence_count
            )
        if self._total_transition_count > 0:
            metrics["seqmaze/valid_transition_rate"] = torch.tensor(
                self._valid_transition_count / self._total_transition_count
            )
        return metrics

    def reset(self) -> None:
        """Reset all accumulated counters."""
        self._sequence_count = 0
        self._eos_present_count = 0
        self._goal_reached_count = 0
        self._valid_transition_count = 0
        self._total_transition_count = 0


# =============================================================================
@dataclass(frozen=True)
class SeqMazeScoreReport:
    """Aggregate path-prediction scores for one seqmaze v1 evaluation pass.

    All fields are scalar (0-d) float32 tensors except path_length_regret,
    which is a scalar float32.

    Attributes:
        sequence_exact: Fraction of exactly correct sequences.
        path_position_accuracy: Fraction of non-PAD positions correct.
        valid_transition_rate: Fraction of valid adjacent transitions.
        reaches_goal: Fraction of sequences reaching the goal.
        path_length_regret: Mean extra tokens beyond shortest path.
        eos_accuracy: Fraction of correct EOS positions.
    """

    sequence_exact: Tensor = field(default_factory=lambda: torch.tensor(0.0))
    path_position_accuracy: Tensor = field(
        default_factory=lambda: torch.tensor(0.0)
    )
    valid_transition_rate: Tensor = field(
        default_factory=lambda: torch.tensor(0.0)
    )
    reaches_goal: Tensor = field(default_factory=lambda: torch.tensor(0.0))
    path_length_regret: Tensor = field(
        default_factory=lambda: torch.tensor(0.0)
    )
    eos_accuracy: Tensor = field(default_factory=lambda: torch.tensor(0.0))

    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        parts = [
            f"sequence_exact={self.sequence_exact.item():.4f}",
            f"path_position_accuracy={self.path_position_accuracy.item():.4f}",
            f"valid_transition_rate={self.valid_transition_rate.item():.4f}",
            f"reaches_goal={self.reaches_goal.item():.4f}",
            f"path_length_regret={self.path_length_regret.item():.4f}",
            f"eos_accuracy={self.eos_accuracy.item():.4f}",
        ]
        return "SeqMazeScoreReport(" + ", ".join(parts) + ")"

    # ------------------------------------------------------------------ #
    def as_dict(self) -> dict[str, Tensor]:
        """Return the canonical key-value dict for logging."""
        return {
            "seqmaze/sequence_exact": self.sequence_exact,
            "seqmaze/path_position_accuracy": self.path_position_accuracy,
            "seqmaze/valid_transition_rate": self.valid_transition_rate,
            "seqmaze/reaches_goal": self.reaches_goal,
            "seqmaze/path_length_regret": self.path_length_regret,
            "seqmaze/eos_accuracy": self.eos_accuracy,
        }


# =============================================================================
@dataclass(frozen=True)
class SeqMazeStepScore:
    """Per-step answer quality for one deliberation step.

    At every control step the model exposes a complete parallel path
    prediction.  This score captures the quality of that prediction so
    the reward projector can translate it into a scalar RL signal.

    All fields are per-sample tensors of shape ``(B,)``.

    Attributes:
        path_exact: (B,) bool — canonicalized prediction matches target exactly.
        goal_reached: (B,) bool — goal token appears before first EOS.
        all_transitions_valid: (B,) bool — every adjacent candidate-token pair
            before first EOS is a valid graph edge.
        eos_present: (B,) bool — at least one EOS token appears.
        correct_tokens: (B,) int64 — number of correctly predicted supervised
            positions (including EOS).
        total_tokens: (B,) int64 — number of supervised positions in the target.
    """

    path_exact: Tensor
    goal_reached: Tensor
    all_transitions_valid: Tensor
    eos_present: Tensor
    correct_tokens: Tensor
    total_tokens: Tensor


# =============================================================================
def build_seqmaze_step_score(
    task_output: SeqMazeTaskOutput,
    targets: SeqMazeTargets,
    successor_indices: Tensor,  # (B, N, K)
    successor_mask: Tensor,  # (B, N, K)
    goal_candidate_index: Tensor,  # (B, N)
    *,
    eos_id: int,
    pad_id: int,
    n_max: int,
) -> SeqMazeStepScore:
    """Compute per-step answer quality from the current path prediction.

    Uses the same canonical decoding and graph-validation primitives as
    offline evaluation.  No second interpretation of correctness.

    Args:
        task_output: Current path-prediction output from the adapter.
        targets: Path supervision targets for the batch.
        successor_indices: (B, N, K) successor candidate indices.
        successor_mask: (B, N, K) valid successor slots.
        goal_candidate_index: (B, N) one-hot or index of the goal token.
        eos_id: EOS token id (= N_max).
        pad_id: PAD token id (= N_max + 1).
        n_max: Maximum candidate nodes.

    Returns:
        SeqMazeStepScore with per-sample fields of shape (B,).
    """
    logits = task_output.path_logits  # (B, T, V)
    pred = logits.argmax(dim=-1)  # (B, T)
    canonical_pred = _canonicalize(pred, eos_id, pad_id)
    canonical_target = _canonicalize(targets.path_index, eos_id, pad_id)
    B, T = canonical_pred.shape
    device = canonical_pred.device

    # path_exact
    path_exact = (canonical_pred == canonical_target).all(dim=-1)  # (B,)

    # correct_tokens and total_tokens
    correct = (canonical_pred == canonical_target) & targets.path_mask
    correct_tokens = correct.sum(dim=-1)  # (B,)
    total_tokens = targets.path_mask.sum(dim=-1)  # (B,)

    # eos_present
    pred_eos_pos = _extract_eos_position(canonical_pred, eos_id)
    eos_present = pred_eos_pos < T  # (B,)

    # goal_reached
    goal_idx = (
        goal_candidate_index.argmax(dim=-1)
        if goal_candidate_index.ndim == 2
        else goal_candidate_index
    )
    goal_reached = torch.zeros(B, dtype=torch.bool, device=device)
    for b in range(B):
        t_len = int(pred_eos_pos[b].item())
        prefix = canonical_pred[b, :t_len]
        goal_reached[b] = (prefix == goal_idx[b]).any()

    # all_transitions_valid
    adj = _build_adjacency(successor_indices, successor_mask, n_max)
    all_transitions_valid = torch.ones(B, dtype=torch.bool, device=device)
    for b in range(B):
        t_len = int(pred_eos_pos[b].item())
        prefix = canonical_pred[b, :t_len]
        is_candidate = prefix < n_max
        cand_tokens = prefix[is_candidate]
        if cand_tokens.shape[0] >= 2:
            for t_idx in range(cand_tokens.shape[0] - 1):
                src = int(cand_tokens[t_idx].item())
                dst = int(cand_tokens[t_idx + 1].item())
                if not (src < n_max and dst < n_max and adj[b, src, dst]):
                    all_transitions_valid[b] = False
                    break

    return SeqMazeStepScore(
        path_exact=path_exact,
        goal_reached=goal_reached,
        all_transitions_valid=all_transitions_valid,
        eos_present=eos_present,
        correct_tokens=correct_tokens,
        total_tokens=total_tokens,
    )


# =============================================================================
def _canonicalize(
    pred: Tensor,  # (B, T) int64 — argmax path token indices
    eos_id: int,
    pad_id: int,
) -> Tensor:
    """Truncate predictions after first EOS; replace remaining positions with PAD.

    Args:
        pred: (B, T) argmax prediction indices.
        eos_id: EOS token id.
        pad_id: PAD token id.

    Returns:
        (B, T) canonicalized prediction.
    """
    B, T = pred.shape
    device = pred.device

    # Find first EOS position per batch row
    is_eos = pred == eos_id  # (B, T)
    # For rows with no EOS, use T as the cutoff (no truncation)
    has_eos = is_eos.any(dim=-1)  # (B,)
    first_eos_pos = is_eos.to(torch.int64).argmax(dim=-1)  # (B,) — T if none

    # Build range mask: positions <= first_eos_pos are kept
    range_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    # For rows with EOS: keep up to and including first EOS
    # For rows without EOS: keep everything
    cutoff = torch.where(has_eos, first_eos_pos + 1, T)  # (B,)
    keep_mask = range_idx < cutoff.unsqueeze(-1)  # (B, T)

    canonical = torch.where(
        keep_mask, pred, torch.tensor(pad_id, device=device)
    )
    return canonical


# =============================================================================
def _extract_eos_position(
    tokens: Tensor,  # (B, T) int64
    eos_id: int,
) -> Tensor:
    """Return the index of the first EOS in each row, or T if not found."""
    is_eos = tokens == eos_id
    has_eos = is_eos.any(dim=-1)
    first_pos = is_eos.to(torch.int64).argmax(dim=-1)
    return torch.where(
        has_eos, first_pos, torch.tensor(tokens.shape[1], device=tokens.device)
    )


# =============================================================================
def _build_adjacency(
    successor_indices: Tensor,  # (B, N, K)
    successor_mask: Tensor,  # (B, N, K)
    n_max: int,
) -> Tensor:
    """Build a dense (B, N, N) adjacency matrix from successor indices.

    adj[b, i, j] = 1 if node j is a valid successor of node i.
    """
    B, N, K = successor_indices.shape
    adj = torch.zeros(
        B, N, n_max, dtype=torch.bool, device=successor_indices.device
    )
    valid = successor_mask  # (B, N, K)
    idx = successor_indices  # (B, N, K)
    # Clamp indices to valid range to avoid scatter OOB
    idx_clamped = idx.clamp(min=0, max=n_max - 1)
    # Expand dims for scatter
    batch_arange = torch.arange(B, device=idx.device).view(-1, 1, 1)
    node_arange = torch.arange(N, device=idx.device).view(1, -1, 1)
    adj[batch_arange, node_arange, idx_clamped] = valid
    return adj


# =============================================================================
def compute_seqmaze_score_report(
    logits: Tensor,  # (B, T, V) — path logits, V = N_max + 2
    targets: SeqMazeTargets,
    successor_indices: Tensor,  # (B, N, K)
    successor_mask: Tensor,  # (B, N, K)
    goal_candidate_index: Tensor,  # (B, N) — one-hot or index; we take argmax
    eos_id: int,
    pad_id: int,
    n_max: int,
) -> SeqMazeScoreReport:
    """Compute all six seqmaze v1 evaluation metrics.

    All metrics are computed on the canonicalized prediction (truncated
    after first EOS, replaced with PAD).

    Args:
        logits: (B, T, V) raw path logits.
        targets: Path supervision targets.
        successor_indices: (B, N, K) successor candidate indices.
        successor_mask: (B, N, K) valid successor slots.
        goal_candidate_index: (B, N) one-hot or index of the goal token.
        eos_id: EOS token id (= N_max).
        pad_id: PAD token id (= N_max + 1).
        n_max: Maximum candidate nodes.

    Returns:
        SeqMazeScoreReport with all six metrics as scalar float32 tensors.
    """
    B, T, V = logits.shape
    device = logits.device

    # Argmax prediction
    pred = logits.argmax(dim=-1)  # (B, T)

    # Canonicalize
    canonical_pred = _canonicalize(pred, eos_id, pad_id)  # (B, T)

    # Target values
    target = targets.path_index  # (B, T)
    target_mask = targets.path_mask  # (B, T) — True for supervised positions
    target_length = targets.path_length  # (B,)

    # --- sequence_exact ---
    # Canonicalize target too (it should already be canonical, but for safety)
    canonical_target = _canonicalize(target, eos_id, pad_id)
    seq_exact = (
        (canonical_pred == canonical_target).all(dim=-1).to(torch.float32)
    )  # (B,)

    # --- path_position_accuracy ---
    # Count correct positions over supervised (masked) positions
    correct_pos = (canonical_pred == canonical_target) & target_mask  # (B, T)
    supervised_count = target_mask.sum(dim=-1).clamp(min=1).to(torch.float32)
    pos_acc = (
        correct_pos.to(torch.float32).sum(dim=-1) / supervised_count
    )  # (B,)

    # --- valid_transition_rate ---
    # Build adjacency matrix
    adj = _build_adjacency(
        successor_indices, successor_mask, n_max
    )  # (B, N, N)

    # Extract prefix before first EOS from canonical prediction
    pred_eos_pos = _extract_eos_position(canonical_pred, eos_id)  # (B,)
    T_actual = torch.where(pred_eos_pos < T, pred_eos_pos, T)  # cap at T

    # For each sample, count valid transitions in the prefix
    # TODO(perf): vectorize the per-sample loops below.  Acceptable for v1
    # research baseline but will bottleneck eval at large batch sizes.
    valid_trans_rates = torch.zeros(B, device=device)
    for b in range(B):
        t_len = int(T_actual[b].item())
        if t_len < 2:
            # 0 or 1 token — no transitions to evaluate
            valid_trans_rates[b] = 1.0
        else:
            prefix = canonical_pred[b, :t_len]  # (t_len,)
            # Filter to candidate indices (ignore EOS in middle counting)
            is_candidate = prefix < n_max
            cand_tokens = prefix[is_candidate]
            if cand_tokens.shape[0] < 2:
                valid_trans_rates[b] = 1.0
            else:
                valid_count = 0
                total_trans = cand_tokens.shape[0] - 1
                for t_idx in range(total_trans):
                    src = int(cand_tokens[t_idx].item())
                    dst = int(cand_tokens[t_idx + 1].item())
                    if src < n_max and dst < n_max and adj[b, src, dst]:
                        valid_count += 1
                valid_trans_rates[b] = (
                    valid_count / total_trans if total_trans > 0 else 1.0
                )

    # --- reaches_goal ---
    # Goal token is the one with goal_flag. We get the candidate index from goal_candidate_index.
    goal_idx = (
        goal_candidate_index.argmax(dim=-1)
        if goal_candidate_index.ndim == 2
        else goal_candidate_index
    )  # (B,)
    goal_reached = torch.zeros(B, device=device)
    for b in range(B):
        t_len = int(T_actual[b].item())
        prefix = canonical_pred[b, :t_len]
        goal_reached[b] = 1.0 if (prefix == goal_idx[b]).any() else 0.0

    # --- path_length_regret ---
    # Target path length is the number of candidate tokens before EOS
    target_eos_pos = _extract_eos_position(canonical_target, eos_id)
    # Predicted EOS position (before canonicalization already)
    pred_prefix_len = pred_eos_pos.to(torch.float32)
    target_prefix_len = target_eos_pos.to(torch.float32)
    # Regret = max(0, pred_len - target_len)
    regret = (pred_prefix_len - target_prefix_len).clamp(min=0.0)

    # --- eos_accuracy ---
    # Is the first predicted EOS at the same position as the first target EOS?
    target_eos = _extract_eos_position(canonical_target, eos_id)
    pred_eos = _extract_eos_position(canonical_pred, eos_id)
    eos_acc = (pred_eos == target_eos).to(torch.float32)

    # Aggregate
    n = float(B)
    return SeqMazeScoreReport(
        sequence_exact=seq_exact.mean(),
        path_position_accuracy=pos_acc.mean(),
        valid_transition_rate=valid_trans_rates.mean(),
        reaches_goal=goal_reached.mean(),
        path_length_regret=regret.mean(),
        eos_accuracy=eos_acc.mean(),
    )


# =============================================================================
__all__ = [
    "SeqMazeProbeScoreReport",
    "compute_edge_score_report",
    "SeqMazeScoreReport",
    "compute_seqmaze_score_report",
    "SEQMAZE_V1_METRIC_SPECS",
    "SEQMAZE_V1_SCORING_SPEC",
    "SeqMazeStepScore",
    "build_seqmaze_step_score",
]
