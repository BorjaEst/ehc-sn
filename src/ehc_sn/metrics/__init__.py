from typing import Literal, Optional, Sequence

from torchmetrics import MetricCollection

from ehc_sn.metrics.adapter import update_metrics_from_step
from ehc_sn.metrics.torchmetrics import LastRatioMetric, RatioMetric
from ehc_sn.metrics.types import HaltedAgg, LossAgg, StepMetrics, TokenAgg

KeyGroups = Literal["act", "rl"]


# =================================================================================================
_COMMON_KEYS = ("all/accuracy", "halted/rate", "halted/avg_steps", "tokens/accuracy", "loss/lm")
_ACT_KEYS = ("loss/q_halt", "loss/q_continue")
_RL_KEYS = ("loss/actor", "loss/critic", "loss/entropy", "loss/q_value")


# =================================================================================================
def build_train_metrics(  # -----------------------------------------------------------------------
    groups: Optional[KeyGroups | Sequence[KeyGroups]] = None,
) -> MetricCollection:  # fmt: skip
    """Build a MetricCollection for training metrics (non-accumulated, direct step ratios)."""
    keys = _groups_to_keys(groups)
    return MetricCollection({k: LastRatioMetric() for k in keys}, compute_groups=[list(keys)])


# =================================================================================================
def build_val_metrics(  # -------------------------------------------------------------------------
    groups: Optional[KeyGroups | Sequence[KeyGroups]] = None,
) -> MetricCollection:  # fmt: skip
    """Build a MetricCollection for validation metrics (accumulated over the epoch)."""
    keys = _groups_to_keys(groups)
    return MetricCollection({k: RatioMetric() for k in keys}, compute_groups=[list(keys)])


# =================================================================================================
def _groups_to_keys(  # ---------------------------------------------------------------------------
    groups: Optional[KeyGroups | Sequence[KeyGroups]],
) -> set[str]:  # fmt: skip
    """Convert a set of key groups to the corresponding set of metric keys."""
    keys = set(_COMMON_KEYS)
    keys = keys | _ACT_KEYS if "act" in groups else keys
    keys = keys | _RL_KEYS if "rl" in groups else keys
    return keys


# =================================================================================================
__all__ = [
    "RatioMetric", "LastRatioMetric", "HaltedAgg", "LossAgg", "StepMetrics", "TokenAgg",
    "update_metrics_from_step", "build_metrics",
]  # fmt: skip
