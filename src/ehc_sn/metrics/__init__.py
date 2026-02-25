from torchmetrics import MetricCollection

from ehc_sn.metrics.adapter import update_metrics_from_step
from ehc_sn.metrics.torchmetrics import LastRatioMetric, RatioMetric
from ehc_sn.metrics.types import HaltedAgg, LossAgg, StepMetrics, TokenAgg

__all__ = [
    "RatioMetric",
    "LastRatioMetric",
    "HaltedAgg",
    "LossAgg",
    "StepMetrics",
    "TokenAgg",
    "update_metrics_from_step",
    "build_metrics",
]


def build_metrics() -> MetricCollection:
    """Construct a MetricCollection with all HRM metrics."""
    return MetricCollection(
        {
            "all/accuracy": LastRatioMetric(),
            "halted/rate": LastRatioMetric(),
            "halted/avg_steps": LastRatioMetric(),
            "tokens/accuracy": LastRatioMetric(),
            "loss/lm": LastRatioMetric(),
            "loss/q_halt": LastRatioMetric(),
            "loss/q_continue": LastRatioMetric(),
        },
        prefix=None,
        postfix=None,
        compute_groups=[
            [
                "all/accuracy",
                "halted/rate",
                "halted/avg_steps",
                "tokens/accuracy",
                "loss/lm",
                "loss/q_halt",
                "loss/q_continue",
            ]
        ],
    )
