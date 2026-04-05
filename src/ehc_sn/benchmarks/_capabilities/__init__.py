"""Internal benchmark capability contracts."""

from ehc_sn.benchmarks._capabilities.batch_prediction import BatchPrediction, BatchPredicts
from ehc_sn.benchmarks._capabilities.online_adaptation import IngestTransition, OnlineAdaptationAgent
from ehc_sn.benchmarks._capabilities.rollout import Acts, ResetState, RolloutAgent

__all__ = [
    "Acts",
    "BatchPrediction",
    "BatchPredicts",
    "IngestTransition",
    "OnlineAdaptationAgent",
    "ResetState",
    "RolloutAgent",
]
