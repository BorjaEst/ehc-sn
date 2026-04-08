"""Internal benchmark role contracts."""

from ehc_sn.benchmarks._capabilities.batch_prediction import BatchPredicts
from ehc_sn.benchmarks._capabilities.episodic_memory import EpisodicMemoryAgent
from ehc_sn.benchmarks._capabilities.online_adaptation import OnlineAdaptationAgent
from ehc_sn.benchmarks._capabilities.rollout import RolloutAgent

__all__ = [
    "BatchPredicts",
    "EpisodicMemoryAgent",
    "OnlineAdaptationAgent",
    "RolloutAgent",
]
