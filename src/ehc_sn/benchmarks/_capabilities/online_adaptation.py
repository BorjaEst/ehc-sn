"""Capability contracts for online-adaptation benchmarks."""

from __future__ import annotations

from typing import Any, Protocol

from ehc_sn.benchmarks._capabilities.rollout import RolloutAgent


class IngestTransition(Protocol):
    """Consume a transition during benchmark-time adaptation."""

    def ingest_transition(self, transition: Any, state: Any) -> Any:
        """Return the updated state after consuming one transition."""


class OnlineAdaptationAgent(RolloutAgent, IngestTransition, Protocol):
    """Composite capability for exposure/probe style benchmarks."""


__all__ = ["IngestTransition", "OnlineAdaptationAgent"]
