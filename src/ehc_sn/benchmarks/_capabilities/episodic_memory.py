"""Capability contracts for TEM-correct episodic-memory benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from torch import Tensor

CueFamily = Literal["observation", "landmark"]


@dataclass(frozen=True)
class EpisodicMemoryStep:
    """One benchmark-owned replay step consumed by an episodic-memory agent."""

    observation: Tensor
    previous_action: Tensor
    episode_start: Tensor
    observation_id: Tensor | None = None
    location_id: Tensor | None = None
    landmark_id: Tensor | None = None


@dataclass(frozen=True)
class EpisodicMemoryQuery:
    """One benchmark-owned readout request issued during M0 evaluation."""

    kind: Literal["current_location", "cue_location"]
    cue_family: CueFamily | None = None
    cue_id: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "current_location":
            if self.cue_family is not None or self.cue_id is not None:
                raise ValueError("current_location queries must not provide cue metadata.")
            return
        if self.cue_family is None or self.cue_id is None:
            raise ValueError("cue_location queries must provide both cue_family and cue_id.")


@dataclass(frozen=True)
class EpisodicMemoryReadout:
    """One benchmark-visible episodic-memory readout."""

    location_id: int | None = None
    diagnostics: dict[str, Any] | None = None


class EpisodicMemoryAgent(Protocol):
    """Single-slot benchmark adapter surface used by the M0 bridge benchmark."""

    def reset_state(self) -> Any:
        """Return a fresh single-slot benchmark state."""

    def ingest_step(self, step: EpisodicMemoryStep, state: Any) -> Any:
        """Return the updated state after consuming one replay step."""

    def readout(self, query: EpisodicMemoryQuery, state: Any) -> EpisodicMemoryReadout:
        """Return one readout for the provided query and current state."""


__all__ = [
    "CueFamily",
    "EpisodicMemoryAgent",
    "EpisodicMemoryQuery",
    "EpisodicMemoryReadout",
    "EpisodicMemoryStep",
]
