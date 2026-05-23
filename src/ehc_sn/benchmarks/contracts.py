"""Benchmark-owned contracts for model-comparison mode.

This module is the canonical benchmark boundary for:

- TOML contract parsing (artifact manifests and track recipes)
- capability contracts for ready tracks
- model-comparison execution binding surfaces

It intentionally does not own evaluator runtime logic and does not modify the
report layer in :mod:`ehc_sn.benchmarks.runner`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, Field, RootModel, model_validator

from ehc_sn.tasks.arena.evaluation import ArenaScoreReport
from ehc_sn.tasks.arena.runtime import (
    ARENA_REPLAY_OPTIONAL_KEYS,
    ARENA_REPLAY_REQUIRED_KEYS,
)
from ehc_sn.tasks.mazehard.evaluation import MazeHardScoreReport
from ehc_sn.tasks.mazehard.runtime import MAZE_HARD_BATCH_KEYS

ModelComparisonMode: TypeAlias = Literal["model-comparison"]
CapabilityKind: TypeAlias = Literal[
    "arena_replay",
    "mazehard_deliberation",
]
ScoreReport: TypeAlias = ArenaScoreReport | MazeHardScoreReport


# =============================================================================
class ArtifactLoadPolicy(RootModel[dict[str, Any]]):
    """Typed load-policy subtable for benchmark artifact manifests."""


# =============================================================================
class ArtifactBenchmarkPolicy(RootModel[dict[str, Any]]):
    """Typed benchmark-policy subtable for benchmark artifact manifests."""


# =============================================================================
class ArtifactProvenance(RootModel[dict[str, Any]]):
    """Typed provenance subtable for benchmark artifact manifests."""


# =============================================================================
class ArtifactManifest(BaseModel, extra="forbid"):
    """Typed schema for benchmark artifact manifests."""

    schema_version: str
    artifact_id: str
    artifact_type: str
    model_family: str
    model_config_path: str
    checkpoint_path: str
    checkpoint_format: str
    supported_tracks: tuple[str, ...]

    load_policy: ArtifactLoadPolicy
    benchmark_policy: ArtifactBenchmarkPolicy
    provenance: ArtifactProvenance


# =============================================================================
class TrackArtifactRules(RootModel[dict[str, Any]]):
    """Typed artifact-rules subtable for benchmark track recipes."""


# =============================================================================
class TrackBridge(RootModel[dict[str, Any]]):
    """Typed bridge subtable for benchmark track recipes."""


# =============================================================================
class TrackExecution(RootModel[dict[str, Any]]):
    """Typed execution subtable for benchmark track recipes."""


# =============================================================================
class TrackData(RootModel[dict[str, Any]]):
    """Typed data subtable for benchmark track recipes."""


# =============================================================================
class TrackAdaptation(RootModel[dict[str, Any]]):
    """Typed adaptation subtable for benchmark track recipes."""


# =============================================================================
class TrackReporting(RootModel[dict[str, Any]]):
    """Typed reporting subtable for benchmark track recipes."""


# =============================================================================
class TrackModelFamilyBinding(RootModel[dict[str, Any]]):
    """Typed per-model-family binding subtable for track recipes."""


# =============================================================================
class TrackRecipe(BaseModel, extra="forbid"):
    """Typed schema for benchmark track recipes in model-comparison mode only."""

    schema_version: str
    recipe_id: str
    track_id: str
    mode: ModelComparisonMode
    capability_kind: str
    compared_artifact_type: str
    fixed_recipe_label: str
    adaptation_protocol_id: str
    primary_metric: str
    supported_model_families: tuple[str, ...]

    artifact_rules: TrackArtifactRules
    bridge: TrackBridge
    execution: TrackExecution
    data: TrackData
    adaptation: TrackAdaptation
    reporting: TrackReporting
    bindings: dict[str, TrackModelFamilyBinding] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_mode_and_bindings(self) -> "TrackRecipe":
        if self.mode != "model-comparison":
            raise ValueError("TrackRecipe.mode must be 'model-comparison'.")
        missing = [
            family
            for family in self.supported_model_families
            if family not in self.bindings
        ]
        if missing:
            raise ValueError(
                "TrackRecipe.bindings is missing model-family entries for: "
                + ", ".join(sorted(missing))
                + "."
            )
        return self


# =============================================================================
def parse_artifact_manifest(path: str | Path) -> ArtifactManifest:
    """Parse and validate one benchmark artifact-manifest TOML file."""
    with Path(path).open("rb") as f:
        payload = tomllib.load(f)
    return ArtifactManifest.model_validate(payload)


# =============================================================================
def parse_track_recipe(path: str | Path) -> TrackRecipe:
    """Parse and validate one benchmark track-recipe TOML file."""
    with Path(path).open("rb") as f:
        payload = tomllib.load(f)
    return TrackRecipe.model_validate(payload)


# =============================================================================
@dataclass(frozen=True)
class ArenaReplayCapabilityContract:
    """Benchmark-facing capability contract for Arena replay."""

    track_id: Literal["arena-struct"] = "arena-struct"
    capability_kind: Literal["arena_replay"] = "arena_replay"
    required_batch_keys: tuple[str, ...] = ARENA_REPLAY_REQUIRED_KEYS
    optional_batch_keys: tuple[str, ...] = ARENA_REPLAY_OPTIONAL_KEYS
    score_report_type: type[ArenaScoreReport] = ArenaScoreReport


# =============================================================================
@dataclass(frozen=True)
class MazeHardDeliberationCapabilityContract:
    """Benchmark-facing capability contract for MazeHard deliberation."""

    track_id: Literal["mazehard-delib"] = "mazehard-delib"
    capability_kind: Literal["mazehard_deliberation"] = "mazehard_deliberation"
    required_batch_keys: tuple[str, ...] = MAZE_HARD_BATCH_KEYS
    optional_batch_keys: tuple[str, ...] = ()
    score_report_type: type[MazeHardScoreReport] = MazeHardScoreReport


READY_CAPABILITY_CONTRACTS: dict[
    CapabilityKind,
    ArenaReplayCapabilityContract | MazeHardDeliberationCapabilityContract,
] = {
    "arena_replay": ArenaReplayCapabilityContract(),
    "mazehard_deliberation": MazeHardDeliberationCapabilityContract(),
}


# =============================================================================
class ScoreAggregator(Protocol):
    """Benchmark-owned score-aggregation surface for model-comparison mode."""

    def __call__(self, score_reports: tuple[ScoreReport, ...]) -> object: ...


# =============================================================================
@dataclass(frozen=True)
class ModelComparisonExecution:
    """Per-execution benchmark resources for model-comparison mode."""

    bridge: object
    controller: object
    bridge_parameter_groups: tuple[dict[str, Any], ...]


# =============================================================================
class ModelComparisonExecutionFactory(Protocol):
    """Factory that materializes fresh execution resources per benchmark run."""

    def __call__(self) -> ModelComparisonExecution: ...


# =============================================================================
@dataclass(frozen=True)
class ModelComparisonExecutionBundle:
    """Benchmark-owned binding bundle for model-comparison execution."""

    model: object
    create_execution: ModelComparisonExecutionFactory
    loaders: dict[str, object]
    score_aggregator: ScoreAggregator


# =============================================================================
class ModelComparisonBinding(Protocol):
    """Benchmark-owned model-comparison binding protocol."""

    def bind_model_comparison(
        self,
        manifest: ArtifactManifest,
        recipe: TrackRecipe,
    ) -> ModelComparisonExecutionBundle: ...


# =============================================================================
def validate_model_comparison_pair(
    manifest: ArtifactManifest,
    recipe: TrackRecipe,
) -> None:
    """Validate artifact/recipe compatibility at the benchmark contract boundary."""
    if recipe.mode != "model-comparison":
        raise ValueError(
            "TrackRecipe.mode must be 'model-comparison' for benchmark binding."
        )
    if recipe.track_id not in manifest.supported_tracks:
        raise ValueError(
            "ArtifactManifest.supported_tracks does not include recipe.track_id: "
            f"{recipe.track_id!r}."
        )
    if manifest.model_family not in recipe.supported_model_families:
        raise ValueError(
            "TrackRecipe.supported_model_families does not include manifest.model_family: "
            f"{manifest.model_family!r}."
        )
    if recipe.capability_kind not in READY_CAPABILITY_CONTRACTS:
        raise ValueError(
            "Unsupported capability_kind for model-comparison mode: "
            f"{recipe.capability_kind!r}."
        )


# =============================================================================
__all__ = [
    "ArenaReplayCapabilityContract",
    "ArtifactBenchmarkPolicy",
    "ArtifactLoadPolicy",
    "ArtifactManifest",
    "ArtifactProvenance",
    "CapabilityKind",
    "MazeHardDeliberationCapabilityContract",
    "ModelComparisonBinding",
    "ModelComparisonExecution",
    "ModelComparisonExecutionBundle",
    "ModelComparisonExecutionFactory",
    "ModelComparisonMode",
    "READY_CAPABILITY_CONTRACTS",
    "ScoreAggregator",
    "ScoreReport",
    "TrackAdaptation",
    "TrackArtifactRules",
    "TrackBridge",
    "TrackData",
    "TrackExecution",
    "TrackModelFamilyBinding",
    "TrackRecipe",
    "TrackReporting",
    "parse_artifact_manifest",
    "parse_track_recipe",
    "validate_model_comparison_pair",
]
