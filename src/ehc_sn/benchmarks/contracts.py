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

from pydantic import BaseModel, Field, model_validator

from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.rollouts.runtime import Runner
from ehc_sn.rollouts.scoring import RolloutScorer
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
    "mazehard_step_evaluator",
]
ScoreReport: TypeAlias = ArenaScoreReport | MazeHardScoreReport


# =============================================================================
class ArtifactLoadPolicy(BaseModel, extra="forbid"):
    """Load-policy subtable for benchmark artifact manifests."""

    freeze_core: bool
    allow_full_resume: bool


# =============================================================================
class ArtifactBenchmarkPolicy(BaseModel, extra="forbid"):
    """Benchmark-policy subtable for benchmark artifact manifests."""

    mode: ModelComparisonMode


# =============================================================================
class ArtifactProvenance(BaseModel, extra="forbid"):
    """Provenance subtable for benchmark artifact manifests."""

    source_run: str


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

    @model_validator(mode="after")
    def _validate_model_comparison_mode(self) -> "ArtifactManifest":
        if self.artifact_type != "model_core":
            raise ValueError(
                "ArtifactManifest.artifact_type must be 'model_core'."
            )
        if self.benchmark_policy.mode != "model-comparison":
            raise ValueError(
                "ArtifactManifest.benchmark_policy.mode must be "
                "'model-comparison'."
            )
        if not self.load_policy.freeze_core:
            raise ValueError(
                "ArtifactManifest.load_policy.freeze_core must be True for "
                "model-comparison benchmarks."
            )
        if self.load_policy.allow_full_resume:
            raise ValueError(
                "ArtifactManifest.load_policy.allow_full_resume must be False "
                "for model-comparison benchmarks."
            )
        return self


# =============================================================================
class TrackArtifactRules(BaseModel, extra="forbid"):
    """Artifact-rules subtable for benchmark track recipes."""

    require_frozen_core: bool
    require_fresh_bridge: bool


# =============================================================================
class TrackBridge(BaseModel, extra="forbid"):
    """Bridge subtable for benchmark track recipes."""

    adapter: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
class TrackExecution(BaseModel, extra="forbid"):
    """Execution subtable for benchmark track recipes."""

    fixed_budget_steps: int | None = Field(default=None, ge=1)
    halt_action: int | None = Field(default=None, ge=0)
    allow_halt: bool | None = None
    explore: bool | None = None
    seed: int | None = None
    reseed_before_evaluation: bool | None = None
    max_batches: int = Field(default=0, ge=0)


# =============================================================================
class TrackData(BaseModel, extra="forbid"):
    """Data subtable for benchmark track recipes."""

    dataset_path: str
    split: str
    batch_size: int = Field(ge=1)
    n_cases: int = Field(ge=0)
    sample_ids: tuple[str, ...] = ()


# =============================================================================
class BridgeAdaptationProtocol(BaseModel, extra="forbid"):
    """Explicit benchmark-time bridge adaptation protocol contract."""

    examples: int = Field(ge=1)
    steps: int = Field(ge=1)
    optimizer: str
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    betas: tuple[float, float]
    seed: int
    reseed_before_adaptation: bool
    stopping_rule: str


# =============================================================================
class TrackAdaptation(BaseModel, extra="forbid"):
    """Adaptation subtable for benchmark track recipes."""

    objective_token_loss: LossType
    protocol: BridgeAdaptationProtocol


# =============================================================================
class TrackReporting(BaseModel, extra="forbid"):
    """Reporting subtable for benchmark track recipes."""

    fixed_recipe: str


# =============================================================================
class TrackModelFamilyBinding(BaseModel, extra="forbid"):
    """Per-model-family binding subtable for track recipes."""

    adapter: dict[str, Any]


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
    adaptation_data: TrackData
    evaluation_data: TrackData
    adaptation: TrackAdaptation
    reporting: TrackReporting
    bindings: dict[str, TrackModelFamilyBinding] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_shared_data_selector(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data
        if "data" not in data:
            return data
        if "adaptation_data" in data or "evaluation_data" in data:
            return data
        migrated = dict(data)
        shared_data = migrated.pop("data")
        migrated["adaptation_data"] = shared_data
        migrated["evaluation_data"] = shared_data
        return migrated

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

        if self.track_id == "mazehard-delib":
            if self.capability_kind != "mazehard_step_evaluator":
                raise ValueError(
                    "mazehard-delib requires capability_kind='mazehard_step_evaluator'."
                )
            if not self.artifact_rules.require_frozen_core:
                raise ValueError(
                    "mazehard-delib requires artifact_rules.require_frozen_core=True."
                )
            if not self.artifact_rules.require_fresh_bridge:
                raise ValueError(
                    "mazehard-delib requires artifact_rules.require_fresh_bridge=True."
                )
            if self.execution.fixed_budget_steps is None:
                raise ValueError(
                    "mazehard-delib requires execution.fixed_budget_steps."
                )
            if self.execution.halt_action is None:
                raise ValueError(
                    "mazehard-delib requires execution.halt_action."
                )
            if self.execution.seed is None:
                raise ValueError("mazehard-delib requires execution.seed.")
            if self.execution.allow_halt is not False:
                raise ValueError(
                    "mazehard-delib requires execution.allow_halt=False."
                )
            if self.execution.explore is not False:
                raise ValueError(
                    "mazehard-delib requires execution.explore=False."
                )
            if self.execution.reseed_before_evaluation is not True:
                raise ValueError(
                    "mazehard-delib requires "
                    "execution.reseed_before_evaluation=True."
                )
            if self.adaptation.protocol.reseed_before_adaptation is not True:
                raise ValueError(
                    "mazehard-delib requires "
                    "adaptation.protocol.reseed_before_adaptation=True."
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
class MazeHardStepEvaluatorContract:
    """Benchmark-facing step-evaluator contract for MazeHard."""

    track_id: Literal["mazehard-delib"] = "mazehard-delib"
    capability_kind: Literal["mazehard_step_evaluator"] = (
        "mazehard_step_evaluator"
    )
    required_batch_keys: tuple[str, ...] = MAZE_HARD_BATCH_KEYS
    optional_batch_keys: tuple[str, ...] = ()
    score_report_type: type[MazeHardScoreReport] = MazeHardScoreReport


READY_CAPABILITY_CONTRACTS: dict[
    CapabilityKind,
    ArenaReplayCapabilityContract | MazeHardStepEvaluatorContract,
] = {
    "arena_replay": ArenaReplayCapabilityContract(),
    "mazehard_step_evaluator": MazeHardStepEvaluatorContract(),
}


# =============================================================================
class ScoreAggregator(Protocol):
    """Benchmark-owned score-aggregation surface for model-comparison mode."""

    def __call__(self, score_reports: tuple[ScoreReport, ...]) -> object: ...


# =============================================================================
class ReplayProviderFactory(Protocol):
    """Factory protocol for task-owned replay evaluation providers."""

    def __call__(
        self,
        *,
        dataset_path: str,
        split: str,
        batch_size: int,
        n_cases: int,
    ) -> object: ...


# =============================================================================
@dataclass(frozen=True)
class ModelComparisonExecutionResources:
    """Typed runtime resources shared by benchmark model-comparison bindings."""

    replay_provider_factory: ReplayProviderFactory
    controller_step_options: dict[str, object]


# =============================================================================
@dataclass(frozen=True)
class ModelComparisonExecution:
    """Per-execution benchmark resources for model-comparison mode."""

    bridge: object
    controller: object
    objective: RolloutScorer
    runner: Runner
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
    resources: ModelComparisonExecutionResources
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
    if not manifest.load_policy.freeze_core:
        raise ValueError(
            "ArtifactManifest.load_policy.freeze_core must be True for "
            "model-comparison benchmarks."
        )
    if manifest.load_policy.allow_full_resume:
        raise ValueError(
            "ArtifactManifest.load_policy.allow_full_resume must be False "
            "for model-comparison benchmarks."
        )


# =============================================================================
__all__ = [
    "ArenaReplayCapabilityContract",
    "ArtifactBenchmarkPolicy",
    "ArtifactLoadPolicy",
    "ArtifactManifest",
    "ArtifactProvenance",
    "CapabilityKind",
    "MazeHardStepEvaluatorContract",
    "ModelComparisonBinding",
    "ModelComparisonExecution",
    "ModelComparisonExecutionBundle",
    "ModelComparisonExecutionResources",
    "ModelComparisonExecutionFactory",
    "ModelComparisonMode",
    "ReplayProviderFactory",
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
