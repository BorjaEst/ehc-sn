"""MazeHard protocol owner for model-comparison benchmark bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ehc_sn.adapters.hrm import (
    MazeHardHRMAdapterSettings,
    MazeHardHRMV1ACTTaskBinding,
    MazeHardHRMV1BridgeAdapter,
    MazeHardHRMV2BridgeAdapter,
)
from ehc_sn.benchmarks.contracts import (
    ArenaScoreReport,
    ArtifactManifest,
    MazeHardScoreReport,
    ModelComparisonBinding,
    ModelComparisonExecution,
    ModelComparisonExecutionBundle,
    ModelComparisonExecutionResources,
    TrackRecipe,
    validate_model_comparison_pair,
)
from ehc_sn.controllers.deliberation.act import (
    ACTController,
    ACTControllerConfig,
)
from ehc_sn.objectives.act import ACTObjective, ACTObjectiveConfig
from ehc_sn.rollouts.runtime import RecurrentRunner
from ehc_sn.tasks.mazehard.providers import MazeHardReplayProvider

from ._shared import (
    binding_config,
    bridge_only_parameter_groups,
    load_frozen_model,
    normalize_model_family,
)

_MAZEHARD_MODEL_FAMILIES: frozenset[str] = frozenset({"hrm-v1", "hrm-v2"})


# =============================================================================
@dataclass(frozen=True)
class MazeHardCaseAggregate:
    """Denominator-aware per-case aggregate for MazeHard benchmark scoring."""

    token_correct_sum: torch.Tensor
    token_count_sum: torch.Tensor
    sequence_accuracy_sum: torch.Tensor
    sequence_exact_sum: torch.Tensor
    sequence_count_sum: torch.Tensor


# =============================================================================
@dataclass(frozen=True)
class _MazeHardACTControlOutput:
    """ACT control payload translated from HRM v2 policy outputs."""

    q_logits: torch.Tensor


# =============================================================================
@dataclass(frozen=True)
class _MazeHardACTBackboneOutput:
    """ACT-compatible backbone payload translated from HRM bridge outputs."""

    task: object
    control: _MazeHardACTControlOutput


# =============================================================================
class _MazeHardHRMV2ACTBridgeAdapter:
    """Translate HRM v2 bridge outputs into ACT backbone output protocol."""

    def __init__(self, bridge: MazeHardHRMV2BridgeAdapter) -> None:
        self._bridge = bridge

    def __getattr__(self, name: str) -> object:
        return getattr(self._bridge, name)

    def forward(  # -----------------------------------------------------------
        self,
        batch: object,
        state: object | None = None,
    ) -> tuple[_MazeHardACTBackboneOutput, object]:
        output, next_state = self._bridge(batch, state)
        translated = _MazeHardACTBackboneOutput(
            task=output.task,
            control=_MazeHardACTControlOutput(q_logits=output.policy.q_values),
        )
        return translated, next_state

    __call__ = forward


# =============================================================================
class MazeHardDelibHRMV1ModelComparisonBinding(ModelComparisonBinding):
    """MazeHard-Delib benchmark binding for the HRM model-comparison slice."""

    def bind_model_comparison(
        self,
        manifest: ArtifactManifest,
        recipe: TrackRecipe,
    ) -> ModelComparisonExecutionBundle:
        validate_model_comparison_pair(manifest, recipe)
        if recipe.track_id != "mazehard-delib":
            raise ValueError(
                "MazeHardDelibHRMV1ModelComparisonBinding only supports track "
                f"'mazehard-delib', got {recipe.track_id!r}."
            )
        if recipe.capability_kind != "mazehard_deliberation":
            raise ValueError(
                "MazeHardDelibHRMV1ModelComparisonBinding requires "
                "capability_kind='mazehard_deliberation'."
            )

        model_family = normalize_model_family(manifest.model_family)
        if model_family not in _MAZEHARD_MODEL_FAMILIES:
            raise ValueError(
                "MazeHardDelib HRM binding only supports model families "
                f"{sorted(_MAZEHARD_MODEL_FAMILIES)!r}, got {model_family!r}."
            )

        family_binding = binding_config(recipe, model_family)
        fixed_budget_steps = recipe.execution.fixed_budget_steps
        halt_action = recipe.execution.halt_action
        allow_halt = recipe.execution.allow_halt
        explore = recipe.execution.explore

        if fixed_budget_steps is None:
            raise ValueError("recipe.execution.fixed_budget_steps is required.")
        if halt_action is None:
            raise ValueError("recipe.execution.halt_action is required.")
        if allow_halt is not False:
            raise ValueError(
                "MazeHard-Delib deterministic benchmark requires allow_halt=False."
            )
        if explore is not False:
            raise ValueError(
                "MazeHard-Delib deterministic benchmark requires explore=False."
            )

        model, _, _ = load_frozen_model(
            model_family,
            manifest,
        )

        def create_execution() -> ModelComparisonExecution:
            bridge = _build_mazehard_bridge(
                model_family,
                model,
                family_binding,
            )
            controller = _build_mazehard_controller(
                model_family,
                bridge,
                fixed_budget_steps=fixed_budget_steps,
                halt_action=halt_action,
            )
            parameter_groups = bridge_only_parameter_groups(model, bridge)
            if not parameter_groups:
                raise ValueError(
                    "MazeHard-Delib HRM binding requires a non-empty "
                    "bridge-only trainable parameter set."
                )
            return ModelComparisonExecution(
                bridge=bridge,
                controller=controller,
                objective=ACTObjective(
                    ACTObjectiveConfig(
                        token_loss=recipe.adaptation.objective_token_loss
                    ),
                    task_binding=MazeHardHRMV1ACTTaskBinding(),
                ),
                runner=RecurrentRunner(),
                bridge_parameter_groups=parameter_groups,
            )

        return ModelComparisonExecutionBundle(
            model=model,
            create_execution=create_execution,
            resources=ModelComparisonExecutionResources(
                replay_provider_factory=MazeHardReplayProvider,
                controller_step_options={
                    "allow_halt": allow_halt,
                    "explore": explore,
                },
            ),
            score_aggregator=_aggregate_mazehard_score_reports,
        )


# =============================================================================
def _build_mazehard_bridge(
    model_family: str,
    model: object,
    family_binding: dict[str, Any],
) -> object:
    adapter_cfg = dict(family_binding.get("adapter", {}))

    if model_family == "hrm-v1":
        settings = MazeHardHRMAdapterSettings.model_validate(adapter_cfg)
        return MazeHardHRMV1BridgeAdapter(model, settings)
    if model_family == "hrm-v2":
        settings = MazeHardHRMAdapterSettings.model_validate(adapter_cfg)
        bridge = MazeHardHRMV2BridgeAdapter(model, settings)
        return _MazeHardHRMV2ACTBridgeAdapter(bridge)
    raise ValueError(f"Unsupported MazeHard model family: {model_family!r}.")


# =============================================================================
def _build_mazehard_controller(
    model_family: str,
    bridge: object,
    *,
    fixed_budget_steps: int,
    halt_action: int,
) -> object:
    if model_family == "hrm-v1":
        return ACTController(
            bridge,
            ACTControllerConfig(
                max_halt_steps=fixed_budget_steps,
                exploration_prob=0.0,
                done_action=halt_action,
            ),
        )
    if model_family == "hrm-v2":
        return ACTController(
            bridge,
            ACTControllerConfig(
                max_halt_steps=fixed_budget_steps,
                exploration_prob=0.0,
                done_action=halt_action,
            ),
        )
    raise ValueError(f"Unsupported MazeHard model family: {model_family!r}.")


# =============================================================================
def _aggregate_mazehard_score_reports(
    score_reports: tuple[
        ArenaScoreReport | MazeHardScoreReport | MazeHardCaseAggregate,
        ...,
    ],
) -> MazeHardScoreReport:
    if not score_reports:
        raise ValueError(
            "MazeHard score aggregation requires at least one report."
        )
    if not all(
        isinstance(report, (MazeHardScoreReport, MazeHardCaseAggregate))
        for report in score_reports
    ):
        raise TypeError(
            "MazeHard score aggregation received a non-MazeHard score."
        )

    reports = tuple(score_reports)

    if all(isinstance(report, MazeHardCaseAggregate) for report in reports):
        token_correct_sum = torch.stack(
            [report.token_correct_sum for report in reports]
        ).sum()
        token_count_sum = (
            torch.stack([report.token_count_sum for report in reports])
            .sum()
            .clamp_min(1.0)
        )
        sequence_accuracy_sum = torch.stack(
            [report.sequence_accuracy_sum for report in reports]
        ).sum()
        sequence_exact_sum = torch.stack(
            [report.sequence_exact_sum for report in reports]
        ).sum()
        sequence_count_sum = (
            torch.stack([report.sequence_count_sum for report in reports])
            .sum()
            .clamp_min(1.0)
        )
        return MazeHardScoreReport(
            tokens_accuracy=token_correct_sum / token_count_sum,
            sequences_accuracy=sequence_accuracy_sum / sequence_count_sum,
            sequences_exact=sequence_exact_sum / sequence_count_sum,
        )

    # Aggregate typed MazeHardScoreReport values.
    typed_reports = tuple(
        report for report in reports if isinstance(report, MazeHardScoreReport)
    )
    tokens = torch.stack(
        [report.tokens_accuracy for report in typed_reports]
    ).mean()
    seq_acc = torch.stack(
        [report.sequences_accuracy for report in typed_reports]
    ).mean()
    seq_exact = torch.stack(
        [report.sequences_exact for report in typed_reports]
    ).mean()
    return MazeHardScoreReport(
        tokens_accuracy=tokens,
        sequences_accuracy=seq_acc,
        sequences_exact=seq_exact,
    )


# =============================================================================
__all__ = [
    "MazeHardCaseAggregate",
    "MazeHardDelibHRMV1ModelComparisonBinding",
]
