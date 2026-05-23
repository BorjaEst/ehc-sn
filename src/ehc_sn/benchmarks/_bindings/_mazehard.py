"""MazeHard protocol owner for model-comparison benchmark bindings."""

from __future__ import annotations

from typing import Any

import torch

from ehc_sn.adapters.mazehard.ehc import (
    MazeHardEHCAdapterSettings,
    MazeHardEHCV1BridgeAdapter,
)
from ehc_sn.adapters.mazehard.hrm import (
    MazeHardHRMAdapterSettings,
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
    TrackRecipe,
    validate_model_comparison_pair,
)
from ehc_sn.controllers.deliberation.act import (
    ACTController,
    ACTControllerConfig,
)
from ehc_sn.controllers.deliberation.actor_critic import (
    DeliberationACController,
    DeliberationACControllerConfig,
)
from ehc_sn.tasks.mazehard.capabilities.deliberation import (
    MazeHardDeliberationCapability,
    MazeHardDeliberationConfig,
)
from ehc_sn.tasks.mazehard.providers import (
    MazeHardFixedProbeProvider,
    MazeHardReplayDiagnosticProvider,
)
from ehc_sn.tasks.mazehard.reward import MazeHardRewardProjector

from ._shared import (
    binding_config,
    bridge_only_parameter_groups,
    load_frozen_model,
    normalize_model_family,
)

_MAZEHARD_MODEL_FAMILIES: frozenset[str] = frozenset(
    {"hrm-v1", "hrm-v2", "ehc-v1"}
)


# =============================================================================
class SharedMazeHardModelComparisonBinding(ModelComparisonBinding):
    """Shared MazeHard benchmark binding for HRM v1/v2 and EHC v1.

    Naming drift note:
    The current capability id is still ``mazehard_deliberation``, but this
    shared benchmark binding enforces a fixed-budget, halt-suppressed protocol
    across families.
    """

    def bind_model_comparison(
        self,
        manifest: ArtifactManifest,
        recipe: TrackRecipe,
    ) -> ModelComparisonExecutionBundle:
        validate_model_comparison_pair(manifest, recipe)
        if recipe.track_id != "mazehard-delib":
            raise ValueError(
                "SharedMazeHardModelComparisonBinding only supports track "
                f"'mazehard-delib', got {recipe.track_id!r}."
            )
        if recipe.capability_kind != "mazehard_deliberation":
            raise ValueError(
                "SharedMazeHardModelComparisonBinding requires "
                "capability_kind='mazehard_deliberation'."
            )

        model_family = normalize_model_family(manifest.model_family)
        if model_family not in _MAZEHARD_MODEL_FAMILIES:
            raise ValueError(
                "MazeHard shared binding only supports model families "
                f"{sorted(_MAZEHARD_MODEL_FAMILIES)!r}, got {model_family!r}."
            )

        family_binding = binding_config(recipe, model_family)
        fixed_budget_steps = _resolve_fixed_budget_steps(recipe, family_binding)
        halt_action = _resolve_halt_action(recipe, family_binding)

        model, loader_fn, loaded_keys = load_frozen_model(
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
            return ModelComparisonExecution(
                bridge=bridge,
                controller=controller,
                bridge_parameter_groups=bridge_only_parameter_groups(
                    model,
                    bridge,
                ),
            )

        return ModelComparisonExecutionBundle(
            model=model,
            create_execution=create_execution,
            loaders={
                "init_only_weight_loader": loader_fn,
                "loaded_key_count": len(loaded_keys),
                "replay_provider_factory": MazeHardReplayDiagnosticProvider,
                "probe_provider_factory": MazeHardFixedProbeProvider,
                "controller_step_options": {
                    "allow_halt": False,
                    "explore": False,
                },
                "fixed_budget_steps": fixed_budget_steps,
                "scoring_protocol": "final-step canonical mazehard score",
            },
            score_aggregator=_aggregate_mazehard_score_reports,
        )


# =============================================================================
def _build_mazehard_bridge(
    model_family: str,
    model: object,
    family_binding: dict[str, Any],
) -> object:
    adapter_cfg = dict(family_binding.get("adapter", {}))
    if "encoder_kind" in family_binding and "encoder_kind" not in adapter_cfg:
        adapter_cfg["encoder_kind"] = family_binding["encoder_kind"]
    if "vocab_size" in family_binding and "vocab_size" not in adapter_cfg:
        adapter_cfg["vocab_size"] = family_binding["vocab_size"]

    if model_family == "hrm-v1":
        settings = MazeHardHRMAdapterSettings.model_validate(adapter_cfg)
        return MazeHardHRMV1BridgeAdapter(model, settings)
    if model_family == "hrm-v2":
        settings = MazeHardHRMAdapterSettings.model_validate(adapter_cfg)
        return MazeHardHRMV2BridgeAdapter(model, settings)
    if model_family == "ehc-v1":
        settings = MazeHardEHCAdapterSettings.model_validate(adapter_cfg)
        return MazeHardEHCV1BridgeAdapter(model, settings)
    raise ValueError(f"Unsupported MazeHard model family: {model_family!r}.")


# =============================================================================
def _resolve_fixed_budget_steps(
    recipe: TrackRecipe,
    family_binding: dict[str, Any],
) -> int:
    candidates = (
        family_binding.get("fixed_budget_steps"),
        recipe.execution.root.get("fixed_budget_steps"),
        recipe.execution.root.get("episode_horizon"),
    )
    for candidate in candidates:
        if candidate is not None:
            value = int(candidate)
            if value < 1:
                raise ValueError(
                    f"fixed_budget_steps must be >= 1, got {value}."
                )
            return value
    return 16


# =============================================================================
def _resolve_halt_action(
    recipe: TrackRecipe,
    family_binding: dict[str, Any],
) -> int:
    candidates = (
        family_binding.get("halt_action"),
        recipe.execution.root.get("halt_action"),
    )
    for candidate in candidates:
        if candidate is not None:
            value = int(candidate)
            if value < 0:
                raise ValueError(f"halt_action must be >= 0, got {value}.")
            return value
    return 0


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

    finalizer = MazeHardDeliberationCapability(
        MazeHardDeliberationConfig(
            halt_action=halt_action,
            episode_horizon=fixed_budget_steps,
        ),
        MazeHardRewardProjector(),
    )
    return DeliberationACController(
        bridge,
        DeliberationACControllerConfig(),
        finalizer,
    )


# =============================================================================
def _aggregate_mazehard_score_reports(
    score_reports: tuple[ArenaScoreReport | MazeHardScoreReport, ...],
) -> MazeHardScoreReport:
    if not score_reports:
        raise ValueError(
            "MazeHard score aggregation requires at least one report."
        )
    if not all(
        isinstance(report, MazeHardScoreReport) for report in score_reports
    ):
        raise TypeError(
            "MazeHard score aggregation received a non-MazeHard score."
        )

    reports = tuple(score_reports)
    tokens = torch.stack([report.tokens_accuracy for report in reports]).mean()
    seq_acc = torch.stack(
        [report.sequences_accuracy for report in reports]
    ).mean()
    seq_exact = torch.stack(
        [report.sequences_exact for report in reports]
    ).mean()
    return MazeHardScoreReport(
        tokens_accuracy=tokens,
        sequences_accuracy=seq_acc,
        sequences_exact=seq_exact,
    )


# =============================================================================
__all__ = ["SharedMazeHardModelComparisonBinding"]
