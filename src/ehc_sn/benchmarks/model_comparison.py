"""Benchmark-owned shared runtime for ready-track model-comparison execution."""

from __future__ import annotations

import random
from collections.abc import Callable
from itertools import cycle, islice
from typing import cast

import torch
from torch.optim import Optimizer

from ehc_sn.benchmarks._bindings._mazehard import MazeHardCaseAggregate
from ehc_sn.benchmarks._bindings.model_comparison import (
    MazeHardDelibHRMV1ModelComparisonBinding,
    SharedArenaReplayModelComparisonBinding,
)
from ehc_sn.benchmarks.contracts import (
    ArenaScoreReport,
    ArtifactManifest,
    BridgeAdaptationProtocol,
    ModelComparisonBinding,
    ModelComparisonExecution,
    ModelComparisonExecutionResources,
    ScoreReport,
    TrackData,
    TrackRecipe,
    validate_model_comparison_pair,
)
from ehc_sn.benchmarks.runner import task_family_for_track
from ehc_sn.tasks.scoring import scoring_spec_for_task
from ehc_sn.eval.contracts import EvaluationCaseResult
from ehc_sn.eval.executor import execute_replay_evaluation_batch
from ehc_sn.tasks.arena.evaluation import (
    ArenaCaseMetrics,
    aggregate_arena_case_metrics,
    build_arena_score_report,
    build_arena_step_score,
)
from ehc_sn.tasks.arena.runtime import coerce_arena_targets
from ehc_sn.tasks.mazehard.evaluation import (
    MazeHardScoreReport,
    build_maze_hard_step_score,
)
from ehc_sn.training.optim import Adam, AdamConfig


# =============================================================================
def run_ready_track_model_comparison(
    manifest: ArtifactManifest,
    recipe: TrackRecipe,
    *,
    binding: ModelComparisonBinding | None = None,
) -> ScoreReport:
    """Run deterministic ready-track model-comparison from manifest+recipe."""
    validate_model_comparison_pair(manifest, recipe)

    # Validate recipe primary_metric against the task scoring spec before
    # any execution — fail fast with a clear error.
    task_family = task_family_for_track(recipe.track_id)
    scoring_spec_for_task(task_family).require_benchmark_metric(
        recipe.primary_metric
    )

    seed = recipe.execution.seed
    if seed is None:
        raise ValueError("recipe.execution.seed is required.")
    # Seed benchmark-owned execution setup so fresh bridge construction is
    # deterministic for the manifest+recipe execution contract.
    _set_seed(seed)

    active_binding = binding or _default_binding_for_track(recipe.track_id)
    bundle = active_binding.bind_model_comparison(manifest, recipe)
    execution = bundle.create_execution()

    provider_factory = cast(
        Callable[..., object],
        bundle.resources.replay_provider_factory,
    )
    adaptation_provider = _build_provider(
        provider_factory,
        recipe.adaptation_data,
    )
    evaluation_provider = _build_provider(
        provider_factory,
        recipe.evaluation_data,
    )

    step_options = _validate_step_options(bundle.resources)

    _adapt_bridge_only(
        execution=execution,
        provider=adaptation_provider,
        recipe=recipe,
        step_options=step_options,
    )

    score_reports: list[ScoreReport] = []
    if recipe.execution.reseed_before_evaluation:
        _set_seed(seed)
    with torch.no_grad():
        for case in evaluation_provider.provide_cases(
            max_batches=recipe.execution.max_batches
        ):
            carry0 = execution.controller.initial_state(case.batch)
            result = execute_replay_evaluation_batch(
                case=case,
                runner=execution.runner,
                controller=execution.controller,
                carry=carry0,
                objective=execution.objective,
                max_rollout_steps=recipe.execution.fixed_budget_steps,
                hard_max_rollout_steps=None,
                runner_options=step_options,
                objective_options={
                    "controller": execution.controller,
                    "td_target": False,
                },
            )
            score_reports.append(
                _score_evaluated_case(
                    result, recipe.track_id, model_family=manifest.model_family
                )
            )

    if not score_reports:
        raise ValueError("Benchmark evaluation produced no score reports.")

    aggregated = bundle.score_aggregator(tuple(score_reports))
    if not isinstance(aggregated, (ArenaScoreReport, MazeHardScoreReport)):
        raise TypeError(
            "Benchmark score aggregator returned invalid score type."
        )
    return aggregated


# =============================================================================
def run_ready_track_model_comparison_seeds(
    manifest: ArtifactManifest,
    recipe: TrackRecipe,
    *,
    seed_count: int,
    binding: ModelComparisonBinding | None = None,
) -> tuple[ScoreReport, ...]:
    """Run ready-track model-comparison for concrete seed_count executions."""
    if seed_count < 1:
        raise ValueError(f"seed_count must be >= 1, got {seed_count}.")
    base_seed = recipe.execution.seed
    if base_seed is None:
        raise ValueError("recipe.execution.seed is required.")

    per_seed_scores: list[ScoreReport] = []
    for seed_offset in range(seed_count):
        run_seed = int(base_seed + seed_offset)
        run_recipe = recipe.model_copy(deep=True)
        run_recipe.execution.seed = run_seed
        run_recipe.adaptation.protocol.seed = run_seed
        per_seed_scores.append(
            run_ready_track_model_comparison(
                manifest,
                run_recipe,
                binding=binding,
            )
        )
    return tuple(per_seed_scores)


# =============================================================================
def _default_binding_for_track(track_id: str) -> ModelComparisonBinding:
    if track_id == "arena-struct":
        return SharedArenaReplayModelComparisonBinding()
    if track_id == "mazehard-delib":
        return MazeHardDelibHRMV1ModelComparisonBinding()
    raise ValueError(
        "No ready-track model-comparison binding is registered for "
        f"track_id={track_id!r}."
    )


# =============================================================================
def _validate_step_options(
    resources: ModelComparisonExecutionResources,
) -> dict[str, object]:
    step_options = resources.controller_step_options
    if not isinstance(step_options, dict):
        raise TypeError("controller_step_options must be a mapping.")
    return step_options


# =============================================================================
def _set_seed(seed: int) -> None:
    """Set benchmark runtime RNG state for reproducible evaluation."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
def _adapt_bridge_only(
    *,
    execution: ModelComparisonExecution,
    provider: object,
    recipe: TrackRecipe,
    step_options: dict[str, object],
) -> None:
    """Adapt bridge-only trainables under the recipe-owned protocol budget."""
    protocol = recipe.adaptation.protocol
    if protocol.reseed_before_adaptation:
        _set_seed(protocol.seed)

    max_batches = _max_adaptation_batches(
        protocol,
        recipe.adaptation_data.batch_size,
    )
    adaptation_cases = tuple(provider.provide_cases(max_batches=max_batches))
    if not adaptation_cases:
        raise ValueError("Bridge adaptation requires at least one case.")

    if not execution.bridge_parameter_groups:
        return

    optimizer = _build_bridge_optimizer(
        execution.bridge_parameter_groups,
        protocol,
    )
    for case in islice(cycle(adaptation_cases), protocol.steps):
        optimizer.zero_grad(set_to_none=True)
        carry0 = execution.controller.initial_state(case.batch)
        result = execute_replay_evaluation_batch(
            case=case,
            runner=execution.runner,
            controller=execution.controller,
            carry=carry0,
            objective=execution.objective,
            max_rollout_steps=recipe.execution.fixed_budget_steps,
            hard_max_rollout_steps=None,
            runner_options=step_options,
            objective_options={
                "controller": execution.controller,
                "td_target": False,
            },
        )
        loss = result.evaluated.loss
        loss.backward()
        optimizer.step()


# =============================================================================
def _max_adaptation_batches(
    protocol: BridgeAdaptationProtocol,
    batch_size: int,
) -> int:
    """Return batch count needed to cover adaptation examples budget."""
    # Ceiling division to cover the requested number of examples.
    return (protocol.examples + batch_size - 1) // batch_size


# =============================================================================
def _build_bridge_optimizer(
    bridge_parameter_groups: tuple[dict[str, object], ...],
    protocol: BridgeAdaptationProtocol,
) -> Optimizer:
    """Build the bridge adaptation optimizer from explicit protocol fields."""
    optimizer_kind = protocol.optimizer.strip().lower()
    if optimizer_kind != "adam":
        raise ValueError(
            "Unsupported bridge adaptation optimizer: "
            f"{protocol.optimizer!r}."
        )
    if protocol.stopping_rule.strip().lower() != "fixed_steps":
        raise ValueError(
            "Unsupported bridge adaptation stopping_rule: "
            f"{protocol.stopping_rule!r}."
        )
    groups = [dict(group) for group in bridge_parameter_groups]
    return Adam(
        groups,
        AdamConfig(
            lr=protocol.learning_rate,
            weight_decay=protocol.weight_decay,
            betas=protocol.betas,
        ),
    )


# =============================================================================
def _build_provider(
    provider_factory: Callable[..., object],
    data_config: TrackData,
) -> object:
    """Build one data provider from explicit track data-selection config."""
    base_kwargs = {
        "dataset_path": data_config.dataset_path,
        "split": data_config.split,
        "batch_size": data_config.batch_size,
    }
    if data_config.sample_ids:
        return provider_factory(
            **base_kwargs,
            sample_ids=list(data_config.sample_ids),
        )
    return provider_factory(
        **base_kwargs,
        n_cases=data_config.n_cases,
    )


# =============================================================================
# Metric-key lookup for per-case aggregation across arena families.
_ARENA_ACC_KEYS: dict[str, tuple[str, str]] = {
    "tem-v1": ("accuracy_obs_inference_all", "accuracy_obs_inference_revisit"),
    "tem-v2": ("accuracy_obs_inference_all", "accuracy_obs_inference_revisit"),
    "ehc-v1": (
        "ehc_accuracy_obs_inference_all",
        "ehc_accuracy_obs_inference_revisit",
    ),
}


def _score_evaluated_case(
    result: EvaluationCaseResult,
    track_id: str,
    *,
    model_family: str | None = None,
) -> ScoreReport:
    """Build one task-owned score report from an evaluated rollout.

    For arena-struct, passes *model_family* to select the correct per-case
    metric keys.
    """
    if track_id == "arena-struct":
        return _score_arena_evaluated_case(result, model_family=model_family)
    if track_id == "mazehard-delib":
        return _score_mazehard_evaluated_case(result)
    raise ValueError(f"Unsupported ready-track id for scoring: {track_id!r}.")


# =============================================================================
def _score_arena_evaluated_case(
    result: EvaluationCaseResult,
    *,
    model_family: str | None = None,
) -> ArenaScoreReport:
    """Build one task-owned Arena score report from an evaluated rollout.

    Uses per-step :class:`~ehc_sn.metrics.step_metrics.RatioStat` extras
    accumulated across all steps in the evaluated chunk rather than the last
    step only.
    """
    evaluated = result.evaluated
    model_family_normalized = (
        model_family.strip().lower().replace("_", "-")
        if model_family
        else "tem-v1"
    )
    acc_keys = _ARENA_ACC_KEYS.get(
        model_family_normalized,
        ("accuracy_obs_inference_all", "accuracy_obs_inference_revisit"),
    )
    case = aggregate_arena_case_metrics(
        evaluated,
        case_id=result.case_id,
        acc_all_key=acc_keys[0],
        acc_revisit_key=acc_keys[1],
    )
    return ArenaScoreReport(
        accuracy_all=torch.tensor(case.accuracy_all),
        accuracy_revisit=torch.tensor(case.accuracy_revisit),
        correct_all=torch.tensor(case.correct_all),
        count_all=torch.tensor(case.count_all),
        correct_revisit=torch.tensor(case.correct_revisit),
        count_revisit=torch.tensor(case.count_revisit),
    )


# =============================================================================
def _score_mazehard_evaluated_case(
    result: EvaluationCaseResult,
) -> MazeHardCaseAggregate:
    """Build denominator-aware MazeHard case aggregates from one rollout."""
    observed = result.evaluated.last_step
    objective_step = observed.outputs
    step_outputs = objective_step.outputs
    if step_outputs is None:
        raise ValueError(
            "ACT objective step did not retain controller outputs for scoring."
        )

    task_output = step_outputs.task
    logits = getattr(task_output, "task_logits", None)
    if logits is None:
        raise ValueError("MazeHard task output must expose task_logits.")

    labels = observed.batch.get("labels")
    if labels is None:
        raise ValueError("MazeHard evaluated batch is missing labels.")

    step_score = build_maze_hard_step_score(logits, labels)
    token_correct_sum = step_score.token_is_correct.to(
        dtype=torch.float32
    ).sum()
    token_count_sum = step_score.valid_mask.to(dtype=torch.float32).sum()
    sequence_accuracy = step_score.sequence_accuracy
    sequence_exact = step_score.sequence_is_correct.to(dtype=torch.float32)
    sequence_count_sum = sequence_accuracy.new_tensor(
        float(sequence_accuracy.shape[0])
    )
    return MazeHardCaseAggregate(
        token_correct_sum=token_correct_sum,
        token_count_sum=token_count_sum,
        sequence_accuracy_sum=sequence_accuracy.sum(),
        sequence_exact_sum=sequence_exact.sum(),
        sequence_count_sum=sequence_count_sum,
    )


# =============================================================================
__all__ = [
    "run_ready_track_model_comparison",
    "run_ready_track_model_comparison_seeds",
]
