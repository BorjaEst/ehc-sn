"""Canonical recipe registry for supported task--model evaluation pairs.

This is the single authority for which evaluation recipes exist, what they
represent, and how to build a resolved ``EvaluationExperiment`` from each.

Usage::

    from ehc_sn.eval.recipes import resolve_recipe

    recipe = resolve_recipe(EvaluationAlias("arena-tem-v1"))
    # recipe.config.task        → "arena"
    # recipe.config.model_family → "tem-v1"
    # recipe.binding.builder    → callable
    # recipe.binding.options_type → ArenaTEMV1EvaluationOptions
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from ehc_sn.eval.contracts import EvaluationExperiment
from ehc_sn.eval.invocation import (
    EvaluationBuildRequest,
    EvaluationOptions,
)
from ehc_sn.eval.recipe_catalogue import load_recipe_config
from ehc_sn.eval.recipe_schema import (
    EvaluationRecipeConfig,
    RecipeContractError,
)
from ehc_sn.metrics.spec import TaskScoringSpec
from ehc_sn.tasks.scoring import scoring_spec_for_task


# =============================================================================
class EvaluationAlias(StrEnum):
    """Canonical public aliases for supported evaluation recipes.

    Each alias selects a predefined (task, model_family, version) triple
    with a corresponding TOML recipe config and Python binding.

    Aliases follow a ``{task}-{model_family}-{version}`` naming convention.
    """

    ARENA_TEM_V1 = "arena-tem-v1"
    ARENA_TEM_V2 = "arena-tem-v2"
    GOALTRACE_HRM_V1 = "goaltrace-hrm-v1"
    MAZEHARD_HRM_V1 = "mazehard-hrm-v1"
    MAZEHARD_HRM_V2 = "mazehard-hrm-v2"
    ROUTEBIND_HRM_V1 = "routebind-hrm-v1"
    SEQMAZE_HRM_V1 = "seqmaze-hrm-v1"
    SEQMAZE_HRM_V2 = "seqmaze-hrm-v2"


# =============================================================================
@dataclass(frozen=True)
class EvaluationRecipeBinding:
    """Executable Python implementation for one evaluation recipe.

    Attributes
    ----------
    task:
        Task-family identifier, derived from the canonical recipe ID.
        Must match the recipe TOML ``task`` field.
    model_family:
        Model-family identifier, derived from the canonical recipe ID.
        Must match the recipe TOML ``model_family`` field.
    options_type:
        Pydantic model type for pair-specific scientific evaluation
        options.  The base :class:`EvaluationOptions` is used for pairs
        without explicit options.
    builder:
        Experiment builder callable.  Signature:
        ``builder(build_request: EvaluationBuildRequest) -> EvaluationExperiment``.
    """

    task: str
    model_family: str
    options_type: type[EvaluationOptions]
    builder: Callable[[EvaluationBuildRequest], EvaluationExperiment]


@dataclass(frozen=True)
class EvaluationRecipe:
    """Recipe configuration joined with executable binding and scoring spec.

    This is the composed type returned by ``resolve_recipe``.
    Declarative data lives in ``config``; executable symbols live in
    ``binding``; metric metadata lives in ``scoring``.
    """

    config: EvaluationRecipeConfig
    binding: EvaluationRecipeBinding
    scoring: TaskScoringSpec | None = None


# =============================================================================
# Recipe binding registry — maps recipe ID → executable symbols only.
#
# Declarative identity (task, model_family, primary_metric, figures, etc.)
# lives in TOML recipe files under config/evaluation/.
# =============================================================================

_EVALUATION_RECIPE_BINDINGS: Mapping[
    EvaluationAlias,
    EvaluationRecipeBinding,
] = {}
"""Mapping from alias to its executable binding (options type + builder).

Populated by the first call to ``resolve_recipe``.  Always use
``resolve_recipe`` to access a full recipe; do not read this dict directly.
"""


def _build_bindings() -> None:
    """Lazy-populate the binding registry.

    Imports are deferred to avoid circular imports at module load time.
    This function is idempotent.
    """
    if _EVALUATION_RECIPE_BINDINGS:
        return  # type: ignore[truthy-function]

    # -- tem-v1 / arena -----------------------------------------------------------
    from ehc_sn.experiments.arena.tem_v1.config import (
        ArenaTEMV1EvaluationOptions,
    )
    from ehc_sn.experiments.arena.tem_v1.evaluation import (
        build_arena_tem_v1_evaluation_experiment,
    )

    _bindings: dict[EvaluationAlias, EvaluationRecipeBinding] = {}

    _bindings[EvaluationAlias.ARENA_TEM_V1] = EvaluationRecipeBinding(
        task="arena",
        model_family="tem-v1",
        options_type=ArenaTEMV1EvaluationOptions,
        builder=build_arena_tem_v1_evaluation_experiment,
    )

    # -- tem-v2 / arena -----------------------------------------------------------
    from ehc_sn.experiments.arena.tem_v2.config import (
        ArenaTEMV2EvaluationOptions,
    )
    from ehc_sn.experiments.arena.tem_v2.evaluation import (
        build_arena_tem_v2_evaluation_experiment,
    )

    _bindings[EvaluationAlias.ARENA_TEM_V2] = EvaluationRecipeBinding(
        task="arena",
        model_family="tem-v2",
        options_type=ArenaTEMV2EvaluationOptions,
        builder=build_arena_tem_v2_evaluation_experiment,
    )

    # -- hrm-v1 / goaltrace -------------------------------------------------------
    from ehc_sn.experiments.goaltrace.hrm_v1.config import (
        GoaltraceHRMV1EvaluationOptions,
    )
    from ehc_sn.experiments.goaltrace.hrm_v1.evaluation import (
        build_goaltrace_hrm_v1_evaluation_experiment,
    )

    _bindings[EvaluationAlias.GOALTRACE_HRM_V1] = EvaluationRecipeBinding(
        task="goaltrace",
        model_family="hrm-v1",
        options_type=GoaltraceHRMV1EvaluationOptions,
        builder=build_goaltrace_hrm_v1_evaluation_experiment,
    )

    # -- hrm-v1 / mazehard --------------------------------------------------------
    from ehc_sn.experiments.mazehard.hrm_v1.config import (
        MazeHardHRMV1EvaluationOptions,
    )
    from ehc_sn.experiments.mazehard.hrm_v1.evaluation import (
        build_mazehard_hrm_v1_evaluation_experiment,
    )

    _bindings[EvaluationAlias.MAZEHARD_HRM_V1] = EvaluationRecipeBinding(
        task="mazehard",
        model_family="hrm-v1",
        options_type=MazeHardHRMV1EvaluationOptions,
        builder=build_mazehard_hrm_v1_evaluation_experiment,
    )

    # -- hrm-v2 / mazehard --------------------------------------------------------
    from ehc_sn.experiments.mazehard.hrm_v2.config import (
        MazeHardHRMV2EvaluationOptions,
    )
    from ehc_sn.experiments.mazehard.hrm_v2.evaluation import (
        build_mazehard_hrm_v2_evaluation_experiment,
    )

    _bindings[EvaluationAlias.MAZEHARD_HRM_V2] = EvaluationRecipeBinding(
        task="mazehard",
        model_family="hrm-v2",
        options_type=MazeHardHRMV2EvaluationOptions,
        builder=build_mazehard_hrm_v2_evaluation_experiment,
    )

    # -- hrm-v1 / routebind -------------------------------------------------------
    from ehc_sn.experiments.routebind.hrm_v1.config import (
        RoutebindHRMV1EvaluationOptions,
    )
    from ehc_sn.experiments.routebind.hrm_v1.evaluation import (
        build_routebind_hrm_v1_evaluation_experiment,
    )

    _bindings[EvaluationAlias.ROUTEBIND_HRM_V1] = EvaluationRecipeBinding(
        task="routebind",
        model_family="hrm-v1",
        options_type=RoutebindHRMV1EvaluationOptions,
        builder=build_routebind_hrm_v1_evaluation_experiment,
    )

    # -- hrm-v1 / seqmaze ---------------------------------------------------------
    from ehc_sn.experiments.seqmaze.hrm_v1.config import (
        SeqMazeHRMV1EvaluationOptions,
    )
    from ehc_sn.experiments.seqmaze.hrm_v1.evaluation import (
        build_seqmaze_hrm_v1_evaluation_experiment,
    )

    _bindings[EvaluationAlias.SEQMAZE_HRM_V1] = EvaluationRecipeBinding(
        task="seqmaze",
        model_family="hrm-v1",
        options_type=SeqMazeHRMV1EvaluationOptions,
        builder=build_seqmaze_hrm_v1_evaluation_experiment,
    )

    # -- hrm-v2 / seqmaze ---------------------------------------------------------
    from ehc_sn.experiments.seqmaze.hrm_v2.config import (
        SeqMazeHRMV2EvaluationOptions,
    )
    from ehc_sn.experiments.seqmaze.hrm_v2.evaluation import (
        build_seqmaze_hrm_v2_evaluation_experiment,
    )

    _bindings[EvaluationAlias.SEQMAZE_HRM_V2] = EvaluationRecipeBinding(
        task="seqmaze",
        model_family="hrm-v2",
        options_type=SeqMazeHRMV2EvaluationOptions,
        builder=build_seqmaze_hrm_v2_evaluation_experiment,
    )

    _EVALUATION_RECIPE_BINDINGS.update(_bindings)  # type: ignore[union-attr]


# =============================================================================
# Contract validation
# =============================================================================


def _validate_recipe_contract(
    config: EvaluationRecipeConfig,
    binding: EvaluationRecipeBinding,
) -> TaskScoringSpec | None:
    """Verify that the recipe TOML metadata agrees with the Python binding
    and that the ``primary_metric`` references a valid benchmark-eligible
    metric from the task's ``TaskScoringSpec``.

    Returns
    -------
    TaskScoringSpec | None
        The task's scoring spec, or ``None`` if the task is not registered.

    Raises
    ------
    RecipeContractError
        If ``task`` or ``model_family`` from the TOML disagrees with the
        binding's canonical values, or if ``primary_metric`` is non-None
        and not found or not benchmark-eligible.
    """
    violations: list[str] = []
    if config.task != binding.task:
        violations.append(
            f"task: TOML says {config.task!r}, binding says {binding.task!r}"
        )
    if config.model_family != binding.model_family:
        violations.append(
            f"model_family: TOML says {config.model_family.value!r}, "
            f"binding says {binding.model_family!r}"
        )

    # Look up the task's scoring spec and validate primary_metric.
    scoring: TaskScoringSpec | None = None
    try:
        scoring = scoring_spec_for_task(config.task)
    except KeyError:
        pass  # Task not in scoring registry — no cross-validation possible.

    if scoring is not None and config.primary_metric is not None:
        try:
            scoring.require_benchmark_metric(config.primary_metric)
        except (KeyError, ValueError) as exc:
            violations.append(str(exc))

    if violations:
        raise RecipeContractError(
            f"Recipe TOML for {config.alias!r} disagrees with Python binding: "
            + "; ".join(violations)
        )

    return scoring


# =============================================================================
# Public API
# =============================================================================


def resolve_recipe(alias: EvaluationAlias) -> EvaluationRecipe:
    """Resolve an alias to a full ``EvaluationRecipe``.

    Combines alias validation, TOML loading, and binding lookup
    in one call.

    Parameters
    ----------
    alias:
        A canonical evaluation alias (e.g. ``EvaluationAlias("arena-tem-v1")``).

    Returns
    -------
    EvaluationRecipe
        Composed recipe with config from TOML and binding from the
        executable registry.

    Raises
    ------
    ValueError
        If the recipe ID is not recognised.
    RecipeCatalogueError
        If the recipe TOML is missing or invalid.
    """
    _build_bindings()
    config = load_recipe_config(alias)
    binding = _EVALUATION_RECIPE_BINDINGS[alias]

    # Cross-validate TOML metadata against the Python binding and
    # primary_metric against the task scoring spec.
    scoring = _validate_recipe_contract(config, binding)

    return EvaluationRecipe(config=config, binding=binding, scoring=scoring)


def list_recipes() -> list[EvaluationAlias]:
    """Return all registered evaluation aliases in display order."""
    _build_bindings()
    return list(_EVALUATION_RECIPE_BINDINGS)


__all__ = [
    "EvaluationAlias",
    "EvaluationRecipe",
    "EvaluationRecipeBinding",
    "RecipeContractError",
    "list_recipes",
    "resolve_recipe",
]
