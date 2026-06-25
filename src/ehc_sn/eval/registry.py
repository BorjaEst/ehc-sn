"""Evaluation experiment registry — typed dispatch by experiment_id.

Lazy-loads experiment config classes and builder functions to avoid
import cycles with ``experiments/``.  Only top-level orchestration
(runners, CLI, artifact reconstruction) should import this module.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Private — populated lazily by _build_registry().
_EVALUATION_EXPERIMENTS: dict[str, tuple[type[BaseModel], Any]] = {}


def _build_registry() -> None:
    """Lazy-populate the evaluation experiment registry.

    Imports are deferred to this point to avoid circular imports at
    module load time.  This function is idempotent.
    """
    if _EVALUATION_EXPERIMENTS:
        return

    # -- hrm-v1 / mazehard --------------------------------------------------------
    from ehc_sn.experiments.mazehard.hrm_v1.config import (
        MazeHardHRMV1EvaluationExperimentConfig,
    )
    from ehc_sn.experiments.mazehard.hrm_v1.evaluation import (
        build_mazehard_hrm_v1_evaluation_experiment,
    )

    _EVALUATION_EXPERIMENTS["hrm-v1-mazehard"] = (
        MazeHardHRMV1EvaluationExperimentConfig,
        build_mazehard_hrm_v1_evaluation_experiment,
    )

    # -- hrm-v2 / mazehard --------------------------------------------------------
    from ehc_sn.experiments.mazehard.hrm_v2.config import (
        MazeHardHRMV2EvaluationExperimentConfig,
    )
    from ehc_sn.experiments.mazehard.hrm_v2.evaluation import (
        build_mazehard_hrm_v2_evaluation_experiment,
    )

    _EVALUATION_EXPERIMENTS["hrm-v2-mazehard"] = (
        MazeHardHRMV2EvaluationExperimentConfig,
        build_mazehard_hrm_v2_evaluation_experiment,
    )

    # -- hrm-v1 / goaltrace -------------------------------------------------------
    from ehc_sn.experiments.goaltrace.hrm_v1.config import (
        GoaltraceHRMV1EvaluationExperimentConfig,
    )
    from ehc_sn.experiments.goaltrace.hrm_v1.evaluation import (
        build_goaltrace_hrm_v1_evaluation_experiment,
    )

    _EVALUATION_EXPERIMENTS["hrm-v1-goaltrace"] = (
        GoaltraceHRMV1EvaluationExperimentConfig,
        build_goaltrace_hrm_v1_evaluation_experiment,
    )

    # -- hrm-v1 / routebind -------------------------------------------------------
    from ehc_sn.experiments.routebind.hrm_v1.config import (
        RoutebindHRMV1EvaluationExperimentConfig,
    )
    from ehc_sn.experiments.routebind.hrm_v1.evaluation import (
        build_routebind_hrm_v1_evaluation_experiment,
    )

    _EVALUATION_EXPERIMENTS["hrm-v1-routebind"] = (
        RoutebindHRMV1EvaluationExperimentConfig,
        build_routebind_hrm_v1_evaluation_experiment,
    )

    # -- hrm-v1 / seqmaze ---------------------------------------------------------
    from ehc_sn.experiments.seqmaze.hrm_v1.config import (
        SeqMazeHRMV1EvaluationExperimentConfig,
    )
    from ehc_sn.experiments.seqmaze.hrm_v1.evaluation import (
        build_seqmaze_hrm_v1_evaluation_experiment,
    )

    _EVALUATION_EXPERIMENTS["hrm-v1-seqmaze"] = (
        SeqMazeHRMV1EvaluationExperimentConfig,
        build_seqmaze_hrm_v1_evaluation_experiment,
    )

    # -- hrm-v2 / seqmaze ---------------------------------------------------------
    from ehc_sn.experiments.seqmaze.hrm_v2.config import (
        SeqMazeHRMV2EvaluationExperimentConfig,
    )
    from ehc_sn.experiments.seqmaze.hrm_v2.evaluation import (
        build_seqmaze_hrm_v2_evaluation_experiment,
    )

    _EVALUATION_EXPERIMENTS["hrm-v2-seqmaze"] = (
        SeqMazeHRMV2EvaluationExperimentConfig,
        build_seqmaze_hrm_v2_evaluation_experiment,
    )

    # -- tem-v1 / arena -----------------------------------------------------------
    from ehc_sn.experiments.arena.tem_v1.config import (
        ArenaTEMV1EvaluationExperimentConfig,
    )
    from ehc_sn.experiments.arena.tem_v1.evaluation import (
        build_arena_tem_v1_evaluation_experiment,
    )

    _EVALUATION_EXPERIMENTS["tem-v1-arena"] = (
        ArenaTEMV1EvaluationExperimentConfig,
        build_arena_tem_v1_evaluation_experiment,
    )

    # -- tem-v2 / arena -----------------------------------------------------------
    from ehc_sn.experiments.arena.tem_v2.config import (
        ArenaTEMV2EvaluationExperimentConfig,
    )
    from ehc_sn.experiments.arena.tem_v2.evaluation import (
        build_arena_tem_v2_evaluation_experiment,
    )

    _EVALUATION_EXPERIMENTS["tem-v2-arena"] = (
        ArenaTEMV2EvaluationExperimentConfig,
        build_arena_tem_v2_evaluation_experiment,
    )


# =============================================================================
# Public API
# =============================================================================


def get_evaluation_experiment_registration(
    experiment_id: str,
) -> tuple[type[BaseModel], Any]:
    """Return ``(config_type, builder)`` for *experiment_id*.

    Raises
    ------
    KeyError
        If *experiment_id* is unknown.  The error message lists available IDs.
    """
    _build_registry()
    if experiment_id not in _EVALUATION_EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment_id {experiment_id!r}. "
            f"Available: {sorted(_EVALUATION_EXPERIMENTS)}."
        )
    return _EVALUATION_EXPERIMENTS[experiment_id]


def list_experiment_ids() -> list[str]:
    """Return all registered experiment identifiers, sorted."""
    _build_registry()
    return sorted(_EVALUATION_EXPERIMENTS)


__all__ = [
    "get_evaluation_experiment_registration",
    "list_experiment_ids",
]
