"""Routing tables for TEM variational objectives."""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import (
    TEM_ACC_OBS_PATH_ALL,
    TEM_ACC_OBS_PATH_REVISIT,
    TEM_ACC_OBS_POST_ALL,
    TEM_ACC_OBS_POST_REVISIT,
    TEM_ACC_OBS_RECALL_ALL,
    TEM_ACC_OBS_RECALL_REVISIT,
    TEM_LOSS_GRID_KL,
    TEM_LOSS_OBS_NLL,
    TEM_LOSS_OBS_PATH,
    TEM_LOSS_OBS_POST,
    TEM_LOSS_OBS_RECALL,
    TEM_LOSS_PLACE_CONSISTENCY,
    TEM_LOSS_PLACE_SENSORY,
    TEM_LOSS_PLACE_TRANSITION,
    TEM_LOSS_REG,
    extra_ratio_paths,
)


# =============================================================================
def _with_namespace(  # -------------------------------------------------------
    namespace: str, routes: tuple[Route, ...]
) -> tuple[Route, ...]:
    """Prefix route keys with a metric namespace."""
    return tuple(
        Route(f"{namespace}/{route.key}", route.num_path, route.den_path)
        for route in routes
    )


# =============================================================================
TEM_STEP_ROUTES: tuple[Route, ...] = (
    # -- Accuracy routes ------------------------------------------------------
    Route(
        "accuracy/obs_path_all",
        *extra_ratio_paths(TEM_ACC_OBS_PATH_ALL),
    ),
    Route(
        "accuracy/obs_path_revisit",
        *extra_ratio_paths(TEM_ACC_OBS_PATH_REVISIT),
    ),
    Route(
        "accuracy/obs_post_all",
        *extra_ratio_paths(TEM_ACC_OBS_POST_ALL),
    ),
    Route(
        "accuracy/obs_post_revisit",
        *extra_ratio_paths(TEM_ACC_OBS_POST_REVISIT),
    ),
    Route(
        "accuracy/obs_recall_all",
        *extra_ratio_paths(TEM_ACC_OBS_RECALL_ALL),
    ),
    Route(
        "accuracy/obs_recall_revisit",
        *extra_ratio_paths(TEM_ACC_OBS_RECALL_REVISIT),
    ),
    # -- Loss grid routes -----------------------------------------------------
    Route(
        "loss/grid_kl_revisit",
        *extra_ratio_paths(TEM_LOSS_GRID_KL),
    ),
    # -- Loss observation routes ----------------------------------------------
    Route(
        "loss/obs_path_revisit",
        *extra_ratio_paths(TEM_LOSS_OBS_PATH),
    ),
    Route(
        "loss/obs_post_revisit",
        *extra_ratio_paths(TEM_LOSS_OBS_POST),
    ),
    Route(
        "loss/obs_nll_revisit",
        *extra_ratio_paths(TEM_LOSS_OBS_NLL),
    ),
    Route(
        "loss/obs_recall_revisit",
        *extra_ratio_paths(TEM_LOSS_OBS_RECALL),
    ),
    # -- Loss place consistency routes ----------------------------------------
    Route(
        "loss/place_consistency_revisit",
        *extra_ratio_paths(TEM_LOSS_PLACE_CONSISTENCY),
    ),
    Route(
        "loss/place_sensory_revisit",
        *extra_ratio_paths(TEM_LOSS_PLACE_SENSORY),
    ),
    Route(
        "loss/place_transition_revisit",
        *extra_ratio_paths(TEM_LOSS_PLACE_TRANSITION),
    ),
    # -- Loss regularization routes -------------------------------------------
    Route(
        "loss/reg_revisit",
        *extra_ratio_paths(TEM_LOSS_REG),
    ),
)

# =============================================================================
TEM_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace(
    "episode", TEM_STEP_ROUTES
)

TEM_PRIMARY_VAL_ROUTE_KEY: str = "episode/accuracy/obs_path_revisit"

# =============================================================================
__all__ = ["TEM_EPISODE_ROUTES", "TEM_PRIMARY_VAL_ROUTE_KEY", "TEM_STEP_ROUTES"]
