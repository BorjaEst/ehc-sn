"""Routing tables for EHC variational objectives."""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import (
    EHC_ACC_OBS_ANCESTRAL_ALL,
    EHC_ACC_OBS_ANCESTRAL_REVISIT,
    EHC_ACC_OBS_INFERENCE_ALL,
    EHC_ACC_OBS_INFERENCE_REVISIT,
    EHC_ACC_OBS_RETRIEVED_ALL,
    EHC_ACC_OBS_RETRIEVED_REVISIT,
    EHC_LOSS_GRID_KL_ALL,
    EHC_LOSS_GRID_KL_REVISIT,
    EHC_LOSS_OBS_NLL_ALL,
    EHC_LOSS_OBS_NLL_REVISIT,
    EHC_LOSS_PLACE_CONSISTENCY_ALL,
    EHC_LOSS_PLACE_CONSISTENCY_REVISIT,
    EHC_LOSS_REG_ALL,
    EHC_LOSS_REG_REVISIT,
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
EHC_STEP_ROUTES: tuple[Route, ...] = (
    Route(
        "accuracy/obs_inference_revisit",
        *extra_ratio_paths(EHC_ACC_OBS_INFERENCE_REVISIT),
    ),
    Route(
        "accuracy/obs_retrieved_revisit",
        *extra_ratio_paths(EHC_ACC_OBS_RETRIEVED_REVISIT),
    ),
    Route(
        "accuracy/obs_ancestral_revisit",
        *extra_ratio_paths(EHC_ACC_OBS_ANCESTRAL_REVISIT),
    ),
    Route(
        "accuracy/obs_inference_all",
        *extra_ratio_paths(EHC_ACC_OBS_INFERENCE_ALL),
    ),
    Route(
        "accuracy/obs_retrieved_all",
        *extra_ratio_paths(EHC_ACC_OBS_RETRIEVED_ALL),
    ),
    Route(
        "accuracy/obs_ancestral_all",
        *extra_ratio_paths(EHC_ACC_OBS_ANCESTRAL_ALL),
    ),
    Route(
        "loss/obs_nll_revisit",
        *extra_ratio_paths(EHC_LOSS_OBS_NLL_REVISIT),
    ),
    Route(
        "loss/grid_kl_revisit",
        *extra_ratio_paths(EHC_LOSS_GRID_KL_REVISIT),
    ),
    Route(
        "loss/place_consistency_revisit",
        *extra_ratio_paths(EHC_LOSS_PLACE_CONSISTENCY_REVISIT),
    ),
    Route(
        "loss/reg_revisit",
        *extra_ratio_paths(EHC_LOSS_REG_REVISIT),
    ),
    Route(
        "loss/obs_nll_all",
        *extra_ratio_paths(EHC_LOSS_OBS_NLL_ALL),
    ),
    Route(
        "loss/grid_kl_all",
        *extra_ratio_paths(EHC_LOSS_GRID_KL_ALL),
    ),
    Route(
        "loss/place_consistency_all",
        *extra_ratio_paths(EHC_LOSS_PLACE_CONSISTENCY_ALL),
    ),
    Route(
        "loss/reg_all",
        *extra_ratio_paths(EHC_LOSS_REG_ALL),
    ),
)

# =============================================================================
EHC_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace(
    "episode", EHC_STEP_ROUTES
)

EHC_PRIMARY_VAL_ROUTE_KEY: str = "episode/accuracy/obs_ancestral_revisit"

# =============================================================================
__all__ = ["EHC_EPISODE_ROUTES", "EHC_PRIMARY_VAL_ROUTE_KEY", "EHC_STEP_ROUTES"]
