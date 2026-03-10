"""Routing tables for TEM variational loss heads."""

from ehc_sn.metrics.adapter import Route
from ehc_sn.metrics.keys import (
    TEM_LOSS_GRID_KL,
    TEM_LOSS_OBS_NLL,
    TEM_LOSS_PLACE_CONSISTENCY,
    TEM_LOSS_REG,
    extra_ratio_paths,
)


def _with_namespace(namespace: str, routes: tuple[Route, ...]) -> tuple[Route, ...]:
    """Prefix route keys with a metric namespace."""
    return tuple(Route(f"{namespace}/{route.key}", route.num_path, route.den_path) for route in routes)


TEM_STEP_ROUTES: tuple[Route, ...] = (
    Route("loss/obs_nll", *extra_ratio_paths(TEM_LOSS_OBS_NLL)),
    Route("loss/grid_kl", *extra_ratio_paths(TEM_LOSS_GRID_KL)),
    Route("loss/place_consistency", *extra_ratio_paths(TEM_LOSS_PLACE_CONSISTENCY)),
    Route("loss/reg", *extra_ratio_paths(TEM_LOSS_REG)),
)

TEM_EPISODE_ROUTES: tuple[Route, ...] = _with_namespace("episode", TEM_STEP_ROUTES)

__all__ = ["TEM_EPISODE_ROUTES", "TEM_STEP_ROUTES"]
