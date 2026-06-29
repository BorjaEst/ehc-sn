"""Locate, load, cache, and validate recipe TOML files.

Usage::

    from ehc_sn.eval.recipe_catalogue import load_recipe_config

    config = load_recipe_config(EvaluationAlias.ARENA_TEM_V1)
    # config.task      -> "arena"
    # config.model_family -> "tem-v1"
    # config.capture.profile -> "diagnostic"

Boundary rules:
    - Must not import from "experiments/", "models/", "tasks/",
      "adapters/", or "lightning/".
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import ValidationError

from ehc_sn.eval.recipe_schema import (
    EvaluationRecipeConfig,
    RecipeCatalogueError,
)

_RECIPE_CATALOGUE_ROOT = Path("config/evaluation/recipes")
_RECIPE_CACHE: dict[str, EvaluationRecipeConfig] = {}


def _recipe_path(alias):
    return _RECIPE_CATALOGUE_ROOT / f"{alias.value}.toml"


def load_recipe_config(alias):
    """Load, validate, and cache the recipe TOML for *alias*."""
    cached = _RECIPE_CACHE.get(alias.value)
    if cached is not None:
        return cached

    path = _recipe_path(alias)

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RecipeCatalogueError(
            f"Recipe TOML not found for {alias.value!r}: {path}"
        )
    except tomllib.TOMLDecodeError as exc:
        raise RecipeCatalogueError(f"Invalid TOML in {path}: {exc}")

    try:
        config = EvaluationRecipeConfig.model_validate(raw)
    except ValidationError as exc:
        raise RecipeCatalogueError(
            f"Recipe validation failed for {alias.value!r} in {path}: {exc}"
        )

    if config.alias != alias.value:
        raise RecipeCatalogueError(
            f"Recipe file {path} declares alias={config.alias!r} "
            f"but was loaded for {alias.value!r}."
        )

    _RECIPE_CACHE[alias.value] = config
    return config


def clear_recipe_cache() -> None:
    """Clear the internal recipe cache."""
    _RECIPE_CACHE.clear()


__all__ = [
    "RecipeCatalogueError",
    "clear_recipe_cache",
    "load_recipe_config",
]
