"""Evaluation experiment configuration loading and resolution.

Provides the single public entry point for loading a TOML evaluation
config and resolving it to an :class:`EvaluationExperiment`.

Usage::

    from ehc_sn.eval.configuration import load_evaluation_experiment

    loaded = load_evaluation_experiment(
        Path("config/evaluation/hrm-v1-mazehard.toml"),
    )
    # loaded.experiment      → EvaluationExperiment
    # loaded.experiment_id   → "hrm-v1-mazehard"
    # loaded.config_digest   → "a1b2c3d4e5f6g7h8"

Boundary rules:
    - Must not import from ``experiments/<task>/<family>/`` directly.
      Dispatches through ``ehc_sn.eval.registry``.
    - Must not be re-exported from ``eval/__init__.py``.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ehc_sn.eval.registry import (
    get_evaluation_experiment_registration,
    list_experiment_ids,
)
from ehc_sn.experiments._infra import EvaluationExperiment


# =============================================================================
class EvaluationConfigurationError(ValueError):
    """Raised when evaluation configuration is invalid, unreadable, or mismatched."""


@dataclass(frozen=True)
class LoadedEvaluationExperiment:
    """A resolved evaluation experiment with configuration provenance.

    Attributes
    ----------
    experiment:
        Fully resolved experiment returned by the experiment-specific builder.
    experiment_id:
        The ``experiment_id`` string extracted from the TOML config.
    config_path:
        Resolved absolute path to the configuration file.
    config_digest:
        First 16 hex characters of the SHA-256 digest of the configuration
        file bytes.  Deterministic — same file always produces the same value.
    """

    experiment: EvaluationExperiment
    experiment_id: str
    config_path: Path
    config_digest: str


# =============================================================================
def load_evaluation_experiment(
    config_path: Path,
    *,
    expected_experiment_id: str | None = None,
) -> LoadedEvaluationExperiment:
    """Load and resolve a TOML evaluation config to an ``EvaluationExperiment``.

    Reads the TOML file, extracts ``experiment_id``, dispatches through
    the experiment registry, validates with the experiment-specific
    Pydantic model, and returns the builder's resolved experiment with
    configuration provenance.

    The ``experiment_id`` key is stripped from the TOML dict *before*
    Pydantic validation, because the Pydantic config models use
    ``extra="forbid"`` and do not declare an ``experiment_id`` field.

    Parameters
    ----------
    config_path:
        Path to the TOML evaluation configuration file.
    expected_experiment_id:
        If provided, the config's ``experiment_id`` must match.  Used
        by per-task aliases to catch misconfiguration.

    Returns
    -------
    LoadedEvaluationExperiment
        Resolved experiment with configuration provenance.

    Raises
    ------
    EvaluationConfigurationError
        If the config is missing, unreadable, invalid TOML, missing a
        non-empty ``experiment_id`` string, has an unexpected
        ``experiment_id``, references an unknown ``experiment_id``, or
        fails experiment-specific Pydantic validation.
    """
    resolved_path = config_path.resolve()

    # ---- Read TOML ----------------------------------------------------------
    try:
        config_map = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationConfigurationError(
            f"Could not read evaluation config: {resolved_path}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise EvaluationConfigurationError(
            f"Invalid TOML in {resolved_path}: {exc}"
        ) from exc

    # ---- Extract experiment_id ----------------------------------------------
    experiment_id = config_map.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise EvaluationConfigurationError(
            f"Config {resolved_path} must declare a non-empty string "
            f"'experiment_id'."
        )
    experiment_id = experiment_id.strip()

    # ---- Validate expected identity -----------------------------------------
    if (
        expected_experiment_id is not None
        and experiment_id != expected_experiment_id
    ):
        raise EvaluationConfigurationError(
            f"Expected experiment_id={expected_experiment_id!r}, "
            f"but {resolved_path} declares {experiment_id!r}."
        )

    # ---- Resolve registry ---------------------------------------------------
    try:
        config_type, builder = get_evaluation_experiment_registration(
            experiment_id
        )
    except KeyError:
        raise EvaluationConfigurationError(
            f"Unknown experiment_id {experiment_id!r}. "
            f"Available: {list_experiment_ids()}."
        ) from None

    # ---- Strip non-model key before Pydantic validation ---------------------
    # The Pydantic config classes use extra="forbid" and do not define
    # experiment_id as a field.  Strip before calling model_validate.
    config_map = {k: v for k, v in config_map.items() if k != "experiment_id"}

    # ---- Validate and build -------------------------------------------------
    try:
        config = config_type.model_validate(config_map)
    except ValidationError as exc:
        raise EvaluationConfigurationError(
            f"Invalid configuration for experiment {experiment_id!r} "
            f"in {resolved_path}."
        ) from exc

    experiment = builder(config)

    # ---- Compute config digest ----------------------------------------------
    config_digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()[:16]

    return LoadedEvaluationExperiment(
        experiment=experiment,
        experiment_id=experiment_id,
        config_path=resolved_path,
        config_digest=config_digest,
    )


__all__ = [
    "EvaluationConfigurationError",
    "LoadedEvaluationExperiment",
    "load_evaluation_experiment",
]
