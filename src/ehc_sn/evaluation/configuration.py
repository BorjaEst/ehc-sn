"""Evaluation experiment configuration loading and resolution.

Provides the single public entry point for loading a TOML evaluation
config and resolving it to an :class:`EvaluationExperiment`.

Usage::

    from ehc_sn.evaluation.configuration import load_evaluation_experiment

    loaded = load_evaluation_experiment(
        Path("evaluation.toml"),
    )
    # loaded.experiment      → EvaluationExperiment
    # loaded.resolved.alias  → "arena-tem-v1"
    # loaded.config_digest   → "a1b2c3d4e5f6g7h8"

Boundary rules:
    - Must not import from ``experiments/<task>/<family>/`` directly.
      Dispatches through ``ehp_sn.evaluation.recipes``.
    - Must not be re-exported from ``evaluation/__init__.py``.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import fields as _dataclass_fields
from dataclasses import is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ehc_sn.evaluation.contracts import EvaluationExperiment
from ehc_sn.evaluation.invocation import (
    EvaluationBuildRequest,
    EvaluationInvocationConfig,
    EvaluationOptions,
    EvaluationRuntimeConfig,
    InspectionConfig,
    OutputConfig,
    ResolvedCapturePlan,
    ResolvedCaseSelection,
    ResolvedDatasetMetadata,
    ResolvedModelArtifact,
    TrackingConfig,
    ValidationConfig,
    merge_evaluation_config,
)
from ehc_sn.evaluation.recipe_catalogue import RecipeCatalogueError
from ehc_sn.evaluation.recipes import (
    EvaluationAlias,
    EvaluationRecipe,
    list_recipes,
    resolve_recipe,
)


# =============================================================================
class EvaluationConfigurationError(ValueError):
    """Raised when evaluation configuration is invalid, unreadable, or mismatched."""


class IncompatibleModelError(ValueError):
    """Raised when a model artifact is incompatible with the selected recipe."""


def validate_compatibility(
    recipe: EvaluationRecipe,
    model: ResolvedModelArtifact,
) -> None:
    """Verify that the resolved model artifact is compatible with the recipe.

    Checks ``model_family`` match and that the artifact's capabilities
    satisfy the recipe's required capabilities.

    Parameters
    ----------
    recipe:
        Resolved recipe with ``config.model_family`` and
        ``config.required_capabilities``.
    model:
        Resolved model artifact with manifest metadata.

    Raises
    ------
    IncompatibleModelError
        If the artifact's model family does not match the recipe, or if
        required capabilities are missing.
    """
    if model.model_family is not None:
        recipe_family = (
            recipe.config.model_family.value
            if hasattr(recipe.config.model_family, "value")
            else str(recipe.config.model_family)
        )
        if model.model_family != recipe_family:
            raise IncompatibleModelError(
                f"Recipe requires model_family={recipe_family!r}, "
                f"but artifact declares model_family={model.model_family!r}."
            )

    if model.capabilities is not None and recipe.config.required_capabilities:
        required = set(
            c.value if hasattr(c, "value") else str(c)
            for c in recipe.config.required_capabilities
        )
        missing = required - model.capabilities
        if missing:
            raise IncompatibleModelError(
                f"Model artifact is missing required capabilities: "
                f"{sorted(missing)}.  "
                f"Artifact capabilities: {sorted(model.capabilities)}."
            )


@dataclass(frozen=True)
class ResolvedEvaluationInvocation:
    """Fully merged and validated evaluation request — the single resolution boundary.

    Produced by ``resolve_evaluation_invocation()`` in this module.
    Every field is already resolved: recipe defaults → invocation TOML → CLI
    overrides have been applied, pair-specific options have been validated,
    and the model artifact has been identified.

    Builders, the CLI, MLflow tracking, and output persistence all consume
    this object rather than reaching back into raw schemas.
    """

    alias: EvaluationAlias
    recipe: EvaluationRecipe
    model: ResolvedModelArtifact
    cases: ResolvedCaseSelection
    runtime: EvaluationRuntimeConfig
    evaluation: EvaluationOptions
    capture: ResolvedCapturePlan
    inspection: InspectionConfig
    validation: ValidationConfig
    output: OutputConfig
    tracking: TrackingConfig


# =============================================================================
# Provenance projection — ResolvedEvaluationInvocation → TOML-safe primitive dict
# =============================================================================

TomlScalar = str | int | float | bool | date | datetime
TomlValue = TomlScalar | list["TomlValue"] | dict[str, "TomlValue"]


def _to_toml_value(value: Any, *, path: str = "$") -> TomlValue:
    """Recursively project *value* into a TOML-compatible primitive tree.

    Parameters
    ----------
    value:
        Any Python value to convert.
    path:
        Dotted field path for error reporting.

    Returns
    -------
    TomlValue
        A tree of only ``str``, ``int``, ``float``, ``bool``,
        ``date``, ``datetime``, ``list``, and ``dict`` values.

    Raises
    ------
    TypeError
        If *value* or any of its children is not representable as a
        TOML value.  The message includes the full field path and the
        qualified type name.
    """
    # --- Enums: unwrap to .value --------------------------------------------
    if isinstance(value, Enum):
        return _to_toml_value(value.value, path=path)

    # --- Path: convert to string --------------------------------------------
    if isinstance(value, Path):
        return str(value)

    # --- Type / class / callable: convert to string -------------------------
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        qual = (
            getattr(value, "__qualname__", None)
            or getattr(value, "__name__", None)
            or str(value)
        )
        mod = getattr(value, "__module__", None)
        return f"{mod}.{qual}" if mod else qual

    # --- None: reject (TOML has no null) ------------------------------------
    if value is None:
        raise TypeError(
            f"{path}: None is not representable in TOML; "
            "omit the field or define an explicit encoding."
        )

    # --- TOML scalars: pass through -----------------------------------------
    if isinstance(value, (str, int, float, bool, date, datetime)):
        return value

    # --- Dataclass: project field by field ----------------------------------
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, TomlValue] = {}
        for f in _dataclass_fields(value):
            v = getattr(value, f.name)
            if v is None:
                continue
            result[f.name] = _to_toml_value(v, path=f"{path}.{f.name}")
        return result

    # --- Pydantic BaseModel: dump then recurse into the dict ----------------
    if isinstance(value, BaseModel) and not isinstance(value, type):
        raw = value.model_dump(mode="python", exclude_none=True)
        return _to_toml_value(raw, path=path)

    # --- Mapping: recurse into items ----------------------------------------
    if isinstance(value, Mapping):
        result: dict[str, TomlValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: TOML table keys must be strings, "
                    f"got {type(key).__name__}."
                )
            result[key] = _to_toml_value(item, path=f"{path}.{key}")
        return result

    # --- List / tuple: recurse into elements --------------------------------
    if isinstance(value, (list, tuple)):
        return [
            _to_toml_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    # --- Sets / frozensets: convert to sorted list --------------------------
    if isinstance(value, (set, frozenset)):
        return [
            _to_toml_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(sorted(value, key=str))
        ]

    # --- Opaque runtime objects: represent by qualified class name ----------
    # Objects like ModelArtifact instances are intentionally not
    # decomposable; their qualified name in provenance is stable and
    # informative.
    return f"{type(value).__module__}.{type(value).__qualname__}"


def invocation_to_provenance(
    invocation: ResolvedEvaluationInvocation,
) -> dict[str, TomlValue]:
    """Project a resolved invocation into a TOML-safe primitive dictionary.

    The returned dict contains only types that ``tomli_w.dumps()`` can
    accept: nested ``dict``, ``list``, ``str``, ``int``, ``float``,
    ``bool``, ``date``, and ``datetime``.

    Parameters
    ----------
    invocation:
        Fully resolved evaluation invocation.

    Returns
    -------
    dict[str, TomlValue]
        TOML-safe primitive tree.
    """
    result: dict[str, TomlValue] = {}
    for f in _dataclass_fields(invocation):
        v = getattr(invocation, f.name)
        if v is None:
            continue
        result[f.name] = _to_toml_value(v, path=f"$.{f.name}")
    return result


@dataclass(frozen=True)
class LoadedEvaluationExperiment:
    """A resolved evaluation experiment with configuration provenance.

    Attributes
    ----------
    experiment:
        Fully resolved experiment returned by the experiment-specific builder.
    resolved:
        The fully resolved and merged evaluation invocation.
    config_path:
        Resolved absolute path to the configuration file.
    config_digest:
        First 16 hex characters of the SHA-256 digest of the configuration
        file bytes.  Deterministic — same file always produces the same value.
    """

    experiment: EvaluationExperiment
    resolved: ResolvedEvaluationInvocation
    config_path: Path
    config_digest: str


# =============================================================================
def _recipe_defaults(
    recipe: EvaluationRecipe,
) -> dict[str, object]:
    """Construct a default invocation mapping from a recipe's TOML fields.

    Recipe top-level fields that overlap with invocation fields are the
    merge base.  Pydantic field defaults from ``EvaluationInvocationConfig``
    provide neutral structural fallbacks for fields not covered by the recipe.

    The merge order after this function is:
        recipe defaults → TOML file values → CLI overrides
    """
    from ehc_sn.evaluation.invocation import (
        CaptureConfig,
        CaseSelectionConfig,
        EvaluationRuntimeConfig,
        InspectionConfig,
        OutputConfig,
    )

    base: dict[str, object] = {
        "alias": recipe.config.alias,
        "cases": dict(CaseSelectionConfig().model_dump()),
        "runtime": dict(EvaluationRuntimeConfig().model_dump()),
        "capture": dict(CaptureConfig().model_dump()),
        "inspection": dict(InspectionConfig().model_dump()),
        "output": dict(OutputConfig().model_dump()),
        "tracking": {"experiment": "ehp-evaluation"},
        "evaluation": {},
    }

    # Overlay recipe TOML fields on top of neutral Pydantic defaults.
    recipe_fields = recipe.config.model_dump(
        mode="python",
        include={"cases", "evaluation", "capture", "inspection", "precision"},
        exclude_none=True,
    )
    for key, value in recipe_fields.items():
        if value is not None:
            base[key] = value

    # Promote precision to runtime if set.
    if recipe.config.precision is not None:
        base.setdefault("runtime", {})["precision"] = (  # type: ignore[index]
            recipe.config.precision
        )

    return base


def resolve_evaluation_invocation(
    *,
    recipe: EvaluationRecipe,
    invocation: EvaluationInvocationConfig,
    model: ResolvedModelArtifact,
) -> ResolvedEvaluationInvocation:
    """Merge, validate, and resolve a full evaluation invocation.

    Applies pair-specific evaluation-option validation and constructs the
    resolved case-selection and capture-plan objects.

    Parameters
    ----------
    recipe:
        Resolved recipe (config + binding).
    invocation:
        Fully merged and validated invocation config.
    model:
        Resolved model artifact reference.

    Returns
    -------
    ResolvedEvaluationInvocation
        Fully resolved invocation with all validation complete.
    """
    # ---- Validate pair-specific evaluation options --------------------------
    try:
        evaluation_options = recipe.binding.options_type.model_validate(
            dict(invocation.evaluation) if invocation.evaluation else {}
        )
    except ValidationError as exc:
        raise EvaluationConfigurationError(
            f"Invalid evaluation options for {recipe.config.alias!r}: {exc}"
        ) from exc

    # ---- Validate model-recipe compatibility --------------------------------
    validate_compatibility(recipe, model)

    # ---- Resolve cases ------------------------------------------------------
    resolved_cases = ResolvedCaseSelection(
        dataset=ResolvedDatasetMetadata(
            name=(
                invocation.cases.dataset.uri
                if invocation.cases.dataset
                else "default"
            ),
            version="unknown",
        ),
        split=invocation.cases.split.value,
        case_ids=tuple(invocation.cases.ids),
        count=invocation.cases.count,
    )

    # ---- Resolve capture ----------------------------------------------------
    resolved_capture = ResolvedCapturePlan(
        profile=invocation.capture.profile or "metrics_only",
        fields=tuple(invocation.capture.include),
        profile_version=invocation.capture.profile_version or 1,
        exclude=tuple(invocation.capture.exclude),
        max_cases=invocation.capture.max_cases,
        max_steps_per_case=invocation.capture.max_steps_per_case,
        storage_budget_bytes=invocation.capture.storage_budget_bytes,
    )

    return ResolvedEvaluationInvocation(
        alias=EvaluationAlias(invocation.alias),
        recipe=recipe,
        model=model,
        cases=resolved_cases,
        runtime=invocation.runtime,
        evaluation=evaluation_options,
        capture=resolved_capture,
        inspection=invocation.inspection,
        validation=invocation.validation,
        output=invocation.output,
        tracking=invocation.tracking,
    )


def build_experiment_from_invocation(
    *,
    resolved: ResolvedEvaluationInvocation,
    config_path: Path | None = None,
) -> LoadedEvaluationExperiment:
    """Build a ``LoadedEvaluationExperiment`` from a resolved invocation.

    Constructs the ``EvaluationBuildRequest``, calls the pair-specific
    builder, and wraps the result with provenance metadata.

    This function does not validate evaluation options, resolve cases, or
    resolve capture profiles — those steps belong to
    ``resolve_evaluation_invocation()``.

    Parameters
    ----------
    resolved:
        Fully resolved evaluation invocation.
    config_path:
        Optional path to the config file for digest computation.  ``None``
        when the invocation was constructed from CLI flags without a file.

    Returns
    -------
    LoadedEvaluationExperiment
        Resolved experiment with configuration provenance.
    """
    # ---- Build request ------------------------------------------------------
    build_request = EvaluationBuildRequest(
        cases=resolved.cases,
        runtime=resolved.runtime,
        options=resolved.evaluation,
        capture=resolved.capture,
        model_artifact=resolved.model,
    )

    # ---- Build experiment ---------------------------------------------------
    experiment = resolved.recipe.binding.builder(build_request)

    # ---- Compute config digest ----------------------------------------------
    if config_path is not None:
        resolved_path = config_path.resolve()
        config_digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()[
            :16
        ]
    else:
        resolved_path = Path()
        config_digest = ""

    return LoadedEvaluationExperiment(
        experiment=experiment,
        resolved=resolved,
        config_path=resolved_path,
        config_digest=config_digest,
    )


def load_evaluation_experiment(
    config_path: Path,
    *,
    expected_alias: str | None = None,
    model: ResolvedModelArtifact | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> LoadedEvaluationExperiment:
    """Load and resolve a TOML evaluation config to an ``EvaluationExperiment``.

    When ``cli_overrides`` is provided, the merge pipeline runs:
    recipe/Pydantic defaults → TOML file values → CLI overrides.
    CLI ``None`` values are treated as "not supplied" and do not override.

    Parameters
    ----------
    config_path:
        Path to the TOML evaluation configuration file.
    expected_alias:
        If provided, the config's ``alias`` value must match.  Used
        to catch misconfiguration.
    model:
        Pre-resolved model artifact.  Required when ``cli_overrides``
        are provided; may be omitted for direct file-only validation.
    cli_overrides:
        Optional sparse dict of CLI flag values to merge on top of
        file values.

    Returns
    -------
    LoadedEvaluationExperiment
        Resolved experiment with configuration provenance.
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

    # ---- Extract alias -----------------------------------------------------
    alias_str = config_map.get("alias")
    if not isinstance(alias_str, str) or not alias_str.strip():
        raise EvaluationConfigurationError(
            f"Config {resolved_path} must declare a non-empty string 'alias'."
        )
    alias_str = alias_str.strip()

    # ---- Validate expected identity -----------------------------------------
    if expected_alias is not None and alias_str != expected_alias:
        raise EvaluationConfigurationError(
            f"Expected alias={expected_alias!r}, "
            f"but {resolved_path} declares {alias_str!r}."
        )

    # ---- Resolve recipe (catalogue + binding) -------------------------------
    try:
        alias_id = EvaluationAlias(alias_str)
        recipe = resolve_recipe(alias_id)
    except (ValueError, RecipeCatalogueError) as exc:
        raise EvaluationConfigurationError(
            f"Unknown alias {alias_str!r}. "
            f"Available: {sorted(a.value for a in list_recipes())}."
        ) from exc

    # ---- Merge: recipe TOML defaults → file values → CLI overrides ----------
    if cli_overrides is not None:
        if model is None:
            raise EvaluationConfigurationError(
                "model must be provided when cli_overrides are supplied."
            )
        recipe_defaults = _recipe_defaults(recipe)
        invocation = merge_evaluation_config(
            recipe_defaults=recipe_defaults,
            file_values=config_map,
            cli_overrides=cli_overrides,
        )
        resolved = resolve_evaluation_invocation(
            recipe=recipe,
            invocation=invocation,
            model=model,
        )
    else:
        # Direct file-only validation (no CLI merge needed).
        try:
            invocation = EvaluationInvocationConfig.model_validate(config_map)
        except ValidationError as exc:
            raise EvaluationConfigurationError(
                f"Invalid invocation config for alias {alias_str!r} "
                f"in {resolved_path}: {exc}"
            ) from exc
        resolved = resolve_evaluation_invocation(
            recipe=recipe,
            invocation=invocation,
            model=model
            or ResolvedModelArtifact(
                requested_uri="",
                resolved_source="",
            ),
        )

    return build_experiment_from_invocation(
        resolved=resolved,
        config_path=resolved_path,
    )


# =============================================================================
__all__ = [
    "EvaluationConfigurationError",
    "IncompatibleModelError",
    "LoadedEvaluationExperiment",
    "build_experiment_from_invocation",
    "load_evaluation_experiment",
    "resolve_evaluation_invocation",
    "validate_compatibility",
]
