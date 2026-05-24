"""Thin CLI wrapper for the Arena-Struct benchmark track."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ehc_sn.benchmarks import (
    build_track_report_from_seed_scores,
    run_ready_track_model_comparison_seeds,
    write_track_report,
)
from ehc_sn.benchmarks.contracts import (
    parse_artifact_manifest,
    parse_track_recipe,
)

_DEFAULT_TRACK_ID = "arena-struct"
_DEFAULT_CONFIG_PATH = "config/benchmarks/arena-struct.toml"


# =============================================================================
# Settings Model
# =============================================================================
class RunArguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    """Script-local CLI/config settings for the Arena-Struct benchmark wrapper."""

    model_config = SettingsConfigDict(extra="forbid")

    @classmethod
    def settings_customise_sources(  # ----------------------------------------
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customise settings sources to prioritize CLI args over env vars and defaults."""
        extra = [
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ]
        return CliSettingsSource(settings_cls), *extra

    config: Path = Field(
        default=Path(_DEFAULT_CONFIG_PATH),
        description="Path to the benchmark wrapper TOML defaults file.",
    )
    track: str | None = Field(
        default=None,
        description="Optional benchmark track override; defaults to the wrapper's canonical track.",
    )
    manifest: Path | None = Field(
        default=None,
        description="Path to benchmark artifact manifest TOML.",
    )
    recipe: Path | None = Field(
        default=None,
        description="Path to benchmark track recipe TOML.",
    )
    fixed_recipe: str | None = Field(
        default=None,
        description="Fixed-recipe label recorded in the benchmark report.",
    )
    seed_count: int | None = Field(
        default=None,
        description="Number of independent seeds represented by this report payload.",
    )
    ood_slice: str | None = Field(
        default=None,
        description="Optional OOD or evaluation-slice tag emitted with the report.",
    )
    output: Path | None = Field(
        default=None,
        description="Optional output JSON path; defaults to reports/benchmarks/<track>-<model>.json.",
    )


# =============================================================================
# Main Entrypoint
# =============================================================================
def main() -> None:
    raise SystemExit(_run())


# =============================================================================
def _run() -> int:
    bootstrap = RunArguments()
    defaults = _load_defaults(bootstrap.config)
    settings = RunArguments(**defaults)

    track = settings.track or _DEFAULT_TRACK_ID
    manifest_path = settings.manifest
    recipe_path = settings.recipe
    seed_count = _resolve_seed_count(
        settings.seed_count
        if settings.seed_count is not None
        else defaults.get("seed_count")
    )
    ood_slice = settings.ood_slice

    if manifest_path is None:
        raise ValueError("--manifest is required unless provided by defaults.")
    if recipe_path is None:
        raise ValueError("--recipe is required unless provided by defaults.")

    manifest = parse_artifact_manifest(manifest_path)
    recipe = parse_track_recipe(recipe_path)
    seed_scores = run_ready_track_model_comparison_seeds(
        manifest,
        recipe,
        seed_count=seed_count,
    )

    track = str(track or recipe.track_id)
    model_family = str(manifest.model_family)
    fixed_recipe = (
        str(settings.fixed_recipe)
        if settings.fixed_recipe is not None
        else str(recipe.reporting.fixed_recipe)
    )

    report = build_track_report_from_seed_scores(
        track,
        model_family,
        seed_scores,
        fixed_recipe=fixed_recipe,
        ood_slice=None if ood_slice is None else str(ood_slice),
    )

    canonical_track = str(report["track_id"])
    canonical_model = str(report["model_family"])
    output_path = (
        settings.output
        if settings.output is not None
        else Path("reports")
        / "benchmarks"
        / f"{canonical_track}-{canonical_model}.json"
    )
    write_track_report(report, output_path)
    return 0


# =============================================================================
def _load_defaults(  # --------------------------------------------------------
    path: Path,
) -> dict[str, Any]:
    """Load benchmark defaults from a TOML file at the given path, returning a
    dict of settings values. Raise a fast, explicit error if the file is
    missing or invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Benchmark defaults file not found: {path}")
    with path.open("rb") as file_obj:
        data = tomllib.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(
            f"Benchmark defaults at {path} must decode to a table/object."
        )
    return data


# =============================================================================
def _resolve_seed_count(  # ---------------------------------------------------
    raw_value: object,
) -> int:
    """Resolve a raw seed_count value from defaults or CLI, coercing to an int
    and providing a default of 1 if the value is None. Raise a fast, explicit
    error if the value is invalid.
    """
    if raw_value is None:
        return 1
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"seed_count must be an integer, got {raw_value!r}."
        ) from exc


# =============================================================================
if __name__ == "__main__":
    main()
