"""Legacy B0 alias wrapper for the MazeHard-Delib benchmark track."""

from __future__ import annotations

import json
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
    build_score_report,
    build_track_report,
    write_track_report,
)

_DEFAULT_TRACK_ID = "b0"
_DEFAULT_CONFIG_PATH = "config/benchmarks/b0-mazehard.toml"


# =============================================================================
# Settings Model
# =============================================================================
class RunArguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    """Script-local CLI/config settings for the legacy B0 compatibility wrapper."""

    model_config = SettingsConfigDict(extra="forbid")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
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
        description="Optional benchmark track override; defaults to the legacy "
        "B0 alias track.",
    )
    model_family: str | None = Field(
        default=None,
        description="Model family identifier to evaluate (for example hrm-v1, "
        "hrm-v2, ehp-v1).",
    )
    score_json: Path | None = Field(
        default=None,
        description="Path to score payload JSON consumed by benchmark score "
        "coercion.",
    )
    fixed_recipe: str = Field(
        default="default",
        description="Fixed-recipe label recorded in the benchmark report.",
    )
    seed_count: int | None = Field(
        default=None,
        description="Number of independent seeds represented by this report "
        "payload.",
    )
    ood_slice: str | None = Field(
        default=None,
        description="Optional OOD or evaluation-slice tag emitted with the "
        "report.",
    )
    output: Path | None = Field(
        default=None,
        description="Optional output JSON path; defaults to "
        "artifacts/benchmarks/<track>-<model>.json.",
    )


# =============================================================================
# Main Entrypoint
# =============================================================================
def main() -> None:
    raise SystemExit(_run())


def _load_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark defaults file not found: {path}")
    with path.open("rb") as file_obj:
        data = tomllib.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(
            f"Benchmark defaults at {path} must decode to a table/object."
        )
    return data


def _resolve_seed_count(raw_value: object) -> int:
    if raw_value is None:
        return 1
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"seed_count must be an integer, got {raw_value!r}."
        ) from exc


def _run() -> int:
    bootstrap = RunArguments()
    defaults = _load_defaults(bootstrap.config)
    settings = RunArguments(**defaults)

    track = settings.track or str(defaults.get("track") or _DEFAULT_TRACK_ID)
    model_family = settings.model_family
    score_json_path = settings.score_json
    fixed_recipe = str(settings.fixed_recipe)
    seed_count = _resolve_seed_count(
        settings.seed_count
        if settings.seed_count is not None
        else defaults.get("seed_count")
    )
    ood_slice = settings.ood_slice

    if model_family is None:
        raise ValueError(
            "--model-family is required unless provided by defaults."
        )
    if score_json_path is None:
        raise ValueError(
            "--score-json is required unless provided by defaults."
        )

    payload = json.loads(score_json_path.read_text(encoding="utf-8"))
    score_report = build_score_report(track, payload)
    report = build_track_report(
        track,
        str(model_family),
        score_report,
        primary_metric="sequences_exact",
        fixed_recipe=fixed_recipe,
        seed_count=seed_count,
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


if __name__ == "__main__":
    main()
