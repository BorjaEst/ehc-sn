#!/usr/bin/env python3
"""CLI entry point for assembling a report run from a ReportSpec TOML.

Usage::

    python scripts/reporting/run_report.py \\
        --config config/reporting/tem_v1_arena_struct_val_8.toml \\
        [--output /path/to/output] \\
        [--no-metrics] \\
        [--no-figures]

Wires existing components — no metric extraction, figure registry, artifact
discovery, or report-writing logic lives here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ehc_sn.reporting import (
    ReportSpec,
    build_report_run,
    load_report_spec,
)
from ehc_sn.reporting.figures import ReportFigureRenderer
from ehc_sn.reporting.metrics import DEFAULT_MAZEHARD_NORMALIZER

# ---------------------------------------------------------------------------
# CLI settings
# ---------------------------------------------------------------------------


class ReportArguments(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    """CLI settings for assembling a report run from a ReportSpec TOML."""

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
        """CLI overrides all other sources."""
        extra = [
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ]
        return CliSettingsSource(settings_cls), *extra

    config: Path = Field(
        ...,
        description="Path to a ReportSpec TOML file.",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing report output (if any).",
    )
    output: Path | None = Field(
        default=None,
        description="Override the output directory from the spec.",
    )
    no_metrics: bool = Field(
        default=False,
        description="Skip metric normalizer injection.",
    )
    no_figures: bool = Field(
        default=False,
        description="Skip figure renderer injection.",
    )


def main() -> None:
    settings = ReportArguments()

    # Load spec.
    spec: ReportSpec = load_report_spec(settings.config)

    # Apply output override via model_copy (no mutation).
    if settings.output is not None:
        spec = spec.model_copy(update={"output_dir": settings.output})

    # Metric normalizers.
    metric_normalizers: object = None
    if not settings.no_metrics:
        metric_normalizers = DEFAULT_MAZEHARD_NORMALIZER

    # Figure renderers — use the canonical report figure renderer.
    figure_renderers: object = None
    if spec.figures.figures and not settings.no_figures:
        figure_renderers = ReportFigureRenderer()

    # Assemble.
    report = build_report_run(
        spec,
        overwrite=settings.overwrite,
        metric_records=None,
        metric_normalizers=metric_normalizers,
        figure_renderers=figure_renderers,
    )

    print(f"Report run written to: {report.root.resolve()!s}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
