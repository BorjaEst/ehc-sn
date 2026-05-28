"""Offline report-figure rendering from persisted evaluation regime artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, model_validator

from ehc_sn.eval.artifacts import (
    LoadedArtifactCase,
    load_artifact_run_cases,
)
from ehc_sn.figures import REGISTRY, FigureContext, list_figures, render
from ehc_sn.figures.sinks import _persist_named_figure_artifacts


# =============================================================================
class OfflineReportFigureContextSettings(BaseModel, extra="forbid"):
    """Report-surface figure context controls for post-hoc rendering."""

    sample_idx: int = Field(
        default=0,
        ge=0,
        description="Starting sample index for report templates that support it.",
    )
    max_items: int | None = Field(
        default=None,
        ge=1,
        description="Optional item cap for report templates that support it.",
    )


# =============================================================================
class OfflineReportFigureEntrySettings(BaseModel, extra="forbid"):
    """Per-figure rendering entry for one offline report run."""

    figure: str = Field(
        ...,
        min_length=1,
        description="One registered report figure name.",
    )
    case_ids: list[str] = Field(
        default_factory=list,
        description="Optional explicit case IDs to render, in deterministic order.",
    )
    max_cases: int = Field(
        default=8,
        ge=1,
        description="Maximum cases rendered when case_ids is not provided.",
    )
    context: OfflineReportFigureContextSettings = Field(
        default_factory=OfflineReportFigureContextSettings,
        description="Figure-context overrides applied for this figure entry.",
    )


# =============================================================================
class OfflineReportRenderSettings(BaseModel, extra="forbid"):
    """TOML settings for one offline report-figure rendering run."""

    run_dir: Path = Field(
        ...,
        description="Path to one persisted eval regime run directory.",
    )
    output_dir: Path | None = Field(
        default=None,
        description="Optional output directory; defaults to <run_dir>/report_figures.",
    )
    entries: list[OfflineReportFigureEntrySettings] = Field(
        default_factory=list,
        description="Per-figure report entries with independent selection/context.",
    )
    save_pdf: bool = Field(
        default=True,
        description="Persist rendered report figures as PDF files.",
    )
    save_png: bool = Field(
        default=False,
        description="Persist rendered report figures as PNG files.",
    )
    png_dpi: int = Field(
        default=160,
        ge=1,
        description="PNG export DPI used when save_png is enabled.",
    )

    @model_validator(mode="after")
    def _validate_report_surface(self) -> "OfflineReportRenderSettings":
        """Validate report-only rendering surface and sink choices."""
        if not self.entries:
            raise ValueError("entries must not be empty.")
        if not self.save_pdf and not self.save_png:
            raise ValueError(
                "At least one output sink must be enabled: save_pdf or save_png."
            )

        available = set(list_figures())
        missing = [
            entry.figure
            for entry in self.entries
            if entry.figure not in available
        ]
        if missing:
            unknown = ", ".join(sorted(set(missing)))
            raise ValueError(f"Unknown figures: {unknown}.")

        non_report = [
            entry.figure
            for entry in self.entries
            if REGISTRY.get(entry.figure).kind != "report"
        ]
        if non_report:
            blocked = ", ".join(sorted(set(non_report)))
            raise ValueError(
                "Offline report rendering only supports report-kind figures: "
                f"{blocked}."
            )

        bounded = [
            entry.figure
            for entry in self.entries
            if REGISTRY.get(entry.figure).input_contract == "bounded_trace"
        ]
        if bounded:
            blocked = ", ".join(sorted(set(bounded)))
            raise ValueError(
                "Offline report rendering does not support bounded-trace "
                f"figures: {blocked}."
            )
        return self


# =============================================================================
def render_report_figures_from_run(
    settings: OfflineReportRenderSettings,
) -> dict[str, int]:
    """Render report-kind figures for one persisted regime run."""
    loaded_cases = load_artifact_run_cases(settings.run_dir)

    output_dir = settings.output_dir or (
        Path(settings.run_dir) / "report_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    rendered_case_ids: set[str] = set()
    for entry in settings.entries:
        selected_cases = _select_cases(
            loaded_cases,
            case_ids=entry.case_ids,
            max_cases=entry.max_cases,
        )
        rendered_case_ids.update(case.case_id for case in selected_cases)

        ctx = FigureContext(
            sample_idx=entry.context.sample_idx,
            max_items=entry.context.max_items,
            split_name=Path(settings.run_dir).name,
        )
        spec = REGISTRY.get(entry.figure)

        for idx, case in enumerate(selected_cases):
            fig = render(
                entry.figure,
                case.trace,
                ctx,
                temporal_semantics=case.temporal_semantics,
            )
            try:
                _persist_named_figure_artifacts(
                    fig,
                    output_dir=output_dir,
                    case_index=idx,
                    case_id=case.case_id,
                    default_filename=spec.default_filename,
                    save_pdf_enabled=settings.save_pdf,
                    save_png_enabled=settings.save_png,
                    png_dpi=settings.png_dpi,
                )
            finally:
                plt.close(fig)
            rendered += 1

    return {
        "n_cases_loaded": len(loaded_cases),
        "n_cases_rendered": len(rendered_case_ids),
        "n_entries_rendered": len(settings.entries),
        "n_figures_rendered": rendered,
    }


# =============================================================================
# =============================================================================
def render_case(
    figure: str,
    case: LoadedArtifactCase,
    *,
    ctx: FigureContext | None = None,
) -> mpl_figure.Figure:
    """Render one registered figure from a loaded evaluation artifact case.

    Args:
        figure: Registered figure name.
        case: Loaded artifact case.
        ctx: Optional figure context; defaults to ``FigureContext()``.

    Returns:
        Matplotlib ``Figure``.

    Raises:
        ValueError: If the case has no trace attached.
    """
    if case.trace is None:
        raise ValueError(
            f"Artifact case {case.case_id!r} has no trace; "
            f"cannot render figure {figure!r}."
        )
    return render(
        figure,
        trace=case.trace,
        ctx=ctx or FigureContext(),
        temporal_semantics=case.temporal_semantics,
    )


# =============================================================================
def render_report(
    artifact_run: str | Path,
    figures: Sequence[str],
    output_dir: str | Path,
    *,
    max_cases: int | None = None,
) -> dict[str, int]:
    """Render report-quality figures from a persisted evaluation artifact run.

    This is a convenience wrapper around
    :func:`render_report_figures_from_run` that builds the required settings
    object internally.

    Args:
        artifact_run: Path to one persisted eval regime run directory.
        figures: Registered report-kind figure names to render.
        output_dir: Output directory for persisted figure files.
        max_cases: Maximum cases rendered per figure entry.  Defaults to 8.

    Returns:
        Dict with keys: ``n_cases_loaded``, ``n_cases_rendered``,
        ``n_entries_rendered``, ``n_figures_rendered``.
    """
    entries = [
        OfflineReportFigureEntrySettings(
            figure=f,
            max_cases=max_cases or 8,
        )
        for f in figures
    ]
    settings = OfflineReportRenderSettings(
        run_dir=Path(artifact_run),
        output_dir=Path(output_dir),
        entries=entries,
    )
    return render_report_figures_from_run(settings)


# =============================================================================
def _select_cases(
    loaded_cases: list[LoadedArtifactCase],
    *,
    case_ids: list[str],
    max_cases: int,
) -> list[LoadedArtifactCase]:
    """Select cases deterministically from loaded persisted trace cases."""
    if case_ids:
        by_id = {case.case_id: case for case in loaded_cases}
        selected: list[LoadedArtifactCase] = []
        for case_id in case_ids:
            case = by_id.get(case_id)
            if case is None:
                raise ValueError(
                    "Requested case_id has no saved trace payload in this run: "
                    f"{case_id!r}."
                )
            selected.append(case)
        return selected
    return loaded_cases[:max_cases]


# =============================================================================
__all__ = [
    "OfflineReportFigureEntrySettings",
    "OfflineReportFigureContextSettings",
    "OfflineReportRenderSettings",
    "LoadedArtifactCase",
    "load_artifact_run_cases",
    "render_case",
    "render_report",
    "render_report_figures_from_run",
]
