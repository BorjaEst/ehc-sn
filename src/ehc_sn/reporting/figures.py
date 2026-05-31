"""Report figure rendering — Protocol, builder-path, and standalone-path.

This module owns all figure rendering for report assembly.  It provides:

1. :class:`ReportFigureRendererRegistry` — structural Protocol for injected
   renderers.
2. :class:`ReportFigureRenderer` — builder-path renderer that delegates to
   :func:`ehc_sn.eval.render.render_regime_preview_figures`.
3. Settings models for the standalone rendering path.
4. :func:`render_report_figures_from_run` — standalone-path rendering for
   CLI/diagnostics.

Usage (builder path)::

    from ehc_sn.reporting.figures import ReportFigureRenderer

    renderer = ReportFigureRenderer()
    build_report_run(spec, figure_renderers=renderer)

Usage (standalone path)::

    from ehc_sn.reporting.figures import (
        OfflineReportRenderSettings,
        render_report_figures_from_run,
    )

    settings = OfflineReportRenderSettings(run_dir=..., entries=[...])
    summary = render_report_figures_from_run(settings)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, model_validator

from ehc_sn.eval.artifacts import (
    LoadedArtifactCase,
    load_artifact_run_cases,
)
from ehc_sn.eval.contracts import EvaluationCaseResult
from ehc_sn.eval.render import render_regime_preview_figures
from ehc_sn.figures import REGISTRY, FigureContext, list_figures, render
from ehc_sn.figures.sinks import _persist_named_figure_artifacts
from ehc_sn.reporting.schema import (
    EvalArtifactReference,
    FigureIndex,
    FigureIndexEntry,
)

# ============================================================================
# Section 1 — Protocol (injection contract)
# ============================================================================


class ReportFigureRendererRegistry(Protocol):
    """Structural protocol for an injected figure-rendering registry.

    The builder calls ``render`` once per selected eval artifact during
    report assembly.  The implementation owns all figure dispatch, case
    selection, rendering, and persistence.

    Conforming objects need not inherit from this class — structural
    sub-typing is sufficient.
    """

    def render(
        self,
        *,
        artifact: EvalArtifactReference,
        output_dir: Path,
        requested_figures: Sequence[str],
        formats: Sequence[str] | None = None,
    ) -> FigureIndex:
        """Render report figures for one eval artifact.

        Parameters
        ----------
        artifact:
            Reference to the eval artifact to render figures from.
        output_dir:
            Target directory for figure files.  Guaranteed to exist
            when called (the builder creates it before invoking the
            renderer).
        requested_figures:
            Figure names the report spec asked for.
        formats:
            Optional list of output formats ("pdf", "png").
            ``None`` defaults to ``["pdf", "png"]``.

        Returns
        -------
        FigureIndex
            Typed manifest of rendered figures for this artifact.
        """
        ...


# ============================================================================
# Section 2 — Builder-path implementation
# ============================================================================


class ReportFigureRenderer:
    """Report-run figure renderer that delegates to the preview renderer.

    Conforms structurally to :class:`ReportFigureRendererRegistry` for
    injection into :func:`~ehc_sn.reporting.builder.build_report_run`.
    """

    def render(
        self,
        *,
        artifact: EvalArtifactReference,
        output_dir: Path,
        requested_figures: Sequence[str],
        formats: Sequence[str] | None = None,
    ) -> FigureIndex:
        """Render requested figures from one eval artifact.

        Parameters
        ----------
        artifact:
            Reference to the eval artifact to render figures from.
        output_dir:
            Target directory for figure files.  Created if needed.
        requested_figures:
            Figure names the report spec asked for.
        formats:
            Optional list of output formats (``\"pdf\"``, ``\"png\"``).
            ``None`` defaults to ``[\"pdf\", \"png\"]``.

        Returns
        -------
        FigureIndex
            Typed manifest of rendered figures for this artifact.
        """
        list_figures()

        task = artifact.task
        regime_id = artifact.regime_id

        case_results = _build_case_results(artifact.path)

        entries: list[FigureIndexEntry] = []
        for figure_name in requested_figures:
            spec = REGISTRY.get(figure_name)
            # Paradigm figures (task-context tag) render only 1 exemplar case.
            is_context = "task-context" in spec.tags if spec else False
            max_cases = 1 if is_context else 4

            actual_formats = formats if formats is not None else ["pdf", "png"]
            save_pdf = "pdf" in actual_formats
            save_png = "png" in actual_formats

            previews = render_regime_preview_figures(
                case_results=case_results,
                figure_names=[figure_name],
                figure_dir=output_dir,
                figure_ctx=FigureContext(),
                max_cases=max_cases,
                save_pdf=save_pdf,
                save_png=save_png,
            )

            for preview in previews:
                if preview.error is not None:
                    continue

                figure_stem = Path(spec.default_filename).stem

                for fmt in actual_formats:
                    filename = (
                        f"{preview.case_index:04d}-"
                        f"{_sanitize(preview.case_id)}-"
                        f"{_sanitize(figure_stem)}.{fmt}"
                    )
                    figure_path = output_dir / filename
                    if not figure_path.exists():
                        continue

                    entries.append(
                        FigureIndexEntry(
                            task=task,
                            regime_id=regime_id,
                            figure_id=preview.figure_name,
                            path=Path("figures") / Path(filename),
                            format=fmt,
                            title=preview.figure_name,
                            description=spec.description or "",
                        )
                    )

        return FigureIndex(entries=entries)


# ============================================================================
# Section 3 — Standalone-path settings
# ============================================================================


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
            if "report" not in REGISTRY.get(entry.figure).allowed_surfaces
        ]
        if non_report:
            blocked = ", ".join(sorted(set(non_report)))
            raise ValueError(
                "Offline report rendering does not allow these figures on "
                f"the report surface: {blocked}."
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


# ============================================================================
# Section 4 — Standalone-path rendering
# ============================================================================


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


# ============================================================================
# Section 5 — Index
# ============================================================================

_DEFAULT_FILENAME = "index.json"


def write_figure_index(
    index: FigureIndex,
    output_dir: Path,
    *,
    filename: str = _DEFAULT_FILENAME,
) -> Path:
    """Serialize *index* as JSON and write to *output_dir*.

    Parameters
    ----------
    index:
        Figure index to persist.
    output_dir:
        Target directory.  Created if it does not exist.
    filename:
        Output filename within *output_dir*.  Defaults to ``"index.json"``.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    raw = json.loads(index.model_dump_json(indent=2))
    output_path.write_text(
        json.dumps(raw, indent=2),
        encoding="utf-8",
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_figure_index(path: Path) -> FigureIndex:
    """Load a :class:`FigureIndex` from a JSON file at *path*.

    Parameters
    ----------
    path:
        Path to a ``figures/index.json`` file.

    Returns
    -------
    FigureIndex
        Parsed and validated figure index.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON is malformed.
    pydantic.ValidationError
        If the content does not conform to ``FigureIndex``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return FigureIndex.model_validate(raw)


# ============================================================================
# Section 7 — Internal helpers
# ============================================================================


def _build_case_results(
    artifact_path: Path,
) -> list[EvaluationCaseResult]:
    """Load eval-artifact cases as ``EvaluationCaseResult`` for the renderer."""
    cases = load_artifact_run_cases(artifact_path)
    return [
        EvaluationCaseResult(
            case_id=c.case_id,
            evaluated=None,
            source_context=c.source_context,
            trace=c.trace,
        )
        for c in cases
    ]


def _sanitize(value: str) -> str:
    """Normalize a string into a safe filename component."""
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "case"


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


# ----------------------------------------------------------------------------
__all__ = [
    "OfflineReportFigureContextSettings",
    "OfflineReportFigureEntrySettings",
    "OfflineReportRenderSettings",
    "ReportFigureRenderer",
    "ReportFigureRendererRegistry",
    "render_report_figures_from_run",
    "load_figure_index",
    "write_figure_index",
]
