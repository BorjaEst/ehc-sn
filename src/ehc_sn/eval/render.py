"""Lightning-agnostic regime preview figure rendering.

Extracts the rendering loop from
:class:`~ehc_sn.callbacks.evaluation.EvaluationRegimesCallback` so that the
callback owns only policy (scheduling, failure handling, logging) and the
rendering logic is a pure function in the ``eval`` package.

Usage::

    from ehc_sn.eval.render import (
        RenderedPreview,
        render_regime_preview_figures,
    )

    previews = render_regime_preview_figures(
        case_results=regime_result.case_results,
        figure_names=["mazehard_prediction_evolution"],
        figure_dir=Path("/output/figures"),
        max_cases=2,
        save_pdf=True,
    )
    for preview in previews:
        if preview.error is not None:
            ...  # caller-owned failure policy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from ehc_sn.eval.contracts import EvaluationCaseResult
from ehc_sn.figures import REGISTRY, FigureContext, render
from ehc_sn.figures.sinks import _persist_named_figure_artifacts


# =============================================================================
@dataclass(frozen=True)
class RenderedPreview:
    """Result of attempting to render one figure from one case trace.

    A successful render has ``error=None``.  A failed render has ``error``
    set to the string representation of the exception.

    Entries are only produced for cases whose ``trace`` attribute is not
    ``None``.
    """

    figure_name: str
    case_id: str
    case_index: int
    error: str | None = None


# =============================================================================
def render_regime_preview_figures(
    *,
    case_results: Sequence[EvaluationCaseResult],
    figure_names: list[str],
    figure_dir: Path,
    figure_ctx: FigureContext | None = None,
    max_cases: int = 2,
    save_pdf: bool = True,
    save_png: bool = False,
    png_dpi: int = 160,
) -> list[RenderedPreview]:
    """Render diagnostic preview figures from regime case traces.

    Each (case, figure_name) pair is attempted independently.  Failures are
    captured in ``RenderedPreview.error`` — the caller owns the failure policy.

    Args:
        case_results: Regime case results; each may have ``trace`` set.
        figure_names: Registered figure names to render.  Must exist in
            ``REGISTRY`` or a ``ValueError`` is raised immediately.
        figure_dir: Output directory for persisted figure files.  Created
            if missing.
        figure_ctx: Render context forwarded to ``render()``.
            Defaults to ``FigureContext()``.
        max_cases: Maximum number of case results to attempt.  Must be >= 1.
        save_pdf: Whether to persist figures as PDF.
        save_png: Whether to persist figures as PNG.
        png_dpi: PNG export DPI when ``save_png`` is ``True``.

    Returns:
        One ``RenderedPreview`` per attempted (case, figure) pair.  Does
        not raise for individual rendering failures.

    Raises:
        ValueError: If any ``figure_name`` is not in ``REGISTRY``, or if
            ``max_cases < 1``.
    """
    if max_cases < 1:
        raise ValueError(f"max_cases must be >= 1, got {max_cases}.")

    _validate_figure_names(figure_names)

    if figure_ctx is None:
        figure_ctx = FigureContext()

    figure_dir.mkdir(parents=True, exist_ok=True)

    results: list[RenderedPreview] = []
    selected = case_results[:max_cases]

    for idx, result in enumerate(selected):
        if result.trace is None:
            continue
        for name in figure_names:
            spec = REGISTRY.get(name)
            fig: plt.Figure | None = None
            try:
                fig = render(name, result.trace, figure_ctx)
                _persist_named_figure_artifacts(
                    fig,
                    output_dir=figure_dir,
                    case_index=idx,
                    case_id=result.case_id,
                    default_filename=spec.default_filename,
                    save_pdf_enabled=save_pdf,
                    save_png_enabled=save_png,
                    png_dpi=png_dpi,
                )
                results.append(
                    RenderedPreview(
                        figure_name=name,
                        case_id=result.case_id,
                        case_index=idx,
                    )
                )
            except Exception as exc:
                results.append(
                    RenderedPreview(
                        figure_name=name,
                        case_id=result.case_id,
                        case_index=idx,
                        error=str(exc),
                    )
                )
            finally:
                if fig is not None:
                    plt.close(fig)

    return results


# =============================================================================
def _validate_figure_names(figure_names: list[str]) -> None:
    """Check that all figure names are registered; raise ``ValueError`` if not.

    This is a separate helper so the validation can run before any filesystem
    or rendering work begins.
    """
    for name in figure_names:
        if name not in REGISTRY:
            raise ValueError(
                f"Figure name {name!r} is not registered. "
                f"Available: {sorted(REGISTRY.list())!r}."
            )


# =============================================================================
__all__ = ["RenderedPreview", "render_regime_preview_figures"]
