"""Figure diagnostics engine — machine-generated checks for rendered figures.

The engine runs post-layout checks on a ``matplotlib.figure.Figure`` and
returns a list of ``Diagnostic`` results.  Each diagnostic has a level
(``FAIL``, ``WARN``, ``INFO``), a code string, a human-readable message,
and an optional location label.

Checks implemented here:

D1 — text_overlap (FAIL): two text artists with intersecting display bboxes.
D2 — artist_outside_figure (FAIL): artist extends beyond figure canvas.
D4 — small_text (WARN): text artist smaller than minimum font size.
D5 — inconsistent_panel_width (WARN): input↔target width deviation > 1%.
D6 — inconsistent_panel_height (WARN): input↔target height deviation > 1%.
D7 — inconsistent_data_extent (WARN): panel x/y limits differ when they
     should be equal.
D11 — hidden_data (WARN): data image extends beyond axes limits (clipping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DiagnosticLevel: type = Literal["FAIL", "WARN", "INFO"]


# =============================================================================
@dataclass(frozen=True)
class Diagnostic:
    """One diagnostic result for a rendered figure.

    Attributes:
        level: Severity — ``FAIL`` (guaranteed problem), ``WARN``
            (likely problem or readability concern), ``INFO``
            (informational).
        code: Machine-readable short string (e.g. ``"text_overlap"``).
        message: Human-readable explanation.
        location: Optional string identifying the affected axes or
            artist (e.g. ``"panel=A"``).
    """

    level: DiagnosticLevel
    code: str
    message: str
    location: str = ""


# =============================================================================
# Minimum values
# =============================================================================

_MIN_FONT_SIZE_PT: float = 6.0
_MAX_PANEL_DEVIATION: float = 0.01


# =============================================================================
class DiagnosticEngine:
    """Runs layout and readability checks on a matplotlib Figure.

    Usage::

        engine = DiagnosticEngine()
        diagnostics = engine.run(fig, panel_a=ax_input, panel_b=ax_target)
        for d in diagnostics:
            print(f"[{d.level}] {d.code}: {d.message}")
    """

    def __init__(
        self,
        min_font_size: float = _MIN_FONT_SIZE_PT,
        max_panel_deviation: float = _MAX_PANEL_DEVIATION,
    ) -> None:
        self.min_font_size = min_font_size
        self.max_panel_deviation = max_panel_deviation

    # ── Public API ──────────────────────────────────────────────────────

    def run(
        self,
        fig: Figure,
        *,
        panel_a: Axes | None = None,
        panel_b: Axes | None = None,
        comparable_axes: list[tuple[Axes, Axes]] | None = None,
    ) -> list[Diagnostic]:
        """Run all available checks on *fig*.

        Args:
            fig: Rendered Matplotlib figure with a drawn canvas.
            panel_a: Optional primary comparison panel A.
            panel_b: Optional primary comparison panel B.
            comparable_axes: Optional list of ``(ax1, ax2)`` pairs
                whose extents should be compared for consistency.

        Returns:
            List of ``Diagnostic`` results (sorted FAIL → WARN → INFO).
        """
        results: list[Diagnostic] = []

        # Fig-level checks ────────────────────────────────────────────
        results.extend(self._check_text_overlap(fig))
        results.extend(self._check_artist_outside_figure(fig))
        results.extend(self._check_small_text(fig))

        # Panel comparison checks ─────────────────────────────────────
        if panel_a is not None and panel_b is not None:
            results.extend(
                self._check_panel_consistency(panel_a, panel_b, "A", "B")
            )

        if comparable_axes:
            for i, (a, b) in enumerate(comparable_axes):
                tag = f"pair_{i}"
                results.extend(self._check_comparable_extents(a, b, tag))

        # Image clipping ──────────────────────────────────────────────
        results.extend(self._check_hidden_data(fig))

        # Sort
        _level_order: dict[DiagnosticLevel, int] = {
            "FAIL": 0,
            "WARN": 1,
            "INFO": 2,
        }
        results.sort(key=lambda d: (_level_order.get(d.level, 99), d.code))
        return results

    # ── Individual checks ───────────────────────────────────────────────

    @staticmethod
    def _check_text_overlap(fig: Figure) -> list[Diagnostic]:
        """D1: detect intersecting text artist bounding boxes."""
        results: list[Diagnostic] = []
        try:
            fig.canvas.draw()
        except Exception:
            return results

        renderer = fig.canvas.get_renderer()
        text_artists: list[tuple] = []

        for ax in fig.get_axes():
            for child in ax.get_children():
                if hasattr(child, "get_text") and hasattr(
                    child, "get_window_extent"
                ):
                    bbox = child.get_window_extent(renderer)
                    if bbox.width > 0 and bbox.height > 0:
                        text_artists.append((child, bbox, ax.get_label() or ""))

        for i in range(len(text_artists)):
            for j in range(i + 1, len(text_artists)):
                a1, b1, loc1 = text_artists[i]
                a2, b2, loc2 = text_artists[j]
                if b1.overlaps(b2):
                    t1 = getattr(a1, "get_text", lambda: "")()
                    t2 = getattr(a2, "get_text", lambda: "")()
                    overlap_area = _bbox_intersection_area(b1, b2)
                    if overlap_area > 0:
                        results.append(
                            Diagnostic(
                                level="FAIL",
                                code="text_overlap",
                                message=(
                                    f"Text '{t1}' ({loc1}) overlaps "
                                    f"'{t2}' ({loc2}), overlap={overlap_area:.0f} px²"
                                ),
                            )
                        )
        return results

    @staticmethod
    def _check_artist_outside_figure(fig: Figure) -> list[Diagnostic]:
        """D2: detect artists extending beyond figure boundaries."""
        results: list[Diagnostic] = []
        try:
            fig.canvas.draw()
        except Exception:
            return results

        renderer = fig.canvas.get_renderer()
        fig_bbox = fig.bbox

        for ax in fig.get_axes():
            for child in ax.get_children():
                try:
                    bbox = child.get_window_extent(renderer)
                except Exception:
                    continue
                if bbox.width == 0 or bbox.height == 0:
                    continue
                if not _bbox_contained_in(fig_bbox, bbox):
                    label = (
                        getattr(child, "get_text", lambda: "")()
                        if hasattr(child, "get_text")
                        else ""
                    )
                    right = bbox.x1 - fig_bbox.x1
                    bottom = bbox.y1 - fig_bbox.y1
                    results.append(
                        Diagnostic(
                            level="FAIL",
                            code="artist_outside_figure",
                            message=(
                                f"Artist{(' ' + repr(label)) if label else ''} "
                                f"extends beyond figure canvas: "
                                f"right={right:.0f}px, bottom={bottom:.0f}px"
                            ),
                            location=ax.get_label() or "",
                        )
                    )
        return results

    def _check_small_text(self, fig: Figure) -> list[Diagnostic]:
        """D4: warn on text artists below minimum font size."""
        results: list[Diagnostic] = []
        for ax in fig.get_axes():
            for child in ax.get_children():
                if hasattr(child, "get_fontsize"):
                    fs = child.get_fontsize()
                    if fs < self.min_font_size:
                        label = ""
                        if hasattr(child, "get_text"):
                            label = child.get_text()
                        results.append(
                            Diagnostic(
                                level="WARN",
                                code="small_text",
                                message=(
                                    f"Text '{label}' at {fs:.1f}pt "
                                    f"< minimum {self.min_font_size:.1f}pt"
                                ),
                                location=ax.get_label() or "",
                            )
                        )
        return results

    @staticmethod
    def _check_panel_consistency(
        ax_a: Axes, ax_b: Axes, name_a: str, name_b: str
    ) -> list[Diagnostic]:
        """D5/D6: check panel width/height consistency."""
        results: list[Diagnostic] = []
        ba = ax_a.get_position()
        bb = ax_b.get_position()

        # D5: width
        w_max = max(ba.width, 1e-9)
        w_dev = abs(ba.width - bb.width) / w_max
        if w_dev > 0.01:
            results.append(
                Diagnostic(
                    level="WARN",
                    code="inconsistent_panel_width",
                    message=(
                        f"Panel {name_a} width={ba.width:.4f} vs "
                        f"Panel {name_b} width={bb.width:.4f}, "
                        f"deviation={w_dev:.2%}"
                    ),
                )
            )
        else:
            results.append(
                Diagnostic(
                    level="INFO",
                    code="panel_width_ok",
                    message=(
                        f"Panel {name_a}/{name_b} width: "
                        f"{ba.width:.4f} / {bb.width:.4f} "
                        f"(deviation={w_dev:.2%})"
                    ),
                )
            )

        # D6: height
        h_max = max(ba.height, 1e-9)
        h_dev = abs(ba.height - bb.height) / h_max
        if h_dev > 0.01:
            results.append(
                Diagnostic(
                    level="WARN",
                    code="inconsistent_panel_height",
                    message=(
                        f"Panel {name_a} height={ba.height:.4f} vs "
                        f"Panel {name_b} height={bb.height:.4f}, "
                        f"deviation={h_dev:.2%}"
                    ),
                )
            )
        else:
            results.append(
                Diagnostic(
                    level="INFO",
                    code="panel_height_ok",
                    message=(
                        f"Panel {name_a}/{name_b} height: "
                        f"{ba.height:.4f} / {bb.height:.4f} "
                        f"(deviation={h_dev:.2%})"
                    ),
                )
            )

        return results

    @staticmethod
    def _check_comparable_extents(
        ax_a: Axes, ax_b: Axes, tag: str
    ) -> list[Diagnostic]:
        """D7: check that comparable axes have matching data extents."""
        results: list[Diagnostic] = []
        xla, xra = ax_a.get_xlim()
        xlb, xrb = ax_b.get_xlim()
        yla, yra = ax_a.get_ylim()
        ylb, yrb = ax_b.get_ylim()

        if abs(xla - xlb) > 0.01 or abs(xra - xrb) > 0.01:
            results.append(
                Diagnostic(
                    level="WARN",
                    code="inconsistent_data_extent",
                    message=(
                        f"Comparable axes ({tag}) x limits differ: "
                        f"A=({xla:.2f}, {xra:.2f}) B=({xlb:.2f}, {xrb:.2f})"
                    ),
                )
            )
        if abs(yla - ylb) > 0.01 or abs(yra - yrb) > 0.01:
            results.append(
                Diagnostic(
                    level="WARN",
                    code="inconsistent_data_extent",
                    message=(
                        f"Comparable axes ({tag}) y limits differ: "
                        f"A=({yla:.2f}, {yra:.2f}) B=({ylb:.2f}, {yrb:.2f})"
                    ),
                )
            )

        return results

    @staticmethod
    def _check_hidden_data(fig: Figure) -> list[Diagnostic]:
        """D11: detect ``AxesImage`` data that gets clipped."""
        results: list[Diagnostic] = []
        for ax in fig.get_axes():
            xl, xr = ax.get_xlim()
            yl, yr = ax.get_ylim()
            for child in ax.get_children():
                if hasattr(child, "get_extent"):
                    try:
                        ext = child.get_extent()
                    except Exception:
                        continue
                    if ext is None or len(ext) < 4:
                        continue
                    ext_left, ext_right, ext_bottom, ext_top = ext[:4]
                    if ext_left < xl - 0.5 or ext_right > xr + 0.5:
                        results.append(
                            Diagnostic(
                                level="WARN",
                                code="hidden_data",
                                message=(
                                    f"Image extent ({ext_left:.1f}, {ext_right:.1f}) "
                                    f"exceeds axes x limits ({xl:.1f}, {xr:.1f})"
                                ),
                                location=ax.get_label() or "",
                            )
                        )
                        break
        return results


# =============================================================================
# Internal helpers
# =============================================================================


def _bbox_contained_in(outer: object, inner: object) -> bool:
    """Return True if *inner* bbox is fully contained in *outer* bbox."""
    return (
        inner.x0 >= outer.x0
        and inner.y0 >= outer.y0
        and inner.x1 <= outer.x1
        and inner.y1 <= outer.y1
    )


def _bbox_intersection_area(bbox_a: object, bbox_b: object) -> float:
    """Compute intersection area (px²) of two display-space bounding boxes.

    Each bbox supports ``.x0``, ``.y0``, ``.x1``, ``.y1`` attributes (the
    standard Matplotlib ``Bbox`` interface).
    """
    x0 = max(bbox_a.x0, bbox_b.x0)  # type: ignore[union-attr]
    y0 = max(bbox_a.y0, bbox_b.y0)  # type: ignore[union-attr]
    x1 = min(bbox_a.x1, bbox_b.x1)  # type: ignore[union-attr]
    y1 = min(bbox_a.y1, bbox_b.y1)  # type: ignore[union-attr]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


# =============================================================================
__all__ = [
    "Diagnostic",
    "DiagnosticEngine",
]
