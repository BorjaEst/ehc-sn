"""Shared task-overview figure template.

Provides the canonical two-row mosaic (visual comparison row + annotation
row), equal-geometry enforcement, annotation-slot rendering, and three-zone
summary layout for all task-overview figures.

Subclasses override:

* ``render_input(ax)``        — draw the input panel on *ax*.
* ``render_target(ax)``       — draw the target panel on *ax*.
* ``annotation_for_input()``  — return ``PanelAnnotation`` for the input
  annotation slot (default ``None``).
* ``annotation_for_target()`` — return ``PanelAnnotation`` for the target
  annotation slot (default ``None``).
* ``objective_text()``        — return list of prose lines (Objective zone).
* ``sample_rows()``           — return list of ``(key, value)`` pairs
  (Sample zone).
* ``contract_notation()``     — return short math-notation string
  (Contract zone).

The template owns layout, annotation rendering, summary zone typography,
and equal-geometry validation.  Subclasses own task-specific visual
encoding, annotation content decisions, and summary text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ehc_sn.figures._contracts import (
    CategoricalLegend,
    ColorStrip,
    ContinuousScale,
    PanelAnnotation,
)
from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel

# ── Geometry constants ──────────────────────────────────────────────────────

_ANNOTATION_HEIGHT_RATIO: float = 0.12
_SUMMARY_WIDTH_RATIO: float = 0.95
_MAP_WIDTH_RATIO: float = 1.0

# ── Summary zone layout ─────────────────────────────────────────────────────

# Fixed gap (axes fraction) between the panel title bottom and the first
# line of objective text — identical across all subclasses.
_TITLE_TEXT_GAP: float = 0.02
# Minimum spacing between stacked zones.
_ZONE_GAP: float = 0.03
# Bottom margin for the contract zone.
_CONTRACT_BOTTOM: float = 0.06
# Top of usable area (just below the title baseline).
_SUMMARY_USABLE_TOP: float = 0.94

# ── Typography ──────────────────────────────────────────────────────────────

_SUMMARY_FONTSIZE: float = 5.5
_CONTRACT_FONTSIZE: float = 5.5
_ANNOTATION_FONTSIZE: float = 5.0


class TaskOverviewTemplate(BaseFigureTemplate):
    """Shared two-row task-overview figure template.

    Row 0: ``input``, ``target``, ``summary`` (visual comparison + text).
    Row 1: ``input_key``, ``target_key``, ``summary`` (annotation + spanning
    summary).
    """

    HEIGHT_FRAC: float = 0.22
    WIDTH_FRAC: float = 1.0
    MOSAIC: Sequence[Sequence[str]] = [
        ["input", "target", "summary"],
        ["input_key", "target_key", "summary"],
    ]
    MOSAIC_KWARGS: dict = {
        "gridspec_kw": {
            "height_ratios": [
                1.0,  # row 0: visual comparison
                _ANNOTATION_HEIGHT_RATIO,
            ],
            "width_ratios": [
                _MAP_WIDTH_RATIO,
                _MAP_WIDTH_RATIO,
                _SUMMARY_WIDTH_RATIO,
            ],
        },
    }

    # ── plot (final) ────────────────────────────────────────────────────

    def plot(self) -> Figure:
        """Render the figure, enforce equal geometry, and invoke post-render
        hook."""
        fig = super().plot()
        self._enforce_equal_geometry()
        self._post_render(fig)
        return fig

    # ── Subclass override points ────────────────────────────────────────

    def render_input(self, ax: Axes) -> None:
        """Draw the input panel on *ax*.  Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} must override render_input()"
        )

    def render_target(self, ax: Axes) -> None:
        """Draw the target panel on *ax*.  Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} must override render_target()"
        )

    def annotation_for_input(self) -> PanelAnnotation:
        """Return the annotation content for the input slot.

        Default: ``None`` (slot hidden).
        """
        return None

    def annotation_for_target(self) -> PanelAnnotation:
        """Return the annotation content for the target slot.

        Default: ``None`` (slot hidden).
        """
        return None

    def objective_text(self) -> list[str]:
        """Return stable prose lines describing the task objective.

        Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override objective_text()"
        )

    def sample_rows(self) -> list[tuple[str, str]]:
        """Return ``(key, value)`` pairs for the Sample zone.

        Values must come from ``self.data`` (selector-produced), not be
        recomputed from raw arrays.  Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override sample_rows()"
        )

    def contract_notation(self) -> str:
        """Return a short math-notation string for the Contract zone.

        Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override contract_notation()"
        )

    def _post_render(self, fig: Figure) -> None:  # noqa: B027
        """Post-render hook called after geometry enforcement.

        Subclasses may override to add figure-level adjustments (e.g.
        goaltrace's internal current/goal legend glyphs).  Default is
        a no-op.
        """

    def summary_title(self) -> str:
        """Return the summary-panel title.

        Default is ``"Summary"``.  Subclasses should override with a
        task-specific title (e.g. ``"Routebind Task"``) so the panel
        title doubles as the task label and saves one line of body text.
        """
        return "Summary"

    # ── Panel methods (@panel, auto-discovered) ─────────────────────────

    @panel(slots=["input"], order=0)
    def _input_panel(self, ax: Axes) -> None:
        """Panel A: delegate to ``render_input``."""
        self._last_input_mappable = self.render_input(ax)

    @panel(slots=["target"], order=1)
    def _target_panel(self, ax: Axes) -> None:
        """Panel B: delegate to ``render_target``."""
        self._last_target_mappable = self.render_target(ax)

    @panel(slots=["input_key"], order=3)
    def _input_key(self, ax: Axes) -> None:
        """Input annotation slot."""
        spec = self.annotation_for_input()
        self._bind_mappable(spec, self._last_input_mappable)
        self._render_annotation(ax, spec)

    @panel(slots=["target_key"], order=4)
    def _target_key(self, ax: Axes) -> None:
        """Target annotation slot."""
        spec = self.annotation_for_target()
        self._bind_mappable(spec, self._last_target_mappable)
        self._render_annotation(ax, spec)

    @panel(slots=["summary"], order=2)
    def _summary_panel(self, ax: Axes) -> None:
        """Panel C: three-zone summary."""
        self._render_summary(ax)

    # ── Internal plumbing ───────────────────────────────────────────────

    @staticmethod
    def _bind_mappable(spec: PanelAnnotation, mappable: object | None) -> None:
        """If *spec* is a ``ContinuousScale`` with no mappable, bind one."""
        if isinstance(spec, ContinuousScale) and spec.mappable is None:
            spec.mappable = mappable

    # ── Annotation rendering ────────────────────────────────────────────

    def _render_annotation(self, ax: Axes, spec: PanelAnnotation) -> None:
        """Render one annotation slot from a ``PanelAnnotation`` spec.

        * ``None``            — hide the axes.
        * ``CategoricalLegend`` — render a horizontal legend.
        * ``ContinuousScale``   — render a horizontal colorbar via
          ``fig.colorbar``.
        """
        if spec is None:
            ax.set_visible(False)
            return

        if isinstance(spec, CategoricalLegend):
            self._render_categorical_legend(ax, spec)
        elif isinstance(spec, ColorStrip):
            self._render_color_strip(ax, spec)
        elif isinstance(spec, ContinuousScale):
            self._render_continuous_scale(ax, spec)
        else:
            raise TypeError(f"Unknown annotation type: {type(spec).__name__}")

        # Transparent full-axes spacer: gives tight-layout a consistent
        # reference bbox regardless of whether the slot contains a compact
        # colour bar, a 2-row legend, or a colour strip.
        ax.add_patch(
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                edgecolor="none",
                clip_on=False,
            )
        )

    def _render_categorical_legend(
        self, ax: Axes, legend: CategoricalLegend
    ) -> None:
        """Render a categorical legend in the annotation slot.

        When *label_stride* > 1, only every *label_stride*-th entry gets a
        text label; unlabeled entries still show a colour swatch so the
        full range of categories remains visually accessible.

        When *compact* is True, entries render as thin ``Line2D`` strokes
        instead of filled ``Patch`` squares.
        """
        stride = legend.label_stride
        compact = legend.compact
        handles = []
        for i, (label, color) in enumerate(legend.entries):
            show_label = label if stride <= 1 or i % stride == 0 else ""
            if compact:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=3.5,
                        label=show_label,
                    )
                )
            else:
                handles.append(mpatches.Patch(color=color, label=show_label))
        ax.legend(
            handles=handles,
            loc="center",
            ncol=legend.ncol,
            fontsize=_ANNOTATION_FONTSIZE,
            framealpha=0.85,
            borderpad=0.2,
            handlelength=0.8,
            handletextpad=0.3,
            title=legend.title,
        )
        ax.axis("off")

    def _render_color_strip(self, ax: Axes, strip: ColorStrip) -> None:
        """Render a ``ColorStrip`` as one or more rows of thin vertical bars.

        Bars span the horizontal axis.  For multi-row strips, labels are
        placed above the top row and below each subsequent row so the
        indexing is unambiguous without a separate title.
        """
        n = len(strip.entries)
        if n == 0:
            ax.set_visible(False)
            return

        n_rows = max(1, strip.rows)
        per_row = (n + n_rows - 1) // n_rows

        bar_h = 0.30 if n_rows == 1 else 0.18
        row_gap = 0.10 if n_rows > 1 else 0.0

        bar_y0 = 0.35 if n_rows == 1 else 0.52

        label_every = strip.label_every
        if label_every is None:
            label_every = max(3, n // 8)

        for row_idx in range(n_rows):
            start = row_idx * per_row
            end = min(start + per_row, n)
            row_n = end - start
            if row_n == 0:
                continue

            row_bar_w = 0.80 / row_n
            row_gap_w = 0.04 / row_n if row_n > 1 else 0.0
            y0 = bar_y0 - row_idx * (bar_h + row_gap)

            for j in range(row_n):
                i = start + j
                x0 = 0.10 + j * (row_bar_w + row_gap_w)
                _label, color = strip.entries[i]
                rect = mpatches.Rectangle(
                    (x0, y0),
                    row_bar_w,
                    bar_h,
                    facecolor=color,
                    edgecolor="none",
                    transform=ax.transAxes,
                    clip_on=False,
                )
                ax.add_patch(rect)

            # Labels: above row 0, below each subsequent row
            row_label_every = max(2, row_n // 8)
            for j in range(0, row_n, row_label_every):
                i = start + j
                label, _color = strip.entries[i]
                x_center = 0.10 + j * (row_bar_w + row_gap_w) + row_bar_w / 2

                if row_idx == 0 and n_rows > 1:
                    label_y = y0 + bar_h + 0.055
                    va = "bottom"
                elif row_idx == 0:
                    label_y = 0.18
                    va = "top"
                else:
                    label_y = y0 - 0.055
                    va = "top"

                ax.text(
                    x_center,
                    label_y,
                    label,
                    ha="center",
                    va=va,
                    fontsize=3.8,
                    transform=ax.transAxes,
                )

        if strip.title:
            ax.text(
                0.5,
                0.92,
                strip.title,
                ha="center",
                va="bottom",
                fontsize=4.2,
                fontstyle="italic",
                transform=ax.transAxes,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def _render_continuous_scale(
        self, ax: Axes, scale: ContinuousScale
    ) -> None:
        """Render a continuous colorbar in the annotation slot.

        The colour bar is drawn inside an inset axes (80 % width, 55 %
        height, centred) so that horizontal padding and vertical
        shrinkage are applied without altering the GridSpec layout.
        All text is rendered at compact sizes so the bar fits within
        the 12 %-height annotation row.
        """
        if scale.mappable is None:
            ax.set_visible(False)
            return
        if self.fig is None:
            return

        ax.axis("off")
        cax = inset_axes(ax, width="80%", height="75%", loc="center")
        cbar = self.fig.colorbar(
            scale.mappable,
            cax=cax,
            orientation="horizontal",
        )
        cbar.ax.tick_params(labelsize=4.0, pad=1.5)
        if scale.label:
            cbar.set_label(scale.label, fontsize=5.0, labelpad=2)
        if scale.ticks is not None:
            cbar.ax.set_xticks(scale.ticks)

    # ── Summary rendering ───────────────────────────────────────────────

    def _render_summary(self, ax: Axes) -> None:
        """Render the three-zone summary panel with fixed title-gap anchoring.

        The Objective zone is always anchored at the same distance below
        the panel title — identical for all subclasses.  Contract is
        anchored at a fixed bottom margin.  Sample floats between them.
        This gives a visually consistent title-to-text gap regardless of
        content length.
        """
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_title(self.summary_title(), fontsize=7, pad=4)

        # ── Gather content ──────────────────────────────────────────────
        objective = self.objective_text()
        sample = self.sample_rows()
        contract = self.contract_notation()

        # ── Objective zone — fixed gap below title ──────────────────────
        objective_top = _SUMMARY_USABLE_TOP - _TITLE_TEXT_GAP
        ax.text(
            0.08,
            objective_top,
            "\n".join(objective),
            transform=ax.transAxes,
            fontsize=_SUMMARY_FONTSIZE,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # ── Contract zone — fixed bottom margin ─────────────────────────
        ax.text(
            0.08,
            _CONTRACT_BOTTOM,
            contract,
            transform=ax.transAxes,
            fontsize=_CONTRACT_FONTSIZE,
            verticalalignment="bottom",
            fontfamily="serif",
        )

        # ── Sample zone — centred between objective and contract ────────
        if sample:
            max_key_len = max(len(k) for k, _ in sample)
            sample_lines = [f"{k:<{max_key_len}}  {v}" for k, v in sample]
            sample_mid = (objective_top + _CONTRACT_BOTTOM) / 2
            ax.text(
                0.08,
                sample_mid,
                "\n".join(sample_lines),
                transform=ax.transAxes,
                fontsize=_SUMMARY_FONTSIZE,
                verticalalignment="center",
                fontfamily="monospace",
            )

    # ── Geometry enforcement ────────────────────────────────────────────

    def _enforce_equal_geometry(self) -> None:
        """Raise ``RuntimeError`` if input and target panels have
        inconsistent dimensions."""
        if "input" not in self.axdict or "target" not in self.axdict:
            return

        ax_in = self.axdict["input"]
        ax_tgt = self.axdict["target"]

        b_in = ax_in.get_position()
        b_tgt = ax_tgt.get_position()

        w_dev = abs(b_in.width - b_tgt.width) / max(b_in.width, 1e-9)
        h_dev = abs(b_in.height - b_tgt.height) / max(b_in.height, 1e-9)

        if w_dev > 0.01 or h_dev > 0.01:
            raise RuntimeError(
                f"Task-overview panel geometry mismatch: "
                f"input=({b_in.width:.4f}, {b_in.height:.4f}) "
                f"target=({b_tgt.width:.4f}, {b_tgt.height:.4f}) "
                f"width_dev={w_dev:.2%} height_dev={h_dev:.2%}"
            )


# ── Exports ─────────────────────────────────────────────────────────────────

__all__ = ["TaskOverviewTemplate"]
