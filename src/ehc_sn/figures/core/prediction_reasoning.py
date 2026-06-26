"""Shared prediction-reasoning figure template.

Two-row mosaic (``(a)`` ground truth | ``(b)`` reasoning evolution) with a
2-row × up-to-8-column snapshot grid inside the evolution panel and a shared
legend slot under the GT panel.

Subclasses override:

* ``render_ground_truth(ax)`` — draw the oracle target on *ax*.
* ``render_snapshot(ax, snapshot_data, *, step_label, metric_label)`` —
  draw one snapshot with optional metric text.
* ``annotation_for_gt()`` — return ``PanelAnnotation`` for the shared
  legend slot (target and predictions use the same units/scale).
* ``_metric_for_snapshot(index)`` — return a per-step metric string or
  ``None``.

The template owns layout, mosaic construction, snapshot selection, iteration
labels, halt/truncation decoration, the shared legend, equal-snapshot
geometry, shared-scale enforcement, and per-step metric typography.
"""

from __future__ import annotations

from abc import ABC
from typing import Sequence

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures._contracts import (
    CategoricalLegend,
    ColorStrip,
    ContinuousScale,
    PanelAnnotation,
)
from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.utils.axes import subdivide_axes

# ── Geometry constants ──────────────────────────────────────────────────────

_ANNOTATION_HEIGHT_RATIO: float = 0.10
_GT_WIDTH_RATIO: float = 1.0
_EVOLUTION_WIDTH_RATIO: float = 2.6
_SNAPSHOT_ROWS: int = 2
_SNAPSHOT_COLS: int = 8
_MAX_SNAPSHOTS: int = _SNAPSHOT_ROWS * _SNAPSHOT_COLS

# ── Typography ──────────────────────────────────────────────────────────────

_SNAPSHOT_TITLE_FONTSIZE: float = 5.5
_SNAPSHOT_METRIC_FONTSIZE: float = 4.2
_SNAPSHOT_METRIC_ALPHA: float = 0.65
_HALT_BORDER_COLOR: str = "#d62728"
_HALT_BORDER_WIDTH: float = 2.5
_TRUNCATED_BORDER_COLOR: str = "#9467bd"
_TRUNCATED_BORDER_WIDTH: float = 2.0
_NORMAL_BORDER_COLOR: str = "none"
_NORMAL_BORDER_WIDTH: float = 0.0


class PredictionReasoningTemplate(BaseFigureTemplate, ABC):
    """Shared two-row prediction-reasoning figure template.

    Mosaic layout::

        ┌─────────────┬───────────────────────────────────┐
        │ (a) GT panel│ (b) Reasoning evolution mosaic    │
        │    1×       │    2 rows × 8 cols (max 16 snaps) │
        ├─────────────┤                                   │
        │   Legend    │                                   │
        └─────────────┴───────────────────────────────────┘

    The legend sits under the GT panel.  Target and predictions share
    the same units/scale, so one legend serves the whole figure.

    Width ratio: ``1.0 : 2.1``.  Height ratio: ``1.0 : 0.12``.
    """

    HEIGHT_FRAC: float = 0.22
    WIDTH_FRAC: float = 1.0

    MOSAIC = [
        ["gt", "evolution"],
        ["legend", "evolution"],
    ]
    MOSAIC_KWARGS: dict = {
        "gridspec_kw": {
            "height_ratios": [
                1.0,  # row 0: visual comparison
                _ANNOTATION_HEIGHT_RATIO,  # row 1: shared legend
            ],
            "width_ratios": [
                _GT_WIDTH_RATIO,
                _EVOLUTION_WIDTH_RATIO,
            ],
        },
    }

    def __init__(self, data: Any, ctx: Any) -> None:
        super().__init__(data, ctx)
        self._gt_mappable: object | None = None

    # ── Subclass override points ────────────────────────────────────────

    def render_ground_truth(self, ax: Axes) -> None:
        """Draw the oracle target panel on *ax*.  Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} must override render_ground_truth()"
        )

    def render_snapshot(
        self,
        ax: Axes,
        snapshot_data: object,
        *,
        step_label: str,
        metric_label: str | None,
    ) -> None:
        """Draw one prediction snapshot on *ax*.  Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} must override render_snapshot()"
        )

    def annotation_for_gt(self) -> PanelAnnotation | None:
        """Return annotation spec for the shared legend slot under the GT panel.

        Since target and predictions share the same units and scale, one
        legend (categorical or continuous) serves the whole figure.

        Default: ``None`` (slot hidden).
        """
        return None

    # ── Data accessors (must be set by subclass before plot) ────────────

    @property
    def _snapshots(self) -> Sequence[object]:
        """Return the sequence of snapshot payloads."""
        return self.data.snapshots

    @property
    def _selected_indices(self) -> Sequence[int]:
        """Return real deliberation step numbers."""
        return self.data.selected_indices

    @property
    def _halt_step(self) -> int | None:
        """Return the halt step, or ``None``."""
        return self.data.halt_step

    @property
    def _truncated(self) -> bool:
        """Return whether the trace was truncated."""
        return self.data.truncated

    def _metric_for_snapshot(self, index: int) -> str | None:
        """Return a short metric string for snapshot *index*, or ``None``.

        Subclasses should override when per-snapshot metrics are available.
        """
        return None

    # ── plot (final) ────────────────────────────────────────────────────

    def plot(self) -> Figure:
        """Render the figure, enforce equal geometry."""
        fig = super().plot()
        self._enforce_evolution_geometry()
        return fig

    # ── Panel methods (@panel, auto-discovered) ─────────────────────────

    @panel(slots=["gt"], order=0)
    def _gt_panel(self, ax: Axes) -> None:
        """Panel (a): delegate to ``render_ground_truth``."""
        self._gt_mappable = self.render_ground_truth(ax)

    @panel(slots=["evolution"], order=1)
    def _evolution_panel(self, ax: Axes) -> None:
        """Panel (b): 2×8 snapshot mosaic with iteration labels."""
        snapshots = self._snapshots
        indices = self._selected_indices
        n_snapshots = len(snapshots)
        halt_step = self._halt_step
        truncated = self._truncated

        axs = subdivide_axes(
            ax,
            nrows=_SNAPSHOT_ROWS,
            ncols=_SNAPSHOT_COLS,
            wspace=0.04,
            hspace=0.10,
            hide_parent=True,
        ).ravel()

        for j in range(min(n_snapshots, _MAX_SNAPSHOTS)):
            actual_step = indices[j]
            is_halted = (
                halt_step is not None
                and not truncated
                and actual_step == halt_step
            )
            is_truncated = (
                truncated
                and actual_step == indices[-1]
                and actual_step == halt_step
            )

            # Build step label
            step_str = f"i={actual_step}"
            if is_halted:
                step_str += " · halt"
            elif is_truncated:
                step_str += " · truncated"

            metric_str = self._metric_for_snapshot(j)
            label = f"{step_str}\n{metric_str}" if metric_str else step_str

            mappable = self.render_snapshot(
                axs[j], snapshots[j], step_label=label, metric_label=metric_str
            )

            # Halt/truncation border
            if is_halted:
                for spine in axs[j].spines.values():
                    spine.set_edgecolor(_HALT_BORDER_COLOR)
                    spine.set_linewidth(_HALT_BORDER_WIDTH)
                    spine.set_visible(True)
                axs[j].patch.set_edgecolor(_HALT_BORDER_COLOR)
                axs[j].patch.set_linewidth(_HALT_BORDER_WIDTH)
            elif is_truncated:
                for spine in axs[j].spines.values():
                    spine.set_edgecolor(_TRUNCATED_BORDER_COLOR)
                    spine.set_linewidth(_TRUNCATED_BORDER_WIDTH)
                    spine.set_visible(True)
                axs[j].patch.set_edgecolor(_TRUNCATED_BORDER_COLOR)
                axs[j].patch.set_linewidth(_TRUNCATED_BORDER_WIDTH)
            else:
                for spine in axs[j].spines.values():
                    spine.set_visible(False)

        # Hide unused snapshot slots
        for j in range(n_snapshots, _MAX_SNAPSHOTS):
            axs[j].axis("off")

    @panel(slots=["legend"], order=2)
    def _legend(self, ax: Axes) -> None:
        """Shared legend slot under the GT panel."""
        spec = self.annotation_for_gt()
        self._bind_mappable(spec, self._gt_mappable)
        self._render_annotation(ax, spec)

    # ── Mappable binding ───────────────────────────────────────────────

    @staticmethod
    def _bind_mappable(spec: PanelAnnotation, mappable: object | None) -> None:
        """If *spec* is a ``ContinuousScale`` with no mappable, bind one."""
        if isinstance(spec, ContinuousScale) and spec.mappable is None:
            spec.mappable = mappable

    # ── Annotation rendering ────────────────────────────────────────────

    def _render_annotation(
        self, ax: Axes, spec: PanelAnnotation | None
    ) -> None:
        """Render one annotation slot from a ``PanelAnnotation`` spec.

        * ``None`` — hide the axes.
        * ``CategoricalLegend`` — render a horizontal legend.
        * ``ContinuousScale`` — render a horizontal colorbar via
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

        # Transparent full-axes spacer for consistent bbox
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
        """Render a categorical legend in the annotation slot."""
        import matplotlib.lines as mlines

        stride = legend.label_stride
        compact = legend.compact
        handles = []
        for i, (label, color) in enumerate(legend.entries):
            show_label = label if stride <= 1 or i % stride == 0 else ""
            if compact:
                handles.append(
                    mlines.Line2D(
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
            fontsize=5.0,
            framealpha=0.85,
            borderpad=0.2,
            handlelength=0.8,
            handletextpad=0.3,
            title=legend.title,
        )
        ax.axis("off")

    def _render_color_strip(self, ax: Axes, strip: ColorStrip) -> None:
        """Render a ``ColorStrip`` as one or more rows of thin vertical bars."""
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
                i_idx = start + j
                x0 = 0.10 + j * (row_bar_w + row_gap_w)
                _label, color = strip.entries[i_idx]
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

            row_label_every = max(2, row_n // 8)
            for j in range(0, row_n, row_label_every):
                i_idx = start + j
                label, _color = strip.entries[i_idx]
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
        """Render a continuous colorbar in the annotation slot."""
        if scale.mappable is None:
            ax.set_visible(False)
            return
        if self.fig is None:
            return

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

    # ── Geometry enforcement ────────────────────────────────────────────

    def _enforce_evolution_geometry(self) -> None:
        """Verify all snapshot axes within the evolution panel have equal
        dimensions."""
        if "evolution" not in self.axdict:
            return
        ax_evo = self.axdict["evolution"]
        children = ax_evo.get_children()
        snapshot_axes = [
            c for c in children if isinstance(c, Axes) and c is not ax_evo
        ]
        if len(snapshot_axes) < 2:
            return

        ref = snapshot_axes[0].get_position()
        for child in snapshot_axes[1:]:
            pos = child.get_position()
            w_dev = abs(pos.width - ref.width) / max(ref.width, 1e-9)
            h_dev = abs(pos.height - ref.height) / max(ref.height, 1e-9)
            if w_dev > 0.01 or h_dev > 0.01:
                raise RuntimeError(
                    f"Snapshot geometry mismatch: "
                    f"reference=({ref.width:.4f}, {ref.height:.4f}) "
                    f"child=({pos.width:.4f}, {pos.height:.4f}) "
                    f"width_dev={w_dev:.2%} height_dev={h_dev:.2%}"
                )


# ── Exports ─────────────────────────────────────────────────────────────────

__all__ = ["PredictionReasoningTemplate"]
