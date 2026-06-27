"""LEC content filtering diagnostic — report-facing figure template.

Shows the LEC transformation cascade across three panels:

    A. Sensory code c       — raw input entering LEC
    B. EMA-filtered state   — output of the stateful EMA filter
    C. Final LEC cells      — after norm (mean-sub + ReLU + L2) + sigmoid(w_f) scaling

Rows are activation dimensions grouped by LEC frequency/filter band.
Columns are rollout time. This is a model-internal activation-flow figure,
not a biological firing-rate raster.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import colorbar, panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.lec import (
    LECContentFilteringFigureData,
    select_lec_content_filtering,
)
from ehc_sn.figures.utils.colors import (
    attach_aligned_colorbar_fmt,
    colormap_with_nan_color,
)
from ehc_sn.figures.utils.labels import format_panel_title, set_panel_title
from ehc_sn.figures.utils.scales import build_shared_minmax
from ehc_sn.traces.trace_tree import TraceTree


@dataclass(frozen=True)
class _StackedBands:
    matrix: np.ndarray
    boundaries: list[tuple[int, int]]
    labels: list[str]


def _freq_label(idx: int) -> str:
    """Frequency-only label for Panel A."""
    return f"$f_{idx}$"


def _alpha_label(idx: int, alpha: np.ndarray) -> str:
    """Alpha-only label for Panel B."""
    label = ""
    if len(alpha) > idx and np.isfinite(alpha[idx]):
        label = f"$\\alpha=${float(alpha[idx]): .2f}, $f_{idx}$"
    return label


def _wf_label(idx: int, w_f: np.ndarray) -> str:
    """Sigmoid-weight label for Panel C, rendered with KaTeX subscript."""
    label = ""
    if len(w_f) > idx and np.isfinite(w_f[idx]):
        label = f"$\\sigma(w_{{\\!f}})=${float(w_f[idx]): .2f}, $f_{idx}$"
    return label


def _stack_bands(
    bands: list[np.ndarray],
    label_fn,
) -> _StackedBands:
    """Stack per-frequency arrays into a rows × time activation matrix.

    Parameters
    ----------
    bands:
        List of arrays shaped (T, n_units). Each list element is one
        frequency/filter band.
    label_fn:
        Callable ``(band_index) -> str`` that produces the y-axis label
        for each band.

    Returns
    -------
    _StackedBands
        matrix:
            Shape (n_rows, T). NaN separator rows are inserted between bands.
        boundaries:
            Inclusive/exclusive row spans for each real frequency band.
        labels:
            Per-band labels produced by ``label_fn``.
    """
    if not bands:
        return _StackedBands(
            matrix=np.empty((0, 0), dtype=float),
            boundaries=[],
            labels=[],
        )

    # Use first non-empty band to determine T.
    non_empty = [b for b in bands if b.size > 0]
    if not non_empty:
        return _StackedBands(
            matrix=np.empty((0, 0), dtype=float),
            boundaries=[],
            labels=[],
        )

    T = int(non_empty[0].shape[0])

    rows: list[np.ndarray] = []
    boundaries: list[tuple[int, int]] = []
    labels: list[str] = []
    row_offset = 0

    for i, arr in enumerate(bands):
        labels.append(label_fn(i))

        if arr.size == 0:
            # Missing band: one NaN row so the band is visible but not fake data.
            band_matrix = np.full((1, T), np.nan, dtype=float)
        else:
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected LEC band {i} to have shape (T, n_units), "
                    f"got shape {arr.shape}."
                )
            if arr.shape[0] != T:
                raise ValueError(
                    f"All LEC bands must share the same time dimension. "
                    f"Band 0 has T={T}; band {i} has T={arr.shape[0]}."
                )
            band_matrix = np.asarray(arr, dtype=float).T  # (n_units, T)

        start = row_offset
        end = start + band_matrix.shape[0]
        rows.append(band_matrix)
        boundaries.append((start, end))
        row_offset = end

        # Separator row between bands, but not after the final band.
        if i < len(bands) - 1:
            rows.append(np.full((1, T), np.nan, dtype=float))
            row_offset += 1

    return _StackedBands(
        matrix=np.vstack(rows),
        boundaries=boundaries,
        labels=labels,
    )


def _ytick_positions(boundaries: list[tuple[int, int]]) -> list[float]:
    """Return vertical center row for each band, to use as ytick positions."""
    return [(start + end) / 2.0 for start, end in boundaries]


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return LECContentFilteringFigure(
        select_lec_content_filtering(trace, ctx), ctx
    ).plot()


class LECContentFilteringFigure(BaseFigureTemplate):
    """LEC transformation cascade: sensory code → EMA filter → final cells."""

    # Three compact horizontal panels. Keep height moderate; the panel content
    # is dense but should not dominate the notebook vertically.
    HEIGHT_FRAC: float = 0.30
    MOSAIC = [["sensory"], ["filtered"], ["cells"]]
    MOSAIC_KWARGS = {"gridspec_kw": {"wspace": 0.0, "hspace": 0.0}}
    SHAREX: bool = True

    _CMAP = "GnBu"

    def __init__(
        self, data: LECContentFilteringFigureData, ctx: FigureContext
    ) -> None:
        super().__init__(data, ctx)

    # -- Stack helpers ---------------------------------------------------------

    @cached_property
    def _sensory(self) -> _StackedBands:
        return _stack_bands(self.data.sensory_by_freq, label_fn=_freq_label)

    @cached_property
    def _filtered(self) -> _StackedBands:
        return _stack_bands(
            self.data.filtered_by_freq,
            label_fn=lambda i: _alpha_label(i, self.data.alpha),
        )

    @cached_property
    def _cells(self) -> _StackedBands:
        return _stack_bands(
            self.data.cells_by_freq,
            label_fn=lambda i: _wf_label(i, self.data.w_f),
        )

    # -- Color limits ----------------------------------------------------------

    @cached_property
    def _activation_limits(self) -> tuple[float, float]:
        """Single shared robust limits across all three panels."""
        return build_shared_minmax(
            [self._sensory.matrix, self._filtered.matrix, self._cells.matrix],
            percentile=(1.0, 99.0),
        )

    def _cmap(self):
        return colormap_with_nan_color(self._CMAP)

    @staticmethod
    def _draw_band_separators(
        ax: Axes,
        *,
        boundaries: list[tuple[int, int]],
    ) -> None:
        """Draw white horizontal separator lines at band boundaries."""
        if not boundaries:
            return
        for start, end in boundaries:
            ax.axhline(start - 0.5, color="white", linewidth=0.6, alpha=0.9)
            ax.axhline(end - 0.5, color="white", linewidth=0.6, alpha=0.9)

    def _draw_panel(
        self,
        ax: Axes,
        *,
        matrix: np.ndarray,
        boundaries: list[tuple[int, int]],
        labels: list[str],
        vmin: float,
        vmax: float,
    ) -> None:
        """Common imshow + separators + ytick logic for all panels."""
        im = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap=self._cmap(),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        self._draw_band_separators(ax, boundaries=boundaries)
        ypos = _ytick_positions(boundaries)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=7, family="monospace")
        attach_aligned_colorbar_fmt(im, vmax)
        return im

    def _apply_colorbars(self, colorbar_groups):
        """Apply a single colorbar with aligned decimal-point formatting.

        Adds ``FormatStrFormatter`` to the colorbar ticks so that decimal
        points are vertically aligned regardless of minus signs or digit
        count.
        """
        for group_state in colorbar_groups.values():
            mappable = group_state.get("mappable")
            axes = group_state.get("axes", [])
            if mappable is None or not axes:
                continue
            axes = list(dict.fromkeys(axes))
            label = group_state.get("label", None)
            cbar = self.fig.colorbar(mappable, ax=axes, label=label, pad=0.02)
            tick_labelsize = group_state.get("tick_labelsize", None)
            if tick_labelsize is not None:
                cbar.ax.tick_params(labelsize=tick_labelsize)
            fmt = getattr(mappable, "_tem_cbar_fmt", None)
            if fmt is not None:
                cbar.ax.yaxis.set_major_formatter(fmt)

    # -- Panel A: Sensory code c -----------------------------------------------

    @colorbar(group="activation", label="Activation")
    @panel()
    def sensory(self, ax: Axes) -> None:
        set_panel_title(ax, format_panel_title("a", "Sensory code c"))

        vmin, vmax = self._activation_limits
        self._draw_panel(
            ax,
            matrix=self._sensory.matrix,
            boundaries=self._sensory.boundaries,
            labels=self._sensory.labels,
            vmin=vmin,
            vmax=vmax,
        )
        ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.margins(x=0, y=0)

    # -- Panel B: EMA-filtered state -------------------------------------------

    @colorbar(group="activation", label="Activation")
    @panel()
    def filtered(self, ax: Axes) -> None:
        set_panel_title(ax, format_panel_title("b", "EMA-filtered state"))

        vmin, vmax = self._activation_limits
        self._draw_panel(
            ax,
            matrix=self._filtered.matrix,
            boundaries=self._filtered.boundaries,
            labels=self._filtered.labels,
            vmin=vmin,
            vmax=vmax,
        )
        ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.margins(x=0, y=0)

    # -- Panel C: Final LEC cells ----------------------------------------------

    @colorbar(group="activation", label="Activation")
    @panel()
    def cells(self, ax: Axes) -> None:
        set_panel_title(
            ax,
            format_panel_title("c", "Final LEC cells"),
        )

        vmin, vmax = self._activation_limits
        self._draw_panel(
            ax,
            matrix=self._cells.matrix,
            boundaries=self._cells.boundaries,
            labels=self._cells.labels,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Time step", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, bottom=True, top=False)
        ax.margins(x=0, y=0)
