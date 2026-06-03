"""LEC content filtering diagnostic — report-facing figure template.

Shows model activation dimensions in the LEC/x content stream:

    a. content-state activations
    b. filtered-state activations
    c. filter effect = filtered - content

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
    LECContentFilteringData,
    select_lec_content_filtering,
)
from ehc_sn.figures.utils.labels import format_panel_title
from ehc_sn.traces.trace_tree import TraceTree


@dataclass(frozen=True)
class _StackedBands:
    matrix: np.ndarray
    boundaries: list[tuple[int, int]]
    labels: list[str]


def _band_label(idx: int, alpha: np.ndarray, w_f: np.ndarray) -> str:
    """Build a compact frequency-band label.

    Examples
    --------
    f0
    f0  α=0.82  w=0.34
    """
    label = f"f{idx}"
    if len(alpha) > idx and np.isfinite(alpha[idx]):
        label += f"  α={float(alpha[idx]):.2f}"
    if len(w_f) > idx and np.isfinite(w_f[idx]):
        label += f"  w={float(w_f[idx]): .2f}"
    return label


def _stack_bands(
    bands: list[np.ndarray],
    *,
    alpha: np.ndarray,
    w_f: np.ndarray,
) -> _StackedBands:
    """Stack per-frequency arrays into a rows × time activation matrix.

    Parameters
    ----------
    bands:
        List of arrays shaped (T, n_units). Each list element is one
        frequency/filter band.

    Returns
    -------
    _StackedBands
        matrix:
            Shape (n_rows, T). NaN separator rows are inserted between bands.
        boundaries:
            Inclusive/exclusive row spans for each real frequency band.
        labels:
            Frequency labels, optionally including gate values.
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
        labels.append(_band_label(i, alpha, w_f))

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


def _robust_activation_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Return robust shared limits for activation panels."""
    vals = np.concatenate(
        [np.asarray(a, dtype=float).ravel() for a in arrays if a.size > 0]
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    lo = float(np.nanpercentile(vals, 1.0))
    hi = float(np.nanpercentile(vals, 99.0))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))

    if hi <= lo:
        hi = lo + 1.0

    # If activations are already sigmoid-like, keep the familiar scale.
    if lo >= 0.0 and hi <= 1.0:
        return 0.0, 1.0

    return lo, hi


def _symmetric_effect_limit(diff: np.ndarray) -> float:
    vals = np.asarray(diff, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(vals), 99.0))
    return max(vmax, 1e-3)


def _copy_cmap_with_bad(name: str):
    """Return a colormap with NaN separator rows rendered as light gray."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="0.92")
    return cmap


def _ytick_positions(boundaries: list[tuple[int, int]]) -> list[float]:
    """Return vertical center row for each band, to use as ytick positions."""
    return [(start + end) / 2.0 for start, end in boundaries]


def _set_panel_title(ax: Axes, label: str) -> None:
    """Place a panel label at the top-right corner inside the axes."""
    ax.text(
        0.040,
        0.900,
        label,
        ha="left",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "none",
            "pad": 2.0,
        },
    )


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return LECContentFilteringFigure(
        select_lec_content_filtering(trace, ctx), ctx
    ).plot()


class LECContentFilteringFigure(BaseFigureTemplate):
    """LEC / x content-state filtering across frequency bands."""

    # Three compact horizontal panels. Keep height moderate; the panel content
    # is dense but should not dominate the notebook vertically.
    HEIGHT_FRAC: float = 0.30
    MOSAIC = [["content"], ["filtered"], ["effect"]]
    MOSAIC_KWARGS = {"gridspec_kw": {"wspace": 0.0, "hspace": 0.0}}
    SHAREX: bool = True

    _CMAP = "GnBu"
    _CMAP_DIVERGING = "RdBu_r"

    def __init__(
        self, data: LECContentFilteringData, ctx: FigureContext
    ) -> None:
        super().__init__(data, ctx)
        self._gates_available = len(data.alpha) > 0 or len(data.w_f) > 0

    @cached_property
    def _content(self) -> _StackedBands:
        return _stack_bands(
            self.data.cells_by_freq,
            alpha=self.data.alpha,
            w_f=self.data.w_f,
        )

    @cached_property
    def _filtered(self) -> _StackedBands:
        return _stack_bands(
            self.data.filtered_by_freq,
            alpha=self.data.alpha,
            w_f=self.data.w_f,
        )

    @cached_property
    def _activation_limits(self) -> tuple[float, float]:
        return _robust_activation_limits(
            self._content.matrix,
            self._filtered.matrix,
        )

    def _activation_cmap(self):
        return _copy_cmap_with_bad(self._CMAP)

    def _effect_cmap(self):
        return _copy_cmap_with_bad(self._CMAP_DIVERGING)

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

    def _draw_unavailable_note(self, ax: Axes) -> None:
        if self._gates_available:
            return
        ax.text(
            0.995,
            0.02,
            "gate params unavailable",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=7,
            style="italic",
            color="0.25",
            bbox={
                "facecolor": "white",
                "alpha": 0.65,
                "edgecolor": "none",
                "pad": 1.2,
            },
        )

    @staticmethod
    def _align_effect_colorbar_ticks(ax: Axes, im: Axes, vmax: float) -> None:
        """Configure effect-panel colorbar tick labels with aligned decimals.

        The diverging colormap colorbar may show labels like ``-0.5``,
        ``0.0``, ``0.5`` where the minus sign on the negative label
        shifts the decimal point.  This hook forces all tick labels to
        the same number of decimal places via ``FormatStrFormatter`` so
        the decimal column is visually aligned.

        The formatter is stored on the ScalarMappable so that
        ``_apply_colorbars`` can retrieve it when creating the colorbar.
        """
        import matplotlib.ticker as ticker

        ndp = max(1, -int(np.floor(np.log10(vmax))) + 1)
        ndp = min(ndp, 3)
        im._tem_cbar_fmt = ticker.FormatStrFormatter(f"% .{ndp}f")

    def _apply_colorbars(self, colorbar_groups):
        """Override base-class colorbar application with tick formatting.

        Creates colorbars identically to the base class, then applies
        ``FormatStrFormatter`` to the effect-panel colorbar for aligned
        decimal points.
        """
        # Replicate base _apply_colorbars logic so we can capture the cbar.
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
            # Apply custom formatter if stored on the mappable.
            fmt = getattr(mappable, "_tem_cbar_fmt", None)
            if fmt is not None:
                cbar.ax.yaxis.set_major_formatter(fmt)

    # -- Panel a: content-state activations ------------------------------------

    @colorbar(group="activation", label="Activation")
    @panel()
    def content(self, ax: Axes) -> None:
        _set_panel_title(
            ax, format_panel_title("a", "Content-state activations")
        )

        vmin, vmax = self._activation_limits
        ax.imshow(
            np.ma.masked_invalid(self._content.matrix),
            aspect="auto",
            cmap=self._activation_cmap(),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        self._draw_band_separators(ax, boundaries=self._content.boundaries)
        self._draw_unavailable_note(ax)

        # Frequency-band ytick labels.
        ypos = _ytick_positions(self._content.boundaries)
        ax.set_yticks(ypos)
        ax.set_yticklabels(self._content.labels, fontsize=7, family="monospace")

        # Upper panel: no x-axis labels/ticks
        ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.margins(x=0, y=0)

    # -- Panel b: filtered-state activations -----------------------------------

    @colorbar(group="activation", label="Activation")
    @panel()
    def filtered(self, ax: Axes) -> None:
        _set_panel_title(
            ax, format_panel_title("b", "Filtered-state activations")
        )

        vmin, vmax = self._activation_limits
        ax.imshow(
            np.ma.masked_invalid(self._filtered.matrix),
            aspect="auto",
            cmap=self._activation_cmap(),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        self._draw_band_separators(ax, boundaries=self._filtered.boundaries)

        # Frequency-band ytick labels.
        ypos = _ytick_positions(self._content.boundaries)
        ax.set_yticks(ypos)
        ax.set_yticklabels(self._content.labels, fontsize=7, family="monospace")

        # Upper panel: no x-axis labels/ticks
        ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.margins(x=0, y=0)

    # -- Panel c: filter effect ------------------------------------------------

    @colorbar(group="effect", label="Filtered \u2212 content")
    @panel()
    def effect(self, ax: Axes) -> None:
        _set_panel_title(ax, format_panel_title("c", "Filter effect"))

        content = self._content.matrix
        filtered = self._filtered.matrix

        if content.shape != filtered.shape:
            raise ValueError(
                "Content and filtered LEC stacks must have the same shape "
                f"for the filter-effect panel. Got content={content.shape}, "
                f"filtered={filtered.shape}."
            )

        diff = filtered - content
        vmax = _symmetric_effect_limit(diff)

        im = ax.imshow(
            np.ma.masked_invalid(diff),
            aspect="auto",
            cmap=self._effect_cmap(),
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )

        self._draw_band_separators(ax, boundaries=self._content.boundaries)

        # Frequency-band ytick labels.
        ypos = _ytick_positions(self._content.boundaries)
        ax.set_yticks(ypos)
        ax.set_yticklabels(self._content.labels, fontsize=7, family="monospace")

        # Upper panel: no x-axis labels/ticks
        ax.set_xlabel("Time step", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, bottom=True, top=False)
        ax.margins(x=0, y=0)

        # -- Align colorbar tick decimals --------------------------------------
        # Ensure tick labels have matching decimal places so the decimal point
        # appears vertically aligned regardless of negative-sign width.
        self._align_effect_colorbar_ticks(ax, im, vmax)
