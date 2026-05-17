from __future__ import annotations

from abc import ABC
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import pub_ready_plots as prp
import scienceplots  # noqa: F401 (registers "science", "nature", ...)
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from ehc_sn.figures.registry import FigureContext


@dataclass(frozen=True)
class _PanelSpec:
    name: str
    fn: Any
    slots: tuple[str, ...]
    primary: str
    order: int | None


class BaseFigureTemplate(ABC):
    """Base class for multi-panel figure templates.

    Subclasses define a MOSAIC (optional) and panel methods annotated with @panel.
    Receives prepared data (``self.data``) extracted by the selector layer.
    """

    WIDTH_FRAC: float = 1.0
    HEIGHT_FRAC: float = 0.15
    SINGLE_COL: bool = False
    SHAREX: bool = False
    SHAREY: bool = False

    PRP_LAYOUT: prp.Layout = prp.Layout.ICML
    MOSAIC: Sequence[Sequence[str]] | str | None = None
    MOSAIC_KWARGS: dict[str, Any] = {}

    def __init__(self, data: Any, ctx: FigureContext) -> None:
        self.data = data
        self.ctx = ctx
        self.fig: Optional[Figure] = None
        self.axdict: dict[str, Axes] = {}

    @property
    def prp_options(self) -> dict[str, Any]:
        return {
            "width_frac": self.WIDTH_FRAC,
            "height_frac": self.HEIGHT_FRAC,
            "single_col": self.SINGLE_COL,
        }

    @property
    def mosaic_options(self) -> dict[str, Any]:
        options = {"sharex": self.SHAREX, "sharey": self.SHAREY}
        options.update(self.MOSAIC_KWARGS)
        return options

    def plot(self) -> Figure:
        """Create, render, and return the final Matplotlib figure."""
        colorbar_groups: dict[str, dict[str, Any]] = {}
        prp_layout = self.ctx.layout if self.ctx.layout is not None else self.PRP_LAYOUT

        with ExitStack() as stack:
            stack.enter_context(plt.style.context(list(self.ctx.styles)))

            context_kwargs: dict[str, Any] = {"nrows": 1, "ncols": 1}
            if self.ctx.dpi is not None:
                context_kwargs["dpi"] = self.ctx.dpi

            cm = prp.get_context(layout=prp_layout, **self.prp_options, **context_kwargs)
            fig, axs = stack.enter_context(cm)
            for ax in axs.ravel() if not isinstance(axs, Axes) else [axs]:
                ax.remove()

            self.fig = fig
            panels = self._discover_panels()
            self.axdict = self._create_layout(self.fig, panels)
            self._validate_slot_claims(panels, set(self.axdict.keys()))

            for panel in self._sorted_panels(panels):
                axes = self._axes_for_panel(panel)
                primary_ax = self.axdict[panel.primary]
                self._render_panel(panel, primary_ax, axes, colorbar_groups)

            self._apply_colorbars(colorbar_groups)
            return self.fig

    def ax(self, name: str) -> Axes:
        if name not in self.axdict:
            raise KeyError(f"Unknown slot '{name}'. Available slots: {sorted(self.axdict.keys())}")
        return self.axdict[name]

    def _render_panel(
        self,
        panel: _PanelSpec,
        ax: Axes,
        axes: Sequence[Axes],
        colorbar_groups: dict[str, dict[str, Any]],
    ) -> None:
        panel.fn(ax)
        meta = getattr(panel.fn, "_tem_colorbar", None)
        if not meta:
            return

        group = meta["group"]
        label = meta.get("label", None)
        tick_labelsize = meta.get("tick_labelsize", None)
        group_state = colorbar_groups.setdefault(
            group,
            {"axes": [], "mappable": None, "label": label, "tick_labelsize": tick_labelsize},
        )
        group_state["axes"].extend(axes)

        if group_state.get("label") is None and label is not None:
            group_state["label"] = label
        if group_state.get("tick_labelsize") is None and tick_labelsize is not None:
            group_state["tick_labelsize"] = tick_labelsize

        mappable = self._find_group_mappable(axes)
        if group_state["mappable"] is None and mappable is not None:
            group_state["mappable"] = mappable

    def _apply_colorbars(self, colorbar_groups: dict[str, dict[str, Any]]) -> None:
        for group_state in colorbar_groups.values():
            mappable = group_state.get("mappable")
            axes = group_state.get("axes", [])
            if mappable is None or not axes:
                continue
            axes = list(dict.fromkeys(axes))
            label = group_state.get("label", None)
            cbar = self.fig.colorbar(mappable, ax=axes, label=label)
            tick_labelsize = group_state.get("tick_labelsize", None)
            if tick_labelsize is not None:
                cbar.ax.tick_params(labelsize=tick_labelsize)

    def _discover_panels(self) -> list[_PanelSpec]:
        panels: list[_PanelSpec] = []
        for name in dir(self):
            attr = getattr(self, name, None)
            meta = getattr(attr, "_tem_panel", None)
            if meta is None:
                continue
            slots = tuple(meta.get("slots", (name,)))
            primary = meta.get("primary") or slots[0]
            if primary not in slots:
                slots = (primary, *slots)
            order = meta.get("order")
            panels.append(_PanelSpec(name=name, fn=attr, slots=slots, primary=primary, order=order))
        return panels

    def _sorted_panels(self, panels: Iterable[_PanelSpec]) -> list[_PanelSpec]:
        def sort_key(p: _PanelSpec) -> tuple[int, int, str]:
            order = p.order if p.order is not None else 10_000
            return (0 if p.order is not None else 1, order, p.name)

        return sorted(panels, key=sort_key)

    def _validate_slot_claims(self, panels: Iterable[_PanelSpec], available: set[str]) -> None:
        slot_to_panel: dict[str, str] = {}
        for p in panels:
            for slot in p.slots:
                if slot not in available:
                    raise ValueError(
                        f"Panel '{p.name}' claims unknown slot '{slot}'. "
                        f"Available slots: {sorted(available)}"
                    )
                if slot in slot_to_panel:
                    raise ValueError(
                        f"Slot '{slot}' claimed by both '{slot_to_panel[slot]}' and '{p.name}'."
                    )
                slot_to_panel[slot] = p.name

    def _axes_for_panel(self, panel: _PanelSpec) -> list[Axes]:
        primary_ax = self.axdict[panel.primary]
        axes = [primary_ax]
        for slot in panel.slots:
            if slot == panel.primary:
                continue
            axes.append(self.axdict[slot])
        return axes

    def _create_layout(self, fig: Figure, panels: Sequence[_PanelSpec]) -> dict[str, Axes]:
        if self.MOSAIC is None:
            mosaic = self._default_mosaic_for_panels(panels)
        else:
            mosaic = self.MOSAIC
        return fig.subplot_mosaic(mosaic, **self.mosaic_options)

    def _default_mosaic_for_panels(self, panels: Sequence[_PanelSpec]) -> list[list[str]]:
        if not panels:
            raise ValueError("Figure template defines no panels and no MOSAIC.")

        ordered = self._sorted_panels(panels)
        ncols = max(len(p.slots) for p in ordered)

        mosaic: list[list[str]] = []
        for p in ordered:
            row = list(p.slots)
            if len(row) < ncols:
                row.extend(["."] * (ncols - len(row)))
            mosaic.append(row)
        return mosaic

    def _find_panel_mappable(self, ax: Axes) -> ScalarMappable | None:
        mappable = getattr(ax, "_tem_colorbar_mappable", None)
        if mappable is not None:
            return mappable
        for child in reversed(getattr(ax, "child_axes", [])):
            mappable = getattr(child, "_tem_colorbar_mappable", None)
            if mappable is not None:
                return mappable
            if getattr(child, "images", None):
                return child.images[-1]
            if getattr(child, "collections", None):
                return child.collections[-1]
        if getattr(ax, "images", None):
            return ax.images[-1]
        if getattr(ax, "collections", None):
            return ax.collections[-1]
        return None

    def _find_group_mappable(self, axes: Sequence[Axes]) -> ScalarMappable | None:
        for axis in axes:
            mappable = self._find_panel_mappable(axis)
            if mappable is not None:
                return mappable
        return None
