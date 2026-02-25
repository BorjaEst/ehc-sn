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
    """

    WIDTH_FRAC: float = 1.0  # Use to configure get_mpl_rcParams
    HEIGHT_FRAC: float = 0.15  # Use to configure get_mpl_rcParams
    SINGLE_COL: bool = False  # Use to configure get_mpl_rcParams
    SHAREX: bool = False  # Use to configure subplot_mosaic
    SHAREY: bool = False  # Use to configure subplot_mosaic

    PRP_LAYOUT: prp.Layout = prp.Layout.ICML
    MOSAIC: Sequence[Sequence[str]] | str | None = None
    MOSAIC_KWARGS: dict[str, Any] = {}

    def __init__(self, trace: Any, ctx: FigureContext) -> None:
        self.trace = trace
        self.ctx = ctx
        self.fig: Optional[Figure] = None
        self.axdict: dict[str, Axes] = {}

    @property
    def prp_options(self) -> dict[str, Any]:
        """Return pub_ready_plots options derived from class attributes."""
        return {
            "width_frac": self.WIDTH_FRAC,
            "height_frac": self.HEIGHT_FRAC,
            "single_col": self.SINGLE_COL,
        }

    @property
    def mosaic_options(self) -> dict[str, Any]:
        """Return subplot_mosaic options derived from class attributes and MOSAIC_KWARGS."""
        options = {"sharex": self.SHAREX, "sharey": self.SHAREY}
        options.update(self.MOSAIC_KWARGS)  # User-provided MOSAIC_KWARGS override class defaults
        return options

    def plot(self) -> Figure:
        """Create, render, and return the final Matplotlib figure."""
        colorbar_groups: dict[str, dict[str, Any]] = {}
        prp_layout = self.ctx.layout if self.ctx.layout is not None else self.PRP_LAYOUT

        with ExitStack() as stack:
            # scienceplots aesthetics and prp + registry overrides
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

            # render each panel
            for panel in self._sorted_panels(panels):
                axes = self._axes_for_panel(panel)
                primary_ax = self.axdict[panel.primary]
                self._render_panel(panel, primary_ax, axes, colorbar_groups)

            self._apply_colorbars(colorbar_groups)
            return self.fig

    def ax(self, name: str) -> Axes:
        """Return the Axes for a given slot name."""
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
        panel.fn(ax)  # Rendering function is expected to modify ax in-place
        meta = getattr(panel.fn, "_tem_colorbar", None)
        if not meta:
            return

        group = meta["group"]
        label = meta.get("label", None)
        group_state = colorbar_groups.setdefault(group, {"axes": [], "mappable": None, "label": label})
        group_state["axes"].extend(axes)

        # Keep first non-empty label (avoid overwriting)
        if group_state.get("label") is None and label is not None:
            group_state["label"] = label

        # Discover a mappable (primary axis first, then additional axes)
        mappable = self._find_group_mappable(axes)

        if group_state["mappable"] is None and mappable is not None:
            group_state["mappable"] = mappable

    def _apply_colorbars(self, colorbar_groups: dict[str, dict[str, Any]]) -> None:
        """Attach grouped colorbars to the figure."""
        for group_state in colorbar_groups.values():
            mappable = group_state.get("mappable")
            axes = group_state.get("axes", [])
            if mappable is None or not axes:
                continue
            # Avoid duplicates in case the same axis is collected multiple times.
            axes = list(dict.fromkeys(axes))
            label = group_state.get("label", None)
            self.fig.colorbar(mappable, ax=axes, label=label)

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
        def sort_key(panel: _PanelSpec) -> tuple[int, int, str]:
            order = panel.order if panel.order is not None else 10_000
            return (0 if panel.order is not None else 1, order, panel.name)

        return sorted(panels, key=sort_key)

    def _validate_slot_claims(self, panels: Iterable[_PanelSpec], available: set[str]) -> None:
        slot_to_panel: dict[str, str] = {}
        for panel in panels:
            for slot in panel.slots:
                if slot not in available:
                    raise ValueError(
                        f"Panel '{panel.name}' claims unknown slot '{slot}'. "
                        f"Available slots: {sorted(available)}"
                    )
                if slot in slot_to_panel:
                    raise ValueError(
                        f"Slot '{slot}' claimed by both '{slot_to_panel[slot]}' and '{panel.name}'."
                    )
                slot_to_panel[slot] = panel.name

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
        """Generate a simple mosaic when a template does not define MOSAIC.

        The default layout stacks panels vertically and allocates one column
        per claimed slot within each panel (padding with '.' for empties).
        """
        if not panels:
            raise ValueError("Figure template defines no panels and no MOSAIC.")

        ordered = self._sorted_panels(panels)
        ncols = max(len(panel.slots) for panel in ordered)

        mosaic: list[list[str]] = []
        for panel in ordered:
            row = list(panel.slots)
            if len(row) < ncols:
                row.extend(["."] * (ncols - len(row)))
            mosaic.append(row)
        return mosaic

    def _find_panel_mappable(self, ax: Axes) -> ScalarMappable | None:
        # A) Preferred: explicit convention used by your plot functions
        mappable = getattr(ax, "_tem_colorbar_mappable", None)
        if mappable is not None:
            return mappable

        # B) Inset axes created via ax.inset_axes(...) end up here
        for child in reversed(getattr(ax, "child_axes", [])):
            mappable = getattr(child, "_tem_colorbar_mappable", None)
            if mappable is not None:
                return mappable
            if getattr(child, "images", None):
                return child.images[-1]
            if getattr(child, "collections", None):
                return child.collections[-1]

        # C) Optional fallback: common Matplotlib artists
        # matrix/matshow creates an AxesImage stored in ax.images
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
