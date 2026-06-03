"""LEC / MEC / HPC representational similarity — report-facing figure template.

Tests whether LEC is more organised by content identity (observation), MEC
by structural identity (location), and HPC shows mixed / conjunctive
organisation.

This figure asks whether each internal representation groups time steps by
sensory identity or by arena location.  For each stream, we compute cosine
similarity between all pairs of time-step states.  The *content effect* is
defined as:

    mean(same observation) − mean(different observation)

The *location effect* is defined as:

    mean(same location) − mean(different location)

A TEM-style factorisation predicts a relative dissociation:
    - LEC should carry more content information.
    - MEC should carry more structural/location information.
    - HPC may combine both.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.analysis.tem_representations import (
    ContentStructureRSAResult,
)
from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.lec import (
    select_lec_content_structure_rsa,
)
from ehc_sn.figures.utils.labels import format_panel_title
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return LECContentStructureRSAFigure(
        select_lec_content_structure_rsa(trace, ctx), ctx
    ).plot()


class LECContentStructureRSAFigure(BaseFigureTemplate):
    """Content vs. location organisation across LEC, MEC, and HPC.

    Panels:
        A — LEC: same-observation vs different-observation similarity
        B — MEC: same-location vs different-location similarity
        C — HPC: content effect and location effect side by side
        D — Summary: content effect vs location effect for all three streams
    """

    HEIGHT_FRAC: float = 0.22
    MOSAIC = [["by_obs", "by_loc", "summary_contrast"]]
    MOSAIC_KWARGS = {"width_ratios": [1.0, 1.0, 0.9]}

    _SYS_NAMES = ["LEC", "MEC", "HPC"]
    _OBS_COLORS = {"Same obs.": "tab:blue", "Diff. obs.": "tab:red"}
    _LOC_COLORS = {"Same loc.": "tab:green", "Diff. loc.": "tab:red"}

    def __init__(
        self, data: ContentStructureRSAResult, ctx: FigureContext
    ) -> None:
        super().__init__(data, ctx)

    # -- Panel A: same/different observation similarity per stream --------------

    @panel()
    def by_obs(self, ax: Axes) -> None:
        ax.set_title(
            format_panel_title("a", "Observation grouping"), fontsize=10
        )
        names = self._SYS_NAMES
        x = np.arange(len(names))
        width = 0.35

        same = [
            self.data.lec.same_observation,
            self.data.mec.same_observation,
            self.data.hpc.same_observation,
        ]
        diff = [
            self.data.lec.different_observation,
            self.data.mec.different_observation,
            self.data.hpc.different_observation,
        ]
        ax.bar(
            x - width / 2,
            same,
            width,
            label="Same obs.",
            color="tab:blue",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            diff,
            width,
            label="Diff. obs.",
            color="tab:red",
            alpha=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Cos. sim.", fontsize=9)
        ax.legend(fontsize=6)

    # -- Panel B: same/different location similarity per stream -----------------

    @panel()
    def by_loc(self, ax: Axes) -> None:
        ax.set_title(format_panel_title("b", "Location grouping"), fontsize=10)
        names = self._SYS_NAMES
        x = np.arange(len(names))
        width = 0.35

        same = [
            self.data.lec.same_location,
            self.data.mec.same_location,
            self.data.hpc.same_location,
        ]
        diff = [
            self.data.lec.different_location,
            self.data.mec.different_location,
            self.data.hpc.different_location,
        ]
        ax.bar(
            x - width / 2,
            same,
            width,
            label="Same loc.",
            color="tab:green",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            diff,
            width,
            label="Diff. loc.",
            color="tab:red",
            alpha=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Cos. sim.", fontsize=9)
        ax.legend(fontsize=6)

    # -- Panel C: content minus location effect per stream ----------------------

    @panel()
    def summary_contrast(self, ax: Axes) -> None:
        ax.set_title(
            format_panel_title("c", "Content vs. location"), fontsize=10
        )
        names = self._SYS_NAMES
        x = np.arange(len(names))
        width = 0.35

        content_ef = [
            self.data.lec.content_effect,
            self.data.mec.content_effect,
            self.data.hpc.content_effect,
        ]
        location_ef = [
            self.data.lec.location_effect,
            self.data.mec.location_effect,
            self.data.hpc.location_effect,
        ]

        ax.bar(
            x - width / 2,
            content_ef,
            width,
            label="Content effect",
            color="tab:blue",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            location_ef,
            width,
            label="Location effect",
            color="tab:green",
            alpha=0.8,
        )
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Effect\n(same − diff)", fontsize=8)
        ax.legend(fontsize=6)
