"""PFC path-memory probe figure — linear decoding evidence from z_H and z_L.

Single-panel (or two-panel) figure showing per-step balanced accuracy for
z_H and z_L from the pfc_path_memory_probe artifact, plus summary annotation.

The figure reads from a persisted probe artifact (NPZ + JSON), not from
the full dense trace.  This is the first probe-backed figure in the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.eval.probes import ProbeArtifact, load_probe_artifact
from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext

# Colour scheme — consistent with other H/L figures.
H_COLOR = "tab:blue"
L_COLOR = "tab:orange"
CHANCE_LINE = "grey"


# =============================================================================
def _resolve_probe_artifact(ctx: FigureContext) -> ProbeArtifact:
    """Locate and load the probe artifact from the figure context.

    Resolves ``<artifact_path>/probes/pfc_path_memory_probe.npz`` from
    ``ctx.artifact_path``.

    Raises
    ------
    ValueError
        If ``ctx.artifact_path`` is ``None``.
    FileNotFoundError
        If the expected NPZ file does not exist on disk.
    """
    if ctx.artifact_path is None:
        raise ValueError(
            "pfc_path_memory_probe requires FigureContext.artifact_path "
            "to locate the probe artifact."
        )

    npz_path = Path(ctx.artifact_path) / "probes" / "pfc_path_memory_probe.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing probe artifact for pfc_path_memory_probe: "
            f"{npz_path}. "
            f"Run compute_and_persist_probes_for_artifact() first."
        )

    return load_probe_artifact(npz_path=npz_path)


# =============================================================================
def plot(trace: object, ctx: FigureContext) -> Figure:
    """Render the PFC path-memory probe figure.

    Args:
        trace: Ignored — this figure reads from probe artifacts, not traces.
        ctx: Figure context with probe artifact location.

    Returns:
        Matplotlib ``Figure``.
    """
    artifact = _resolve_probe_artifact(ctx)
    return PfcPathMemoryProbeFigure(artifact, ctx).plot()


# =============================================================================
class PfcPathMemoryProbeFigure(BaseFigureTemplate):
    """Two-panel figure: balanced accuracy over steps + summary annotation."""

    HEIGHT_FRAC: float = 0.22
    MOSAIC = [
        ["ba_plot", "summary"],
    ]
    MOSAIC_KWARGS = {
        "width_ratios": [2.5, 1.0],
    }

    def __init__(self, artifact: ProbeArtifact, ctx: FigureContext) -> None:
        super().__init__(artifact, ctx)
        self._artifact = artifact

    # ------------------------------------------------------------------
    @panel()
    def ba_plot(self, ax: Axes) -> None:
        """Panel A: balanced accuracy over recurrent steps for z_H and z_L."""
        arr = self._artifact.arrays
        meta = self._artifact.metadata
        steps = np.arange(meta["steps"])

        h_ba = arr["h_balanced_accuracy_mean"]
        h_std = arr["h_balanced_accuracy_std"]
        l_ba = arr["l_balanced_accuracy_mean"]
        l_std = arr["l_balanced_accuracy_std"]

        ax.plot(
            steps, h_ba, color=H_COLOR, linewidth=1.5, label="z_H (high-level)"
        )
        ax.fill_between(
            steps,
            h_ba - h_std,
            h_ba + h_std,
            color=H_COLOR,
            alpha=0.15,
        )

        ax.plot(
            steps, l_ba, color=L_COLOR, linewidth=1.5, label="z_L (low-level)"
        )
        ax.fill_between(
            steps,
            l_ba - l_std,
            l_ba + l_std,
            color=L_COLOR,
            alpha=0.15,
        )

        # Chance line.
        ax.axhline(
            y=0.5,
            color=CHANCE_LINE,
            linestyle="--",
            linewidth=0.8,
            label="chance",
        )

        # Best-step markers.
        h_best = int(arr["h_best_step"])
        l_best = int(arr["l_best_step"])
        ax.scatter(
            h_best,
            h_ba[h_best],
            color=H_COLOR,
            marker="*",
            s=60,
            zorder=5,
        )
        ax.scatter(
            l_best,
            l_ba[l_best],
            color=L_COLOR,
            marker="*",
            s=60,
            zorder=5,
        )

        ax.set_xlabel("Recurrent step")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title("Path memory linear decoding")
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize="x-small", loc="lower right")
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    @panel()
    def summary(self, ax: Axes) -> None:
        """Panel B: compact annotation table with key metrics."""
        arr = self._artifact.arrays
        meta = self._artifact.metadata
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        h_best = int(arr["h_best_step"])
        l_best = int(arr["l_best_step"])
        h_mean = float(arr["h_balanced_accuracy_mean"].mean())
        l_mean = float(arr["l_balanced_accuracy_mean"].mean())

        lines = [
            f"z_H mean BA: {h_mean:.3f}",
            f"z_H best BA: {arr['h_balanced_accuracy_mean'][h_best]:.3f} (step {h_best})",
            f"z_H final BA: {arr['h_final_score']:.3f}",
            "",
            f"z_L mean BA: {l_mean:.3f}",
            f"z_L best BA: {arr['l_balanced_accuracy_mean'][l_best]:.3f} (step {l_best})",
            f"z_L final BA: {arr['l_final_score']:.3f}",
            "",
            f"n_tokens: {meta['n_tokens']}",
            f"n_positive: {meta['n_positive']}",
            f"n_splits: {meta['n_splits']}",
            f"features: {meta['n_features']}",
        ]

        title = "Path memory probe"
        ax.text(
            0.05,
            0.95,
            title,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            verticalalignment="top",
        )
        ax.text(
            0.05,
            0.50,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="center",
            fontfamily="monospace",
        )
