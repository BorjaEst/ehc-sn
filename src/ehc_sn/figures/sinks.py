"""Figure persistence helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import matplotlib.figure as mpl_figure
from lightning.pytorch.loggers import TensorBoardLogger


def make_figure_path(base_dir: Path, name: str, step: Optional[int] = None) -> Path:
    """Build a figure output path.

    Args:
        base_dir: Base output directory.
        name: Base filename.
        step: Optional step number.

    Returns:
        Path for the PDF output.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-step={step}" if step is not None else ""
    return base_dir / "figures" / f"{name}{suffix}.pdf"


def save_pdf(fig: mpl_figure.Figure, path: Path) -> None:
    """Save a figure as a PDF.

    Args:
        fig: Matplotlib figure.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def save_png(fig: mpl_figure.Figure, path: Path, *, dpi: int) -> None:
    """Save a figure as a PNG with an explicit export DPI.

    Args:
        fig: Matplotlib figure.
        path: Output path.
        dpi: Export DPI for rasterization.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)


def _persist_named_figure_artifacts(
    fig: mpl_figure.Figure,
    *,
    output_dir: Path,
    case_index: int,
    case_id: str,
    default_filename: str,
    save_pdf_enabled: bool,
    save_png_enabled: bool,
    png_dpi: int,
) -> None:
    """Persist one rendered figure with deterministic case/scoped naming."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_stem = _sanitize_filename_component(Path(default_filename).stem)
    stem = (
        f"{case_index:04d}-"
        f"{_sanitize_filename_component(case_id)}-"
        f"{figure_stem}"
    )
    if save_pdf_enabled:
        save_pdf(fig, output_dir / f"{stem}.pdf")
    if save_png_enabled:
        save_png(fig, output_dir / f"{stem}.png", dpi=png_dpi)


def _sanitize_filename_component(value: str) -> str:
    """Normalize a string into a safe filename component."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "case"


def log_tensorboard_figure(
    logger: TensorBoardLogger,
    tag: str,
    fig: mpl_figure.Figure,
    global_step: Optional[int] = None,
) -> None:
    """Log a Matplotlib figure to TensorBoard.

    Args:
        logger: TensorBoard logger instance.
        tag: TensorBoard tag.
        fig: Matplotlib figure.
        global_step: Optional step.
    """
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "add_figure"):
        experiment.add_figure(tag, fig, global_step=global_step)
