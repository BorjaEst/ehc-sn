"""Figure utility functions."""

from __future__ import annotations

import re

from matplotlib.axes import Axes


def format_panel_title(letter: str, title: str) -> str:
    """Return a lowercase title-prefix panel label.

    Produces strings like ``"a. Content-state activity"``, following the
    professional multi-panel convention of lowercase bold letters embedded
    in the panel title rather than floating black letters.

    Parameters
    ----------
    letter : str
        Single lowercase letter (e.g. ``"a"``, ``"b"``).
    title : str
        Panel title (e.g. ``"Content-state activity"``).

    Returns
    -------
    str
        Formatted string ``"{letter}. {title}"``.

    Raises
    ------
    ValueError
        If *letter* is not a single ASCII lowercase letter, or if *title* is
        empty or whitespace-only.
    TypeError
        If either argument is not a string.
    """
    if not isinstance(letter, str):
        raise TypeError(f"letter must be a str, got {type(letter).__name__}")
    if not isinstance(title, str):
        raise TypeError(f"title must be a str, got {type(title).__name__}")
    if not re.match(r"^[a-z]$", letter):
        raise ValueError(
            f"letter must be a single lowercase letter, got {letter!r}"
        )
    if not title or title.strip() == "":
        raise ValueError("title must not be empty")
    return f"{letter}. {title}"


def add_panel_label(ax: Axes, label: str, fontsize: int = 12) -> None:
    """Add a visible panel letter to the top-left corner of an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    label : str
        Panel label (e.g. ``"A"``, ``"B"``).
    fontsize : int
        Font size in points.
    """
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
        fontsize=fontsize,
    )
