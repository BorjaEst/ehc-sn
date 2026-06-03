"""Figure utility helpers."""

from ehc_sn.figures.utils.axes import (
    configure_environment_axes,
    mosaic_axes,
    subdivide_axes,
)
from ehc_sn.figures.utils.colors import (
    attach_aligned_colorbar_fmt,
    colormap_with_nan_color,
)
from ehc_sn.figures.utils.labels import format_panel_title, set_panel_title
from ehc_sn.figures.utils.scales import (
    build_shared_minmax,
    build_shared_norm,
    symmetric_diverging_limit,
)

__all__ = [
    "attach_aligned_colorbar_fmt",
    "build_shared_minmax",
    "build_shared_norm",
    "colormap_with_nan_color",
    "configure_environment_axes",
    "format_panel_title",
    "mosaic_axes",
    "set_panel_title",
    "subdivide_axes",
    "symmetric_diverging_limit",
]
