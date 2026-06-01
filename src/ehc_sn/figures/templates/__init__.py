"""Figure templates package.

Each template module exposes a ``plot(trace, ctx)`` function.
"""

from ehc_sn.figures.templates import mec_autocorr_mosaic, mec_grid_metrics

__all__ = ["mec_autocorr_mosaic", "mec_grid_metrics"]
