"""
Entorhinal-Hippocampal Complex (EHC) Spatial Navigation library.

This library provides tools for modeling spatial navigation mechanisms
based on neural representations found in the entorhinal-hippocampal complex.

The library is organized into several main components:

Modules
-------
mec : module
    Medial Entorhinal Cortex models, including grid cells.
hpc : module
    Hippocampus models, including place cells.
stdp : module
    Spike-Timing-Dependent Plasticity mechanisms.
autoencoder : module
    Autoencoder components for neural representation.
utils : module
    Utility functions and visualization tools.

Examples
--------
>>> import ehc_sn
>>> from ehc_sn.mec import GridCellsLayer, MECNetwork
>>> from ehc_sn.hpc import PlaceCellsLayer, HPCNetwork
"""

from importlib.metadata import version as _version

try:
    __version__ = _version("ehc_sn")
except Exception:  # pragma: no cover
    # package is not installed
    with open(__import__("os").path.join(__import__("os").path.dirname(__file__), "VERSION")) as f:
        __version__ = f.read().strip()

from ehc_sn import hpc, mec

__all__ = ["mec", "hpc"]
