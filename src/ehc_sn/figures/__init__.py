"""Figures module for ehc_sn library.

This module provides figure classes for visualizing different types of data and results
from the entorhinal-hippocampal circuit models. Each figure class encapsulates both
the plotting logic and configuration parameters.

Available figure types:
- ReconstructionTraceFigure: 1D signal reconstruction comparison
- ReconstructionMapFigure: 2D spatial data reconstruction comparison
- DecoderMontageFigure: Decoder weight montage showing individual latent unit reconstructions
- BinaryMapFigure: Binary 2D spatial data visualization
- SparsityFigure: Neural activation sparsity analysis

Key Features:
    - One class per figure type with encapsulated plotting logic
    - Pydantic-based parameter configuration and validation
    - Matplotlib/Seaborn integration for publication-quality plots
    - Modular design allowing easy extension of new figure types
    - Consistent API across all figure classes

Design Principles:
    - Each figure accepts tensors directly (no adapters needed)
    - Figures create their own canvas but can work with external axes
    - Minimal inheritance hierarchy with shared utilities in BaseFigure
    - Domain-specific figures organized by data dimensionality

Usage:
    >>> from ehc_sn.figures import ReconstructionTraceFigure, ReconstructionTraceParams
    >>>
    >>> params = ReconstructionTraceParams(n_samples=4, title="Results")
    >>> figure = ReconstructionTraceFigure(params)
    >>> fig = figure.plot(inputs, outputs)
    >>> fig.show()

Extension:
    To add new figure types:
    1. Create new module: figures/my_figure.py
    2. Inherit from BaseFigure and implement plot() method
    3. Define specific parameters class if needed
    4. Add imports to this __init__.py
"""

from ehc_sn.figures.binary_map import BinaryMapFigure, BinaryMapParams
from ehc_sn.figures.decoder_montage import DecoderMontageFigure, DecoderMontageParams
from ehc_sn.figures.reconstruction_1d import ReconstructionTraceFigure, ReconstructionTraceParams
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure, ReconstructionMapParams
from ehc_sn.figures.sparsity import SparsityFigure, SparsityParams

__all__ = [
    "BinaryMapFigure",
    "BinaryMapParams",
    "DecoderMontageFigure",
    "DecoderMontageParams",
    "ReconstructionTraceFigure",
    "ReconstructionTraceParams",
    "ReconstructionMapFigure",
    "ReconstructionMapParams",
    "SparsityFigure",
    "SparsityParams",
]
