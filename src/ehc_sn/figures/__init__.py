"""Visualization and plotting tools for entorhinal-hippocampal circuit analysis.

This module provides comprehensive visualization capabilities for analyzing and
presenting results from entorhinal-hippocampal circuit models. It includes tools
for plotting cognitive maps, model performance metrics, neural activations, and
other spatial navigation-related visualizations.

The figures module integrates with matplotlib and seaborn to provide publication-
quality plots and interactive visualizations for model analysis, debugging, and
result presentation. All plotting functions are designed to work seamlessly with
the data structures and outputs from the EHC modeling framework.

Key Features:
    - Cognitive map visualization with customizable styling and overlays
    - Model performance plotting (loss curves, accuracy metrics, etc.)
    - Neural activation visualization (firing patterns, connectivity maps)
    - Spatial navigation trajectory plotting and analysis
    - Publication-ready figure generation with consistent styling

Modules:
    cognitive_maps: Visualization tools for spatial cognitive representations

Visualization Principles:
    - Consistent color schemes and styling across all plots
    - Support for both static and interactive visualizations
    - Configurable plot parameters via Pydantic models
    - Integration with matplotlib/seaborn ecosystem
    - Publication-quality output formatting

Examples:
    >>> from ehc_sn.figures import cognitive_maps
    >>> from ehc_sn.figures.cognitive_maps import CognitiveMapFigure
    >>>
    >>> # Create and display cognitive map visualization
    >>> fig = CognitiveMapFigure(title="Spatial Navigation Results")
    >>> fig.plot_obstacle_map(obstacle_data)
    >>> fig.show()

References:
    - Visualization best practices for neuroscience data
    - Spatial navigation analysis and presentation methods
"""

from ehc_sn.figures import cognitive_maps
