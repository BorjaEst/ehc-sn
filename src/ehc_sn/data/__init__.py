"""Data module for spatial navigation and cognitive mapping datasets.

This module provides data generation, management, and loading functionality for
spatial navigation tasks in the entorhinal-hippocampal circuit modeling. It includes
tools for creating cognitive maps, obstacle maps, and other spatial representations
used for training and evaluating neural network models.

The data module integrates with PyTorch Lightning's DataModule framework to provide
efficient data loading, preprocessing, and augmentation for spatial navigation tasks.
All data generators support various spatial representations including binary obstacle
maps, probability maps, and multi-channel cognitive representations.

Key Features:
    - Cognitive map generation with configurable complexity and structure
    - Obstacle map creation for spatial navigation challenges
    - PyTorch Lightning DataModule integration for efficient training
    - Configurable data preprocessing and augmentation pipelines
    - Support for various spatial representation formats

Modules:
    cognitive_maps: Generation and management of cognitive spatial representations
    obstacle_maps: Creation of obstacle-based spatial navigation environments
    _base: Base classes and utilities for data module implementation

Classes:
    DataModule: Base Lightning DataModule for spatial navigation datasets
    DataModuleParams: Configuration parameters for data module initialization

Examples:
    >>> from ehc_sn.data import cognitive_maps, DataModule
    >>> from ehc_sn.data.cognitive_maps import CognitiveMapParams
    >>>
    >>> # Create cognitive map dataset
    >>> params = CognitiveMapParams(map_size=(32, 16), num_obstacles=5)
    >>> datamodule = DataModule(params)
    >>> datamodule.setup()
    >>> train_loader = datamodule.train_dataloader()

References:
    - O'Keefe, J., & Nadel, L. (1978). The hippocampus as a cognitive map.
    - Hafting, T., et al. (2005). Microstructure of a spatial map in the entorhinal cortex.
"""

from ehc_sn.data import cognitive_maps, obstacle_maps
from ehc_sn.data._base import DataModule, DataModuleParams

__all__ = [
    "cognitive_maps",
    "obstacle_maps",
    "DataModule",
    "DataModuleParams",
]
