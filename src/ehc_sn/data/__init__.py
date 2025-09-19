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
    simple_example: 1D synthetic data generation for autoencoder baselines
    obstacle_maps: 2D obstacle cognitive map generation for spatial navigation
    _base: Base classes and utilities for data module implementation

Classes:
    DataModule: Base Lightning DataModule for spatial navigation datasets
    DataModuleParams: Configuration parameters for data module initialization

Examples:
    >>> from ehc_sn.data.simple_example import DataGenerator, DataParams
    >>> from ehc_sn.data.obstacle_maps import ObstacleMapGenerator, ObstacleMapParams
    >>> from ehc_sn.core.datamodule import DataModuleParams
    >>>
    >>> # Create 1D synthetic dataset
    >>> data_params = DataModuleParams(batch_size=32, train_size=1000)
    >>> simple_params = DataParams(input_dim=50, noise_level=0.1)
    >>> datamodule = DataGenerator.create(data_params, simple_params)
    >>>
    >>> # Create 2D obstacle map dataset
    >>> obstacle_params = ObstacleMapParams(grid_size=(16, 16), obstacle_density=0.3)
    >>> datamodule = ObstacleMapGenerator.create(data_params, obstacle_params)

References:
    - O'Keefe, J., & Nadel, L. (1978). The hippocampus as a cognitive map.
    - Hafting, T., et al. (2005). Microstructure of a spatial map in the entorhinal cortex.
"""

# Core data module components
from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.data.obstacle_maps import DataGenerator as ObstacleMapGenerator
from ehc_sn.data.obstacle_maps import DataParams as ObstacleMapParams

# Data generators
from ehc_sn.data.simple_example import DataGenerator, DataParams

__all__ = [
    # Core components
    "BaseDataModule",
    "DataModuleParams",
    # 1D synthetic data
    "DataGenerator",
    "DataParams",
    # 2D obstacle maps
    "ObstacleMapGenerator",
    "ObstacleMapParams",
]
