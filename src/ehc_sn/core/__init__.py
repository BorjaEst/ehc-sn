"""
Neural network models for spatial navigation.

This module contains base classes for building and training neural network models
for spatial navigation, including components for both hippocampal and entorhinal cortex
representations.
"""

from .drtp import DRTPFunction, DRTPLayer, drtp_layer

__all__ = ["DRTPFunction", "DRTPLayer", "drtp_layer"]
