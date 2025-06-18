"""Enumerations used throughout the entorhinal-hippocampal circuit library."""

from enum import Enum, auto


class RegionType(Enum):
    """Neural regions in the entorhinal-hippocampal circuit."""

    MEC_LAYER_II = auto()
    MEC_LAYER_III = auto()
    MEC_LAYER_VB = auto()
    DG = auto()
    CA3 = auto()
    CA2 = auto()
    CA1 = auto()
    SUBICULUM = auto()


class TrainingMethod(Enum):
    """Training methods supported by the models."""

    BACKPROPAGATION = auto()
    DRTP = auto()  # Direct Random Target Projection
