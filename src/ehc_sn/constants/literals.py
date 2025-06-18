"""Literal string and numeric constants used throughout the library."""

from typing import Final, Literal

# Region names as strings
DG_NAME: Final[str] = "dentate_gyrus"
CA1_NAME: Final[str] = "ca1"
CA2_NAME: Final[str] = "ca2"
CA3_NAME: Final[str] = "ca3"
SUB_NAME: Final[str] = "subiculum"
MEC_II_NAME: Final[str] = "mec_layer_ii"
MEC_III_NAME: Final[str] = "mec_layer_iii"
MEC_VB_NAME: Final[str] = "mec_layer_vb"

# Connection types
CONNECTION_TRAINABLE: Final[str] = "trainable"
CONNECTION_FIXED: Final[str] = "fixed"

# Model types
ModelTypeOptions = Literal["baseline", "sparse_encoder", "full"]
TrainingMethodOptions = Literal["backpropagation", "drtp"]
