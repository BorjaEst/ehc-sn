from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tomli
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pydantic import BaseModel, Field


class ItemMemoryStateConfig(BaseModel):
    cmap_hpc: str = Field(
        default="viridis",
        description="Colormap for hippocampal place cell activity",
    )
    cmap_mec: str = Field(
        default="plasma",
        description="Colormap for MEC grid cell activity",
    )
    figsize: Tuple[int, int] = Field(
        default=(15, 10),
        description="Figure size for item memory state plots",
    )


class MemoryRetrievalConfig(BaseModel):
    n_frames: int = Field(
        default=5,
        description="Number of frames to visualize in memory retrieval",
    )
    cmap_hpc: str = Field(
        default="viridis",
        description="Colormap for hippocampal place cell activity",
    )
    cmap_mec: str = Field(
        default="coolwarm",
        description="Colormap for MEC grid cell activity",
    )
    figsize: Tuple[int, int] = Field(
        default=(15, 10),
        description="Figure size for memory retrieval visualization",
    )
