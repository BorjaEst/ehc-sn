from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import lightning as L
import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.optim import Optimizer

from ehc_sn.controllers.rl import RLController, RLControllerConfig
from ehc_sn.data.schema import CHANNEL_SOLUTION, O_ID
from ehc_sn.data.transforms import channels_to_grid
from ehc_sn.envs.mazehard import EnvConfig, MazeHardEnv
from ehc_sn.heads.rl import RLLossConfig, RLLossHead
from ehc_sn.metrics import build_train_metrics, build_val_metrics, update_metrics_from_step
from ehc_sn.metrics.routes import RL_EPISODE_ROUTES, RL_STEP_ROUTES
from ehc_sn.metrics.traces import build_trace_spec
from ehc_sn.models.hrm.hrm_v2 import Batch, HRModelV2, ModelSettings_V2
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.training.step_loop import StepLoop


# =================================================================================================
def supervised_maze_tokenize(  # ------------------------------------------------------------------
    channels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:  # fmt: skip
    """Convert raw maze channels into flattened input/label token sequences.

    Uses :func:`~ehc_sn.data.transforms.channels_to_grid` to merge topology,
    start, and goals into a canonical ``int32`` grid, then flattens to a 1-D
    token sequence.  The label sequence overwrites solution-path cells with
    :data:`O_ID` (HRM-private supervision token).

    Args:
        channels: Raw NPZ channel dict (as returned by ``MazeDataset``).

    Returns:
        ``{"inputs": int32 (H*W,), "labels": int32 (H*W,)}``.
    """
    grid = channels_to_grid(channels)["grid"]  # (H, W) int32
    inputs = grid.ravel()
    labels = inputs.copy()
    if CHANNEL_SOLUTION in channels:
        labels[channels[CHANNEL_SOLUTION].ravel() > 0] = O_ID
    return {"inputs": inputs, "labels": labels}
