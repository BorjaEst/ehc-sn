""" """

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class NavigationTaskOutput:
    """ """

    task_action_logits: Tensor
    obs_logits: Tensor
