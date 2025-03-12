"""The configuration module for the EHC-SN model."""

import os
import torch


# Device configuration for PyTorch hardware acceleration
DEVICE = os.getenv("DEVICE", "cuda")
device = DEVICE if torch.cuda.is_available() else "cpu"
