"""The configuration module for the EHC-SN model."""

import os

import torch

# Device configuration for PyTorch hardware acceleration
DEVICE = os.getenv("DEVICE", "cuda")
device = DEVICE if torch.cuda.is_available() else "cpu"

# Set the default floating-point precision for matrix multiplication
FLOAT_MATMUL_PRECISION = os.getenv("FLOAT_MATMUL_PRECISION", "medium")
torch.set_float32_matmul_precision(FLOAT_MATMUL_PRECISION)

# Enable or disable the use of the cuDNN backend for operations
CUDNN_BENCHMARK = os.getenv("CUDNN_BENCHMARK", "False")
cudnn_benchmark = bool(CUDNN_BENCHMARK)
torch.backends.cudnn.benchmark = cudnn_benchmark
