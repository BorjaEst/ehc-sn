#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=38
#SBATCH --time=48:00:00
#SBATCH --job-name=hrm-v1-baseline.gpu4
#SBATCH --output=slurm-%x-%j.gpu4.out
#SBATCH --error=slurm-%x-%j.gpu4.err


# Set strict bash options for better error handling
set -euo pipefail

# Avoid inheriting conflicting modules from the submission environment.
module purge

# Load HAICORE modules (NVIDIA tools)
module load toolkit/nvidia-hpc-sdk/25.3

# Ensure we run from the repo root where .venv exists
cd "${SLURM_SUBMIT_DIR}"

# Activate the workspace venv (workspace-installed libs)
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# Set environment variables for optimal performance
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HRM_V1_CONFIGURATION_PATH="config/training.hrm-v1.toml"

# Precision reproducibility: must be set before CUDA kernels initialise.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

# Run the training script with srun for proper resource allocation and binding
srun --cpu-bind=cores \
    python scripts/training/hrm_v1_baseline.py \
    --trainer_strategy=ddp \
    --trainer_devices=4 \
    --trainer_num_nodes=1 \
    --trainer_accelerator=gpu \
    --max_steps=200000 \
    --val_check_interval=100 \
    --log_every_n_steps=20
