#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=38
#SBATCH --time=48:00:00
#SBATCH --job-name=hrm-exp01
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err


# Set strict bash options for better error handling
set -euo pipefail

# Load HAICORE modules (micromamba + NVIDIA tools)
module load micromamba_icon
module load toolkit/nvidia-hpc-sdk/25.3

# Ensure we run from the repo root where .venv exists
cd "${SLURM_SUBMIT_DIR}"

# Activate the workspace venv (workspace-installed libs)
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# Set environment variables for optimal performance
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NCCL_ASYNC_ERROR_HANDLING=1
export EXP01_CONFIGURATION_PATH="config/haicore-full:4_hrm.toml"

# Run the training script with srun for proper resource allocation and binding
srun --cpu-bind=cores \
    python experiments/exp01_baseline.py \
    --trainer_strategy=ddp \
    --trainer_devices=4 \
    --trainer_num_nodes=1 \
    --trainer_accelerator=gpu \
    --max_steps=200000 \
    --val_check_interval=60 \
    --log_every_n_steps=20 \
