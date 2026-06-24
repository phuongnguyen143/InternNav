#!/bin/bash
#SBATCH --job-name=dualvln-train
#SBATCH --partition=main
#SBATCH --nodelist=worker-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/khangnh11/VR/logs/%j_%x.out
#SBATCH --error=/home/khangnh11/VR/logs/%j_%x.err

set -euo pipefail

# Repository location
REPO_ROOT="/home/khangnh11/VR/InternNav"
cd "${REPO_ROOT}"

# Ensure the log directory exists
mkdir -p /home/khangnh11/VR/logs

# Activate the environment
# source .venvs/bin/activate

# Optional: print job information
echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Job name:     ${SLURM_JOB_NAME}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "Repository:   ${REPO_ROOT}"
echo "========================================"

# Training configuration (1x H100)
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
# /home/khangnh11/VR/InternNav/checkpoints/InternVLA-N1-DualVLN-bkhn-ver3.0-v1
export SYSTEM2_CKPT="checkpoints/InternVLA-N1-w-NavDP"
export SYSTEM1="navdp_async"

export VLN_DATASETS="r2r_125cm_0_30%0"
export VLN_DATASETS_CUSTOM="bkhn_125cm_0_30"

export INTERNAV_R2R_DATA_PATH="/mnt/data/sftp/data/tungns30/intern_n1/vln_ce/traj_data/r2r"
export INTERNAV_RXR_DATA_PATH="/mnt/data/sftp/data/tungns30/intern_n1/vln_ce/traj_data/rxr"
export INTERNAV_SCALEVLN_DATA_PATH="/mnt/data/sftp/data/tungns30/intern_n1/vln_ce/traj_data/scalevln"
export INTERNAV_CUSTOM_BKHN_DATA_PATH="/mnt/data/sftp/data/khangnh11/vr-office"

export RUN_NAME="InternVLA-N1-DualVLN-office-round1-v1"
export OUTPUT_DIR="checkpoints/${RUN_NAME}"
export REPORT_TO="tensorboard"

# Uncomment for a quick smoke test
# export MAX_STEPS=1

bash scripts/train/qwenvl_train/train_dual_system_h100.sh

echo "========================================"
echo "Training completed"
echo "End time: $(date)"
echo "Output:   ${OUTPUT_DIR}"
echo "========================================"
