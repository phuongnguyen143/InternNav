#!/bin/bash
#SBATCH --job-name=dualvln-eval
#SBATCH --partition=main
#SBATCH --nodelist=worker-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
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

# Evaluation configuration
export CUDA_VISIBLE_DEVICES=0

export MODEL_PATH="checkpoints/InternVLA-N1-DualVLN-train-only-30deg-scratch-navdp-v1/checkpoint-1000"
export SYSTEM1="navdp_async"
export BATCH_SIZE=8
export DATALOADER_WORKERS=8

export VLN_DATASETS="r2r_125cm_0_30%0"
export VLN_DATASETS_CUSTOM="bkhn_125cm_0_30"

export INTERNAV_R2R_DATA_PATH="/mnt/data/sftp/data/tungns30/intern_n1/vln_ce/traj_data/r2r"
export INTERNAV_RXR_DATA_PATH="/mnt/data/sftp/data/tungns30/intern_n1/vln_ce/traj_data/rxr"
export INTERNAV_SCALEVLN_DATA_PATH="/mnt/data/sftp/data/tungns30/intern_n1/vln_ce/traj_data/scalevln"
export INTERNAV_CUSTOM_BKHN_DATA_PATH="/mnt/data/sftp/data/khangnh11/bk_ver2.0_test"

export RUN_NAME="InternVLA-N1-DualVLN-H100-Eval-w-NavDP-30deg"
export OUTPUT_DIR="logs/${RUN_NAME}"

# Uncomment for a quick smoke test
# export MAX_EVAL_STEPS=10

bash scripts/train/qwenvl_train/eval_dual_system_h100.sh

echo "========================================"
echo "Evaluation completed"
echo "End time: $(date)"
echo "Output:   ${OUTPUT_DIR}"
echo "========================================"

