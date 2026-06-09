#!/bin/bash
#   chmod +x scripts/train/qwenvl_train/eval_dual_system_h100.sh
#   bash scripts/train/qwenvl_train/eval_dual_system_h100.sh
#
#   MAX_EVAL_STEPS=10 bash scripts/train/qwenvl_train/eval_dual_system_h100.sh
#
#   sbatch scripts/train/qwenvl_train/eval_dual_system_h100.sh
#
#   CUDA_VISIBLE_DEVICES=1 MODEL_PATH=checkpoints/InternVLA-N1-DualVLN bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# --- SLURM (no-op when not submitted via sbatch) ---
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEFAULT_DATA_ROOT="${REPO_ROOT}/data/InternData-N1/vln_pe"
export INTERNAV_R2R_DATA_PATH="${INTERNAV_R2R_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/r2r}"
export INTERNAV_RXR_DATA_PATH="${INTERNAV_RXR_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/rxr}"
export INTERNAV_SCALEVLN_DATA_PATH="${INTERNAV_SCALEVLN_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/scalevln}"

model_path="${MODEL_PATH:-checkpoints/InternVLA-N1-DualVLN}"
if [[ -d "${model_path}" ]]; then
    model_path="$(cd "${model_path}" && pwd)"
elif [[ -d "${REPO_ROOT}/${model_path}" ]]; then
    model_path="$(cd "${REPO_ROOT}/${model_path}" && pwd)"
fi

# system1 options: navdp_async, nextdit_async, nextdit
system1="${SYSTEM1:-navdp_async}"

batch_size="${BATCH_SIZE:-4}"
max_pixels="${MAX_PIXELS:-313600}"
min_pixels="${MIN_PIXELS:-3136}"
resize_h="${RESIZE_H:-384}"
resize_w="${RESIZE_W:-384}"
num_history="${NUM_HISTORY:-8}"
model_max_length="${MODEL_MAX_LENGTH:-8192}"
data_augmentation="${DATA_AUGMENTATION:-False}"
dataloader_workers="${DATALOADER_WORKERS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

print_gpu_preflight() {
    echo ""
    echo "=== GPU preflight (shell) ==="
    echo "hostname: $(hostname)"
    if command -v free >/dev/null 2>&1; then
        free -h | awk '/^Mem:/{printf "  RAM: %s used / %s total (%s avail)\n", $3, $2, $7}'
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' '{printf "  GPU %s: %s | VRAM %s/%s MB | util %s%% | temp %sC | power %sW\n", $1,$2,$3,$4,$5,$6,$7}'
        echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    else
        echo "  WARN: nvidia-smi not found" >&2
    fi
    echo "=== End GPU preflight ==="
    echo ""
}

vln_datasets="${VLN_DATASETS:-r2r_125cm_0_30%30,r2r_60cm_15_15%30,rxr_125cm_0_30%30,rxr_60cm_15_15%30,scalevln_125cm_0_30%30,scalevln_60cm_30_30%30}"

run_name="${RUN_NAME:-InternVLA-N1-DualVLN-H100-Eval}"
output_dir="${OUTPUT_DIR:-logs/${run_name}}"

extra_args=()
if [ -n "${MAX_EVAL_STEPS:-}" ]; then
    extra_args+=(--max_eval_steps "${MAX_EVAL_STEPS}")
fi

echo "Repo root:       ${REPO_ROOT}"
echo "Model path:      ${model_path}"
if [[ "${model_path}" == /* ]] && [[ ! -d "${model_path}" ]]; then
    echo "ERROR: Model checkpoint path does not exist: ${model_path}" >&2
    exit 1
fi
echo "System 1:        ${system1}"
echo "Data root:       ${DEFAULT_DATA_ROOT}"
echo "R2R data path:   ${INTERNAV_R2R_DATA_PATH}"
if [[ ! -d "${INTERNAV_R2R_DATA_PATH}" ]]; then
    echo "ERROR: R2R data path does not exist: ${INTERNAV_R2R_DATA_PATH}" >&2
    exit 1
fi

echo "Datasets:        ${vln_datasets}"
echo "Batch size:      ${batch_size}"
echo "Resize:          ${resize_h}x${resize_w}  history=${num_history}  max_len=${model_max_length}"
echo "Augmentation:    ${data_augmentation}"
echo "Output dir:      ${output_dir}"
echo "Metrics log:     ${output_dir}/eval_metrics.jsonl"
echo "TensorBoard:     ${output_dir}/tensorboard"
if [ -n "${MAX_EVAL_STEPS:-}" ]; then
    echo "Max eval steps:  ${MAX_EVAL_STEPS} (smoke test)"
else
    echo "Max eval steps:  full dataset"
fi

print_gpu_preflight

python internnav/trainer/internvla_n1_evaluator.py \
    --model_name_or_path "${model_path}" \
    --vln_dataset_use "${vln_datasets}" \
    --data_flatten False \
    --bf16 \
    --num_history "${num_history}" \
    --data_augmentation "${data_augmentation}" \
    --resize_h "${resize_h}" \
    --resize_w "${resize_w}" \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only True \
    --system1 "${system1}" \
    --output_dir "${output_dir}" \
    --per_device_eval_batch_size "${batch_size}" \
    --max_pixels "${max_pixels}" \
    --min_pixels "${min_pixels}" \
    --model_max_length "${model_max_length}" \
    --dataloader_num_workers "${dataloader_workers}" \
    "${extra_args[@]}"
