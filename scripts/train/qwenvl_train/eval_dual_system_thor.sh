#!/bin/bash
# Offline dual-system eval on LeRobot VLN data (Jetson Thor / single GPU).
#
# Run (from InternNav root):
#   chmod +x scripts/train/qwenvl_train/eval_dual_system_thor.sh
#   bash scripts/train/qwenvl_train/eval_dual_system_thor.sh
#
# Smoke test (10 batches):
#   MAX_EVAL_STEPS=10 bash scripts/train/qwenvl_train/eval_dual_system_thor.sh
#
# Evaluate a specific training run:
#   MODEL_PATH=checkpoints/InternVLA-N1-DualVLN-Jetson/checkpoint-5000 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEFAULT_DATA_ROOT="${REPO_ROOT}/data/InternData-N1/vln_pe"
export INTERNAV_R2R_DATA_PATH="${INTERNAV_R2R_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/r2r}"
export INTERNAV_RXR_DATA_PATH="${INTERNAV_RXR_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/rxr}"
export INTERNAV_SCALEVLN_DATA_PATH="${INTERNAV_SCALEVLN_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/scalevln}"

# Trained dual-system checkpoint (output of train_dual_system_thor.sh).
model_path="${MODEL_PATH:-checkpoints/InternVLA-N1-DualVLN}"
if [[ -d "${model_path}" ]]; then
    model_path="$(cd "${model_path}" && pwd)"
elif [[ -d "${REPO_ROOT}/${model_path}" ]]; then
    model_path="$(cd "${REPO_ROOT}/${model_path}" && pwd)"
fi

# system1 options: navdp_async, nextdit_async, nextdit
system1="${SYSTEM1:-navdp_async}"

batch_size=1
max_pixels="${MAX_PIXELS:-78400}"
min_pixels="${MIN_PIXELS:-3136}"
resize_h="${RESIZE_H:-224}"
resize_w="${RESIZE_W:-224}"
num_history="${NUM_HISTORY:-4}"
model_max_length="${MODEL_MAX_LENGTH:-2048}"
data_augmentation="${DATA_AUGMENTATION:-False}"
dataloader_workers="${DATALOADER_WORKERS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

print_jetson_preflight() {
    echo ""
    echo "=== Jetson preflight (shell) ==="
    echo "hostname: $(hostname)"
    if [[ -r /etc/nv_tegra_release ]]; then
        echo "tegra: $(head -1 /etc/nv_tegra_release)"
    fi
    if command -v jetson_release >/dev/null 2>&1; then
        jetson_release 2>/dev/null | head -3 | sed 's/^/  /'
    fi
    if command -v nvpmodel >/dev/null 2>&1; then
        nvpmodel -q 2>/dev/null | grep -i "power mode" | sed 's/^/  /' || true
    fi
    if command -v free >/dev/null 2>&1; then
        free -h | awk '/^Mem:/{printf "  RAM: %s used / %s total (%s avail)\n", $3, $2, $7}'
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' '{printf "  GPU: %s | VRAM %s/%s MB | util %s%% | temp %sC | power %sW\n", $1,$2,$3,$4,$5,$6}'
    fi
    echo "=== End Jetson preflight ==="
    echo ""
}

# data — use a held-out slice via the % suffix, or a different dataset key
vln_datasets="${VLN_DATASETS:-r2r_125cm_0_30%10}"

run_name="${RUN_NAME:-InternVLA-N1-DualVLN-Jetson-Eval}"
output_dir="${OUTPUT_DIR:-logs/${run_name}}"

extra_args=()
if [ -n "${MAX_EVAL_STEPS:-}" ]; then
    extra_args+=(--max_eval_steps "${MAX_EVAL_STEPS}")
fi

echo "Repo root:       ${REPO_ROOT}"
echo "Model path:      ${model_path}"
if [[ "${model_path}" == /* ]] && [[ ! -d "${model_path}" ]]; then
    echo "ERROR: Model checkpoint path does not exist: ${model_path}" >&2
    echo "Train first with scripts/train/qwenvl_train/train_dual_system_thor.sh" >&2
    exit 1
fi
echo "System 1:        ${system1}"
echo "Data root:       ${DEFAULT_DATA_ROOT}"
echo "R2R data path:   ${INTERNAV_R2R_DATA_PATH}"
if [[ ! -d "${INTERNAV_R2R_DATA_PATH}" ]]; then
    echo "ERROR: R2R data path does not exist: ${INTERNAV_R2R_DATA_PATH}" >&2
    exit 1
fi
scene_dirs=$(find "${INTERNAV_R2R_DATA_PATH}" -maxdepth 1 -mindepth 1 -type d | wc -l)
tarballs=$(find "${INTERNAV_R2R_DATA_PATH}" -maxdepth 1 -name '*.tar.gz' | wc -l)
echo "R2R scenes:      ${scene_dirs} extracted dirs, ${tarballs} tarballs"
if [[ "${scene_dirs}" -eq 0 ]]; then
    echo "WARN: no extracted scene dirs — extract tarballs before eval" >&2
fi
echo "Datasets:        ${vln_datasets}"
echo "Resize:          ${resize_h}x${resize_w}  history=${num_history}  max_len=${model_max_length}"
echo "Augmentation:    ${data_augmentation}"
echo "Output dir:      ${output_dir}"
echo "Metrics log:     ${output_dir}/eval_metrics.jsonl"
if [ -n "${MAX_EVAL_STEPS:-}" ]; then
    echo "Max eval steps:  ${MAX_EVAL_STEPS} (smoke test)"
else
    echo "Max eval steps:  full dataset"
fi

print_jetson_preflight

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
