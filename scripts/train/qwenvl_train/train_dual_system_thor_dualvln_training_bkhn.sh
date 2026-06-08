#!/bin/bash
# System 1 NavDP 
#
# Run 
#   chmod +x scripts/train/qwenvl_train/train_dual_system_thor.sh
#   bash scripts/train/qwenvl_train/train_dual_system_thor.sh
#
# Smoke test (1 step, small dataset):
#   MAX_STEPS=1 bash scripts/train/qwenvl_train/train_dual_system_thor.sh
#
# If zero3_offload_opt OOMs, try full CPU offload:
#   DEEPSPEED_CONFIG=scripts/train/qwenvl_train/zero3_offload_torch.json MAX_STEPS=1 bash ...
#
# If you have enough VRAM for everything on GPU:
#   DEEPSPEED_CONFIG=scripts/train/qwenvl_train/zero2.json bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# Reduce CUDA fragmentation; SIGKILL (-9) during training usually means OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEFAULT_DATA_ROOT="${REPO_ROOT}/data/InternData-N1/vln_ce"
export INTERNAV_R2R_DATA_PATH="${INTERNAV_R2R_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/r2r}"
export INTERNAV_RXR_DATA_PATH="${INTERNAV_RXR_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/rxr}"
export INTERNAV_SCALEVLN_DATA_PATH="${INTERNAV_SCALEVLN_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/scalevln}"

# Dual-system loads full System2 + trains NavDP — prefer optimizer CPU offload on Jetson.
deepspeed="${DEEPSPEED_CONFIG:-scripts/train/qwenvl_train/zero3_offload_opt_torch.json}"

# Frozen System 2 checkpoint (local path or HF hub id).
system2_ckpt="${SYSTEM2_CKPT:-checkpoints/InternVLA-N1-System2}"
if [[ -d "${system2_ckpt}" ]]; then
    system2_ckpt="$(cd "${system2_ckpt}" && pwd)"
elif [[ -d "${REPO_ROOT}/${system2_ckpt}" ]]; then
    system2_ckpt="$(cd "${REPO_ROOT}/${system2_ckpt}" && pwd)"
fi

# system1 options: navdp_async, nextdit_async, nextdit
system1="${SYSTEM1:-navdp_async}"

lr="${LR:-1e-4}"
batch_size=1
grad_accum_steps="${GRAD_ACCUM_STEPS:-1}"
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

# data
vln_datasets="${VLN_DATASETS:-r2r_125cm_0_30%10}"

run_name="${RUN_NAME:-InternVLA-N1-DualVLN-Jetson}"
output_dir="${OUTPUT_DIR:-checkpoints/${run_name}}"

extra_args=()
if [ -n "${MAX_STEPS:-}" ]; then
    extra_args+=(--max_steps "${MAX_STEPS}")
    num_epochs="1.0"
    save_steps=1000000
    report_to="none"
else
    num_epochs="${NUM_TRAIN_EPOCHS:-0.05}"
    save_steps=5000
    report_to="${REPORT_TO:-none}"
fi

echo "Repo root:       ${REPO_ROOT}"
echo "System 2 ckpt:   ${system2_ckpt}"
if [[ "${system2_ckpt}" == /* ]] && [[ ! -d "${system2_ckpt}" ]]; then
    echo "ERROR: System 2 checkpoint path does not exist: ${system2_ckpt}" >&2
    echo "Download from https://huggingface.co/InternRobotics/InternVLA-N1-System2" >&2
    exit 1
fi
echo "System 1:        ${system1} (train NavDP, System 2 frozen)"
echo "DeepSpeed:       ${deepspeed}"
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
    echo "WARN: no extracted scene dirs — extract tarballs before training" >&2
fi
echo "Datasets:        ${vln_datasets}"
echo "Resize:          ${resize_h}x${resize_w}  history=${num_history}  max_len=${model_max_length}"
echo "Tune System 2:   vision=False mlp=False llm=False  aug=${data_augmentation}"
echo "Grad accum:      ${grad_accum_steps}"
echo "Output dir:      ${output_dir}"
echo "Metrics log:     ${output_dir}/training_metrics.jsonl"
if [ -n "${MAX_STEPS:-}" ]; then
    echo "Max steps:       ${MAX_STEPS} (smoke test)"
else
    echo "Epochs:          ${num_epochs}"
fi

print_jetson_preflight

torchrun --nnodes=1 --nproc_per_node=1 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed "${deepspeed}" \
    --model_name_or_path "${system2_ckpt}" \
    --vln_dataset_use "${vln_datasets}" \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
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
    --num_train_epochs "${num_epochs}" \
    --per_device_train_batch_size "${batch_size}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${grad_accum_steps}" \
    --max_pixels "${max_pixels}" \
    --min_pixels "${min_pixels}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${save_steps}" \
    --save_total_limit 2 \
    --learning_rate "${lr}" \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
    --logging_steps 1 \
    --logging_first_step True \
    --include_num_input_tokens_seen True \
    --model_max_length "${model_max_length}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${dataloader_workers}" \
    --run_name "${run_name}" \
    --report_to "${report_to}" \
    "${extra_args[@]}"
