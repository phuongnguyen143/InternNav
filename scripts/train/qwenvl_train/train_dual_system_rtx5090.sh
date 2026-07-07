#!/bin/bash
#   chmod +x scripts/train/qwenvl_train/train_dual_system_rtx5090.sh
#   bash scripts/train/qwenvl_train/train_dual_system_rtx5090.sh
#
#   sbatch with 2 GPUs, then: bash scripts/train/qwenvl_train/train_dual_system_rtx5090.sh

# #SBATCH -J internvla-dual-rtx5090
# #SBATCH -p gpu_partition
# #SBATCH -N 1
# #SBATCH --gres=gpu:rtx5090:2
# #SBATCH --cpus-per-task=16
# #SBATCH --ntasks-per-node=1
# #SBATCH -o ./slurm-%j.out
# #SBATCH -e ./slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

detect_gpus_on_node() {
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        echo "${SLURM_GPUS_ON_NODE}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}'
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
    else
        echo 2
    fi
}

NNODES="${NNODES:-1}"
_detected_gpus="$(detect_gpus_on_node)"
NPROC_PER_NODE="${NPROC_PER_NODE:-${_detected_gpus}}"
if [[ "${NPROC_PER_NODE}" -gt 2 ]]; then
    NPROC_PER_NODE=2
fi
if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    NPROC_PER_NODE=1
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    NNODES="${SLURM_NNODES:-${NNODES}}"
    NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-${NPROC_PER_NODE}}"
    if [[ "${NPROC_PER_NODE}" -gt 2 ]]; then
        NPROC_PER_NODE=2
    fi
    MASTER_ADDR="$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)"
    MASTER_PORT="${MASTER_PORT:-$((RANDOM % 101 + 20001))}"
fi

total_gpus=$((NPROC_PER_NODE * NNODES))
TARGET_EFFECTIVE_BATCH="${TARGET_EFFECTIVE_BATCH:-64}"

vln_datasets="${VLN_DATASETS:-r2r_125cm_0_30%80}"
DEFAULT_DATA_ROOT="data/intern_n1/vln_ce"
export INTERNAV_R2R_DATA_PATH="${INTERNAV_R2R_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/r2r}"
export INTERNAV_RXR_DATA_PATH="${INTERNAV_RXR_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/rxr}"
export INTERNAV_SCALEVLN_DATA_PATH="${INTERNAV_SCALEVLN_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/scalevln}"

vln_dataset_custom="${VLN_DATASETS_CUSTOM:-office_125cm_0_30}"
DEFAULT_CUSTOM_DATA_ROOT="data"
export INTERNAV_CUSTOM_BKHN_DATA_PATH="${INTERNAV_CUSTOM_BKHN_DATA_PATH:-${DEFAULT_CUSTOM_DATA_ROOT}/vr-office}"

if [[ "${LOW_MEM:-False}" == "True" ]]; then
    _default_deepspeed="scripts/train/qwenvl_train/zero2.json"
    _default_batch_size=1
    _default_resize=224
    _default_model_max_length=2048
    _default_max_pixels=78400
    _default_num_history=4
    _default_data_aug=False
    _default_torch_empty_cache=50
    _default_dataloader_workers=2
    _default_omp_threads=2
elif [[ "${HIGH_MEM:-False}" == "True" ]]; then
    _default_deepspeed="scripts/train/qwenvl_train/zero2.json"
    _default_batch_size=2
    _default_resize=336
    _default_model_max_length=8192
    _default_max_pixels=313600
    _default_num_history=8
    _default_data_aug=True
    _default_torch_empty_cache=0
    _default_dataloader_workers=4
    _default_omp_threads=8
else
    _default_deepspeed="scripts/train/qwenvl_train/zero2.json"
    _default_batch_size=2
    _default_resize=384
    _default_model_max_length=8192
    _default_max_pixels=313600
    _default_num_history=8
    _default_data_aug=True
    _default_torch_empty_cache=25
    _default_dataloader_workers=2
    _default_omp_threads=4
fi

_default_grad_accum=$(( (TARGET_EFFECTIVE_BATCH + total_gpus * _default_batch_size - 1) / (total_gpus * _default_batch_size) ))

deepspeed="${DEEPSPEED_CONFIG:-${DEEPSPEED:-${_default_deepspeed}}}"


# /home/phuongnh/khang/InternNav/checkpoints/dit/260626/only-s1/InternVLA-N1-DualVLN-office-rtx5090-v2
# /home/phuongnh/khang/InternNav/checkpoints/DualVLN-pixel-goal-v3
system2_ckpt="${SYSTEM2_CKPT:-checkpoints/DualVLN-pixel-goal-v3}"
if [[ -d "${system2_ckpt}" ]]; then
    system2_ckpt="$(cd "${system2_ckpt}" && pwd)"
elif [[ -d "${REPO_ROOT}/${system2_ckpt}" ]]; then
    system2_ckpt="$(cd "${REPO_ROOT}/${system2_ckpt}" && pwd)"
fi

# system1 options: navdp_async, nextdit_async, nextdit
system1="${SYSTEM1:-nextdit_async}"

# use_pixel_goal_for_s1=False -> VLM latent (default), | True -> pixel (x,y)
# Dual w S1 training needs traj gt on every sample:
#   pixel_goal_only=True  (required for both latent and pixel S1 modes)
use_pixel_goal_for_s1="${USE_PIXEL_GOAL_FOR_S1:-True}"
pixel_goal_only="${PIXEL_GOAL_ONLY:-True}"
lr="${LR:-2e-5}"
batch_size="${BATCH_SIZE:-${_default_batch_size}}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-${_default_grad_accum}}"
max_pixels="${MAX_PIXELS:-${_default_max_pixels}}"
min_pixels="${MIN_PIXELS:-3136}"
resize_h="${RESIZE_H:-${_default_resize}}"
resize_w="${RESIZE_W:-${_default_resize}}"
num_history="${NUM_HISTORY:-${_default_num_history}}"
model_max_length="${MODEL_MAX_LENGTH:-${_default_model_max_length}}"
data_augmentation="${DATA_AUGMENTATION:-${_default_data_aug}}"
dataloader_workers="${DATALOADER_WORKERS:-${_default_dataloader_workers}}"
torch_empty_cache_steps="${TORCH_EMPTY_CACHE_STEPS:-${_default_torch_empty_cache}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${_default_omp_threads}}"

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
    else
        echo "  WARN: nvidia-smi not found" >&2
    fi
    echo "  nnodes=${NNODES}  nproc_per_node=${NPROC_PER_NODE}  (RTX 5090 preset)"
    echo "=== End GPU preflight ==="
    echo ""
}

run_name="${RUN_NAME:-DualVLN-pixel-goal-v4}"
output_dir="${OUTPUT_DIR:-checkpoints/${run_name}}"
mkdir -p "${output_dir}"
run_log_file="${RUN_LOG_FILE:-${output_dir}/train.log}"
exec > >(tee "${run_log_file}") 2>&1

extra_args=()
if [[ "${use_pixel_goal_for_s1}" == "True" ]]; then
    extra_args+=(--use_pixel_goal_for_s1 True)
fi
if [ -n "${MAX_STEPS:-}" ]; then
    extra_args+=(--max_steps "${MAX_STEPS}")
    num_epochs="1.0"
    save_steps=1000000
    report_to="none"
else
    num_epochs="${NUM_TRAIN_EPOCHS:-10}"
    save_steps=300
    report_to="${REPORT_TO:-tensorboard}"
fi

effective_batch=$((batch_size * grad_accum_steps * total_gpus))

echo "Repo root:       ${REPO_ROOT}"
echo "System 2 ckpt:   ${system2_ckpt}"
if [[ "${system2_ckpt}" == /* ]] && [[ ! -d "${system2_ckpt}" ]]; then
    echo "ERROR: System 2 checkpoint path does not exist: ${system2_ckpt}" >&2
    echo "Download from https://huggingface.co/InternRobotics/InternVLA-N1-System2" >&2
    exit 1
fi
echo "System 1:        ${system1}"
echo "S1 goal cond:    use_pixel_goal_for_s1=${use_pixel_goal_for_s1}  pixel_goal_only=${pixel_goal_only}"
echo "DeepSpeed:       ${deepspeed}"
echo "Data root:       ${DEFAULT_DATA_ROOT}"
echo "R2R data path:   ${INTERNAV_R2R_DATA_PATH}"
if [[ ! -d "${INTERNAV_R2R_DATA_PATH}" ]]; then
    echo "ERROR: R2R data path does not exist: ${INTERNAV_R2R_DATA_PATH}" >&2
    exit 1
fi

echo "Datasets:        ${vln_datasets}"
echo "Custom datasets: ${vln_dataset_custom}"
echo "Batch size:      ${batch_size} (per device)  grad_accum=${grad_accum_steps}  effective=${effective_batch}"
echo "Resize:          ${resize_h}x${resize_w}  history=${num_history}  max_len=${model_max_length}  max_pixels=${max_pixels}"
echo "Tune System 2:   vision=True mlp=True llm=False  aug=${data_augmentation}"
echo "LOW_MEM:         ${LOW_MEM:-False}  HIGH_MEM: ${HIGH_MEM:-False}"
echo "Output dir:      ${output_dir}"
echo "Run log:         ${run_log_file}"
echo "Metrics log:     ${output_dir}/training_metrics.jsonl"
if [ -n "${MAX_STEPS:-}" ]; then
    echo "Max steps:       ${MAX_STEPS} (smoke test)"
else
    echo "Epochs:          ${num_epochs}"
fi

print_gpu_preflight

torchrun_args=(
    --nnodes="${NNODES}"
    --nproc_per_node="${NPROC_PER_NODE}"
)
if [[ -n "${SLURM_JOB_ID:-}" ]] && [[ "${NNODES}" -gt 1 ]]; then
    torchrun_args+=(
        --rdzv_id="${SLURM_JOB_ID}"
        --rdzv_backend=c10d
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
    )
else
    torchrun_args+=(
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
    )
fi

launch_cmd=(torchrun "${torchrun_args[@]}")
if [[ -n "${SLURM_JOB_ID:-}" ]] && [[ -z "${INTERNAV_NO_SRUN:-}" ]] && [[ ! -t 0 ]]; then
    launch_cmd=(srun "${launch_cmd[@]}")
fi

"${launch_cmd[@]}" \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed "${deepspeed}" \
    --model_name_or_path "${system2_ckpt}" \
    --vln_dataset_use "${vln_datasets}" \
    --vln_dataset_custom "${vln_dataset_custom}" \
    --data_flatten False \
    --remove_unused_columns False \
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
    --pixel_goal_only "${pixel_goal_only}" \
    --system1 "${system1}" \
    --output_dir "${output_dir}" \
    --num_train_epochs "${num_epochs}" \
    --per_device_train_batch_size "${batch_size}" \
    --per_device_eval_batch_size $((batch_size * 2)) \
    --gradient_accumulation_steps "${grad_accum_steps}" \
    --max_pixels "${max_pixels}" \
    --min_pixels "${min_pixels}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${save_steps}" \
    --save_total_limit 5 \
    --learning_rate "${lr}" \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-06}' \
    --logging_steps 1 \
    --logging_first_step True \
    --include_num_input_tokens_seen True \
    --model_max_length "${model_max_length}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${dataloader_workers}" \
    --run_name "${run_name}" \
    --report_to "${report_to}" \
    "${extra_args[@]}"
