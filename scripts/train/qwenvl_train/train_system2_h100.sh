#!/bin/bash
# System 2 (VLM planner) training on H100.
#
#   chmod +x scripts/train/qwenvl_train/train_system2_h100.sh
#   FREEZE_ALL=False bash scripts/train/qwenvl_train/train_system2_h100.sh
#
#   MAX_STEPS=1 FREEZE_ALL=True bash scripts/train/qwenvl_train/train_system2_h100.sh

#
#   NPROC_PER_NODE=8 bash scripts/train/qwenvl_train/train_system2_h100.sh
#
#
#   sbatch scripts/train/qwenvl_train/train_system2_h100.sh
#
#   if we currently on srun
#   srun --pty --partition=main --nodes=1 --ntasks=1 --gpus=nvidia_h100_80gb_hbm3:1 \
#        --cpus-per-task=16 --mem=128G --time=48:00:00 bash -i
#   bash scripts/train/qwenvl_train/train_system2_h100.sh

# #SBATCH -J internvla-system2-h100
# #SBATCH -p gpu_partition
# #SBATCH -N 1
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=16
# #SBATCH --ntasks-per-node=1
# #SBATCH -o ./slurm-%j.out
# #SBATCH -e ./slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    NNODES="${SLURM_NNODES:-${NNODES}}"
    NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-${NPROC_PER_NODE}}"
    MASTER_ADDR="$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)"
    MASTER_PORT="${MASTER_PORT:-$((RANDOM % 101 + 20001))}"
fi

DEFAULT_DATA_ROOT="/mnt/data/sftp/data/tungns30/intern_n1/vln_ce"
export INTERNAV_R2R_DATA_PATH="${INTERNAV_R2R_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/r2r}"
export INTERNAV_RXR_DATA_PATH="${INTERNAV_RXR_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/rxr}"
export INTERNAV_SCALEVLN_DATA_PATH="${INTERNAV_SCALEVLN_DATA_PATH:-${DEFAULT_DATA_ROOT}/traj_data/scalevln}"

total_gpus=$((NPROC_PER_NODE * NNODES))
deepspeed="scripts/train/qwenvl_train/zero2.json"

llm="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
if [[ -d "${llm}" ]]; then
    llm="$(cd "${llm}" && pwd)"
elif [[ -d "${REPO_ROOT}/${llm}" ]]; then
    llm="$(cd "${REPO_ROOT}/${llm}" && pwd)"
fi

lr="${LR:-2e-5}"
vision_tower_lr="${VISION_TOWER_LR:-5e-6}"
batch_size="${BATCH_SIZE:-8}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-8}"
max_pixels="${MAX_PIXELS:-313600}"
min_pixels="${MIN_PIXELS:-3136}"
resize_h="${RESIZE_H:-384}"
resize_w="${RESIZE_W:-384}"
num_history="${NUM_HISTORY:-8}"
model_max_length="${MODEL_MAX_LENGTH:-8192}"
data_augmentation="${DATA_AUGMENTATION:-True}"
dataloader_workers="${DATALOADER_WORKERS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

if [[ "${FREEZE_ALL:-False}" == "True" ]]; then
    tune_mm_vision=False
    tune_mm_mlp=False
    tune_mm_llm=False
else
    tune_mm_vision="${TUNE_MM_VISION:-True}"
    tune_mm_mlp="${TUNE_MM_MLP:-True}"
    tune_mm_llm="${TUNE_MM_LLM:-False}"
fi
frozen_smoke=false
if [[ "${tune_mm_vision}" == "False" && "${tune_mm_mlp}" == "False" && "${tune_mm_llm}" == "False" ]]; then
    frozen_smoke=true
fi
if [[ "${frozen_smoke}" == "true" && "${FREEZE_ALL:-False}" != "True" && -z "${MAX_STEPS:-}" ]]; then
    echo "ERROR: All System2 tune flags are False (vision/mlp/llm) but FREEZE_ALL is not True." >&2
    echo "       Unset TUNE_MM_* overrides or set FREEZE_ALL=False for real training:" >&2
    echo "         unset FREEZE_ALL TUNE_MM_VISION TUNE_MM_MLP TUNE_MM_LLM" >&2
    echo "         FREEZE_ALL=False bash scripts/train/qwenvl_train/train_system2_h100.sh" >&2
    exit 1
fi

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
    echo "  nnodes=${NNODES}  nproc_per_node=${NPROC_PER_NODE}"
    echo "=== End GPU preflight ==="
    echo ""
}

vln_datasets="${VLN_DATASETS:-r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30}"

run_name="${RUN_NAME:-InternVLA-N1-System2-train}"
output_dir="${OUTPUT_DIR:-checkpoints/${run_name}}"

extra_args=()
if [ -n "${MAX_STEPS:-}" ]; then
    extra_args+=(--max_steps "${MAX_STEPS}")
    num_epochs="1.0"
    save_steps=1000000
    report_to="none"
else
    num_epochs="${NUM_TRAIN_EPOCHS:-1.0}"
    save_steps=5000
    report_to="${REPORT_TO:-tensorboard}"
fi

echo "Repo root:       ${REPO_ROOT}"
echo "Model:           ${llm}"
if [[ "${llm}" == /* ]] && [[ ! -d "${llm}" ]]; then
    echo "ERROR: Local model path does not exist: ${llm}" >&2
    exit 1
fi
if [[ "${frozen_smoke}" == "true" ]]; then
    echo "Mode:            frozen smoke (no DeepSpeed, no optimizer updates)"
else
    echo "DeepSpeed:       ${deepspeed}"
fi
echo "Data root:       ${DEFAULT_DATA_ROOT}"
echo "R2R data path:   ${INTERNAV_R2R_DATA_PATH}"
if [[ ! -d "${INTERNAV_R2R_DATA_PATH}" ]]; then
    echo "ERROR: R2R data path does not exist: ${INTERNAV_R2R_DATA_PATH}" >&2
    exit 1
fi

echo "Datasets:        ${vln_datasets}"
effective_batch=$((batch_size * grad_accum_steps * NPROC_PER_NODE * NNODES))
echo "Batch size:      ${batch_size} (per device)  grad_accum=${grad_accum_steps}  effective=${effective_batch}"
echo "Launch:          ${NNODES} node(s) x ${NPROC_PER_NODE} GPU(s)"
echo "Resize:          ${resize_h}x${resize_w}  history=${num_history}  max_len=${model_max_length}"
echo "Attention:       ${ATTN_IMPLEMENTATION}"
echo "FREEZE_ALL:      ${FREEZE_ALL:-False}"
echo "Tune System 2:   vision=${tune_mm_vision} mlp=${tune_mm_mlp} llm=${tune_mm_llm}  aug=${data_augmentation}"
echo "Output dir:      ${output_dir}"
echo "Metrics log:     ${output_dir}/training_metrics.jsonl"
if [ -n "${MAX_STEPS:-}" ]; then
    echo "Max steps:       ${MAX_STEPS} (smoke test)"
else
    echo "Epochs:          ${num_epochs}"
fi

print_gpu_preflight

deepspeed_args=()
if [[ "${frozen_smoke}" != "true" ]]; then
    deepspeed_args=(--deepspeed "${deepspeed}")
fi

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
    "${deepspeed_args[@]}" \
    --model_name_or_path "${llm}" \
    --vln_dataset_use "${vln_datasets}" \
    --data_flatten False \
    --tune_mm_vision "${tune_mm_vision}" \
    --tune_mm_mlp "${tune_mm_mlp}" \
    --tune_mm_llm "${tune_mm_llm}" \
    --bf16 \
    --num_history "${num_history}" \
    --data_augmentation "${data_augmentation}" \
    --resize_h "${resize_h}" \
    --resize_w "${resize_w}" \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
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
    --vision_tower_lr "${vision_tower_lr}" \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_first_step True \
    --include_num_input_tokens_seen True \
    --model_max_length "${model_max_length}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${dataloader_workers}" \
    --run_name "${run_name}" \
    --report_to "${report_to}" \
    "${extra_args[@]}"
