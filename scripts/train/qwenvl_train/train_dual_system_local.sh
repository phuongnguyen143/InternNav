#!/usr/bin/env bash
 
set -euo pipefail
 
# Local torchrun launcher for train_dual_system.sh.
 
RUN_NAME="InternVLA-N1-DualVLN"
OUTPUT_DIR="checkpoints/InternVLA-N1-DualVLN-local-v2"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=""
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
 
DEEPSPEED_CONFIG="scripts/train/qwenvl_train/zero2.json"
SYSTEM2_CKPT="checkpoints/InternVLA-N1-System2"
SYSTEM1="nextdit_async"
# VLN_DATASETS="r2r_125cm_0_30%30,r2r_60cm_15_15%30,rxr_125cm_0_30%30,rxr_60cm_15_15%30,scalevln_125cm_0_30%30,scalevln_60cm_30_30%30"
VLN_DATASETS="bkhn_125cm_0_30"
 
LR="1e-4"
BATCH_SIZE="2"
GRAD_ACCUM_STEPS="1"
EPOCHS="3.0"
SAVE_STEPS="5000"
MAX_PIXELS="313600"
MIN_PIXELS="3136"
DATALOADER_NUM_WORKERS="8"
REPORT_TO="${REPORT_TO:-none}"
 
EXTRA_ARGS=()
 
usage() {
    cat <<EOF
Usage: $0 [options] [-- extra trainer args]
 
Train the InternVLA-N1 dual-system model locally with torchrun.
 
Options:
  --name NAME             Run name used for logs/checkpoints. Default: ${RUN_NAME}
  --output-dir DIR        Output directory. Default: ${OUTPUT_DIR}
  --gpus IDS              Comma-separated local GPU ids. Default: ${GPU_IDS}
  --num-gpus N            Number of torchrun processes. Default: count from --gpus
  --master-addr ADDR      Torch distributed master address. Default: ${MASTER_ADDR}
  --master-port PORT      Torch distributed master port. Default: ${MASTER_PORT}
  --deepspeed PATH        DeepSpeed config path. Default: ${DEEPSPEED_CONFIG}
  --no-deepspeed          Run without passing a DeepSpeed config
  --system2-ckpt PATH     System2 checkpoint/model path. Default: ${SYSTEM2_CKPT}
  --system1 NAME          System1 backend. Default: ${SYSTEM1}
  --datasets LIST         Comma-separated VLN dataset list. Default: ${VLN_DATASETS}
  --batch-size N          Per-device train batch size. Default: ${BATCH_SIZE}
  --grad-accum-steps N    Gradient accumulation steps. Default: ${GRAD_ACCUM_STEPS}
  --lr VALUE              Learning rate. Default: ${LR}
  --epochs VALUE          Number of train epochs. Default: ${EPOCHS}
  --save-steps N          Save checkpoint every N steps. Default: ${SAVE_STEPS}
  --workers N             Dataloader workers. Default: ${DATALOADER_NUM_WORKERS}
  --report-to TARGET      Trainer reporting target. Default: ${REPORT_TO}
  -h, --help              Show this help message.
 
Examples:
  $0 --gpus 0
  $0 --gpus 0,1 --master-port 12345 --report-to wandb
  $0 --system2-ckpt /path/to/InternVLA-N1-System2 --no-deepspeed
EOF
}
 
count_gpus() {
    local ids="${1// /}"
 
    if [[ -z "${ids}" ]]; then
        echo 1
        return
    fi
 
    local -a parts
    IFS=',' read -r -a parts <<<"${ids}"
    echo "${#parts[@]}"
}
 
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            RUN_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --deepspeed)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --no-deepspeed)
            DEEPSPEED_CONFIG=""
            shift
            ;;
        --system2-ckpt)
            SYSTEM2_CKPT="$2"
            shift 2
            ;;
        --system1)
            SYSTEM1="$2"
            shift 2
            ;;
        --datasets)
            VLN_DATASETS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum-steps)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --save-steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --workers)
            DATALOADER_NUM_WORKERS="$2"
            shift 2
            ;;
        --report-to)
            REPORT_TO="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            exit 1
            ;;
    esac
done
 
if [[ -z "${NUM_GPUS}" ]]; then
    NUM_GPUS="$(count_gpus "${GPU_IDS}")"
fi
 
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "Error: --num-gpus must be a positive integer, got '${NUM_GPUS}'"
    exit 1
fi
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"
 
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="checkpoints/${RUN_NAME}"
fi
 
TORCHRUN_LOG_DIR="${OUTPUT_DIR}/torchrun_logs"
mkdir -p "${TORCHRUN_LOG_DIR}"
 
if ! command -v torchrun >/dev/null 2>&1; then
    echo "Error: torchrun was not found. Activate the training environment first."
    exit 1
fi
 
if [[ -n "${DEEPSPEED_CONFIG}" && ! -f "${DEEPSPEED_CONFIG}" ]]; then
    echo "Error: DeepSpeed config was not found: ${DEEPSPEED_CONFIG}"
    exit 1
fi
 
if [[ ! -e "${SYSTEM2_CKPT}" ]]; then
    echo "Warning: system2 checkpoint path does not exist locally: ${SYSTEM2_CKPT}"
    echo "         Pass --system2-ckpt PATH if your checkpoint lives elsewhere."
fi
 
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
 
TRAIN_SCRIPT="internnav/trainer/internvla_n1_trainer.py"
TRAIN_ARGS=(
    --model_name_or_path "${SYSTEM2_CKPT}"
    --vln_dataset_use "${VLN_DATASETS}"
    --data_flatten False
    --tune_mm_vision False
    --tune_mm_mlp False
    --tune_mm_llm False
    --bf16
    --num_history 8
    --data_augmentation True
    --resize_h 384
    --resize_w 384
    --sample_step 4
    --num_future_steps 4
    --predict_step_num 32
    --pixel_goal_only True
    --system1 "${SYSTEM1}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${EPOCHS}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --per_device_eval_batch_size "$((BATCH_SIZE * 2))"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --max_pixels "${MAX_PIXELS}"
    --min_pixels "${MIN_PIXELS}"
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 5
    --learning_rate "${LR}"
    --weight_decay 0
    --warmup_ratio 0.003
    --max_grad_norm 1
    --lr_scheduler_type "cosine_with_min_lr"
    --lr_scheduler_kwargs '{"min_lr": 1e-05}'
    --logging_steps 1
    --model_max_length 8192
    --gradient_checkpointing True
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --run_name "${RUN_NAME}"
    --report_to "${REPORT_TO}"
)
 
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
    TRAIN_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}" "${TRAIN_ARGS[@]}")
fi
 
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
 
echo "Starting local InternVLA-N1 dual-system training"
echo "  run name: ${RUN_NAME}"
echo "  output dir: ${OUTPUT_DIR}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  nproc_per_node: ${NUM_GPUS}"
echo "  rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  torchrun logs: ${TORCHRUN_LOG_DIR}"
echo "  system2 checkpoint: ${SYSTEM2_CKPT}"
echo "  system1: ${SYSTEM1}"
echo "  report_to: ${REPORT_TO}"
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
    echo "  deepspeed: ${DEEPSPEED_CONFIG}"
else
    echo "  deepspeed: disabled"
fi
 
torchrun \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --log_dir="${TORCHRUN_LOG_DIR}" \
    --tee=3 \
    "${TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}"