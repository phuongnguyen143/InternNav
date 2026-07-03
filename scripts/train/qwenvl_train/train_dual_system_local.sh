#!/usr/bin/env bash

set -euo pipefail

# Local torchrun launcher for InternVLA-N1 dual-system (VLN) training.

RUN_NAME="InternVLA-N1-DualVLN-BKHN-finetune"
OUTPUT_DIR="checkpoints/${RUN_NAME}-local"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=""
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

DEEPSPEED_CONFIG="scripts/train/qwenvl_train/zero2.json"
MODEL_PATH="/home/lenguyen1/hoangpqn/vln/InternNav/checkpoints/base_model/InternVLA-N1-w-NavDP"
VLN_DATASETS="bkhn_125cm_0_30"

SYSTEM1="nextdit_async"

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

Train InternVLA-N1 dual-system (System1+System2) locally with torchrun.

Options:
  --name NAME             Run name used for logs/checkpoints. Default: ${RUN_NAME}
  --output-dir DIR        Output directory. Default: ${OUTPUT_DIR}
  --gpus IDS              Comma-separated local GPU ids. Default: ${GPU_IDS}
  --num-gpus N            Number of torchrun processes. Default: count from --gpus
  --master-addr ADDR      Torch distributed master address. Default: ${MASTER_ADDR}
  --master-port PORT      Torch distributed master port. Default: ${MASTER_PORT}
  --deepspeed PATH        DeepSpeed config path. Default: ${DEEPSPEED_CONFIG}
  --no-deepspeed          Run without passing a DeepSpeed config
  --model-path PATH       System2 checkpoint path (used as base). Default: ${MODEL_PATH}
  --datasets LIST         Comma-separated VLN dataset list. Default: ${VLN_DATASETS}
  --system1 NAME          System1 module: nextdit_async, navdp_async, nextdit. Default: ${SYSTEM1}
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
  $0 --gpus 0,1 --batch-size 1 --grad-accum-steps 2 --report-to wandb
  $0 --model-path /path/to/InternVLA-N1-System2 --no-deepspeed
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
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            VLN_DATASETS="$2"
            shift 2
            ;;
        --system1)
            SYSTEM1="$2"
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

if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Warning: System2 checkpoint path does not exist locally: ${MODEL_PATH}"
    echo "         Pass --model-path PATH if your InternVLA-N1-System2 checkpoint lives elsewhere."
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MASTER_ADDR
export MASTER_PORT
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TRAIN_SCRIPT="internnav/trainer/internvla_n1_trainer.py"
TRAIN_ARGS=(
    --model_name_or_path "${MODEL_PATH}"
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
echo "  system2 checkpoint: ${MODEL_PATH}"
echo "  system1: ${SYSTEM1}"
echo "  datasets: ${VLN_DATASETS}"
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