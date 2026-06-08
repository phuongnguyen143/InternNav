#!/bin/bash

#   chmod +x scripts/train/base_train/start_train_thor.sh
#   bash scripts/train/base_train/start_train_thor.sh --model navdp --name navdp_jetson

#   BATCH_SIZE=1 NUM_WORKERS=0 bash scripts/train/base_train/start_train_thor.sh --model navdp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export INTERNAV_JETSON_THOR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# Reduce CUDA fragmentation; SIGKILL (-9) during training usually means OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Jetson-safe training defaults (train.py reads these when INTERNAV_JETSON_THOR=1).
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

NAME="${NAME:-rdp_train}"
MODEL="${MODEL:-rdp}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1" >&2
            echo "Usage: $0 [--name RUN_NAME] [--model {cma,cma_plus,seq2seq,seq2seq_plus,rdp,navdp}]" >&2
            exit 1
            ;;
    esac
done

case $MODEL in
    cma|cma_plus|seq2seq|seq2seq_plus|rdp|navdp) ;;
    *)
        echo "Error: Unsupported model type: $MODEL" >&2
        exit 1
        ;;
esac

DEFAULT_R2R_LEROBOT="${REPO_ROOT}/data/vln_pe/traj_data/r2r"
DEFAULT_NAVDP_ROOT="${REPO_ROOT}/data/InternData-N1/vln_n1"
DEFAULT_NAVDP_JSON="${REPO_ROOT}/data/datasets/navdp_dataset_lerobot.json"

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

validate_data_paths() {
    case $MODEL in
        navdp)
            if [[ ! -d "${DEFAULT_NAVDP_ROOT}" ]]; then
                echo "ERROR: NavDP root dir does not exist: ${DEFAULT_NAVDP_ROOT}" >&2
                exit 1
            fi
            if [[ ! -f "${DEFAULT_NAVDP_JSON}" ]]; then
                echo "ERROR: NavDP dataset manifest not found: ${DEFAULT_NAVDP_JSON}" >&2
                echo "Create it or set dataset_navdp in scripts/train/base_train/configs/navdp.py" >&2
                exit 1
            fi
            scene_dirs=$(find "${DEFAULT_NAVDP_ROOT}" -maxdepth 2 -mindepth 1 -type d 2>/dev/null | wc -l)
            echo "NavDP root:      ${DEFAULT_NAVDP_ROOT} (${scene_dirs} subdirs)"
            echo "NavDP manifest:  ${DEFAULT_NAVDP_JSON}"
            ;;
        *)
            
            ;;
    esac
}

echo "Repo root:       ${REPO_ROOT}"
echo "Model:           ${MODEL}"
echo "Run name:        ${NAME}"
echo "GPUs:            1 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Batch size:      ${BATCH_SIZE}"
echo "Data workers:    ${NUM_WORKERS}"
echo "Report to:       ${REPORT_TO}"
if [[ -n "${NUM_TRAIN_EPOCHS:-}" ]]; then
    echo "Epochs:          ${NUM_TRAIN_EPOCHS} (override)"
fi


print_jetson_preflight

echo "Starting ${MODEL} training on Jetson Thor (single-process python)..."
python scripts/train/base_train/train.py \
    --name "${NAME}" \
    --model-name "${MODEL}"
