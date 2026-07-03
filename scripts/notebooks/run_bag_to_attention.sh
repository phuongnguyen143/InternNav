#!/usr/bin/env bash
# Extract rosbag RGB-D -> prepare inference scene -> run attention-map inference.
#
# Example:
#   ./run_bag_to_attention.sh \
#     --bag /home/lenguyen1/hoangpqn/vln/DATA/debug/20260630/internnav_20260630_111135 \
#     --instruction "walk straight then turn right at the door" \
#     --instruction /path/to/instruction.txt \
#     --trim-start 18 --trim-end 7 \
#     --stride 3 --look-down-every 10 \
#     --attn-layers 6,15,24
#
# Skip steps (reuse existing outputs):
#   ./run_bag_to_attention.sh --bag ... --instruction ... --skip-extract --skip-prepare

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS_ROOT="$PROJECT_ROOT/scripts"

# Defaults (ZED waist camera bag)
RGB_TOPIC="/camera/waist_front_zed_stream/left/color/rect/image/compressed"
DEPTH_TOPIC="/camera/waist_front_zed_stream/depth/depth_registered"
TRIM_START="18"
TRIM_END="7"
STRIDE="3"
LOOK_DOWN_EVERY="10"
ATTN_LAYERS="6,15,24"
DEVICE="cuda:1"
MODEL_PATH="$PROJECT_ROOT/checkpoints/base_model/InternVLA-N1-DualVLN"

BAG_PATH=""
INSTRUCTION=""
EXTRACT_DIR=""
SCENE_DIR=""
SAVE_DIR=""
RUN_NAME=""

SKIP_EXTRACT=0
SKIP_PREPARE=0
SKIP_INFER=0
NO_TRIM=0

usage() {
    cat <<'EOF'
Usage: run_bag_to_attention.sh --bag BAG_DIR --instruction INSTRUCTION.txt [options]

Required:
  --bag PATH              Rosbag2 directory (folder with metadata.yaml + .db3)
  --instruction TEXT|FILE Navigation instruction text, or path to instruction.txt

Optional paths (auto-derived from --bag if omitted):
  --extract-dir PATH      extract_bag_frames output dir
  --scene-dir PATH        prepare_extract_for_inference output dir
  --save-dir PATH         inference annotated output base dir

Extraction:
  --rgb-topic TOPIC       (default: ZED compressed RGB)
  --depth-topic TOPIC     (default: ZED depth_registered)
  --trim-start SEC        Seconds to skip at bag start (default: 18)
  --trim-end SEC          Seconds to skip at bag end (default: 7)
  --no-trim               Extract full bag (trim start/end = 0)

Prepare:
  --stride N              Keep every Nth extracted frame (default: 3)
  --look-down-every N     Mark every Nth output frame as look_down (default: 10)

Inference:
  --attn-layers LIST      Comma-separated layers (default: 6,15,24)
  --device DEV            Torch device (default: cuda:1)
  --model-path PATH       InternVLA checkpoint dir

Skip steps:
  --skip-extract
  --skip-prepare
  --skip-infer

Environment:
  Source ROS before running if needed, e.g.:
    source /opt/ros/humble/setup.bash
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag) BAG_PATH="$2"; shift 2 ;;
        --instruction) INSTRUCTION="$2"; shift 2 ;;
        --extract-dir) EXTRACT_DIR="$2"; shift 2 ;;
        --scene-dir) SCENE_DIR="$2"; shift 2 ;;
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --rgb-topic) RGB_TOPIC="$2"; shift 2 ;;
        --depth-topic) DEPTH_TOPIC="$2"; shift 2 ;;
        --trim-start) TRIM_START="$2"; shift 2 ;;
        --trim-end) TRIM_END="$2"; shift 2 ;;
        --no-trim) NO_TRIM=1; shift ;;
        --stride) STRIDE="$2"; shift 2 ;;
        --look-down-every) LOOK_DOWN_EVERY="$2"; shift 2 ;;
        --attn-layers) ATTN_LAYERS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --skip-extract) SKIP_EXTRACT=1; shift ;;
        --skip-prepare) SKIP_PREPARE=1; shift ;;
        --skip-infer) SKIP_INFER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$BAG_PATH" || -z "$INSTRUCTION" ]]; then
    echo "ERROR: --bag and --instruction are required." >&2
    usage
    exit 1
fi

BAG_PATH="$(cd "$BAG_PATH" && pwd)"

if [[ -f "$INSTRUCTION" ]]; then
    INSTRUCTION="$(readlink -f "$INSTRUCTION")"
    INSTRUCTION_LABEL="$INSTRUCTION"
elif [[ ${#INSTRUCTION} -gt 80 ]]; then
    INSTRUCTION_LABEL="(inline) ${INSTRUCTION:0:80}..."
else
    INSTRUCTION_LABEL="(inline) $INSTRUCTION"
fi

if [[ ! -d "$BAG_PATH" ]]; then
    echo "ERROR: bag not found: $BAG_PATH" >&2
    exit 1
fi
if [[ -z "$INSTRUCTION" ]]; then
    echo "ERROR: --instruction is required (text or path to instruction.txt)." >&2
    exit 1
fi

BAG_NAME="$(basename "$BAG_PATH")"
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="$BAG_NAME"
fi

if [[ -z "$EXTRACT_DIR" ]]; then
    EXTRACT_DIR="$BAG_PATH/keyframe_output_${RUN_NAME}"
fi
if [[ -z "$SCENE_DIR" ]]; then
    SCENE_DIR="$EXTRACT_DIR/internnav_scene_for_inference_stride${STRIDE}"
fi
if [[ -z "$SAVE_DIR" ]]; then
    SAVE_DIR="$SCRIPT_DIR/output_${RUN_NAME}_stride${STRIDE}"
fi

EXTRACT_DIR="$(readlink -f "$EXTRACT_DIR")"
SCENE_DIR="$(readlink -f "$SCENE_DIR")"
SAVE_DIR="$(readlink -f "$SAVE_DIR")"

echo "================================================================"
echo "Bag -> Attention pipeline"
echo "  bag:         $BAG_PATH"
echo "  instruction: $INSTRUCTION_LABEL"
echo "  extract:     $EXTRACT_DIR"
echo "  scene:       $SCENE_DIR"
echo "  infer out:   $SAVE_DIR"
echo "  stride:      $STRIDE  look_down_every: $LOOK_DOWN_EVERY"
echo "  attn layers: $ATTN_LAYERS  device: $DEVICE"
echo "================================================================"

# ------------------------------------------------------------------
# Step 1: extract synchronized RGB + depth from rosbag
# ------------------------------------------------------------------
if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
    echo ""
    echo "[1/3] Extracting frames from rosbag..."
    EXTRACT_ARGS=(
        "$BAG_PATH"
        --output-dir "$EXTRACT_DIR"
        --rgb-topic "$RGB_TOPIC"
        --depth-topic "$DEPTH_TOPIC"
    )
    if [[ "$NO_TRIM" -eq 1 ]]; then
        EXTRACT_ARGS+=(--no-trim)
    else
        EXTRACT_ARGS+=(--trim-start "$TRIM_START" --trim-end "$TRIM_END")
    fi
    (cd "$SCRIPTS_ROOT" && python instruction_generator/extract_bag_frames.py "${EXTRACT_ARGS[@]}")
else
    echo "[1/3] Skipping extract (--skip-extract)"
fi

# ------------------------------------------------------------------
# Step 2: convert to debug_raw_* scene layout for inference
# ------------------------------------------------------------------
if [[ "$SKIP_PREPARE" -eq 0 ]]; then
    echo ""
    echo "[2/3] Preparing inference scene folder..."
    (cd "$SCRIPT_DIR" && python prepare_extract_for_inference.py \
        --extract-dir "$EXTRACT_DIR" \
        --output-dir "$SCENE_DIR" \
        --instruction "$INSTRUCTION" \
        --stride "$STRIDE" \
        --look-down-every "$LOOK_DOWN_EVERY")
else
    echo "[2/3] Skipping prepare (--skip-prepare)"
fi

# ------------------------------------------------------------------
# Step 3: InternVLA inference + attention maps
# ------------------------------------------------------------------
if [[ "$SKIP_INFER" -eq 0 ]]; then
    echo ""
    echo "[3/3] Running inference with attention maps..."
    (cd "$PROJECT_ROOT" && python scripts/notebooks/inference_only_attention_map.py \
        --attention \
        --attn-layers "$ATTN_LAYERS" \
        --scene-dir "$SCENE_DIR" \
        --save-dir "$SAVE_DIR" \
        --model-path "$MODEL_PATH" \
        --device "$DEVICE")
else
    echo "[3/3] Skipping inference (--skip-infer)"
fi

echo ""
echo "Done."
echo "  Scene:          $SCENE_DIR"
echo "  Inference base: $SAVE_DIR"
echo "  Attention maps: $SAVE_DIR/<timestamp>/attention_maps/{instruction,vision}/"
