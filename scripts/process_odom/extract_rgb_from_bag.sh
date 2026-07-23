#!/usr/bin/env bash
# Extract compressed RGB (+ aligned depth if present) from a ROS2 rosbag2.
#
# Wrapper around scripts/instruction_generator/extract_bag_frames.py with
# defaults matching RealSense bags like:
#   /home/lenguyen1/Downloads/realsense_20260718_164645
#     topics:
#       /front/camera/color/image_raw/compressed
#       /front/camera/aligned_depth_to_color/image_raw/compressedDepth
#
# Example:
#   ./extract_rgb_from_bag.sh \
#     --bag /home/lenguyen1/Downloads/realsense_20260718_164645
#
#   ./extract_rgb_from_bag.sh \
#     --bag /home/lenguyen1/Downloads/realsense_20260718_164645 \
#     --output-dir /home/lenguyen1/hoangpqn/vln/DATA/raw_img_extract/realsense_20260718_164645 \
#     --stride 3 \
#     --no-trim
#
# Outputs (for DROID-W / InternNav):
#   <output_dir>/tmp/rgb_frames/frame_XXXXXX.jpg
#   <output_dir>/tmp/depth_frames/frame_XXXXXX.png
#   <output_dir>/frames.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTRACT_PY="${SCRIPTS_ROOT}/instruction_generator/extract_bag_frames.py"

# Defaults for /front RealSense bags (see Downloads/realsense_20260718_164645)
BAG_PATH="/home/lenguyen1/Downloads/realsense_20260718_164645"
RGB_TOPIC="/front/camera/color/image_raw/compressed"
DEPTH_TOPIC="/front/camera/aligned_depth_to_color/image_raw/compressedDepth"
OUTPUT_DIR=""
STORAGE_ID="auto"
TRIM_START="0"
TRIM_END="0"
NO_TRIM=1
WRITE_DEPTH_PREVIEW=0
SYNC_SLOP="0.05"
STRIDE="1"
SOURCE_ROS=1
ROS_SETUP="/opt/ros/humble/setup.bash"

usage() {
  cat <<'EOF'
Usage: extract_rgb_from_bag.sh [options]

Extract compressed RGB (and synced depth) from a ROS2 rosbag2 directory.

Options:
  --bag PATH              Rosbag2 directory (metadata.yaml + .db3/.mcap)
                          Default: /home/lenguyen1/Downloads/realsense_20260718_164645
  --output-dir PATH       Extract output directory
                          Default: <process_odom>/extract_out/<bag_basename>
  --rgb-topic TOPIC       Compressed RGB topic
                          Default: /front/camera/color/image_raw/compressed
  --depth-topic TOPIC     Compressed depth topic
                          Default: /front/camera/aligned_depth_to_color/image_raw/compressedDepth
  --storage-id ID         auto | sqlite3 | mcap (default: auto)
  --sync-slop SEC         Max RGB/depth time diff (default: 0.05)
  --stride N              Save every Nth synced frame (default: 1 = all)
  --trim-start SEC        Skip first N seconds (default: 0 with --no-trim)
  --trim-end SEC          Skip last N seconds (default: 0 with --no-trim)
  --no-trim               Extract full bag (default)
  --trim                  Enable default 20s head/tail trim from extract_bag_frames
  --write-depth-preview   Also write tmp/depth_full.mp4
  --no-source-ros         Do not source /opt/ros/humble/setup.bash
  --ros-setup PATH        ROS setup.bash to source
  -h, --help              Show this help

Environment:
  Needs ROS2 (rosbag2_py) + OpenCV Python. Example:
    source /opt/ros/humble/setup.bash
    conda activate internnav   # if you use that env for cv2 / utils

After extract, point DROID-W input_folder at:
  <output_dir>/tmp/rgb_frames

If you used --stride N, set DROID-W config stride: 1 (subsampling already done).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag) BAG_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --rgb-topic) RGB_TOPIC="$2"; shift 2 ;;
    --depth-topic) DEPTH_TOPIC="$2"; shift 2 ;;
    --storage-id) STORAGE_ID="$2"; shift 2 ;;
    --sync-slop) SYNC_SLOP="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --trim-start) TRIM_START="$2"; NO_TRIM=0; shift 2 ;;
    --trim-end) TRIM_END="$2"; NO_TRIM=0; shift 2 ;;
    --no-trim) NO_TRIM=1; shift ;;
    --trim) NO_TRIM=0; TRIM_START="20"; TRIM_END="20"; shift ;;
    --write-depth-preview) WRITE_DEPTH_PREVIEW=1; shift ;;
    --no-source-ros) SOURCE_ROS=0; shift ;;
    --ros-setup) ROS_SETUP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${BAG_PATH}" ]]; then
  echo "Error: bag directory not found: ${BAG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${BAG_PATH}/metadata.yaml" ]]; then
  echo "Error: missing metadata.yaml in ${BAG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${EXTRACT_PY}" ]]; then
  echo "Error: extractor not found: ${EXTRACT_PY}" >&2
  exit 1
fi

BAG_BASENAME="$(basename "${BAG_PATH}")"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${SCRIPT_DIR}/extract_out/${BAG_BASENAME}"
fi

if [[ "${SOURCE_ROS}" -eq 1 ]]; then
  if [[ -f "${ROS_SETUP}" ]]; then
    # ROS setup.bash references optional unset vars (e.g. AMENT_TRACE_SETUP_FILES).
    # Temporarily allow unbound vars while sourcing under `set -u`.
    set +u
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"
    set -u
  else
    echo "Warning: ROS setup not found at ${ROS_SETUP}; continuing anyway" >&2
  fi
fi

export PYTHONPATH="${SCRIPTS_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo " Extract compressed RGB from rosbag2"
echo "============================================================"
echo "  bag:         ${BAG_PATH}"
echo "  output:      ${OUTPUT_DIR}"
echo "  rgb topic:   ${RGB_TOPIC}"
echo "  depth topic: ${DEPTH_TOPIC}"
echo "  storage:     ${STORAGE_ID}"
echo "  stride:      ${STRIDE}"
if [[ "${NO_TRIM}" -eq 1 ]]; then
  echo "  trim:        none (full bag)"
else
  echo "  trim:        start=${TRIM_START}s end=${TRIM_END}s"
fi
echo "============================================================"

CMD=(
  python "${EXTRACT_PY}" extract "${BAG_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --rgb-topic "${RGB_TOPIC}"
  --depth-topic "${DEPTH_TOPIC}"
  --storage-id "${STORAGE_ID}"
  --sync-slop "${SYNC_SLOP}"
  --stride "${STRIDE}"
)

if [[ "${NO_TRIM}" -eq 1 ]]; then
  CMD+=(--no-trim)
else
  CMD+=(--trim-start "${TRIM_START}" --trim-end "${TRIM_END}")
fi

if [[ "${WRITE_DEPTH_PREVIEW}" -eq 1 ]]; then
  CMD+=(--write-depth-preview)
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

RGB_DIR="${OUTPUT_DIR}/tmp/rgb_frames"
N_RGB=0
if [[ -d "${RGB_DIR}" ]]; then
  N_RGB="$(find "${RGB_DIR}" -maxdepth 1 -type f \( -name 'frame_*.jpg' -o -name 'frame*.jpg' \) | wc -l | tr -d ' ')"
fi

echo
echo "Done."
echo "  RGB frames:   ${RGB_DIR}  (${N_RGB} files)"
echo "  Depth frames: ${OUTPUT_DIR}/tmp/depth_frames"
echo "  frames.json:  ${OUTPUT_DIR}/frames.json"
echo
echo "DROID-W tip: set data.input_folder to:"
echo "  ${RGB_DIR}"
echo "  (files are named frame_XXXXXX.jpg — matches dataset: youtube)"
