#!/usr/bin/env bash
# End-to-end visual odometry pipeline (process_odom):
#   rosbag2 → RGB extract → DROID-W SLAM → odometry_camera.txt → floor_trajectory.txt
#
# Does NOT run keyframe extraction (use instruction_generator/run_extract_keyframe.sh after).
#
# Example:
#   ./run_visual_odom_pipeline.sh \
#     --bag /home/lenguyen1/Downloads/realsense_20260718_164645 \
#     --extract-stride 3 \
#     --droid-stride 1
#
# Resume (skip steps that already produced outputs):
#   ./run_visual_odom_pipeline.sh --bag ... --skip-extract --skip-slam

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DROID_W="${SCRIPT_DIR}/mapping/DROID-W"

CONDA_INTERNAV="${CONDA_INTERNAV:-internnav}"
CONDA_DROID_W="${CONDA_DROID_W:-droid-w}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

BAG_PATH=""
SCENE_ID=""
EXTRACT_DIR=""
EXTRACT_STRIDE="3"
DROID_STRIDE="1"
DROID_CONFIG=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"

RGB_TOPIC="/front/camera/color/image_raw/compressed"
DEPTH_TOPIC="/front/camera/aligned_depth_to_color/image_raw/compressedDepth"

SKIP_EXTRACT=0
SKIP_SLAM=0
SKIP_CONVERT=0
SKIP_FLOOR=0
WRITE_SCENE=0

CAM_H=480
CAM_W=640
CAM_FX=386.847
CAM_FY=386.440
CAM_CX=319.564
CAM_CY=245.442

DATA_ROOT="${VLN_DATA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)/DATA}"

usage() {
  cat <<'EOF'
Usage: run_visual_odom_pipeline.sh --bag PATH [options]

Runs: extract_rgb_from_bag → DROID-W run.py → convert_poses_to_odom_droidw
      → project_slam_path (floor_trajectory + floor_calibration).

Required:
  --bag PATH              Rosbag2 directory (metadata.yaml + .db3/.mcap)

Paths (defaults derived from bag basename):
  --scene-id ID           Scene / folder name (default: basename of --bag)
  --extract-dir PATH      Default: <process_odom>/extract_out/<scene-id>

Subsampling (avoid double stride — see README):
  --extract-stride N      Passed to extract_rgb_from_bag.sh (default: 3)
  --droid-stride N        DROID-W yaml + convert_poses stride (default: 1 if extract-stride>1)

DROID-W:
  --droid-config PATH     Use existing yaml; else writes configs/custom_<scene-id>.yaml
  --gpu ID                CUDA_VISIBLE_DEVICES (default: 0 or env)

RealSense /front topics (defaults):
  --rgb-topic TOPIC
  --depth-topic TOPIC

Camera intrinsics for generated DROID config (640x480 RealSense-ish defaults):
  --cam-h --cam-w --cam-fx --cam-fy --cam-cx --cam-cy

Skip steps (outputs must already exist):
  --skip-extract --skip-slam --skip-convert --skip-floor

Other:
  --write-scene-yaml      Write utils/configs/scenes/<scene-id>.yaml for keyframes
  -h, --help

Environments:
  CONDA_INTERNAV (default: internnav) — extract, convert, floor export
  CONDA_DROID_W (default: droid-w) — WildGS-SLAM

Next step (keyframes):
  cd ../instruction_generator && ./run_extract_keyframe.sh <scene-id>
EOF
}

conda_activate() {
  local env_name="$1"
  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found (need env: ${env_name})" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${env_name}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag) BAG_PATH="$2"; shift 2 ;;
    --scene-id) SCENE_ID="$2"; shift 2 ;;
    --extract-dir) EXTRACT_DIR="$2"; shift 2 ;;
    --extract-stride) EXTRACT_STRIDE="$2"; shift 2 ;;
    --droid-stride) DROID_STRIDE="$2"; shift 2 ;;
    --droid-config) DROID_CONFIG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --rgb-topic) RGB_TOPIC="$2"; shift 2 ;;
    --depth-topic) DEPTH_TOPIC="$2"; shift 2 ;;
    --cam-h) CAM_H="$2"; shift 2 ;;
    --cam-w) CAM_W="$2"; shift 2 ;;
    --cam-fx) CAM_FX="$2"; shift 2 ;;
    --cam-fy) CAM_FY="$2"; shift 2 ;;
    --cam-cx) CAM_CX="$2"; shift 2 ;;
    --cam-cy) CAM_CY="$2"; shift 2 ;;
    --skip-extract) SKIP_EXTRACT=1; shift ;;
    --skip-slam) SKIP_SLAM=1; shift ;;
    --skip-convert) SKIP_CONVERT=1; shift ;;
    --skip-floor) SKIP_FLOOR=1; shift ;;
    --write-scene-yaml) WRITE_SCENE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${BAG_PATH}" ]]; then
  echo "Error: --bag is required" >&2
  usage >&2
  exit 1
fi
if [[ ! -d "${BAG_PATH}" ]]; then
  echo "Error: bag not found: ${BAG_PATH}" >&2
  exit 1
fi

if [[ -z "${SCENE_ID}" ]]; then
  SCENE_ID="$(basename "${BAG_PATH}")"
fi
if [[ -z "${EXTRACT_DIR}" ]]; then
  EXTRACT_DIR="${SCRIPT_DIR}/extract_out/${SCENE_ID}"
fi

RGB_FRAMES="${EXTRACT_DIR}/tmp/rgb_frames"
FRAMES_JSON="${EXTRACT_DIR}/frames.json"
ODOM_TXT="${EXTRACT_DIR}/odometry_camera.txt"
FLOOR_TRAJ="${EXTRACT_DIR}/floor_trajectory.txt"

if [[ -z "${DROID_CONFIG}" ]]; then
  DROID_CONFIG="${DROID_W}/configs/custom_${SCENE_ID}.yaml"
fi

SLAM_RUN_DIR="${DROID_W}/output/${SCENE_ID}/${SCENE_ID}"
POSES_TXT="${SLAM_RUN_DIR}/traj/est_poses_full.txt"
SLAM_CFG="${SLAM_RUN_DIR}/cfg.yaml"

export PYTHONPATH="${SCRIPTS_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "============================================================"
echo " Visual odom pipeline"
echo "============================================================"
echo "  scene:         ${SCENE_ID}"
echo "  bag:           ${BAG_PATH}"
echo "  extract_dir:   ${EXTRACT_DIR}"
echo "  extract_stride:${EXTRACT_STRIDE}"
echo "  droid_stride:  ${DROID_STRIDE}"
echo "  droid_config:  ${DROID_CONFIG}"
echo "  gpu:           ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

# --- 1. Extract ---
if [[ "${SKIP_EXTRACT}" -eq 0 ]]; then
  echo "[1/4] Extract RGB (+ depth) from bag..."
  conda_activate "${CONDA_INTERNAV}"
  "${SCRIPT_DIR}/extract_rgb_from_bag.sh" \
    --bag "${BAG_PATH}" \
    --output-dir "${EXTRACT_DIR}" \
    --stride "${EXTRACT_STRIDE}" \
    --rgb-topic "${RGB_TOPIC}" \
    --depth-topic "${DEPTH_TOPIC}" \
    --no-trim
else
  echo "[1/4] Skip extract (--skip-extract)"
fi
if [[ ! -f "${FRAMES_JSON}" ]]; then
  echo "Error: missing ${FRAMES_JSON}" >&2
  exit 1
fi

# --- 2. DROID-W SLAM ---
if [[ "${SKIP_SLAM}" -eq 0 ]]; then
  if [[ ! -f "${DROID_CONFIG}" ]]; then
    echo "  Writing DROID-W config: ${DROID_CONFIG}"
    mkdir -p "$(dirname "${DROID_CONFIG}")"
    cat > "${DROID_CONFIG}" <<EOF
inherit_from: ./configs/droid_w.yaml

# Generated by run_visual_odom_pipeline.sh
scene: ${SCENE_ID}
dataset: youtube

data:
  input_folder: ${RGB_FRAMES}
  output: ./output/${SCENE_ID}

cam:
  H: ${CAM_H}
  W: ${CAM_W}
  fx: ${CAM_FX}
  fy: ${CAM_FY}
  cx: ${CAM_CX}
  cy: ${CAM_CY}
  H_out: 360
  W_out: 640

stride: ${DROID_STRIDE}
max_frames: -1
save_gt_poses: False

tracking:
  buffer: 350
  force_keyframe_every_n_frames: -1
EOF
  fi
  echo "[2/4] DROID-W SLAM..."
  conda_activate "${CONDA_DROID_W}"
  cd "${DROID_W}"
  python run.py --config "${DROID_CONFIG}"
  cd "${SCRIPT_DIR}"
else
  echo "[2/4] Skip SLAM (--skip-slam)"
fi
if [[ ! -f "${POSES_TXT}" ]]; then
  echo "Error: missing ${POSES_TXT}" >&2
  exit 1
fi

# --- 3. Poses → odometry_camera.txt ---
if [[ "${SKIP_CONVERT}" -eq 0 ]]; then
  echo "[3/4] Convert est_poses_full.txt → odometry_camera.txt..."
  conda_activate "${CONDA_INTERNAV}"
  python "${SCRIPT_DIR}/convert_poses_to_odom_droidw.py" \
    --poses "${POSES_TXT}" \
    --frames-json "${FRAMES_JSON}" \
    --output "${ODOM_TXT}" \
    --stride "${DROID_STRIDE}"
else
  echo "[3/4] Skip convert (--skip-convert)"
fi
if [[ ! -f "${ODOM_TXT}" ]]; then
  echo "Error: missing ${ODOM_TXT}" >&2
  exit 1
fi

# --- 4. Floor trajectory ---
if [[ "${SKIP_FLOOR}" -eq 0 ]]; then
  echo "[4/4] Export floor_trajectory.txt..."
  conda_activate "${CONDA_INTERNAV}"
  CFG_FOR_FLOOR="${SLAM_CFG}"
  if [[ ! -f "${CFG_FOR_FLOOR}" ]]; then
    CFG_FOR_FLOOR="${DROID_CONFIG}"
  fi
  python "${SCRIPT_DIR}/project_slam_path.py" \
    --odom "${ODOM_TXT}" \
    --frames-json "${FRAMES_JSON}" \
    --rgb-dir "${RGB_FRAMES}" \
    --config "${CFG_FOR_FLOOR}" \
    --export-floor-trajectory "${EXTRACT_DIR}"
else
  echo "[4/4] Skip floor export (--skip-floor)"
fi

# --- Optional scene yaml for keyframe_extractor ---
if [[ "${WRITE_SCENE}" -eq 1 ]]; then
  SCENE_YAML="${SCRIPTS_ROOT}/utils/configs/scenes/${SCENE_ID}.yaml"
  echo "  Writing scene config: ${SCENE_YAML}"
  mkdir -p "$(dirname "${SCENE_YAML}")"
  cat > "${SCENE_YAML}" <<EOF
scene: ${SCENE_ID}
odom_apply_body2optical: false

ros:
  rgb_topic: ${RGB_TOPIC}
  depth_topic: ${DEPTH_TOPIC}

paths:
  bag: ${BAG_PATH}
  camera_odom: ${ODOM_TXT}
  floor_trajectory: ${FLOOR_TRAJ}
  output_dir: ${DATA_ROOT}/process_keyframe/${SCENE_ID}
  frames_json: ${FRAMES_JSON}
  rgb_dir: ${RGB_FRAMES}
  droid_cfg: ${SLAM_CFG}
  projected_frames: ${DATA_ROOT}/projected_frame/${SCENE_ID}
EOF
fi

echo
echo "Done."
echo "  extract:       ${EXTRACT_DIR}"
echo "  SLAM poses:    ${POSES_TXT}"
echo "  camera odom:   ${ODOM_TXT}"
echo "  floor traj:    ${FLOOR_TRAJ}"
echo "  calibration:   ${EXTRACT_DIR}/floor_calibration.json"
echo
echo "Keyframes (separate step):"
echo "  cd ${SCRIPTS_ROOT}/instruction_generator"
echo "  ./run_extract_keyframe.sh ${SCENE_ID}"
echo "  (needs utils/configs/scenes/${SCENE_ID}.yaml — use --write-scene-yaml on this script)"
