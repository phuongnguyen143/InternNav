#!/bin/bash
# Config-driven keyframe extraction launcher.
#
# STEP 0 (once per scene, PCD-free SLAM path):
#   python ../process_odom/project_slam_path.py --scene office_round1 --export-floor-trajectory ...
#   # writes floor_trajectory.txt + floor_calibration.json
#   # or if you only have floor_trajectory.txt:
#   python ../process_odom/derive_floor_calibration.py --scene office_round1
#
# STEP 0 (PCD path):
#   python ../process_odom/precompute_floor_trajectory.py --scene office_round1
#
# STEP 1:
#   ./run_extract_keyframe.sh office_round1
#
# SHUTDOWN: Press Ctrl+B then S

set -euo pipefail

SESSION="keyframe"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$SCRIPT_DIR"

SCENE="${1:-${VLN_SCENE:-}}"
if [[ -z "$SCENE" || "$SCENE" == "--scene" ]]; then
    if [[ "${1:-}" == "--scene" && -n "${2:-}" ]]; then
        SCENE="$2"
    else
        echo "ERROR: scene name required."
        echo ""
        echo "USAGE: $0 <scene>"
        echo "       $0 --scene <scene>"
        echo "       VLN_SCENE=<scene> $0"
        echo ""
        echo "Example: $0 office_round1"
        exit 1
    fi
fi

export PYTHONPATH="$SCRIPTS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLN_SCENE="$SCENE"

read_paths() {
    python3 - <<PY
from utils.config import load_config
cfg = load_config("${SCENE}")
paths = cfg.get("paths", default={})
for key in ("bag", "camera_odom", "floor_trajectory", "output_dir"):
  print(paths.get(key, ""))
ros = cfg.get("ros", default={})
print(ros.get("bag_play_rate", 5.0))
PY
}

mapfile -t _CFG_LINES < <(read_paths)
BAG_PATH="${_CFG_LINES[0]}"
CAMERA_ODOM="${_CFG_LINES[1]}"
FLOOR_TRAJ="${_CFG_LINES[2]}"
KEYFRAME_OUT="${_CFG_LINES[3]}"
BAG_RATE="${_CFG_LINES[4]}"

if [[ -z "$BAG_PATH" || -z "$CAMERA_ODOM" || -z "$FLOOR_TRAJ" || -z "$KEYFRAME_OUT" ]]; then
    echo "ERROR: scene config missing required paths (bag, camera_odom, floor_trajectory, output_dir)"
    exit 1
fi

if [[ ! -e "$BAG_PATH" ]]; then
    echo "ERROR: Bag path does not exist: $BAG_PATH"
    exit 1
fi
if [[ ! -f "$CAMERA_ODOM" ]]; then
    echo "ERROR: Camera odom file does not exist: $CAMERA_ODOM"
    exit 1
fi
if [[ ! -f "$FLOOR_TRAJ" ]]; then
    echo "ERROR: Floor trajectory file does not exist: $FLOOR_TRAJ"
    exit 1
fi

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP_ZSH="/opt/ros/${ROS_DISTRO}/setup.zsh"
PANE_ENV="$WORK_DIR/keyframe_pane_env.zsh"
TMUX_SHELL="$(command -v zsh || command -v bash)"

if [[ ! -f "$ROS_SETUP_ZSH" ]]; then
    echo "ERROR: ROS setup not found: $ROS_SETUP_ZSH"
    echo "       Install ROS 2 ${ROS_DISTRO} or set ROS_DISTRO to your distro."
    exit 1
fi
if [[ ! -x "$PANE_ENV" ]]; then
    chmod +x "$PANE_ENV"
fi

export CONDA_BASE ROS_DISTRO VLN_SCENE
PANE_CMD="$PANE_ENV"

mkdir -p "$KEYFRAME_OUT"
FLOOR_CAL="$(dirname "$FLOOR_TRAJ")/$(python3 -c "from utils.config import load_config; print(load_config('${SCENE}').floor_calibration_filename())")"
if [[ ! -f "$FLOOR_CAL" && -f "$FLOOR_TRAJ" ]]; then
    echo "  deriving floor_calibration from floor_trajectory (no PCD) ..."
    python3 "$SCRIPTS_ROOT/process_odom/derive_floor_calibration.py" \
        --floor_trajectory "$FLOOR_TRAJ" \
        --output_dir "$(dirname "$FLOOR_TRAJ")"
fi
if [[ -f "$FLOOR_CAL" ]]; then
    cp "$FLOOR_CAL" "$KEYFRAME_OUT/"
    echo "  copied $(basename "$FLOOR_CAL") -> $KEYFRAME_OUT/"
fi
cp "$FLOOR_TRAJ" "$KEYFRAME_OUT/" 2>/dev/null || true

echo ""
echo "  scene:        $SCENE"
echo "  bag:          $BAG_PATH"
echo "  camera odom:  $CAMERA_ODOM"
echo "  floor traj:   $FLOOR_TRAJ"
echo "  output_dir:   $KEYFRAME_OUT"
echo ""

tmux kill-session -t $SESSION 2>/dev/null || true
sleep 0.5

tmux new-session  -d -s $SESSION -x 220 -y 50
tmux set-option -t $SESSION default-shell "$TMUX_SHELL"
tmux split-window -v -t $SESSION:0.0
tmux split-window -v -t $SESSION:0.1
tmux select-layout -t $SESSION even-vertical

tmux send-keys -t $SESSION:0.0 \
  "$PANE_CMD ros2 bag play $BAG_PATH --rate $BAG_RATE" Enter

tmux send-keys -t $SESSION:0.1 \
  "$PANE_CMD python3 trajectory_publishers.py floor --ros-args -p floor_trajectory_file:=$FLOOR_TRAJ" Enter

tmux send-keys -t $SESSION:0.2 \
  "$PANE_CMD python3 keyframe_extractor.py --ros-args -p camera_odom_file:=$CAMERA_ODOM -p output_dir:=$KEYFRAME_OUT" Enter

tmux bind-key -T prefix S run-shell " \
  tmux send-keys -t $SESSION:0.2 C-c ; \
  sleep 8 ; \
  tmux send-keys -t $SESSION:0.0 C-c ; \
  tmux send-keys -t $SESSION:0.1 C-c \
"

tmux select-pane -t $SESSION:0.2

echo "=== tmux session '$SESSION' started ==="
echo "  Pane 0 → ros2 bag play"
echo "  Pane 1 → trajectory_publishers.py floor"
echo "  Pane 2 → keyframe_extractor.py"
echo ""
echo "  When done: Ctrl+B then S"
echo ""

tmux attach-session -t $SESSION
