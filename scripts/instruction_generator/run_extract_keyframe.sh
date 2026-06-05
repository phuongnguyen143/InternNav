#!/bin/bash
# run_keyframe_extraction.sh
#
# STEP 0 (once per scene, offline — can take several minutes on large PCD):
#   python precompute_floor_trajectory.py \
#     --pcd /path/to/scene.pcd \
#     --camera_odom /path/to/odometry_camera.txt \
#     --output_dir /path/to/scene_dir/
#
# STEP 1 (live bag playback):
#   ./run.sh <bag_path> <camera_odom.txt> <floor_trajectory.txt>
#
# EXAMPLE:
#   ./run.sh .../bkhn_round2 \
#            .../odometry_bkhn_round2_point2plane.txt \
#            .../floor_trajectory.txt
#
# SHUTDOWN: Press Ctrl+B then S to gracefully stop keyframe_extractor.py

SESSION="keyframe"
WORK_DIR="/home/lenguyen1/hoangpqn/vln/InternNav/scripts/instruction_generator"
KEYFRAME_OUT="$WORK_DIR/keyframe_output"

BAG_PATH="${1}"
CAMERA_ODOM="${2}"
FLOOR_TRAJ="${3}"

if [[ -z "$BAG_PATH" || -z "$CAMERA_ODOM" || -z "$FLOOR_TRAJ" ]]; then
    echo "ERROR: Missing arguments."
    echo ""
    echo "USAGE: $0 <bag_path> <camera_odom.txt> <floor_trajectory.txt>"
    echo ""
    echo "  Run precompute_floor_trajectory.py first to create floor_trajectory.txt"
    echo "  and floor_calibration.json in the same directory as the trajectory."
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
CONDA_INIT="source $CONDA_BASE/etc/profile.d/conda.sh && conda activate internnav"

mkdir -p "$KEYFRAME_OUT"
FLOOR_CAL="$(dirname "$FLOOR_TRAJ")/floor_calibration.json"
if [[ -f "$FLOOR_CAL" ]]; then
    cp "$FLOOR_CAL" "$KEYFRAME_OUT/"
    echo "  copied floor_calibration.json -> keyframe_output/"
fi

echo ""
echo "  bag:          $BAG_PATH"
echo "  camera odom:  $CAMERA_ODOM"
echo "  floor traj:   $FLOOR_TRAJ"
echo ""

tmux kill-session -t $SESSION 2>/dev/null
sleep 0.5

tmux new-session  -d -s $SESSION -x 220 -y 50
tmux split-window -v -t $SESSION:0.0
tmux split-window -v -t $SESSION:0.1
tmux select-layout -t $SESSION even-vertical

tmux send-keys -t $SESSION:0.0 \
  "$CONDA_INIT && ros2 bag play $BAG_PATH --rate 5" Enter

tmux send-keys -t $SESSION:0.1 \
  "$CONDA_INIT && cd $WORK_DIR && python3 trajectory_publishers.py floor --ros-args -p floor_trajectory_file:=$FLOOR_TRAJ" Enter

tmux send-keys -t $SESSION:0.2 \
  "$CONDA_INIT && cd $WORK_DIR && python3 keyframe_extractor.py --ros-args -p camera_odom_file:=$CAMERA_ODOM -p output_dir:=$KEYFRAME_OUT" Enter

tmux bind-key -T prefix S run-shell " \
  tmux send-keys -t $SESSION:0.2 C-c ; \
  sleep 8 ; \
  tmux send-keys -t $SESSION:0.0 C-c ; \
  tmux send-keys -t $SESSION:0.1 C-c \
"

tmux select-pane -t $SESSION:0.2

echo "=== tmux session '$SESSION' started ==="
echo "  Pane 0 → ros2 bag play"
  echo "  Pane 1 → trajectory_publishers.py floor (fast, no PCD)"
  echo "  Pane 2 → keyframe_extractor.py"
echo ""
echo "  When done: Ctrl+B then S"
echo ""

tmux attach-session -t $SESSION
