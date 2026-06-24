#!/usr/bin/env zsh
# Source ROS + conda for keyframe tmux panes, then exec the given command.
set -eo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.zsh"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
SCRIPT_DIR="${0:A:h}"
SCRIPTS_ROOT="${SCRIPT_DIR:h}"

if [[ ! -f "$ROS_SETUP" ]]; then
    echo "ERROR: ROS setup not found: $ROS_SETUP" >&2
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate internnav
set +u
source "$ROS_SETUP"
export PYTHONPATH="${SCRIPTS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLN_SCENE="${VLN_SCENE:-}"

cd "$SCRIPT_DIR"
exec "$@"
