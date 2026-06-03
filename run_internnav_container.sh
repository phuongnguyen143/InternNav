#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${INTERNNAV_IMAGE:-internnav}"
CONTAINER_NAME="${INTERNNAV_CONTAINER_NAME:-internnav}"
REPO_DIR="${INTERNNAV_REPO_DIR:-${SCRIPT_DIR}}"
CONTAINER_REPO_DIR="${INTERNNAV_CONTAINER_REPO_DIR:-/root/InternNav}"
NETWORK_MODE="${INTERNNAV_NETWORK:-host}"
XAUTH_DOCKER="${INTERNNAV_XAUTH_DOCKER:-/tmp/.docker.xauth-internnav}"

DOCKER_ARGS=(
    --rm
    -it
    --gpus all
    --ipc host
    --name "${CONTAINER_NAME}"
    --workdir "${CONTAINER_REPO_DIR}"
    --network ${NETWORK_MODE}
    --volume "${REPO_DIR}:${CONTAINER_REPO_DIR}"
    --env "NVIDIA_DRIVER_CAPABILITIES=all"
    --env "PYTHONPATH=${CONTAINER_REPO_DIR}:${CONTAINER_REPO_DIR}/third_party/diffusion-policy:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages"
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    -v "$(pwd)/cyclonedds_config.xml:/etc/cyclonedds.xml"
    -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml 
    -e CODEX_HOME=/tmp/codex-home/.codex 
    -v "$HOME/.codex_internnav:/tmp/codex-home/.codex"
)

if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    touch "${XAUTH_DOCKER}"
    chmod 600 "${XAUTH_DOCKER}"

    if command -v xauth >/dev/null 2>&1; then
        xauth nlist "${DISPLAY}" 2>/dev/null \
            | sed -e 's/^..../ffff/' \
            | xauth -f "${XAUTH_DOCKER}" nmerge - 2>/dev/null || true
    fi

    if command -v xhost >/dev/null 2>&1; then
        xhost +SI:localuser:root >/dev/null 2>&1 || true
    fi

    DOCKER_ARGS+=(
        --env "DISPLAY=${DISPLAY}"
        --env "QT_X11_NO_MITSHM=1"
        --env "XAUTHORITY=/tmp/.docker.xauth"
        --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw"
        --volume "${XAUTH_DOCKER}:/tmp/.docker.xauth:ro"
    )
fi

if [[ $# -eq 0 ]]; then
    exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" /bin/bash
fi

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "$@"
