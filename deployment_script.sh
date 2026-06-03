#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_NAME="${INTERNNAV_CONTAINER_NAME:-internnav}"
CONTAINER_REPO_DIR="${INTERNNAV_CONTAINER_REPO_DIR:-/root/InternNav}"
BAGS_DIR="${INTERNNAV_BAGS_DIR:-/tmp/bags}"
RUN_NAME="${INTERNNAV_RUN_NAME:-internnav_$(date +%Y%m%d_%H%M%S)}"
BAG_OUTPUT="${BAGS_DIR}/${RUN_NAME}"
LOG_DIR="${BAGS_DIR}/${RUN_NAME}_logs"
PID_DIR="${LOG_DIR}/pids"
SERVER_START_DELAY="${INTERNNAV_SERVER_START_DELAY:-5}"
SERVER_READY_TIMEOUT="${INTERNNAV_SERVER_READY_TIMEOUT:-300}"
SERVER_PORT="${INTERNNAV_SERVER_PORT:-5801}"
PROCESS_START_DELAY="${INTERNNAV_PROCESS_START_DELAY:-1}"
STOP_TIMEOUT="${INTERNNAV_STOP_TIMEOUT:-20}"
TERM_TIMEOUT="${INTERNNAV_TERM_TIMEOUT:-5}"

STARTED_LABELS=()

RGB_TOPIC="/camera/waist_front_zed_stream/left/color/rect/image"
COMPRESSED_RGB_TOPIC="${RGB_TOPIC}/compressed"
CAMERA_INFO_TOPIC="${RGB_TOPIC}/camera_info"
DEPTH_TOPIC="/camera/waist_front_zed_stream/depth/depth_registered"
POINT_CLOUD_TOPIC="/camera_on_back_ob/zed_node/point_cloud/cloud_registered"

TOPICS=(
    "${RGB_TOPIC}"
    "${COMPRESSED_RGB_TOPIC}"
    "${CAMERA_INFO_TOPIC}"
    "${DEPTH_TOPIC}"
    "${POINT_CLOUD_TOPIC}"
    "/internvla_n1/pixel_goal_image"
    "/graph_msf/opt_odometry_world_base_filtered"
    "/vln_path"
    "/cmd_vel/stop"
    "/cmd_vel/nav"
    "/tf"
    "/tf_static"
)

quote_args() {
    printf "%q " "$@"
}

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

ensure_container_running() {
    if container_running; then
        return
    fi

    if [[ -x "${SCRIPT_DIR}/run_internnav_container.sh" ]]; then
        echo "InternNav container '${CONTAINER_NAME}' is not running; starting it."
        "${SCRIPT_DIR}/run_internnav_container.sh"
    fi

    if ! container_running; then
        echo "InternNav container '${CONTAINER_NAME}' is not running." >&2
        echo "Start it with: ${SCRIPT_DIR}/run_internnav_container.sh" >&2
        exit 1
    fi
}

container_command_prefix() {
    printf "set -e; "
    printf "mkdir -p %q; " "${LOG_DIR}"
    printf "mkdir -p %q; " "${PID_DIR}"
    printf "cd %q; " "${CONTAINER_REPO_DIR}"
    printf "source /opt/ros/humble/setup.bash 2>/dev/null || true; "
}

run_in_container_detached() {
    local label="$1"
    shift

    local log_file="${LOG_DIR}/${label}.log"
    local pid_file="${PID_DIR}/${label}.pid"
    local command
    local pid

    command="$(container_command_prefix)"
    command+="setsid $(quote_args "$@")"
    command+="> $(printf "%q" "${log_file}") 2>&1"
    command+=" & pid=\$!; "
    command+="echo \"\${pid}\" > $(printf "%q" "${pid_file}"); "
    command+="echo \"\${pid}\""

    pid="$(docker exec --workdir "${CONTAINER_REPO_DIR}" "${CONTAINER_NAME}" bash -lc "${command}")"
    STARTED_LABELS+=("${label}")
    echo "Started ${label}; pid: ${pid}; log: ${log_file}"
}

verify_process() {
    local label="$1"
    local pattern="$2"
    local log_file="${LOG_DIR}/${label}.log"
    local quoted_pattern

    quoted_pattern="$(printf "%q" "${pattern}")"

    if docker exec "${CONTAINER_NAME}" bash -lc "pgrep -af -- ${quoted_pattern} >/dev/null"; then
        return 0
    fi

    echo "Process '${label}' is not running. Check log: ${log_file}" >&2
    return 1
}

process_group_running() {
    local pid="$1"

    docker exec "${CONTAINER_NAME}" bash -lc \
        "stat=\"\$(ps -o stat= -p ${pid} 2>/dev/null || true)\"; [[ \"\${stat}\" == Z* ]] && exit 1; kill -0 -- -${pid} 2>/dev/null || kill -0 ${pid} 2>/dev/null"
}

signal_process_group() {
    local pid="$1"
    local signal="$2"

    docker exec "${CONTAINER_NAME}" bash -lc \
        "kill -${signal} -- -${pid} 2>/dev/null || kill -${signal} ${pid} 2>/dev/null || true"
}

initial_stop_signal() {
    local label="$1"

    case "${label}" in
        http_internvla_server)
            echo "TERM"
            ;;
        *)
            echo "INT"
            ;;
    esac
}

stop_process() {
    local label="$1"
    local pid_file="${PID_DIR}/${label}.pid"
    local quoted_pid_file
    local pid
    local elapsed=0
    local term_elapsed=0
    local signal

    if ! container_running; then
        echo "Container '${CONTAINER_NAME}' is not running; cannot stop ${label}."
        return
    fi

    quoted_pid_file="$(printf "%q" "${pid_file}")"
    pid="$(docker exec "${CONTAINER_NAME}" bash -lc "if [[ -s ${quoted_pid_file} ]]; then cat ${quoted_pid_file}; fi" 2>/dev/null || true)"

    if [[ -z "${pid}" ]]; then
        echo "No PID file for ${label}; skipping."
        return
    fi

    if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
        echo "Invalid PID '${pid}' for ${label}; skipping." >&2
        return
    fi

    if ! process_group_running "${pid}"; then
        echo "${label} is already stopped."
        docker exec "${CONTAINER_NAME}" bash -lc "rm -f ${quoted_pid_file}" >/dev/null 2>&1 || true
        return
    fi

    signal="$(initial_stop_signal "${label}")"
    echo "Stopping ${label} with SIG${signal}."
    signal_process_group "${pid}" "${signal}"

    while (( elapsed < STOP_TIMEOUT )); do
        if ! process_group_running "${pid}"; then
            echo "Stopped ${label}."
            docker exec "${CONTAINER_NAME}" bash -lc "rm -f ${quoted_pid_file}" >/dev/null 2>&1 || true
            return
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done

    if [[ "${signal}" != "TERM" ]]; then
        echo "${label} did not stop after ${STOP_TIMEOUT}s; sending SIGTERM."
        signal_process_group "${pid}" TERM
    fi

    while (( term_elapsed < TERM_TIMEOUT )); do
        if ! process_group_running "${pid}"; then
            echo "Stopped ${label}."
            docker exec "${CONTAINER_NAME}" bash -lc "rm -f ${quoted_pid_file}" >/dev/null 2>&1 || true
            return
        fi

        sleep 1
        term_elapsed=$((term_elapsed + 1))
    done

    echo "${label} did not stop after SIGTERM; sending SIGKILL."
    signal_process_group "${pid}" KILL
    docker exec "${CONTAINER_NAME}" bash -lc "rm -f ${quoted_pid_file}" >/dev/null 2>&1 || true
}

stop_deployment() {
    local idx

    echo "Stopping InternNav deployment processes."
    for ((idx=${#STARTED_LABELS[@]} - 1; idx >= 0; idx--)); do
        stop_process "${STARTED_LABELS[idx]}"
    done
    echo "Stop requests sent."
}

ask_to_stop() {
    local answer

    if [[ ! -t 0 ]]; then
        echo "No interactive stdin; leaving deployment processes running."
        return
    fi

    echo
    if ! read -r -p "Stop all InternNav deployment processes now? [y/N] " answer; then
        answer=""
    fi
    case "${answer,,}" in
        y|yes)
            stop_deployment
            ;;
        *)
            echo "Leaving deployment processes running."
            echo "PID files: ${PID_DIR}"
            ;;
    esac
}

wait_for_http_server() {
    local elapsed=0

    echo "Waiting for HTTP server on 127.0.0.1:${SERVER_PORT}."
    while (( elapsed < SERVER_READY_TIMEOUT )); do
        if docker exec "${CONTAINER_NAME}" python3 -c "import socket; socket.create_connection(('127.0.0.1', ${SERVER_PORT}), 1).close()" >/dev/null 2>&1; then
            return 0
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo "HTTP server did not become ready within ${SERVER_READY_TIMEOUT}s." >&2
    echo "Check log: ${LOG_DIR}/http_internvla_server.log" >&2
    return 1
}

ensure_container_running

run_in_container_detached \
    rosbag_record \
    ros2 bag record -o "${BAG_OUTPUT}" "${TOPICS[@]}"

sleep "${PROCESS_START_DELAY}"
verify_process rosbag_record "ros2 bag record"

run_in_container_detached \
    http_internvla_server \
    python3 scripts/realworld/http_internvla_server_new.py

sleep "${SERVER_START_DELAY}"
verify_process http_internvla_server "http_internvla_server_new.py"
wait_for_http_server

run_in_container_detached \
    http_internvla_client \
    python3 scripts/realworld/http_internvla_client.py

sleep "${PROCESS_START_DELAY}"
verify_process http_internvla_client "http_internvla_client.py"

run_in_container_detached \
    plotjuggler \
    ros2 run plotjuggler plotjuggler -l cmd_vel.xml

run_in_container_detached \
    rviz2 \
    rviz2 -d vln_rviz.rviz

sleep "${PROCESS_START_DELAY}"
verify_process plotjuggler "plotjuggler"
verify_process rviz2 "rviz2"

echo "InternNav deployment started successfully."
echo "Bag output: ${BAG_OUTPUT}"
echo "Logs: ${LOG_DIR}"
ask_to_stop
