#!/usr/bin/env python3
"""Extract synchronized RGB + depth frames from a ROS2 rosbag (mcap/sqlite3).

This is the offline counterpart to the live keyframe_extractor stream path
(`ros2 bag play` + subscribed topics). Output matches the tmp layout used by
keyframe_extractor before finalize:

  <output_dir>/
    tmp/rgb_frames/frame_*.jpg
    tmp/depth_frames/frame_*.png
    tmp/depth_full.mp4          # optional debug preview
    frames.json                 # frame_idx, timestamp list

After WildGS-SLAM, attach bag timestamps to estimated poses (GaussTrace odom format):

  python extract_bag_frames.py export-odom \\
      --frames-json ./keyframe_output_offline/frames.json \\
      --poses ./output/wildgs_slam_custom_helmet/office/traj/est_poses_full.txt \\
      --output ./keyframe_output_offline/odometry_camera.txt \\
      --stride 3

Example:
  python extract_bag_frames.py /path/to/bkhn_round2 \\
      --output-dir ./keyframe_output_offline

  # Two-camera bag (sqlite3, rs1 topics):
  python extract_bag_frames.py /path/to/realsense_bag \\
      --rgb-topic /rs1/rs1/color/image_raw/compressed \\
      --depth-topic /rs1/rs1/depth/image_rect_raw/compressedDepth

  # Full bag (no 20s head/tail trim):
  python extract_bag_frames.py /path/to/bag --no-trim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from utils.config import get_config
from utils.trajectory_io import export_odom_from_tum_and_frames
from frame_utils import (
    depth_m_to_preview_bgr,
    open_mp4_writer,
    ros_stamp_to_sec,
    save_depth_frame_mm,
    save_rgb_frame,
    sync_rgb_depth_messages,
)


def detect_storage_id(bag_path: Path, storage_id: str) -> str:
    """Resolve rosbag2 storage plugin from metadata.yaml or bag file extensions."""
    if storage_id != "auto":
        return storage_id

    metadata_path = bag_path / "metadata.yaml"
    if metadata_path.is_file():
        for line in metadata_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("storage_identifier:"):
                detected = stripped.split(":", 1)[1].strip()
                if detected:
                    return detected

    if any(bag_path.glob("*.db3")):
        return "sqlite3"
    if any(bag_path.glob("*.mcap")):
        return "mcap"

    raise RuntimeError(
        f"Could not detect rosbag2 storage format in {bag_path}. " "Pass --storage-id mcap or sqlite3 explicitly."
    )


def read_bag_topic_messages(
    bag_path: Path,
    topics: List[str],
    storage_id: str,
) -> Dict[str, List[Tuple]]:
    try:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "rosbag2_py is required. Source your ROS2 workspace first, e.g.\n" "  source /opt/ros/humble/setup.bash"
        ) from exc

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id=storage_id),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in topic_types]
    if missing:
        available = sorted(topic_types)
        raise RuntimeError(f"Topics not found in bag: {missing}\nAvailable topics:\n  " + "\n  ".join(available))

    type_map = {name: get_message(topic_types[name]) for name in topics}
    buckets: Dict[str, List[Tuple]] = {name: [] for name in topics}

    while reader.has_next():
        topic, data, _bag_ts = reader.read_next()
        if topic not in buckets:
            continue
        msg = deserialize_message(data, type_map[topic])
        ts = ros_stamp_to_sec(msg.header.stamp)
        if topic.endswith("compressedDepth") or "compressedDepth" in (msg.format or ""):
            buckets[topic].append((ts, bytes(msg.data), msg.format or ""))
        else:
            buckets[topic].append((ts, bytes(msg.data)))

    for name in topics:
        buckets[name].sort(key=lambda item: item[0])
    return buckets


def filter_rgb_messages_by_time(
    rgb_messages: List[Tuple[float, bytes]],
    trim_start_sec: float,
    trim_end_sec: float,
) -> Tuple[List[Tuple[float, bytes]], float, float]:
    """Drop the first trim_start_sec and last trim_end_sec of the RGB stream."""
    if not rgb_messages:
        return [], 0.0, 0.0
    if trim_start_sec <= 0 and trim_end_sec <= 0:
        return rgb_messages, rgb_messages[0][0], rgb_messages[-1][0]

    t_min = rgb_messages[0][0]
    t_max = rgb_messages[-1][0]
    ts_lo = t_min + trim_start_sec
    ts_hi = t_max - trim_end_sec
    duration = t_max - t_min

    if ts_lo >= ts_hi:
        raise RuntimeError(
            f"Trim window is empty: bag duration={duration:.1f}s, "
            f"trim_start={trim_start_sec:.1f}s, trim_end={trim_end_sec:.1f}s "
            f"(need at least {trim_start_sec + trim_end_sec:.1f}s total)"
        )

    trimmed = [(ts, data) for ts, data in rgb_messages if ts_lo <= ts <= ts_hi]
    if not trimmed:
        raise RuntimeError(
            f"No RGB messages in trimmed window [{ts_lo:.3f}, {ts_hi:.3f}] "
            f"(bag [{t_min:.3f}, {t_max:.3f}], duration={duration:.1f}s)"
        )

    print(
        f"Trimming bag: skip first {trim_start_sec:.1f}s and last {trim_end_sec:.1f}s "
        f"-> keep {ts_lo - t_min:.1f}s to {duration - trim_end_sec:.1f}s "
        f"({len(trimmed)}/{len(rgb_messages)} RGB messages)"
    )
    return trimmed, ts_lo, ts_hi


def extract_frames(
    bag_path: Path,
    output_dir: Path,
    rgb_topic: str,
    depth_topic: str,
    storage_id: str,
    fps: float,
    sync_slop_sec: float,
    write_depth_preview: bool,
    trim_start_sec: float,
    trim_end_sec: float,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    rgb_frames_dir = tmp_dir / "rgb_frames"
    depth_frames_dir = tmp_dir / "depth_frames"
    rgb_frames_dir.mkdir(parents=True, exist_ok=True)
    depth_frames_dir.mkdir(parents=True, exist_ok=True)

    buckets = read_bag_topic_messages(
        bag_path,
        [rgb_topic, depth_topic],
        storage_id=storage_id,
    )

    rgb_messages = buckets[rgb_topic]
    depth_messages = buckets[depth_topic]
    if not rgb_messages:
        raise RuntimeError(f"No RGB messages on topic: {rgb_topic}")
    if not depth_messages:
        raise RuntimeError(f"No depth messages on topic: {depth_topic}")

    rgb_messages, _ts_lo, _ts_hi = filter_rgb_messages_by_time(
        rgb_messages,
        trim_start_sec=trim_start_sec,
        trim_end_sec=trim_end_sec,
    )

    depth_writer = None
    frame_records = []
    count = 0

    for frame in sync_rgb_depth_messages(
        rgb_messages,
        depth_messages,
        sync_slop_sec=sync_slop_sec,
    ):
        if write_depth_preview and depth_writer is None:
            h, w = frame.rgb_bgr.shape[:2]
            depth_writer = open_mp4_writer(tmp_dir / "depth_full.mp4", fps, (w, h))

        save_rgb_frame(rgb_frames_dir / f"frame_{count:06d}.jpg", frame.rgb_bgr)
        save_depth_frame_mm(depth_frames_dir / f"frame_{count:06d}.png", frame.depth_m)

        if depth_writer is not None:
            depth_writer.write(depth_m_to_preview_bgr(frame.depth_m))

        frame_records.append(
            {
                "frame_idx": count,
                "timestamp": frame.timestamp,
            }
        )
        count += 1

        if count % 100 == 0:
            print(f"  extracted frame {count}", flush=True)

    if depth_writer is not None:
        depth_writer.release()

    with open(output_dir / "frames.json", "w") as f:
        json.dump(frame_records, f, indent=2)

    print(f"Extracted {count} synchronized frames")
    print(f"  RGB images:     {rgb_frames_dir}")
    print(f"  Depth PNGs:     {depth_frames_dir}")
    if write_depth_preview:
        print(f"  Depth preview:  {tmp_dir / 'depth_full.mp4'}")
    print(f"  Frame metadata: {output_dir / 'frames.json'}")
    return count


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    ros = cfg.ros
    keyframe = cfg.keyframe
    parser = argparse.ArgumentParser(
        description="Extract synchronized RGB + depth frames from a ROS2 bag, "
        "or export camera odometry with bag timestamps.",
    )
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract synchronized RGB + depth frames from a ROS2 bag (default)",
    )
    extract_parser.add_argument("bag_path", type=Path, help="Path to rosbag2 directory or file")
    extract_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./keyframe_output_offline"),
        help="Output directory (default: ./keyframe_output_offline)",
    )
    extract_parser.add_argument("--rgb-topic", default=ros.get("rgb_topic"))
    extract_parser.add_argument("--depth-topic", default=ros.get("depth_topic"))
    extract_parser.add_argument(
        "--storage-id",
        default="auto",
        choices=("auto", "mcap", "sqlite3"),
        help="rosbag2 storage id (default: auto-detect from metadata.yaml or file extension)",
    )
    extract_parser.add_argument("--fps", type=float, default=keyframe.get("record_fps"))
    extract_parser.add_argument(
        "--sync-slop",
        type=float,
        default=ros.get("sync_slop_sec"),
        help="Max RGB/depth timestamp difference in seconds",
    )
    extract_parser.add_argument(
        "--write-depth-preview",
        action="store_true",
        help="Also write tmp/depth_full.mp4 debug preview video",
    )
    extract_parser.add_argument(
        "--trim-start",
        type=float,
        default=20.0,
        help="Seconds to skip from the start of the bag (default: 20)",
    )
    extract_parser.add_argument(
        "--trim-end",
        type=float,
        default=20.0,
        help="Seconds to skip from the end of the bag (default: 20)",
    )
    extract_parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Extract the full bag (sets --trim-start and --trim-end to 0)",
    )

    export_parser = subparsers.add_parser(
        "export-odom",
        help="Merge WildGS-SLAM TUM poses with frames.json timestamps into odom txt",
    )
    export_parser.add_argument(
        "--frames-json",
        type=Path,
        required=True,
        help="frames.json from extract (per-frame ROS timestamps)",
    )
    export_parser.add_argument(
        "--poses",
        type=Path,
        required=True,
        help="WildGS-SLAM est_poses_full.txt (frame_idx tx ty tz qx qy qz qw)",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output odometry txt (timestamp + 4x4 matrix, GaussTrace format)",
    )
    export_parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride used by WildGS-SLAM (must match custom_droid_w.yaml stride)",
    )

    return parser.parse_args()


def run_export_odom(args: argparse.Namespace) -> int:
    if not args.frames_json.is_file():
        print(f"Error: frames.json not found: {args.frames_json}", file=sys.stderr)
        return 1
    if not args.poses.is_file():
        print(f"Error: poses file not found: {args.poses}", file=sys.stderr)
        return 1

    try:
        output_path = export_odom_from_tum_and_frames(
            tum_poses_file=args.poses,
            frames_json=args.frames_json,
            output_file=args.output,
            stride=args.stride,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Camera odometry written to {output_path}")
    return 0


def main() -> int:
    # Backward compatible: `python extract_bag_frames.py <bag>` => extract subcommand.
    if len(sys.argv) > 1 and sys.argv[1] not in ("extract", "export-odom", "-h", "--help"):
        sys.argv.insert(1, "extract")

    args = parse_args()

    if args.command == "export-odom":
        return run_export_odom(args)

    if not args.bag_path.exists():
        print(f"Error: bag path not found: {args.bag_path}", file=sys.stderr)
        return 1

    try:
        storage_id = detect_storage_id(args.bag_path, args.storage_id)
        if args.storage_id == "auto":
            print(f"Detected rosbag2 storage: {storage_id}")

        count = extract_frames(
            args.bag_path,
            args.output_dir,
            rgb_topic=args.rgb_topic,
            depth_topic=args.depth_topic,
            storage_id=storage_id,
            fps=args.fps,
            sync_slop_sec=args.sync_slop,
            write_depth_preview=args.write_depth_preview,
            trim_start_sec=0.0 if args.no_trim else args.trim_start,
            trim_end_sec=0.0 if args.no_trim else args.trim_end,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0 if count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
