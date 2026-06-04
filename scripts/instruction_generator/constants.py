"""Shared constants for the instruction_generator pipeline."""

from __future__ import annotations

# Output filenames
FLOOR_CALIBRATION_FILENAME = "floor_calibration.json"
FLOOR_TRAJECTORY_FILENAME = "floor_trajectory.txt"

# Robot camera-to-base extrinsics (GaussTrace/image_projector.py)
CAMERA_TO_BASE_TRANSLATION = (0.1067, 0.0, 0.77566)
CAMERA_PITCH_RAD = 0.0

# ROS topic defaults
DEFAULT_RGB_TOPIC = "/camera/camera/color/image_raw/compressed"
DEFAULT_DEPTH_TOPIC = (
    "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth"
)
DEFAULT_ODOM_MATCHED_TOPIC = "/odom_txt/matched"
DEFAULT_FRAME_ID = "map"
DEFAULT_CHILD_FRAME_ID = "base_link"
DEFAULT_MAX_TIME_DIFF = 0.05

# Keyframe extractor
COMPRESSED_DEPTH_HEADER_SIZE = 12
DEFAULT_KEYFRAMES_PER_EPISODE = 30
DEFAULT_RECORD_FPS = 10.0
DEFAULT_DEPTH_VIS_SCALE = 10000.0
DEFAULT_SYNC_SLOP_SEC = 0.05
DEFAULT_OFFLINE_MATCH_MAX_DT = 0.5
