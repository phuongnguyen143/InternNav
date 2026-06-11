"""Shared depth encode/decode for RealSense compressedDepth and LeRobot uint16 mm PNGs."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

COMPRESSED_DEPTH_HEADER_SIZE = 12
PNG_MAGIC = b"\x89PNG"
MM_PER_METER = 1000.0
MAX_DEPTH_MM = 65535
LEGACY_DEPTH_VIS_SCALE = 10000.0


def decode_compressed_depth(data: bytes, format_hint: str = "") -> Optional[np.ndarray]:
    """Decode ROS sensor_msgs/CompressedImage compressedDepth payload to float32 meters.

    The payload is not a raw PNG: the first 12 bytes are three float32 quantization
    parameters, followed by the compressed PNG bytes.
    """
    if len(data) <= COMPRESSED_DEPTH_HEADER_SIZE:
        return None

    depth_quant_a = 0.0
    depth_quant_b = 0.0
    image_data = data

    fmt = format_hint or ""
    has_depth_header = "compressedDepth" in fmt
    if not has_depth_header and len(data) > 16:
        has_depth_header = data[12:16] == PNG_MAGIC

    if has_depth_header:
        depth_quant_a, depth_quant_b, _ = struct.unpack("<fff", data[:COMPRESSED_DEPTH_HEADER_SIZE])
        image_data = data[COMPRESSED_DEPTH_HEADER_SIZE:]

    depth = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        return None

    if depth_quant_a != 0.0:
        depth = depth.astype(np.float32)
        valid = depth != 0
        depth_out = np.zeros_like(depth, dtype=np.float32)
        depth_out[valid] = depth_quant_a / (depth[valid].astype(np.float32) - depth_quant_b)
        depth = depth_out
    else:
        depth = depth.astype(np.float32)

    depth[~np.isfinite(depth)] = 0.0
    return depth


def meters_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    """Convert float32 depth in meters to uint16 millimeters (0 = invalid)."""
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    valid = depth_m > 0
    if valid.any():
        scaled = np.rint(depth_m[valid].astype(np.float64) * MM_PER_METER)
        depth_mm[valid] = np.clip(scaled, 0, MAX_DEPTH_MM).astype(np.uint16)
    return depth_mm


def uint16_mm_to_meters(depth_mm: np.ndarray) -> np.ndarray:
    """Convert uint16 millimeters to float32 meters."""
    depth_m = np.zeros(depth_mm.shape, dtype=np.float32)
    valid = depth_mm > 0
    depth_m[valid] = depth_mm[valid].astype(np.float32) / MM_PER_METER
    return depth_m


def save_depth_png_mm(path: Path, depth_m: np.ndarray) -> None:
    """Save depth as 16-bit PNG with pixel values in millimeters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = meters_to_uint16_mm(depth_m)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"Failed to write depth PNG: {path}")


def load_depth_png_mm(path: Path) -> Tuple[np.ndarray, dict]:
    """Load a LeRobot/sim depth PNG (uint16 mm) and return float32 meters + metadata."""
    data = path.read_bytes()
    meta = _decode_png_file_meta(data)
    raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise ValueError(f"cv2.imread failed: {path}")

    meta["raw"] = raw
    meta["raw_dtype"] = str(raw.dtype)
    meta["shape"] = raw.shape

    if raw.dtype == np.uint16 or raw.max() > 255:
        depth_m = uint16_mm_to_meters(raw)
        meta["unit_source"] = "png_uint16_mm"
    else:
        raise ValueError(
            f"Expected uint16 mm depth PNG at {path}, got dtype={raw.dtype} max={raw.max()}. "
            "Re-run keyframe extraction with the updated pipeline."
        )

    return depth_m, meta


def _decode_png_file_meta(data: bytes) -> dict:
    meta: dict = {
        "format": None,
        "quant_a": 0.0,
        "quant_b": 0.0,
        "quant_c": 0.0,
    }
    if data[:4] == PNG_MAGIC:
        meta["format"] = "png_uint16_mm"
    elif len(data) > 16 and data[12:16] == PNG_MAGIC:
        meta["format"] = "compressed_depth"
        meta["quant_a"], meta["quant_b"], meta["quant_c"] = struct.unpack(
            "<fff", data[:COMPRESSED_DEPTH_HEADER_SIZE]
        )
    else:
        raise ValueError(
            "Unrecognized depth file: expected PNG magic at byte 0 or byte 12 "
            f"(got {data[:4]!r} at 0, {data[12:16]!r} at 12)"
        )
    return meta


def decode_depth_bytes(data: bytes, unit: str = "auto") -> Tuple[np.ndarray, dict]:
    """Decode depth from raw file bytes (PNG mm or embedded compressedDepth payload)."""
    meta = _decode_png_file_meta(data)

    if meta["format"] == "png_uint16_mm":
        image_data = data
    else:
        image_data = data[COMPRESSED_DEPTH_HEADER_SIZE:]

    raw = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise ValueError("cv2.imdecode failed")

    meta["raw"] = raw
    meta["raw_dtype"] = str(raw.dtype)
    meta["shape"] = raw.shape

    if meta["format"] == "compressed_depth" and meta["quant_a"] != 0.0:
        depth = np.zeros(raw.shape, dtype=np.float32)
        valid = raw != 0
        depth[valid] = meta["quant_a"] / (raw[valid].astype(np.float32) - meta["quant_b"])
        meta["unit_source"] = "compressed_depth_meters"
    elif unit == "raw":
        return raw.astype(np.float32), meta
    elif unit == "m":
        depth = raw.astype(np.float32)
        meta["unit_source"] = "assumed_meters"
    else:
        depth = uint16_mm_to_meters(raw.astype(np.uint16))
        meta["unit_source"] = "auto_mm_to_m"

    depth[~np.isfinite(depth)] = 0.0
    return depth, meta


def decode_depth_image(path: Path, unit: str = "auto") -> Tuple[np.ndarray, dict]:
    return decode_depth_bytes(path.read_bytes(), unit=unit)


def legacy_depth_vis_to_uint16_mm(
    depth_u8: np.ndarray,
    vis_scale: float = LEGACY_DEPTH_VIS_SCALE,
) -> np.ndarray:
    """Convert 8-bit depth preview frames (from legacy depth.mp4) to uint16 mm."""
    depth_m = np.zeros(depth_u8.shape, dtype=np.float32)
    valid = depth_u8 > 0
    depth_m[valid] = depth_u8[valid].astype(np.float32) * vis_scale / 255.0
    return meters_to_uint16_mm(depth_m)


def resize_depth_nearest(depth: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """Resize depth with nearest-neighbor (size_wh = width, height)."""
    w, h = size_wh
    if depth.shape[1] == w and depth.shape[0] == h:
        return depth
    if depth.dtype != np.uint16:
        depth = meters_to_uint16_mm(depth.astype(np.float32))
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
