#!/usr/bin/env python3
"""Convert extract_bag_frames output -> inference_only_attention_map scene folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


def write_instruction(instruction: str | None, output_dir: Path) -> Path | None:
    """Copy instruction from file or write literal text to output_dir/instruction.txt."""
    if not instruction or not instruction.strip():
        return None

    out_path = output_dir / "instruction.txt"
    candidate = Path(instruction)
    if candidate.is_file():
        shutil.copy2(candidate, out_path)
    else:
        out_path.write_text(instruction.strip() + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--extract-dir",
        type=Path,
        required=True,
        help="e.g. .../keyframe_output_internnav_20260625",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="e.g. .../assets/internnav_20260625_scene",
    )
    p.add_argument(
        "--instruction",
        default=None,
        help=(
            "Navigation instruction: literal text (written to instruction.txt) "
            "or path to an existing instruction.txt file"
        ),
    )
    p.add_argument(
        "--look-down-every",
        type=int,
        default=10,
        help="Mark every Nth frame with _look_down (0 = disable)",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First debug_raw index (default 1 -> 0001)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth extracted frame (default: 1 = all frames)",
    )
    args = p.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    rgb_src = args.extract_dir / "tmp" / "rgb_frames"
    depth_src = args.extract_dir / "tmp" / "depth_frames"
    if not rgb_src.is_dir():
        raise FileNotFoundError(f"Missing {rgb_src}")

    out_rgb = args.output_dir
    out_depth = args.output_dir / "depth"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(rgb_src.glob("frame_*.jpg"))
    if not rgb_files:
        raise FileNotFoundError(f"No frame_*.jpg in {rgb_src}")
    if args.stride > 1:
        rgb_files = rgb_files[:: args.stride]

    for i, rgb_file in enumerate(rgb_files, start=1):
        frame_id = str(args.start_index + i - 1).zfill(4)
        is_look_down = args.look_down_every > 0 and i % args.look_down_every == 0
        stem = f"debug_raw_{frame_id}" + ("_look_down" if is_look_down else "")
        rgb_name = f"{stem}.jpg"
        depth_name = f"{stem}.png"

        Image.open(rgb_file).convert("RGB").save(out_rgb / rgb_name, quality=95)

        depth_file = depth_src / f"{rgb_file.stem}.png"
        if depth_file.is_file():
            shutil.copy2(depth_file, out_depth / depth_name)

    instruction_path = write_instruction(args.instruction, args.output_dir)

    print(f"Wrote {len(rgb_files)} RGB frames -> {out_rgb}")
    print(f"Depth -> {out_depth}")
    if instruction_path is not None:
        print(f"Instruction: {instruction_path}")
    if args.stride > 1:
        print(f"Stride: every {args.stride} source frame(s)")
    if args.look_down_every > 0:
        n_ld = sum(
            1 for i in range(1, len(rgb_files) + 1) if i % args.look_down_every == 0
        )
        print(f"Look-down frames: {n_ld} (every {args.look_down_every})")


if __name__ == "__main__":
    main()
