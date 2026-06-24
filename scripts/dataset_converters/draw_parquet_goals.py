#!/usr/bin/env python3
"""Draw parquet pixel goals on LeRobot RGB frames for debugging.

Reads goal.{setting} and relative_goal_frame_id.{setting} from episode parquet,
loads the matching observation.images.rgb JPGs, and saves annotated images.

Example:
  python scripts/dataset_converters/draw_parquet_goals.py \\
    --lerobot_root /path/to/final/office_round1 \\
    --episode_index 1 \\
    --out_dir /tmp/goal_debug_ep1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPTS_ROOT))

INVALID_GOAL = (-1, -1)
ACTION_NAMES = {-1: "START", 1: "FWD", 2: "LEFT", 3: "RIGHT"}


def list_episode_indices(lerobot_root: Path) -> List[int]:
    data_dir = lerobot_root / "data"
    if not data_dir.exists():
        return []
    indices: List[int] = []
    for pq in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        stem = pq.stem  # episode_000001
        indices.append(int(stem.split("_")[-1]))
    return sorted(indices)


def parquet_path(lerobot_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return (
        lerobot_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def rgb_image_path(
    lerobot_root: Path,
    episode_index: int,
    local_frame: int,
    height_cm: int,
    pitch_deg: int,
) -> Path:
    chunk = episode_index // 1000
    return (
        lerobot_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.rgb.{height_cm}cm_{pitch_deg}deg"
        / f"episode_{episode_index:06d}_{local_frame}.jpg"
    )


def _scalar_cell(value) -> int:
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return int(value[0]) if len(value) else -1
    return int(value)


def _goal_cell(value) -> Tuple[int, int]:
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return int(value[0]), int(value[1])
    return int(value), -1


def load_episode_parquet(
    lerobot_root: Path,
    episode_index: int,
    setting: str,
) -> Dict[str, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required: pip install pandas pyarrow") from exc

    pq_path = parquet_path(lerobot_root, episode_index)
    if not pq_path.is_file():
        raise FileNotFoundError(pq_path)

    df = pd.read_parquet(pq_path)
    goal_col = f"goal.{setting}"
    rel_col = f"relative_goal_frame_id.{setting}"
    if goal_col not in df.columns:
        available = [c for c in df.columns if c.startswith("goal.")]
        raise KeyError(f"{goal_col} not in parquet. Available goal columns: {available}")

    goals = np.array([_goal_cell(g) for g in df[goal_col].tolist()], dtype=np.int32)
    rel_ids = (
        np.array([_scalar_cell(v) for v in df[rel_col].tolist()], dtype=np.int32)
        if rel_col in df.columns
        else np.full(len(df), -1, dtype=np.int32)
    )
    actions = (
        np.array([_scalar_cell(v) for v in df["action"].tolist()], dtype=np.int32)
        if "action" in df.columns
        else np.full(len(df), -1, dtype=np.int32)
    )
    return {"goals": goals, "rel_ids": rel_ids, "actions": actions, "n_frames": len(df)}


def draw_parquet_goal_on_image(
    bgr: np.ndarray,
    goal: Tuple[int, int],
    local_frame: int,
    *,
    dot_radius: int = 8,
    show_text: bool = True,
) -> np.ndarray:
    """Draw parquet (u, v) directly on the image — no re-projection."""
    out = bgr.copy()
    u, v = int(goal[0]), int(goal[1])
    h, w = out.shape[:2]
    valid = 0 <= u < w and 0 <= v < h

    if valid:
        cv2.circle(out, (u, v), dot_radius, (0, 0, 255), -1)  # red filled dot
        cv2.circle(out, (u, v), dot_radius + 2, (255, 255, 255), 1)
    elif u >= 0 and v >= 0:
        # on-image coords but out of bounds
        cv2.putText(
            out,
            f"OOB goal ({u},{v})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    elif show_text:
        cv2.putText(
            out,
            "goal [-1,-1]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    if show_text:
        label = f"f{local_frame} ({u},{v})" if valid else f"f{local_frame} invalid"
        cv2.putText(
            out,
            label,
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def make_contact_sheet(images: List[np.ndarray], out_path: Path, cols: int = 4) -> None:
    if not images:
        return
    cols = max(1, min(cols, len(images)))
    rows = int(math.ceil(len(images) / cols))
    h, w = images[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = cv2.resize(img, (w, h))
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def process_episode(
    lerobot_root: Path,
    episode_index: int,
    out_dir: Path,
    setting: str,
    height_cm: int,
    pitch_deg: int,
    frame_stride: int,
    max_frames: Optional[int],
    all_frames: bool,
    show_text: bool = True,
) -> Dict:
    labels = load_episode_parquet(lerobot_root, episode_index, setting)
    n_frames = int(labels["n_frames"])
    goals: np.ndarray = labels["goals"]
    rel_ids: np.ndarray = labels["rel_ids"]
    actions: np.ndarray = labels["actions"]

    ep_out = out_dir / f"episode_{episode_index:06d}"
    ep_out.mkdir(parents=True, exist_ok=True)

    if all_frames:
        frame_ids = list(range(n_frames))
    else:
        frame_ids = list(range(0, n_frames, max(1, frame_stride)))
        if max_frames is not None:
            frame_ids = frame_ids[:max_frames]

    annotated: List[np.ndarray] = []
    manifest = []
    saved = 0
    valid_saved = 0

    for local_i in frame_ids:
        img_path = rgb_image_path(lerobot_root, episode_index, local_i, height_cm, pitch_deg)
        if not img_path.is_file():
            print(f"[WARN] missing image: {img_path}")
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] failed to read: {img_path}")
            continue

        goal = (int(goals[local_i, 0]), int(goals[local_i, 1]))
        rel_id = int(rel_ids[local_i])
        action = int(actions[local_i])

        vis = draw_parquet_goal_on_image(bgr, goal, local_i, show_text=show_text)
        out_path = ep_out / f"frame_{local_i:04d}.jpg"
        cv2.imwrite(str(out_path), vis)
        annotated.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        saved += 1
        if goal[0] >= 0 and goal[1] >= 0:
            valid_saved += 1

        manifest.append(
            {
                "local_frame": local_i,
                "goal": list(goal),
                "relative_goal_frame_id": rel_id,
                "action": action,
                "image": str(img_path),
                "annotated": str(out_path),
            }
        )

    if annotated:
        make_contact_sheet(annotated, ep_out / "contact_sheet.jpg", cols=4)

    summary = {
        "episode_index": episode_index,
        "setting": setting,
        "n_frames_parquet": n_frames,
        "n_valid_goals": int(np.sum((goals[:, 0] >= 0) & (goals[:, 1] >= 0))),
        "frames_saved": saved,
        "valid_goals_drawn": valid_saved,
        "output_dir": str(ep_out),
    }
    (ep_out / "manifest.json").write_text(json.dumps({"summary": summary, "frames": manifest}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw parquet pixel goals on LeRobot RGB frames.")
    parser.add_argument(
        "--lerobot_root",
        type=str,
        required=True,
        help="LeRobot scene root (contains data/ and videos/)",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=None,
        help="Episode index (e.g. 1). Omit with --all_episodes.",
    )
    parser.add_argument(
        "--all_episodes",
        action="store_true",
        help="Process every episode parquet under data/",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="125cm_30deg",
        help="Camera setting suffix for goal.{setting} column (default: 125cm_30deg)",
    )
    parser.add_argument("--height", type=int, default=125, help="RGB folder height cm")
    parser.add_argument("--pitch", type=int, default=30, help="RGB folder pitch deg")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=10,
        help="Save every N-th frame (ignored with --all_frames)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=24,
        help="Max frames per episode to save (ignored with --all_frames)",
    )
    parser.add_argument(
        "--all_frames",
        action="store_true",
        help="Annotate every frame in the episode",
    )
    parser.add_argument(
        "--no_text",
        action="store_true",
        help="Only draw the dot, no frame/coord label",
    )
    args = parser.parse_args()

    lerobot_root = Path(args.lerobot_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_episodes:
        episode_indices = list_episode_indices(lerobot_root)
        if not episode_indices:
            raise FileNotFoundError(f"No episode parquet files under {lerobot_root / 'data'}")
    elif args.episode_index is not None:
        episode_indices = [args.episode_index]
    else:
        parser.error("Pass --episode_index N or --all_episodes")

    summaries = []
    for ep_idx in episode_indices:
        print(f"\n[episode {ep_idx:06d}]")
        summary = process_episode(
            lerobot_root,
            ep_idx,
            out_dir,
            setting=args.setting,
            height_cm=args.height,
            pitch_deg=args.pitch,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            all_frames=args.all_frames,
            show_text=not args.no_text,
        )
        summaries.append(summary)
        print(
            f"  parquet frames: {summary['n_frames_parquet']} | "
            f"valid goals: {summary['n_valid_goals']} | "
            f"saved: {summary['frames_saved']} -> {summary['output_dir']}"
        )

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
