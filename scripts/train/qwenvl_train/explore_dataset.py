#!/usr/bin/env python3
"""

Quick start (from InternNav repo root):
  export INTERNAV_R2R_DATA_PATH="${PWD}/data/InternData-N1/vln_ce/traj_data/r2r" python scripts/train/qwenvl_train/explore_dataset.py

  INTERNAV_R2R_DATA_PATH="${PWD}/data/round1_bkhn/traj_data/r2r" python scripts/train/qwenvl_train/explore_dataset.py
  python scripts/train/qwenvl_train/explore_dataset.py

With a local Qwen checkpoint (needed for image preprocessing + token shapes):
  python scripts/train/qwenvl_train/explore_dataset.py \\
    --model_name checkpoints/Qwen2.5-VL-7B-Instruct \\
    --sample-indices 0 1 2 \\
    --collate

we have
  1. Dataset config resolution (env vars, sampling %)
  2. On-disk LeRobot layout + parquet schema
  3. Episode index breakdown (pixel-goal / turn / stop samples)
  4. Single-sample fields (images, chat, tokens)
  5. Collated batch keys (what Trainer passes to model.forward)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from internnav.dataset.internvla_n1_lerobot_dataset import (  # noqa: E402
    IGNORE_INDEX,
    TRAJ_TOKEN_INDEX,
    data_list,
    debug_dataset_path,
    get_annotations_from_lerobot_data,
    make_supervised_data_module,
)


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _tensor_summary(name: str, value: Any) -> None:
    if value is None:
        print(f"  {name}: None")
        return
    if torch.is_tensor(value):
        print(f"  {name}: Tensor shape={tuple(value.shape)} dtype={value.dtype}")
        return
    if isinstance(value, (list, tuple)):
        print(f"  {name}: {type(value).__name__} len={len(value)}")
        return
    print(f"  {name}: {type(value).__name__} = {value!r}")


@dataclass
class ExploreDataArgs:
    """Minimal stand-in for DataArguments used by NavPixelGoalDataset."""

    vln_dataset_use: str = "r2r_125cm_0_30%10"
    iign_dataset_use: str = ""
    sample_step: int = 4
    num_history: int = 1
    predict_step_num: int = 32
    pixel_goal_only: bool = False
    num_future_steps: int = 4
    data_augmentation: bool = False
    resize_h: int = 224
    resize_w: int = 224
    max_pixels: int = 78400
    min_pixels: int = 3136
    data_flatten: bool = False
    model_type: str = "qwen2.5vl"
    image_processor: Any = None
    transform_train: Any = None


def build_data_args(args: argparse.Namespace) -> ExploreDataArgs:
    data_args = ExploreDataArgs(
        vln_dataset_use=args.vln_datasets,
        sample_step=args.sample_step,
        num_history=args.num_history,
        predict_step_num=args.predict_step_num,
        pixel_goal_only=args.pixel_goal_only,
        num_future_steps=args.num_future_steps,
        data_augmentation=args.data_augmentation,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        data_flatten=args.data_flatten,
    )
    if args.data_augmentation:
        data_args.transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ColorJitter(brightness=0.2, saturation=0.2),
                v2.RandomPosterize(bits=4),
                v2.RandomAdjustSharpness(sharpness_factor=1.5),
                v2.RandomAutocontrast(),
                v2.ToPILImage(),
                v2.Resize((args.resize_h, args.resize_w)),
            ]
        )
    else:
        data_args.transform_train = v2.Resize((args.resize_h, args.resize_w))
    return data_args


def load_tokenizer_and_processor(model_name: Optional[str]):
    if not model_name:
        return None, None
    model_path = Path(model_name)
    if not model_path.is_absolute():
        candidate = REPO_ROOT / model_name
        if candidate.exists():
            model_path = candidate
    if not model_path.exists():
        print(f"WARN: model path not found: {model_name} — skipping tokenizer/image_processor")
        return None, None

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path),
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    image_processor = transformers.AutoProcessor.from_pretrained(str(model_path)).image_processor
    return tokenizer, image_processor


def print_env_and_config(vln_datasets: str) -> List[Dict]:
    _banner("1. Dataset configuration")
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"VLN_DATASETS: {vln_datasets}")
    print(f"INTERNAV_R2R_DATA_PATH: {os.environ.get('INTERNAV_R2R_DATA_PATH', '(not set)')}")
    print(f"INTERNAV_RXR_DATA_PATH: {os.environ.get('INTERNAV_RXR_DATA_PATH', '(not set)')}")
    print(f"INTERNAV_SCALEVLN_DATA_PATH: {os.environ.get('INTERNAV_SCALEVLN_DATA_PATH', '(not set)')}")
    print()
    print("Registered dataset keys (from internvla_n1_lerobot_dataset.data_dict):")
    print("  r2r_125cm_0_30, r2r_125cm_0_45, r2r_60cm_15_15, r2r_60cm_30_30")
    print("  rxr_125cm_0_30, rxr_125cm_0_45, rxr_60cm_15_15, rxr_60cm_30_30")
    print("  scalevln_125cm_0_30, scalevln_125cm_0_45, scalevln_60cm_30_30")
    print()

    configs = data_list(vln_datasets.split(","))
    for i, cfg in enumerate(configs):
        setting = f"{cfg['height']}cm_{cfg['pitch_2']}deg"
        print(f"  [{i}] resolved config:")
        print(f"      data_path:     {cfg['data_path']}")
        print(f"      camera height: {cfg['height']} cm")
        print(f"      pitch pair:    {cfg['pitch_1']}° (history rgb) / {cfg['pitch_2']}° (lookdown)")
        print(f"      setting key:   {setting}")
        print(f"      sampling_rate: {cfg.get('sampling_rate', 1.0)}")
    return configs


def print_index_breakdown(configs: Sequence[Dict], data_args: ExploreDataArgs) -> None:
    _banner("2. Training sample index (before __getitem__)")
    print(
        "NavPixelGoalDataset scans every episode parquet and builds sliding-window "
        "training examples every sample_step frames.\n"
    )
    print("Sample types:")
    print("  pixel_goal — agent has a 2D waypoint in the lookdown image; target is 'x y' coords")
    print("  turn       — no pixel goal yet; target is turn symbols (← →)")
    print("  stop       — end of episode; target is STOP (replicated 5× unless pixel_goal_only)")
    print()
    print(f"  sample_step={data_args.sample_step}  num_history={data_args.num_history}")
    print(f"  num_future_steps={data_args.num_future_steps}  pixel_goal_only={data_args.pixel_goal_only}")
    print()

    total = 0
    for cfg in configs:
        setting = f"{cfg['height']}cm_{cfg['pitch_2']}deg"
        data_path = cfg["data_path"]
        if not os.path.isdir(data_path):
            print(f"SKIP {data_path} — directory missing")
            continue

        annotations = get_annotations_from_lerobot_data(data_path, setting)
        height = cfg["height"]
        pitch_1 = cfg["pitch_1"]
        pitch_2 = cfg["pitch_2"]
        sample_step = data_args.sample_step
        num_future_steps = data_args.num_future_steps
        pixel_goal_only = data_args.pixel_goal_only

        pixel_goal_list = []
        turn_list = []
        stop_list = []

        for item in annotations["episodes"]:
            actions = item["actions"][1:] + [0]
            pixel_goals = item["pixel_goals"]
            actions_len = len(actions)
            if actions_len < 4:
                continue

            num_rounds = actions_len // sample_step
            for n in range(num_rounds + 1):
                if n * sample_step == actions_len or n * sample_step == actions_len - 1:
                    continue
                start_frame_id = n * sample_step
                action_flag = actions[start_frame_id]
                pixel_goal = pixel_goals[start_frame_id]
                if pixel_goal[0] == -1:
                    if action_flag == 1:
                        continue
                    end_frame_id = min(actions_len, start_frame_id + num_future_steps)
                    turn_actions = []
                    for idx in range(start_frame_id, end_frame_id):
                        if actions[idx] == 1:
                            break
                        turn_actions.append(actions[idx])
                    turn_list.append((item["id"], start_frame_id, turn_actions))
                else:
                    goal_len = pixel_goal[0]
                    if goal_len < 3:
                        continue
                    pixel_goal_list.append((item["id"], start_frame_id, goal_len, pixel_goal[1]))

            stop_list.append((item["id"], actions_len - 1))

        list_data_dict = list(pixel_goal_list)
        if not pixel_goal_only:
            list_data_dict += turn_list
            list_data_dict += stop_list * 5

        sampling_rate = cfg.get("sampling_rate", 1.0)
        final_count = int(len(list_data_dict) * sampling_rate) if sampling_rate < 1.0 else len(list_data_dict)
        total += final_count

        print(f"  {data_path} [{setting}]")
        print(f"    episodes loaded: {len(annotations['episodes'])}")
        print(f"    raw pixel_goal:  {len(pixel_goal_list)}")
        print(f"    raw turn:        {len(turn_list)}")
        print(f"    raw stop:        {len(stop_list)} (×5 in mix → {len(stop_list) * 5})")
        print(f"    after mix:       {len(list_data_dict)}")
        print(f"    after sampling:  {final_count} (rate={sampling_rate})")
        if pixel_goal_list:
            ep_id, start, goal_len, action = pixel_goal_list[0]
            print(f"    example pixel_goal[0]: ep={ep_id} frame={start} goal_len={goal_len} action={action}")
        if turn_list:
            ep_id, start, turns = turn_list[0]
            print(f"    example turn[0]:       ep={ep_id} frame={start} turns={turns}")
        print()

    print(f"Estimated total training samples (all configs): {total}")


def decode_labels(tokenizer, labels: torch.Tensor) -> str:
    mask = labels != IGNORE_INDEX
    if not mask.any():
        return "(no supervised tokens — all IGNORE_INDEX)"
    ids = labels[mask].tolist()
    return tokenizer.decode(ids, skip_special_tokens=False)


def print_sample_detail(dataset, tokenizer, index: int) -> None:
    _banner(f"3. Sample #{index} (__getitem__)")
    raw = dataset.list_data_dict[index]
    ep_id, data_path, video, height, pitch_1, pitch_2, instruction, frame_range, action, pose = raw
    start_frame_id, end_frame_id = frame_range

    print("Index entry (list_data_dict[i]):")
    print(f"  episode_id:    {ep_id}")
    print(f"  data_path:     {data_path}")
    print(f"  video chunk:   {video}")
    print(f"  height:        {height} cm")
    print(f"  pitch_1/2:     {pitch_1} / {pitch_2}")
    print(f"  instruction:   {instruction[:120]}{'...' if len(instruction) > 120 else ''}")
    print(f"  frame range:   [{start_frame_id}, {end_frame_id})")
    print(f"  action/pose:   action={action!r}  pose={'present' if pose is not None else 'None'}")
    print()

    print("Image file pattern (rgb history + lookdown at start_frame):")
    print(
        f"  {video}/observation.images.rgb.{height}cm_{pitch_1}deg/episode_{ep_id:06d}_<frame>.jpg"
    )
    print(
        f"  {video}/observation.images.rgb.{height}cm_{pitch_2}deg/episode_{ep_id:06d}_<frame>.jpg  (lookdown)"
    )
    print(
        f"  depth: .../observation.images.depth.{height}cm_{pitch_2}deg/episode_{ep_id:06d}_<frame>.png"
    )
    print()

    sample = dataset[index]
    print("Returned dict keys:", sorted(sample.keys()))
    for key in ("input_ids", "labels", "position_ids", "pixel_values", "image_grid_thw"):
        _tensor_summary(key, sample.get(key))
    for key in ("traj_images", "traj_depths", "traj_poses"):
        if key in sample:
            _tensor_summary(key, sample[key])

    if tokenizer is not None:
        input_ids = sample["input_ids"].squeeze(0)
        labels = sample["labels"].squeeze(0)
        print()
        print(f"  input_ids length: {input_ids.numel()}")
        print(f"  supervised label tokens: {(labels != IGNORE_INDEX).sum().item()}")
        print("  decoded target (labels != -100):")
        print("  ---")
        print(f"  {decode_labels(tokenizer, labels)}")
        print("  ---")


def print_collated_batch(data_module: Dict, indices: Sequence[int]) -> None:
    _banner("4. Collated batch (DataCollator → model.forward keys)")
    dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]
    instances = [dataset[i] for i in indices]
    batch = collator(instances)

    print(f"Batch size: {len(indices)}  indices: {list(indices)}")
    print()
    print("Batch keys and shapes:")
    for key, value in batch.items():
        _tensor_summary(key, value)

    print()
    print("Model consumption (System-2 / Qwen2.5-VL path):")
    print("  input_ids, attention_mask, position_ids  → text + special vision/traj token slots")
    print("  pixel_values, image_grid_thw             → vision encoder (IMAGE_TOKEN_INDEX patches)")
    print("  labels                                     → causal LM loss (IGNORE_INDEX = masked)")
    if "t_s_pos" in batch:
        print("  t_s_pos                                    → index where <traj> tokens are appended")
        print("  traj_images, traj_depths, traj_poses       → System-1 head (dual-system / pixel_goal_only)")
        print("  video_frame_num                            → valid frames per traj sequence")
    else:
        print("  (no traj_* keys — System-1 disabled; system1=none in train_system2_thor.sh)")




def parse_args() -> argparse.Namespace:
    default_r2r = str(REPO_ROOT / "data/InternData-N1/vln_ce/traj_data/r2r")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vln-datasets", default=os.environ.get("VLN_DATASETS", "r2r_125cm_0_30%10"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", ""), help="Qwen2.5-VL path for tokenizer/processor")
    parser.add_argument("--sample-indices", type=int, nargs="*", default=[0], help="Dataset indices to load")
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--num-history", type=int, default=1)
    parser.add_argument("--predict-step-num", type=int, default=32)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--pixel-goal-only", action="store_true", default=False)
    parser.add_argument("--data-augmentation", action="store_true", default=False)
    parser.add_argument("--resize-h", type=int, default=224)
    parser.add_argument("--resize-w", type=int, default=224)
    parser.add_argument("--max-pixels", type=int, default=78400)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--data-flatten", action="store_true", default=False)
    parser.add_argument("--collate", action="store_true", help="Also run DataCollator on sample-indices")
    parser.add_argument("--skip-filesystem", action="store_true", help="Skip on-disk schema checks")
    parser.add_argument("--skip-index", action="store_true", help="Skip full episode index scan")
    parser.add_argument("--set-r2r-path", default="./data/round1_bkhn/traj_data/", help=f"Set INTERNAV_R2R_DATA_PATH (default: {default_r2r})")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.set_r2r_path:
        os.environ["INTERNAV_R2R_DATA_PATH"] = args.set_r2r_path
    elif "INTERNAV_R2R_DATA_PATH" not in os.environ:
        os.environ["INTERNAV_R2R_DATA_PATH"] = str(REPO_ROOT / "data/InternData-N1/vln_ce/traj_data/r2r")

    configs = print_env_and_config(args.vln_datasets)

    if not args.skip_filesystem:
        _banner("Filesystem + parquet schema probe")
        for cfg in configs:
            setting = f"{cfg['height']}cm_{cfg['pitch_2']}deg"
            debug_dataset_path(cfg["data_path"], setting)

    data_args = build_data_args(args)

    if not args.skip_index:
        print_index_breakdown(configs, data_args)

    tokenizer, image_processor = load_tokenizer_and_processor(args.model_name or None)
    if image_processor is None:
        print(
            "\nWARN: No image_processor — cannot call dataset.__getitem__ (needs Qwen processor). "
            "Pass --model-name checkpoints/Qwen2.5-VL-7B-Instruct to load real samples."
        )
        return 0

    data_args.image_processor = image_processor
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module["train_dataset"]

    _banner("Dataset ready")
    print(f"CombinedDataset length: {len(dataset)}")
    if len(dataset) == 0:
        print("ERROR: empty dataset — extract scene tarballs and verify parquet columns.")
        return 1

    # NavPixelGoalDataset is nested inside CombinedDataset
    nav_ds = None
    for ds in dataset.datasets:
        if type(ds).__name__ == "NavPixelGoalDataset":
            nav_ds = ds
            break
    target_ds = nav_ds if nav_ds is not None else dataset

    for idx in args.sample_indices:
        if idx < 0 or idx >= len(dataset):
            print(f"SKIP sample index {idx} — out of range [0, {len(dataset)})")
            continue
        print_sample_detail(target_ds, tokenizer, idx)

    if args.collate and args.sample_indices:
        valid = [i for i in args.sample_indices if 0 <= i < len(dataset)]
        if valid:
            print_collated_batch(data_module, valid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
