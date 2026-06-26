import argparse
import gc
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import torch
from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

from frame_utils import write_rgb_mp4_segment
from prompts import TRAJECTORY_PROMPT_AFTER, TRAJECTORY_PROMPT_BEFORE

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_WINDOW_SIZE = 7
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_NUM_VIDEO_FRAMES = 64
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_MODEL_PATH = "/home/lenguyen1/hoangpqn/models/llava-onevision-qwen2-7b-ov-hf"
DEFAULT_ROOT_DIR = Path(
    "/home/lenguyen1/hoangpqn/vln/InternNav/scripts/instruction_generator/keyframe_output/episodes"
)
DEFAULT_DEVICE = "cuda:1"

_KEYFRAME_FRAME_IDX_RE = re.compile(r"_(\d{6})\.(?:jpg|png)$", re.IGNORECASE)


def parse_keyframe_frame_idx(path: Path) -> int | None:
    match = _KEYFRAME_FRAME_IDX_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def collect_keyframe_paths(episode_dir: Path) -> list[tuple[int, Path]]:
    """Return (global_frame_idx, path) pairs sorted by frame index, deduplicated."""
    paths = list(episode_dir.glob("kf_*.jpg")) + list(episode_dir.glob("kf_*.png"))
    by_frame_idx: dict[int, Path] = {}
    skipped = 0

    for path in paths:
        frame_idx = parse_keyframe_frame_idx(path)
        if frame_idx is None:
            skipped += 1
            continue
        by_frame_idx[frame_idx] = path

    if skipped:
        print(f"  Warning: skipped {skipped} keyframe file(s) with unparseable names")

    if len(paths) > len(by_frame_idx):
        dropped = len(paths) - len(by_frame_idx) - skipped
        if dropped:
            print(
                f"  Warning: deduplicated {dropped} duplicate keyframe(s) "
                f"({len(paths)} files -> {len(by_frame_idx)} unique frames)"
            )

    return [(i, by_frame_idx[i]) for i in sorted(by_frame_idx)]


def split_keyframes_into_chunks(
    keyframes: list[tuple[int, Path]],
    window_size: int,
    overlap: int,
) -> list[tuple[int, list[tuple[int, Path]]]]:
    """Split keyframes into overlapping windows.

    Returns list of (start_index, chunk_keyframes). Step size is window_size - overlap.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= window_size:
        raise ValueError("overlap must be < window_size")

    if len(keyframes) <= window_size:
        return [(0, keyframes)]

    step = window_size - overlap
    chunks: list[tuple[int, list[tuple[int, Path]]]] = []
    start = 0
    while start < len(keyframes):
        chunk = keyframes[start : start + window_size]
        chunks.append((start, chunk))
        if start + len(chunk) >= len(keyframes):
            break
        start += step
    return chunks


def extract_chunk_video(
    episode_dir: Path,
    episode_start_global: int,
    chunk_keyframes: list[tuple[int, Path]],
    dst_clip: Path,
) -> dict | None:
    """Extract dense rgb.mp4 segment from first to last keyframe in chunk."""
    rgb_path = episode_dir / "rgb.mp4"
    if not rgb_path.exists():
        print(f"  Error: missing {rgb_path}")
        return None

    chunk_start_global = chunk_keyframes[0][0]
    chunk_end_global = chunk_keyframes[-1][0]
    local_start = chunk_start_global - episode_start_global
    local_end = chunk_end_global - episode_start_global

    cap = cv2.VideoCapture(str(rgb_path))
    if not cap.isOpened():
        print(f"  Error: cannot open {rgb_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    dst_clip.parent.mkdir(parents=True, exist_ok=True)
    frame_count = write_rgb_mp4_segment(
        rgb_path,
        dst_clip,
        local_start,
        local_end,
        fps,
        (width, height),
    )
    if frame_count == 0:
        print(f"  Warning: no frames written for clip {dst_clip.name}")
        return None

    return {
        "global_frame_range": f"{chunk_start_global}-{chunk_end_global}",
        "local_frame_range": f"{local_start}-{local_end}",
        "frame_count": frame_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate navigation instructions from dense video chunks "
            "using LLaVA-OneVision."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Episode directory or parent directory containing episode_* subdirs. "
            "If omitted, all episodes under --root-dir are processed."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help=f"Base directory for relative episode paths (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Local path to LLaVA-OneVision model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Number of keyframes per chunk (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=(
            "Shared keyframes between consecutive chunks "
            f"(default: {DEFAULT_CHUNK_OVERLAP}; must be < window-size)"
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_VIDEO_FRAMES,
        help=(
            "Frames sampled from each chunk video for LLaVA "
            f"(default: {DEFAULT_NUM_VIDEO_FRAMES})"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum tokens to generate per chunk (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help=f"Repetition penalty during generation (default: {DEFAULT_REPETITION_PENALTY})",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"CUDA device for inference (default: {DEFAULT_DEVICE})",
    )
    return parser.parse_args()


class LlavaOnevisionLocal:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        device: str = DEFAULT_DEVICE,
    ):
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

        self.processor = LlavaOnevisionProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            torch_dtype=torch.float16,
            device_map=device,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, video_path: Path, num_frames: int) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRAJECTORY_PROMPT_BEFORE},
                    {"type": "video", "path": str(video_path)},
                    {"type": "text", "text": TRAJECTORY_PROMPT_AFTER},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            num_frames=num_frames,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=self.repetition_penalty,
            use_cache=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        response = self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()

        return response


def run_episode(
    episode_dir: Path,
    llava: LlavaOnevisionLocal,
    window_size: int,
    chunk_overlap: int,
    num_frames: int,
):
    print(f"\n{'=' * 80}")
    print(f"EPISODE: {episode_dir.name}")

    chunk_clips_dir = episode_dir / "_chunk_clips"
    if chunk_clips_dir.exists():
        shutil.rmtree(chunk_clips_dir)

    for stale in (episode_dir / "instructions.json", episode_dir / "instructions.txt"):
        if stale.exists():
            stale.unlink()

    if not (episode_dir / "rgb.mp4").exists():
        print("No rgb.mp4 found, skipping.")
        return

    keyframes = collect_keyframe_paths(episode_dir)

    if len(keyframes) == 0:
        print("No keyframe images found, skipping.")
        return

    episode_start_global = keyframes[0][0]
    print(
        f"Found {len(keyframes)} keyframes, window_size={window_size}, "
        f"chunk_overlap={chunk_overlap}, num_frames={num_frames}"
    )

    chunks = split_keyframes_into_chunks(keyframes, window_size, chunk_overlap)
    print(f"Total chunks: {len(chunks)}")

    results = []

    for chunk_idx, (kf_start, chunk_keyframes) in enumerate(chunks):
        kf_end = kf_start + len(chunk_keyframes) - 1
        clip_path = chunk_clips_dir / f"chunk_{chunk_idx:04d}.mp4"
        print(
            f"\n[{chunk_idx + 1}/{len(chunks)}] keyframes {kf_start}–{kf_end} "
            f"({len(chunk_keyframes)} keyframes)"
        )

        clip_meta = extract_chunk_video(
            episode_dir,
            episode_start_global,
            chunk_keyframes,
            clip_path,
        )
        if clip_meta is None:
            print("  Skipping chunk (clip extraction failed).")
            continue

        print(
            f"  Clip: global frames {clip_meta['global_frame_range']} "
            f"({clip_meta['frame_count']} dense frames) -> {clip_path.name}"
        )

        try:
            instruction = llava.generate(video_path=clip_path, num_frames=num_frames)
            print(f"  → {instruction}")

            results.append(
                {
                    "chunk_idx": chunk_idx,
                    "keyframe_range": f"{kf_start}-{kf_end}",
                    "chunk_overlap": chunk_overlap,
                    "global_frame_range": clip_meta["global_frame_range"],
                    "local_frame_range": clip_meta["local_frame_range"],
                    "dense_frame_count": clip_meta["frame_count"],
                    "keyframes": [p.name for _, p in chunk_keyframes],
                    "video_clip": f"_chunk_clips/{clip_path.name}",
                    "num_frames": num_frames,
                    "instruction": instruction,
                }
            )

        except Exception as e:
            print(f"  FAILED: {e}")

        torch.cuda.empty_cache()
        gc.collect()

    if not results:
        print("No instructions generated.")
        return

    save_json = episode_dir / "instructions.json"
    with open(save_json, "w") as f:
        json.dump(results, f, indent=2)

    save_txt = episode_dir / "instructions.txt"
    with open(save_txt, "w") as f:
        for item in results:
            f.write(
                f"[chunk_{item['chunk_idx']:04d} keyframes {item['keyframe_range']} "
                f"global {item['global_frame_range']}] {item['instruction']}\n"
            )

    print(f"\nSaved {len(results)} chunk instructions:")
    print(f"  {save_json}")
    print(f"  {save_txt}")


def resolve_input_path(input_arg: str | None, root_dir: Path) -> Path | None:
    if input_arg is None:
        return None

    input_path = Path(input_arg)
    if not input_path.is_absolute():
        input_path = root_dir / input_path
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory")
        sys.exit(1)
    return input_path


def run_episodes(
    input_path: Path | None,
    llava: LlavaOnevisionLocal,
    window_size: int,
    chunk_overlap: int,
    num_frames: int,
    root_dir: Path,
):
    if input_path is None:
        episode_dirs = sorted(
            [
                x
                for x in root_dir.iterdir()
                if x.is_dir() and x.name.startswith("episode_")
            ]
        )
        print(f"Found {len(episode_dirs)} episodes in {root_dir}")
    else:
        episode_dirs = sorted(
            [
                x
                for x in input_path.iterdir()
                if x.is_dir() and x.name.startswith("episode_")
            ]
        )
        if episode_dirs:
            print(f"Found {len(episode_dirs)} episodes in {input_path}")
        else:
            run_episode(
                input_path,
                llava,
                window_size=window_size,
                chunk_overlap=chunk_overlap,
                num_frames=num_frames,
            )
            return

    for ep in episode_dirs:
        try:
            run_episode(
                ep,
                llava,
                window_size=window_size,
                chunk_overlap=chunk_overlap,
                num_frames=num_frames,
            )
        except Exception as e:
            print(f"Episode failed: {ep.name} — {e}")


if __name__ == "__main__":
    args = parse_args()

    if args.chunk_overlap >= args.window_size:
        print(
            f"Error: --chunk-overlap ({args.chunk_overlap}) must be < "
            f"--window-size ({args.window_size})"
        )
        sys.exit(1)

    llava = LlavaOnevisionLocal(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
    )

    input_path = resolve_input_path(args.input, args.root_dir)
    run_episodes(
        input_path,
        llava,
        window_size=args.window_size,
        chunk_overlap=args.chunk_overlap,
        num_frames=args.num_frames,
        root_dir=args.root_dir,
    )
