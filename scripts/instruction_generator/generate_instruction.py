import argparse
import gc
import json
import re
import sys
import torch

from pathlib import Path
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

from prompts import TRAJECTORY_PROMPT_BEFORE, TRAJECTORY_PROMPT_AFTER

DEFAULT_MAX_NEW_TOKENS = 96
DEFAULT_WINDOW_SIZE = 6
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


def collect_keyframe_paths(episode_dir: Path) -> list[Path]:
    """Return keyframe images sorted by global frame index, deduplicated."""
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

    return [by_frame_idx[i] for i in sorted(by_frame_idx)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate navigation instructions from keyframe images using LLaVA-OneVision.",
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
        help=f"Number of keyframe images per chunk (default: {DEFAULT_WINDOW_SIZE})",
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
    def generate(self, images) -> str:
        conversation = [
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": TRAJECTORY_PROMPT_BEFORE}]
                    + [{"type": "image"} for _ in images]
                    + [{"type": "text", "text": TRAJECTORY_PROMPT_AFTER}]
                ),
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=images, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

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


def load_images(image_paths):
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((384, 384))
            images.append(img)
        except Exception as e:
            print(f"Failed loading {p}: {e}")
    return images


def run_episode(episode_dir: Path, llava: LlavaOnevisionLocal, window_size: int):
    print(f"\n{'=' * 80}")
    print(f"EPISODE: {episode_dir.name}")

    for stale in (episode_dir / "instructions.json", episode_dir / "instructions.txt"):
        if stale.exists():
            stale.unlink()

    frame_paths = collect_keyframe_paths(episode_dir)

    if len(frame_paths) == 0:
        print("No keyframe images found, skipping.")
        return

    print(f"Found {len(frame_paths)} keyframe images, window_size={window_size}")

    # chunk into non-overlapping windows: [0:8], [8:16], [16:24], ...
    chunks = [
        frame_paths[i : i + window_size]
        for i in range(0, len(frame_paths), window_size)
    ]
    print(f"Total chunks: {len(chunks)}")

    results = []

    for chunk_idx, chunk_paths in enumerate(chunks):
        start = chunk_idx * window_size
        end = start + len(chunk_paths) - 1
        print(
            f"\n[{chunk_idx + 1}/{len(chunks)}] frames {start}–{end} ({len(chunk_paths)} images)"
        )

        images = load_images(chunk_paths)
        if len(images) == 0:
            print("  No valid images, skipping chunk.")
            continue

        try:
            instruction = llava.generate(images=images)
            print(f"  → {instruction}")

            results.append(
                {
                    "chunk_idx": chunk_idx,
                    "frame_range": f"{start}-{end}",
                    "frames": [p.name for p in chunk_paths],
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

    # save JSON with all chunk results
    save_json = episode_dir / "instructions.json"
    with open(save_json, "w") as f:
        json.dump(results, f, indent=2)

    # save TXT with one instruction per line, ready for summarization
    save_txt = episode_dir / "instructions.txt"
    with open(save_txt, "w") as f:
        for item in results:
            f.write(
                f"[chunk_{item['chunk_idx']:04d} frames {item['frame_range']}] {item['instruction']}\n"
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
            run_episode(input_path, llava, window_size=window_size)
            return

    for ep in episode_dirs:
        try:
            run_episode(ep, llava, window_size=window_size)
        except Exception as e:
            print(f"Episode failed: {ep.name} — {e}")


if __name__ == "__main__":
    args = parse_args()

    llava = LlavaOnevisionLocal(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
    )

    input_path = resolve_input_path(args.input, args.root_dir)
    run_episodes(
        input_path, llava, window_size=args.window_size, root_dir=args.root_dir
    )
