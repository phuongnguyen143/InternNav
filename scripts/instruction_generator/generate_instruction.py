import gc
import json
import sys
import torch
import os

from pathlib import Path
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

from prompt.trajectory_prompt_v1 import TRAJECTORY_PROMPT_BEFORE, TRAJECTORY_PROMPT_AFTER

BASEPATH = os.path.dirname(os.path.abspath(__file__))
print(BASEPATH)

MAX_NEW_TOKENS = 96
WINDOW_SIZE = 8  # how many frames per chunk

ROOT_DIR = Path("/home/lenguyen1/hoangpqn/vln/InternNav/scripts/instruction_generator/keyframe_output/episodes")


class LlavaOnevisionLocal:
    def __init__(self, model_path: str = "/home/lenguyen1/hoangpqn/models/llava-onevision-qwen2-7b-ov-hf"):

        self.processor = LlavaOnevisionProcessor.from_pretrained(model_path, local_files_only=True, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            torch_dtype=torch.float16,
            device_map="cuda:0",
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
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
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


def run_episode(episode_dir: Path, llava: LlavaOnevisionLocal, window_size: int = WINDOW_SIZE):
    print(f"\n{'='*80}")
    print(f"EPISODE: {episode_dir.name}")

    frame_paths = sorted(episode_dir.glob("kf_*.jpg"))
    if len(frame_paths) == 0:
        frame_paths = sorted(episode_dir.glob("kf_*.png"))

    if len(frame_paths) == 0:
        print("No keyframe images found, skipping.")
        return

    print(f"Found {len(frame_paths)} keyframe images, window_size={window_size}")

    # chunk into non-overlapping windows: [0:6], [6:12], [12:18], ...
    chunks = [frame_paths[i: i + window_size] for i in range(0, len(frame_paths), window_size)]
    print(f"Total chunks: {len(chunks)}")

    results = []

    for chunk_idx, chunk_paths in enumerate(chunks):
        start = chunk_idx * window_size
        end = start + len(chunk_paths) - 1
        print(f"\n[{chunk_idx + 1}/{len(chunks)}] frames {start}–{end} ({len(chunk_paths)} images)")

        images = load_images(chunk_paths)
        if len(images) == 0:
            print("  No valid images, skipping chunk.")
            continue

        try:
            instruction = llava.generate(images=images)
            print(f"  → {instruction}")

            results.append({
                "chunk_idx": chunk_idx,
                "frame_range": f"{start}-{end}",
                "frames": [p.name for p in chunk_paths],
                "instruction": instruction,
            })

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
            f.write(f"[chunk_{item['chunk_idx']:04d} frames {item['frame_range']}] {item['instruction']}\n")

    print(f"\nSaved {len(results)} chunk instructions:")
    print(f"  {save_json}")
    print(f"  {save_txt}")


if __name__ == "__main__":
    llava = LlavaOnevisionLocal(
        model_path="/home/lenguyen1/hoangpqn/models/llava-onevision-qwen2-7b-ov-hf"
    )

    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else WINDOW_SIZE

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if not input_path.is_absolute():
            input_path = ROOT_DIR / input_path
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            sys.exit(1)

        # if the given dir contains episode_ subdirs → run all of them
        # if the given dir IS an episode (has kf_* images) → run it directly
        episode_dirs = sorted([x for x in input_path.iterdir() if x.is_dir() and x.name.startswith("episode_")])

        if episode_dirs:
            print(f"Found {len(episode_dirs)} episodes in {input_path}")
            for ep in episode_dirs:
                try:
                    run_episode(ep, llava, window_size=window_size)
                except Exception as e:
                    print(f"Episode failed: {ep.name} — {e}")
        else:
            # treat the path itself as a single episode
            run_episode(input_path, llava, window_size=window_size)

    else:
        episode_dirs = sorted(
            [x for x in ROOT_DIR.iterdir() if x.is_dir() and x.name.startswith("episode_")]
        )
        print(f"Found {len(episode_dirs)} episodes in ROOT_DIR")
        for ep in episode_dirs:
            try:
                run_episode(ep, llava, window_size=window_size)
            except Exception as e:
                print(f"Episode failed: {ep.name} — {e}")