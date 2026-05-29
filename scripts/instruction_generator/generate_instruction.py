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
TEMPERATURE = 0.0
SUBCLIPS_PER_INSTRUCTION = 2
FRAMES_PER_SUBCLIP = 4

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
            attn_implementation="flash_attention_2"
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
        response = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()

        return response


def sample_frames(frame_paths, max_frames):
    if len(frame_paths) <= max_frames:
        return frame_paths
    indices = torch.linspace(0, len(frame_paths) - 1, steps=max_frames).long()

    return [frame_paths[i] for i in indices]


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


def run_episode(episode_dir: Path, llava: LlavaOnevisionLocal):

    print(f"\n{'='*80}")
    print(f"EPISODE: {episode_dir.name}")

    subclip_dirs = sorted([x for x in episode_dir.iterdir() if x.is_dir() and x.name.startswith("subclip_")])

    if len(subclip_dirs) == 0:
        print("No subclips found.")
        return

    print(f"Found {len(subclip_dirs)} subclips")

    instructions = []

    grouped_subclips = [
        subclip_dirs[i : i + SUBCLIPS_PER_INSTRUCTION] for i in range(0, len(subclip_dirs), SUBCLIPS_PER_INSTRUCTION)
    ]

    for group_idx, group in enumerate(grouped_subclips):

        print(f"\n[{group_idx+1}/{len(grouped_subclips)}] " f"Processing subclips:")

        all_images = []

        group_metadata = []

        for subclip_dir in group:

            print(f"  - {subclip_dir.name}")

            frame_paths = sorted(subclip_dir.glob("*.jpg"))

            if len(frame_paths) == 0:
                frame_paths = sorted(subclip_dir.glob("*.png"))

            sampled_paths = sample_frames(frame_paths, FRAMES_PER_SUBCLIP)

            images = load_images(sampled_paths)

            all_images.extend(images)

            meta_path = subclip_dir / "metadata.json"

            if meta_path.exists():
                with open(meta_path, "r") as f:
                    group_metadata.append(json.load(f))

        if len(all_images) == 0:
            print("  No valid images.")
            continue

        try:

            instruction = llava.generate(images=all_images)

            print(f"\nInstruction:")
            print(instruction)

            result = {
                "group_idx": group_idx,
                "subclips": [x.name for x in group],
                "instruction": instruction,
                "metadata": group_metadata,
            }

            instructions.append(result)

        except Exception as e:

            print(f"FAILED: {e}")

        torch.cuda.empty_cache()
        gc.collect()

    save_json = episode_dir / "instructions.json"

    with open(save_json, "w") as f:
        json.dump(instructions, f, indent=2)

    save_txt = episode_dir / "instructions.txt"

    with open(save_txt, "w") as f:

        for item in instructions:

            f.write(f"[{item['group_idx']:04d}] " f"{item['instruction']}\n")

    print(f"\nSaved:")
    print(f"  {save_json}")
    print(f"  {save_txt}")


if __name__ == "__main__":
    llava = LlavaOnevisionLocal(model_path="/home/lenguyen1/hoangpqn/models/llava-onevision-qwen2-7b-ov-hf")
    if len(sys.argv) > 1:
        ep = Path(sys.argv[1])
        if not ep.is_absolute():
            ep = ROOT_DIR / ep
        if not ep.is_dir():
            print(f"Error: {ep} is not a directory")
            sys.exit(1)

        run_episode(ep, llava)
    else:

        episode_dirs = sorted([x for x in ROOT_DIR.iterdir() if x.is_dir() and x.name.startswith("episode_")])

        print(f"Found {len(episode_dirs)} episodes")

        for ep in episode_dirs:

            try:
                run_episode(ep, llava)

            except Exception as e:
                print(f"Episode failed: {ep.name} — {e}")
