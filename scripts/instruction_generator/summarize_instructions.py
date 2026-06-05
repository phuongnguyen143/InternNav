import gc
import sys
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import SUMMARIZE_PROMPT

QWEN_MODEL_PATH = "/home/lenguyen1/hoangpqn/models/Qwen2-72B-Instruct-AWQ"


class QwenLocal:
    def __init__(self, model_path: str = QWEN_MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        print("Model loaded")

    @torch.inference_mode()
    def summarize(self, instructions: list[str]) -> str:
        numbered = "\n".join(f"{i+1}. {x}" for i, x in enumerate(instructions))
        prompt = SUMMARIZE_PROMPT.format(instructions=numbered)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def process_txt(txt_path: Path, qwen: QwenLocal):
    """Summarize a single instructions.txt and save summary.txt next to it."""
    instructions = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]

    if not instructions:
        print(f"  [SKIP] No instructions found in {txt_path}")
        return

    print(f"  Loaded {len(instructions)} instructions")
    for i, inst in enumerate(instructions):
        print(f"    {i+1}. {inst}")

    summary = qwen.summarize(instructions)

    print(f"  Summary: {summary}")

    save_path = txt_path.parent / "summary.txt"
    save_path.write_text(summary)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("  # single txt file")
        print("  python summarize_instructions.py /path/to/instructions.txt")
        print("  # single episode folder")
        print("  python summarize_instructions.py /path/to/episode_0001/")
        print("  # folder containing many episode_ subdirs")
        print("  python summarize_instructions.py /path/to/episodes/")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: path not found: {input_path}")
        sys.exit(1)

    txt_files = []

    if input_path.is_file():
        txt_files.append(input_path)

    elif input_path.is_dir():
        episode_dirs = sorted([x for x in input_path.iterdir() if x.is_dir() and x.name.startswith("episode_")])

        if episode_dirs:
            for ep in episode_dirs:
                txt = ep / "instructions.txt"
                if txt.exists():
                    txt_files.append(txt)
                else:
                    print(f"  [WARN] No instructions.txt in {ep.name}, skipping.")
        else:
            txt = input_path / "instructions.txt"
            if txt.exists():
                txt_files.append(txt)
            else:
                print(f"Error: no instructions.txt found in {input_path}")
                sys.exit(1)

    if not txt_files:
        print("Error: no instructions.txt files found.")
        sys.exit(1)

    print(f"Found {len(txt_files)} instruction file(s) to summarize")

    qwen = QwenLocal()

    for txt_path in txt_files:
        print(f"\n{'='*60}")
        print(f"Episode: {txt_path.parent.name}")
        try:
            process_txt(txt_path, qwen)
        except Exception as e:
            print(f"  FAILED: {e}")

    del qwen
    torch.cuda.empty_cache()
    gc.collect()
