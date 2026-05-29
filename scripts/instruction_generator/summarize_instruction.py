import gc
import sys
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
# ============================================================
# CONFIG
# ============================================================

QWEN_MODEL_PATH = "/home/lenguyen1/hoangpqn/models/Qwen2-72B-Instruct-AWQ"

SUMMARIZE_PROMPT_TEMPLATE = """You are a navigation instruction summarizer.

Below are sequential navigation instructions describing parts of a single trajectory:
{instructions}

Summarize all of them into ONE fluent, long-horizon navigation instruction.
- Cover the full path from start to end
- Mention key landmarks and turns in order
- Natural language, one sentence or two at most
- No bullet points, no numbering

Output only the final instruction, nothing else.
"""

# ============================================================
# QWEN MODEL
# ============================================================

class QwenLocal:

    def __init__(self, model_path: str = QWEN_MODEL_PATH):
        print(f"[Qwen] Loading from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            # ),
            device_map="cuda:0",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        print("[Qwen] Loaded.")

    @torch.inference_mode()
    def summarize(self, instructions: list[str]) -> str:
        numbered = "\n".join(f"{i+1}. {x}" for i, x in enumerate(instructions))
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(instructions=numbered)

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
        return self.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python summarize_instructions.py <path/to/instructions_raw.txt>")
        print("       python summarize_instructions.py <path/to/episode_folder/>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # Accept either a .txt file or an episode folder
    if input_path.is_dir():
        txt_path = input_path / "instructions_raw.txt"
    else:
        txt_path = input_path

    if not txt_path.exists():
        print(f"Error: file not found: {txt_path}")
        sys.exit(1)

    # Read instructions (one per line, skip blank lines)
    instructions = [
        line.strip()
        for line in txt_path.read_text().splitlines()
        if line.strip()
    ]

    if not instructions:
        print("Error: no instructions found in file.")
        sys.exit(1)

    print(f"Loaded {len(instructions)} instructions from: {txt_path}")
    for i, inst in enumerate(instructions):
        print(f"  {i+1}. {inst}")

    # Summarize
    qwen = QwenLocal()
    summary = qwen.summarize(instructions)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  {summary}")
    print('='*60)

    # Save next to the input file
    save_path = txt_path.parent / "instructions.txt"
    save_path.write_text(summary)
    print(f"\nSaved: {save_path}")

    # Cleanup
    del qwen
    torch.cuda.empty_cache()
    gc.collect()