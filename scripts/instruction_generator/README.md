# Navigation Instruction Generation Pipeline

A pipeline for automatically generating natural language navigation instructions from egocentric video captured on a mobile robot in indoor environments (school/university buildings).

---

## Overview

The pipeline consists of three stages:

```
Keyframe Extraction  →  Subclip Division  →  Instruction Generation  →  Summarization
```

1. **Keyframe Extraction** — detects and saves key navigation moments from real-time RGB and odometry streams using turn and distance thresholds
2. **Subclip Division** — groups keyframes into episodes of 30 and splits the raw frames between each keyframe pair into structured subclips
3. **Instruction Generation** — generates per-subclip navigation instructions using LLaVA by interleaving image frames inside the prompt
4. **Summarization** — consolidates all subclip instructions into a single long-horizon instruction using Qwen2-72B

---

## Repository Structure

```
scripts/instruction_generator/
├── keyframe_extractor.py       # Stage 1 — ROS2 node for keyframe extraction
├── generate_instruction.py     # Stage 2+3 — Subclip instruction generation (LLaVA)
├── summarize_instructions.py   # Stage 4 — Long-horizon summarization (Qwen2-72B)
└── sample_frames.py            # Utility — downsample frames in a folder

keyframe_output/
├── all_frames/                 # All raw frames saved during recording
├── keyframes/                  # Labelled keyframe images
├── keyframes.json              # Keyframe metadata (position, yaw, timestamp)
├── trajectory.png              # Top-down trajectory visualization
└── episodes/
    ├── episodes.json           # Episode and subclip metadata
    └── episode_XXXX/
        ├── subclip_00/         # Raw frames between keyframe 0 and 1
        ├── subclip_01/         # Raw frames between keyframe 1 and 2
        ├── ...
        ├── instructions.json   # Per-group instructions with metadata
        ├── instructions_raw.txt  # Raw per-subclip instructions from LLaVA
        └── instructions.txt    # Final summarized long-horizon instruction
```

---

## Requirements

### ROS2
```
ROS2 Humble or later
sensor_msgs
geometry_msgs
```

### Python
```bash
pip install torch transformers accelerate bitsandbytes pillow opencv-python matplotlib
```

### Models

| Model | Purpose | Size |
|---|---|---|
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | Visual instruction generation | ~7B (4-bit: ~5GB) |
| `Qwen/Qwen2-72B-Instruct-AWQ` | Instruction summarization | ~72B (AWQ: ~38GB) |

Download models locally before running:
```bash
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --local-dir /path/to/models/llava-onevision-qwen2-7b-ov-hf

huggingface-cli download Qwen/Qwen2-72B-Instruct-AWQ \
    --local-dir /path/to/models/Qwen2-72B-Instruct-AWQ
```

---

## Usage

### Stage 1 — Keyframe Extraction

Launch the ROS2 node while the robot is navigating:

```bash
ros2 bag play bkhn_round2 --rate 3
python depth_republish.py
ros2 run image_transport republish compressed raw \
  --ros-args \
  --remap in/compressed:=/camera/camera/color/image_raw/compressed \
  --remap out:=/camera/camera/color/image_raw/raw
python odometry_publisher.py /home/lenguyen1/hoangpqn/GaussTrace/dataset/raw/scenes/BKHN_data/bkhn_round1/odometry_bkhn_round2_point2plane.txt
python extract_keyframe.py
```

Press `Ctrl+C` to stop recording. The node will automatically run `finalize()` to extract keyframes and build episodes.

**Tuning keyframe density** (edit `KeyframeConfig` in `keyframe_extractor.py`):

| Parameter | Default | Effect |
|---|---|---|
| `sharp_turn_thresh_deg` | 25.0 | Lower = more keyframes on turns |
| `curvature_thresh_deg` | 25.0 | Lower = more keyframes on curves |
| `max_dist_between_keyframes` | 5.0 m | Lower = denser keyframes |
| `min_dist_between_keyframes` | 2.0 m | Higher = fewer keyframes |
| `keyframes_per_episode` | 30 | Keyframes per episode |

---

### Stage 2+3 — Instruction Generation (LLaVA)

Generates one navigation instruction per subclip group by interleaving image frames inside the prompt.

**Single episode:**
```bash
python generate_instruction.py episode_0000
# or full path
python generate_instruction.py /path/to/episodes/episode_0000
```

**All episodes:**
```bash
python generate_instruction.py
```

Output per episode:
- `instructions.json` — full metadata with subclip names, frame counts, and raw model output
- `instructions_raw.txt` — one instruction per line, used as input to the summarizer

**Configuration** (top of `generate_instruction.py`):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/path/to/llava-...` | Local path to LLaVA model |
| `SUBCLIPS_PER_INSTRUCTION` | `2` | Subclips grouped per instruction |
| `FRAMES_PER_SUBCLIP` | `4` | Max frames sampled per subclip |
| `MAX_NEW_TOKENS` | `96` | Max tokens in generated instruction |

---

### Stage 4 — Summarization (Qwen2-72B)

Reads all subclip instructions from a single episode and summarizes them into one fluent long-horizon instruction.

**Single episode (pass folder):**
```bash
python summarize_instructions.py /path/to/episodes/episode_0000/
```

**Single episode (pass file directly):**
```bash
python summarize_instructions.py /path/to/episodes/episode_0000/instructions_raw.txt
```

**Batch — run on all episodes:**
```bash
for ep in /path/to/episodes/*/; do
    python summarize_instructions.py "$ep"
done
```

Output: overwrites `instructions.txt` in the same folder with the final summarized instruction.

**Prompt used:**
```
Summarize all of them into ONE fluent, long-horizon navigation instruction.
- Cover the full path from start to end
- Mention key landmarks and turns in order
- Natural language, one sentence or two at most
```

**Configuration** (top of `summarize_instructions.py`):

| Parameter | Default | Description |
|---|---|---|
| `QWEN_MODEL_PATH` | `/path/to/Qwen2-72B...` | Local path to Qwen2-72B-AWQ model |
| `max_new_tokens` | `128` | Max tokens in summarized instruction |

---

## Output Format

Each episode produces:

**`instructions_raw.txt`** — one instruction per subclip group from LLaVA:
```
[0000] Walk straight ahead, passing the bulletin board on your left, and stop at the staircase.
[0001] Turn left at the glass door and continue down the corridor toward the exit sign.
[0002] Walk straight ahead toward the elevator lobby and stop at the reception desk.
```

**`instructions.txt`** — single long-horizon instruction from Qwen2-72B:
```
Walk straight ahead past the bulletin board, turn left at the glass door, continue down
the corridor past the exit sign, and stop at the reception desk in the elevator lobby.
```

---

## Known Limitations

- LLaVA shows bias toward predicting right-turn actions even when not clearly visible in frames
- Instruction quality depends on keyframe density — too few frames per subclip reduces visual context
- Qwen2-72B requires ~40GB VRAM; both models cannot be loaded simultaneously on dual RTX 4090 (48GB total) — the pipeline unloads LLaVA before loading Qwen

---

## Hardware

Tested on:
- 2× NVIDIA GeForce RTX 4090 (24GB each)
- Ubuntu 22.04, ROS2 Humble
- Python 3.10, PyTorch 2.x, Transformers ≥ 4.51.3