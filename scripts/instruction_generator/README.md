# Navigation Instruction Generation Pipeline

A pipeline for automatically generating natural language navigation instructions from egocentric video captured on a mobile robot in indoor environments (school/university buildings).

---

## Overview

The pipeline consists of three stages:

```
Keyframe Extraction  →  Subclip Division  →  Instruction Generation  →  Summarization
```

1. **Keyframe Extraction** — detects and saves key navigation moments from real-time RGB and odometry streams using turn and distance thresholds
2. **Subclip Division** — groups keyframes into episodes of 10 and splits the raw frames between each keyframe pair into structured subclips
3. **Instruction Generation** — generates per-subclip navigation instructions using LLaVA by interleaving image frames inside the prompt
4. **Summarization** — consolidates all subclip instructions into a single long-horizon instruction using Qwen2-72B

---

## Repository Structure

```
scripts/
├── utils/                        # Shared library (config, trajectory_io, floor_pose, …)
│   └── configs/scenes/office_round1.yaml
├── instruction_generator/
│   ├── frame_utils.py            # RGB/depth decode, mp4 helpers
│   ├── keyframe_extractor.py
│   ├── run_extract_keyframe.sh   # ./run_extract_keyframe.sh office_round1
│   └── …
└── process_odom/
    ├── precompute_floor_trajectory.py
    ├── project_slam_path.py
    └── mapping/

keyframe output (example: DATA/process_keyframe/office_round1/)
├── all_frames/                 # All raw frames saved during recording
├── keyframes/                  # Labelled keyframe images
├── keyframes.json              # Keyframe metadata (position, yaw, timestamp)
├── floor_calibration.json      # Floor plane (copy from precompute output_dir)
├── poses.json                  # Floor pose + world_x/y/z + camera_matrix + action_matrix
├── floor_trajectory.txt        # Copy or symlink from precompute (used at finalize)
├── trajectory.png              # Top-down trajectory visualization
└── episodes/
    ├── episodes.json           # Episode and subclip metadata
    └── episode_XXXX/
        ├── rgb.mp4             # Episode RGB video (horizon camera)
        ├── depth_frames/       # uint16 mm PNGs: frame_XXXXXX.png (used by rosbag2lerobot)
        ├── subclip_00/         # Raw frames between keyframe 0 and 1
        ├── subclip_01/         # Raw frames between keyframe 1 and 2
        ├── ...
        ├── kf_*.jpg            # Keyframe images (up to ~11 per episode)
        ├── instructions.json   # Per-chunk instructions with metadata
        └── instructions.txt    # Per-chunk instructions from LLaVA (one line per chunk)
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

Camera odometry (`T_world_cam`) and embodiment trajectory on the floor are **different**:

| Data | Source | Used for |
|------|--------|----------|
| Camera pose | `odometry_*.txt` | `poses.json` → `camera_matrix` → LeRobot `pose.{h}cm_{p}deg` (OpenCV optical) |
| Floor embodiment | `floor_trajectory.txt` (precomputed) | `poses.json` x,y,yaw → keyframes; `action_matrix` → LeRobot `action` |

Floor estimation is **offline** (slow on large PCDs). Do not run it inside ROS nodes during bag play.

#### Step 0 — Precompute (once per scene)

```bash
cd scripts/process_odom
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"

# PCD-based floor (slow):
python precompute_floor_trajectory.py --scene office_round1

# Or odometry-only export (no PCD):
python project_slam_path.py --scene office_round1 --export-floor-trajectory
```

Writes `floor_calibration.json` and `floor_trajectory.txt`.

**`floor_trajectory.txt` format** (2 lines per sample):

```text
<timestamp>
<x> <y> <yaw> <legacy_z> <world_x> <world_y> <world_z>
```

- `x, y, yaw` — floor-frame embodiment pose (for actions / top-down plots)
- `legacy_z` — world-Z of projected base point (kept for compatibility)
- `world_x/y/z` — **3D point on the floor plane** in map/SLAM frame (used for pixel-goal projection)

Older 3- or 4-field rows still parse; `world_*` is recomputed from `(x,y)` + calibration when missing.

Large clouds use coarser patch stride automatically; override with `--stride` / `--patch_radius`.

**Trajectory smoothing** (applied before save; timestamps unchanged):

| Flag | Default | Description |
|------|---------|-------------|
| `--smooth` | `moving_average` | `none`, `moving_average`, or `bspline` |
| `--smooth_window` | `5` | Odd window for moving average (bspline fallback when N < 4) |
| `--smooth_s` | `1.0` | B-spline smoothing factor (`0` = interpolate, larger = smoother) |

```bash
# B-spline smoothing:
python precompute_floor_trajectory.py ... --smooth bspline --smooth_s 2.0

# Disable smoothing:
python precompute_floor_trajectory.py ... --smooth none
```

Smoothing params are recorded in `floor_calibration.json` under `trajectory_smooth`.

Requires `open3d` in the `internnav` conda environment.

#### Step 1 — Live bag + keyframes

```bash
cd scripts/instruction_generator
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
./run_extract_keyframe.sh office_round1
```

**Alternative — offline frame extraction (no live ROS playback):**

```bash
source /opt/ros/humble/setup.bash
conda activate internnav
cd scripts/instruction_generator

python extract_bag_frames.py /path/to/bkhn_round2 \
  --output-dir ./keyframe_output_offline \
  --storage-id mcap \
  --write-depth-preview
```

Writes `tmp/rgb_full.mp4`, `tmp/depth_frames/frame_*.png`, and `frames.json`.
Run keyframe finalize / pose merge separately, or continue with the live tmux flow below.

Live tmux flow:

- Pane 1: `trajectory_publishers.py floor` (loads txt only, no PCD at startup)
- Pane 2: `keyframe_extractor.py` merges camera odom and floor trajectory at `finalize()` into `poses.json`
- `camera_matrix` is stored as **OpenCV optical** `T_world_cam` with `"camera_frame": "optical"` (BKHN: body→optical at finalize; office: SLAM optical odom as-is)

**Migrate legacy poses.json** (body-frame `camera_matrix`):

```bash
python instruction_generator/normalize_poses_json.py \
  --scene bkhn_round1 \
  --poses-json DATA/process_keyframe/bkhn_round1 \
  --camera-odom DATA/raw_rosbag/bkhn_round1/odometry_bkhn_round1_point2plane.txt \
  --in-place
```
- Depth from `/camera/camera/aligned_depth_to_color/image_raw/compressedDepth` is decoded to meters and saved as `tmp/depth_frames/frame_XXXXXX.png` (uint16 mm). Episode export copies these into `episodes/*/depth_frames/`. A temporary `depth_full.mp4` preview may be written for debugging only.

Press `Ctrl+B` then `S` to stop and save.

Place `floor_trajectory.txt` and `floor_calibration.json` in `output_dir` before extraction (or copy from precompute `output_dir`). At finalize, each pose gets:

| Field | Source |
|-------|--------|
| `x, y, yaw, z` | Matched `floor_trajectory.txt` entry |
| `world_x, world_y, world_z` | On-floor 3D point from trajectory |
| `camera_matrix` | Matched camera odom → **OpenCV optical** `T_world_cam` (`camera_frame: optical`) |
| `action_matrix` | Base on floor plane from `(x,y,yaw)` |

#### LeRobot conversion (pose vs action vs pixel goals)

See [`../dataset_converters/README.md`](../dataset_converters/README.md).

```bash
python ../dataset_converters/rosbag2lerobot.py \
  --keyframe_root ./keyframe_output_round2_bkhn \
  --lerobot_out ../dataset_converters/lerobot_data_1 \
  --scene_id round2_bkhn \
  --goal_lookahead 200 \
  --overwrite
```

- `action` ← discrete steps from floor `(x,y,yaw)`
- `pose.*` ← OpenCV optical `T_world_cam` per height/pitch (measured odom at goal setting, else synthetic from `action_matrix`)
- `goal.125cm_30deg` ← project `world_x/y/z` at frame `i + lookahead` into image `i` (640×480)
- `relative_goal_frame_id.125cm_30deg` ← fixed lookahead (e.g. 200) or `-1`

**Tuning keyframe density** (edit `utils/configs/base.yaml` under `keyframe`, or override in a scene yaml):

| Parameter | Default | Effect |
|---|---|---|
| `keyframe.sharp_turn_thresh_deg` | 25.0 | Lower = more keyframes on turns |
| `keyframe.curvature_thresh_deg` | 25.0 | Lower = more keyframes on curves |
| `keyframe.max_dist_between_keyframes` | 6.0 m | Lower = denser keyframes |
| `keyframe.min_dist_between_keyframes` | 3.0 m | Higher = fewer keyframes |
| `keyframe.keyframes_per_episode` | 10 | Keyframes per episode (~11 with shared boundary) |

`merge_window_frames` remains in `keyframe_selection.py` (`KeyframeConfig`).

Re-running keyframe extraction clears stale `kf_*.jpg` and episode videos in each `episode_XXXX/` folder automatically.

---

### Stage 2+3 — Instruction Generation (LLaVA)

Generates one navigation instruction per chunk by extracting a **dense video clip** from `rgb.mp4` spanning the first-to-last keyframe in each window, then feeding that clip to LLaVA-OneVision video mode (uniform subsample of `--num-frames` frames).

Requires `rgb.mp4` in each episode folder (written by keyframe extraction).

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

**Example:** 11 keyframes with `--window-size 6 --chunk-overlap 2` → 3 overlapping chunks (step = 4):
- Chunk 0: keyframes 0–5, video global frames of kf0 → kf5
- Chunk 1: keyframes 4–9 (overlaps kf 4–5 with chunk 0)
- Chunk 2: keyframes 8–10 (overlaps kf 8–9 with chunk 1)

Set `--chunk-overlap 0` for non-overlapping chunks (legacy behavior).

Output per episode:
- `instructions.json` — metadata per chunk (keyframe range, global frame range, clip path, model output)
- `instructions.txt` — one instruction per chunk window
- `_chunk_clips/chunk_XXXX.mp4` — extracted dense clips (regenerated each run)

**Re-run after changing extraction settings:** run keyframe extraction first, then `generate_instruction.py`. Existing mixed keyframes from old runs are not fixed by instruction generation alone.

**Configuration** (top of `generate_instruction.py`):

| Parameter | Default | Description |
|---|---|---|
| `DEFAULT_MODEL_PATH` | local LLaVA path | Local path to LLaVA-OneVision model |
| `DEFAULT_WINDOW_SIZE` | `6` | Keyframes per chunk (defines clip span) |
| `DEFAULT_CHUNK_OVERLAP` | `2` | Shared keyframes between consecutive chunks |
| `DEFAULT_NUM_VIDEO_FRAMES` | `32` | Frames sampled from each clip for LLaVA |
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Max tokens in generated instruction |

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

**`instructions.txt`** — one instruction per chunk from LLaVA (`generate_instruction.py`):
```
[chunk_0000 keyframes 0-5 global 0-115] Turn left and walk forward past the desk...
[chunk_0001 keyframes 6-10 global 228-380] Continue straight down the corridor...
```

**`summary.txt`** — optional single long-horizon instruction from Qwen2-72B (`summarize_instructions.py`):
```
Walk straight ahead past the bulletin board, turn left at the glass door, continue down
the corridor past the exit sign, and stop at the reception desk in the elevator lobby.
```

---

## Known Limitations

- LLaVA shows bias toward predicting right-turn actions even when not clearly visible in frames
- Instruction quality depends on keyframe density and `--num-frames` — sparse keyframes with low `num_frames` reduces visual context
- Qwen2-72B requires ~40GB VRAM; both models cannot be loaded simultaneously on dual RTX 4090 (48GB total) — the pipeline unloads LLaVA before loading Qwen

---

## Hardware

Tested on:
- 2× NVIDIA GeForce RTX 4090 (24GB each)
- Ubuntu 22.04, ROS2 Humble
- Python 3.10, PyTorch 2.x, Transformers ≥ 4.51.3

