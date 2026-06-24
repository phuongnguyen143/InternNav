# LeRobot / InternVLA-N1 dataset conversion

Convert `keyframe_output_*` episodes into **InternVLA-N1 System2** LeRobot format (same parquet schema as reference scene `GdvgFV5R1Z5`).

**Prerequisites:** floor trajectory + keyframe extraction. See [`../process_odom/README.md`](../process_odom/README.md) and [`../instruction_generator/README.md`](../instruction_generator/README.md).

---

## End-to-end pipeline

```
process_odom/                          instruction_generator/
  precompute_floor_trajectory.py    →      keyframe_extractor.py
  or project_slam_path.py --export         (live bag or offline frames)
       │                                        │
       ▼                                        ▼
  floor_calibration.json              poses.json + episodes/
  floor_trajectory.txt                  rgb.mp4 + depth_frames/*.png
       │                                        │
       └────────────────┬───────────────────────┘
                        ▼
              dataset_converters/
                rosbag2lerobot.py  →  LeRobot parquet + 640×480 RGB/depth
                visualize_pixel_goals.py   (optional debug)
                draw_parquet_goals.py      (optional debug)
                decode_depth_image.py      (inspect depth PNGs)
```

---

## Files

| File | Role |
|------|------|
| `rosbag2lerobot.py` | Main converter: frames, parquet labels, meta JSON |
| `internvla_labels.py` | Discrete actions, camera extrinsics per setting, pixel goals |
| `decode_depth_image.py` | CLI to inspect LeRobot / keyframe depth PNGs |
| `visualize_pixel_goals.py` | Overlay projected goals + optional floor-path debug |
| `draw_parquet_goals.py` | Draw stored parquet goal pixels on exported RGB JPGs |
| `../utils/depth_codec.py` | Shared RealSense `compressedDepth` decode + uint16 mm PNG I/O |

---

## Quick start

```bash
conda activate internnav
cd vln/InternNav
export PYTHONPATH="scripts:scripts/dataset_converters/lerobot/src:$PYTHONPATH"

# 1) After keyframe extraction + floor precompute
python scripts/dataset_converters/rosbag2lerobot.py \
  --keyframe_root scripts/instruction_generator/keyframe_output_round2_bkhn \
  --lerobot_out scripts/dataset_converters/lerobot_data_1 \
  --scene_id round2_bkhn \
  --scene bkhn_round1 \
  --goal_lookahead 200 \
  --overwrite

# 2) Debug pixel goals (re-projection)
python scripts/dataset_converters/visualize_pixel_goals.py \
  --keyframe_root scripts/instruction_generator/keyframe_output_round2_bkhn \
  --lerobot_root scripts/dataset_converters/lerobot_data_1/round2_bkhn \
  --episode_index 0 \
  --lookahead 200 \
  --draw_path \
  --frame_stride 40 \
  --out_dir scripts/dataset_converters/goal_debug_ep0

# 3) Debug stored parquet goals (no re-projection)
python scripts/dataset_converters/draw_parquet_goals.py \
  --lerobot_root scripts/dataset_converters/lerobot_data_1/round2_bkhn \
  --episode_index 0 \
  --setting 125cm_30deg \
  --out_dir scripts/dataset_converters/goal_debug_ep0_parquet
```

---

## Output layout

```
{lerobot_out}/{scene_id}/
  meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl
  data/chunk-*/episode_*.parquet
  videos/chunk-*/observation.images.rgb.{h}cm_{p}deg/episode_{ep}_{frame}.jpg
  videos/chunk-*/observation.images.depth.{h}cm_{p}deg/episode_{ep}_{frame}.png
```

---

## Parquet columns (per frame)

| Column | Source | Notes |
|--------|--------|-------|
| `action` | Floor `(x,y,yaw)` trajectory | Habitat-style: `-1` start, `1` forward, `2` left, `3` right |
| `pose.{h}cm_{p}deg` | `action_matrix` + synthetic mount | 4×4 world camera extrinsic per setting |
| `goal.{h}cm_{p}deg` | Floor path projection | Pixel `(u,v)` in **640×480**; `[-1,-1]` if invalid |
| `relative_goal_frame_id.{h}cm_{p}deg` | Lookahead search | e.g. `200`, or `-1` if goal invalid |

Only **`goal.125cm_30deg`** is populated (primary goal setting for `bkhn_125cm_0_30` training). Other settings get `[-1,-1]`.

---

## Pixel goal computation (`internvla_labels.py`)

### Target 3D point

For frame `i`, take the on-floor world point at frame `w = i + lookahead`:

1. Prefer `poses[w].world_x/y/z` from `poses.json` (written by keyframe extractor from `floor_trajectory.txt`).
2. Else `floor_xy_to_world_on_plane(poses[w].x, poses[w].y, floor_plane)`.

This is the precomputed embodiment path on the floor plane — **not** raw camera odometry.

### Projection

For each frame `i`, project a lookahead ground contact point into image `i`:

1. **Target 3D** at frame `i + lookahead`:
   - **SLAM / office** (`odom_apply_body2optical: false`, `goal_use_ground_contact: true`): `ground_world_from_camera_c2w(poses[w].camera_matrix)` — pitch+offset on optical c2w (same as `project_slam_path.py`).
   - **LiDAR / BKHN** (`odom_apply_body2optical: true`): `poses[w].world_x/y/z` from `floor_trajectory.txt` on the estimated floor plane. If off-screen, falls back to optical ground contact.
2. **Camera** at frame `i`: `poses[i].camera_matrix` (SLAM optical c2w; skip `R_body2optical` when `odom_apply_body2optical: false`).
3. `p_cam = inv(T_world_cam) @ target`, pinhole `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`.
4. Valid if `Z > 0` and `(u,v)` inside 640×480.

Intrinsics: use scene `paths.droid_cfg` when `--scene` is set; otherwise default RealSense K from `rosbag2lerobot.py`.

### Lookahead

**Max** frame offset (default **200**). With **adaptive** mode (default), the converter searches offsets from `min(lookahead, episode_end - i)` down to **1** and picks the **farthest visible** ground projection into image `i`.

- Valid → `relative_goal_frame_id` = chosen offset (≤ lookahead), goal = `(u, v)`
- No visible point in window / past episode start → `goal = [-1,-1]`, `relative_goal_frame_id = -1`

Tune with `--goal_lookahead`. Use `--no-goal-lookahead-adaptive` for strict fixed-offset behavior.

---

## Image resize and intrinsics

| Stage | Resolution |
|-------|------------|
| Source `rgb.mp4` (keyframe extractor) | 1280×720 |
| Saved LeRobot JPGs | **640×480** (`cv2.resize`) |

Goals are computed in **640×480 pixel space** by scaling camera intrinsics (not by rescaling pixels after projection):

```text
K_native  from RealSense bag (1280×720)
K_scaled  = scale(K_native, (1280,720) → (640,480))
project with K_scaled, bounds 640×480
```

Default `K_native` is hardcoded in `rosbag2lerobot.py`. Override:

```bash
--camera_intrinsic path/to/K.json   # 3×3 JSON
# or
--camera_intrinsic "fx,0,cx,0,fy,cy,0,0,1"
```

---

## Discrete actions

Between consecutive floor poses `(x,y,yaw)`:

| Value | Meaning |
|-------|---------|
| `-1` | Episode / segment start |
| `1` | Move forward (~0.25 m) |
| `2` | Turn left (~15°) |
| `3` | Turn right (~15°) |

Dense rosbag frames: sub-steps carry forward the last discrete action when motion is below threshold.

---

## `rosbag2lerobot.py` CLI

| Flag | Default | Description |
|------|---------|-------------|
| `--keyframe_root` | `./keyframe_output` | Dir with `poses.json`, `episodes/`, `floor_calibration.json` |
| `--lerobot_out` | `./lerobot_data` | Output parent directory |
| `--scene_id` | `round2_bkhn` | Scene folder name under `lerobot_out` |
| `--scene` | — | Scene config (e.g. `office_round1`); reads `odom_apply_body2optical`, `goal_use_ground_contact` |
| `--fps` | `30` | Metadata FPS (see note below) |
| `--goal_lookahead` | `200` | Max frames ahead for pixel goals |
| `--no-goal-lookahead-adaptive` | off | Disable adaptive search; require exact offset |
| `--height` | `125` | Robot camera height (cm) for goal setting |
| `--pitch_lookdown` | `30` | Pitch (deg) for primary goal setting |
| `--camera_intrinsic` | scaled RealSense K | Native or pre-scaled 3×3 |
| `--floor_calibration` | `<keyframe_root>/floor_calibration.json` | Floor plane (auto-derived from `floor_trajectory.txt` if missing) |
| `--body2optical` | off | Force ROS body→OpenCV optical on `camera_matrix` (LiDAR odom) |
| `--no-body2optical` | off | Skip transform (SLAM/DROID optical odom) |
| `--overwrite` | off | Replace existing scene output |

### RGB / depth folders

Exported for training config `bkhn_125cm_0_30`:

- `observation.images.rgb.125cm_0deg` — from horizon video
- `observation.images.rgb.125cm_30deg` — **duplicate** of 0° frames (format compatibility)
- `observation.images.depth.125cm_30deg` — from `episodes/*/depth_frames/` (uint16 **millimeters**, same as sim `GdvgFV5R1Z5`)

Depth pipeline (RealSense `compressedDepth` → train):

1. `keyframe_extractor.py` decodes ROS payload to meters, saves `depth_frames/frame_XXXXXX.png` as uint16 mm
2. `rosbag2lerobot.py` resizes depth with **nearest-neighbor** to 640×480 and writes LeRobot PNGs
3. Training divides by `1000` in `preprocess_depth_image_v2` (meters)

Legacy `depth.mp4` (8-bit preview) is **not** accepted — re-run keyframe extraction after updating.

Inspect depth:

```bash
python scripts/dataset_converters/decode_depth_image.py \
  scripts/dataset_converters/lerobot_data_1/round2_bkhn/videos/chunk-000/observation.images.depth.125cm_30deg/episode_000000_0.png
```

Pixel goals use the real `camera_matrix` at each frame (matches the exported RGB).

---

## `visualize_pixel_goals.py`

Recomputes goals from `poses.json` and overlays them on frames.

| Overlay | Meaning |
|---------|---------|
| **Red cross** | Projected goal (`internvla_labels`) |
| **Green tilted X** | Parquet goal (if `--lerobot_root` set) |
| **Yellow arrow** | Bottom-center (robot foot) → goal |
| **Green line** (`--draw_path`) | Projected floor trajectory segment |
| **White dot** | Image center reference only (not the goal) |

Outputs: per-frame JPGs, `contact_sheet.jpg`, `floor_trajectory.png`, `manifest.json`.

---

## `draw_parquet_goals.py`

Lightweight debug: reads `goal.{setting}` from parquet and draws a **red dot** at `(u, v)` on the exported LeRobot RGB JPG. No re-projection.

```bash
python scripts/dataset_converters/draw_parquet_goals.py \
  --lerobot_root /path/to/final/office_round1 \
  --episode_index 1 \
  --setting 125cm_30deg \
  --out_dir /tmp/goal_debug
```

| Overlay | Meaning |
|---------|---------|
| **Red dot** | Parquet `goal.{setting}` at pixel `(u, v)` exactly as stored |

Use `--all_frames` to annotate every frame, or `--all_episodes` for the whole scene.

---

## Training

Point `NavPixelGoalDataset` at the converted scene. Primary setting: **`125cm_30deg`**.

Training resizes images again inside the vision processor; parquet goals remain in **640×480 JPEG coordinates** (same convention as InternData-N1 reference).

---

## Known limitations

1. **`info.json` fps** is set to 30; rosbag sync is often ~15 Hz. Consider fixing metadata or downsampling frames in a future pass.
2. **End of episode / sharp turns**: fixed lookahead 200 may project off-screen → many `[-1,-1]` goals near boundaries.
3. **Re-run converter** after changing `internvla_labels.py` or `poses.json`; parquet is not updated automatically. This includes the ROS→Habitat body rotation applied to `pose.*` columns — existing parquet must be regenerated with `--overwrite`.
4. **`poses.json` must include** `world_x/y/z` and `camera_matrix` (re-run keyframe `finalize()` after precompute).

---

## Code review checklist

- [ ] `floor_calibration.json` present in `keyframe_root` (or `floor_trajectory.txt` with world_x/y/z for auto-derive)
- [ ] `poses.json` has `world_x`, `world_y`, `world_z`, `camera_matrix` per frame
- [ ] `floor_trajectory.txt` in keyframe root matches precompute output
- [ ] Scene YAML `odom_apply_body2optical` matches odometry source ([`process_odom/README.md`](../process_odom/README.md))
- [ ] `rosbag2lerobot.py --goal_lookahead` matches training expectation
- [ ] `visualize_pixel_goals.py`: red cross aligns with green path on `--draw_path`
- [ ] Parquet `goal.125cm_30deg` matches recomputed projection after `--overwrite`
