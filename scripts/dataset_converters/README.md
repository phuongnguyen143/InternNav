# LeRobot / InternVLA-N1 dataset conversion

Convert `keyframe_output_*` episodes into **InternVLA-N1 System2** LeRobot format (same parquet schema as reference scene `GdvgFV5R1Z5`).

## Pipeline overview

```
precompute_floor_trajectory.py     →  floor_calibration.json + floor_trajectory.txt
keyframe_extractor.py (live bag)   →  poses.json + episode rgb.mp4
rosbag2lerobot.py                  →  LeRobot parquet + 640×480 frame JPGs
visualize_pixel_goals.py           →  debug overlays (optional)
```

Upstream details: [`../instruction_generator/README.md`](../instruction_generator/README.md).

---

## Files

| File | Role |
|------|------|
| `rosbag2lerobot.py` | Main converter: frames, parquet labels, meta JSON |
| `internvla_labels.py` | Discrete actions, camera extrinsics per setting, **pixel goals** |
| `visualize_pixel_goals.py` | Overlay projected goals + optional floor-path debug |

---

## Quick start

```bash
conda activate internnav
cd vln/InternNav

# 1) After keyframe extraction + floor precompute (see instruction_generator README)
python scripts/dataset_converters/rosbag2lerobot.py \
  --keyframe_root scripts/instruction_generator/keyframe_output_round2_bkhn \
  --lerobot_out scripts/dataset_converters/lerobot_data_1 \
  --scene_id round2_bkhn \
  --goal_lookahead 200 \
  --overwrite

# 2) Debug pixel goals
python scripts/dataset_converters/visualize_pixel_goals.py \
  --keyframe_root scripts/instruction_generator/keyframe_output_round2_bkhn \
  --lerobot_root scripts/dataset_converters/lerobot_data_1/round2_bkhn \
  --episode_index 0 \
  --lookahead 200 \
  --draw_path \
  --frame_stride 40 \
  --out_dir scripts/dataset_converters/goal_debug_ep0
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
| `relative_goal_frame_id.{h}cm_{p}deg` | Fixed lookahead | e.g. `200`, or `-1` if goal invalid |

Only **`goal.125cm_30deg`** is populated (primary goal setting for `bkhn_125cm_0_30` training). Other settings get `[-1,-1]`.

---

## Pixel goal computation (`internvla_labels.py`)

### Target 3D point

For frame `i`, take the **on-floor world point** at frame `w = i + lookahead`:

1. Prefer `poses[w].world_x/y/z` from `poses.json` (written by keyframe extractor from `floor_trajectory.txt`).
2. Else `floor_xy_to_world_on_plane(poses[w].x, poses[w].y, floor_plane)`.

This is the precomputed embodiment path on the floor plane — **not** raw camera odometry.

### Projection (matches `GaussTrace/image_projector.py`)

1. `T_world_cam_optical = camera_matrix @ R_body2optical.T`
2. `p_cam = inv(T_world_cam_optical) @ [world_x, world_y, world_z, 1]`
3. Pinhole: `u = fx·X/Z + cx`, `v = fy·Y/Z + cy` (Z-forward optical frame)
4. Valid if `Z > 0` and `(u,v)` inside image bounds

Uses **`poses[i].camera_matrix`** (camera SLAM odom at frame `i`).

### Lookahead

**Fixed** frame offset (default **200**). No adaptive “first visible point” search.

- Valid → `relative_goal_frame_id = lookahead`, goal = `(u, v)`
- Off-screen / behind camera / past episode end → `goal = [-1,-1]`, `relative_goal_frame_id = -1`

Tune with `--goal_lookahead`.

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
| `--scene_id` | `round2_bkhn` | Scene folder name |
| `--fps` | `30` | Metadata FPS (see note below) |
| `--goal_lookahead` | `200` | Fixed frames ahead for pixel goals |
| `--height` | `125` | Robot camera height (cm) for goal setting |
| `--pitch_lookdown` | `30` | Pitch (deg) for primary goal setting |
| `--camera_intrinsic` | scaled RealSense K | Native or pre-scaled 3×3 |
| `--floor_calibration` | `<keyframe_root>/floor_calibration.json` | Floor plane |
| `--overwrite` | off | Replace existing scene output |

### RGB / depth folders

Exported for training config `bkhn_125cm_0_30`:

- `observation.images.rgb.125cm_0deg` — from horizon video
- `observation.images.rgb.125cm_30deg` — **duplicate** of 0° frames (format compatibility)
- `observation.images.depth.125cm_30deg` — from depth video

Pixel goals use the **real** `camera_matrix` at each frame (matches the exported RGB).

---

## `visualize_pixel_goals.py`

| Overlay | Meaning |
|---------|---------|
| **Red cross** | Projected goal (`internvla_labels`) |
| **Green tilted X** | Parquet goal (if `--lerobot_root` set) |
| **Yellow arrow** | Bottom-center (robot foot) → goal |
| **Green line** (`--draw_path`) | Projected floor trajectory segment |
| **White dot** | Image center reference only (not the goal) |

Outputs: per-frame JPGs, `contact_sheet.jpg`, `floor_trajectory.png`, `manifest.json`.

---

## Training

Point `NavPixelGoalDataset` at the converted scene. Primary setting: **`125cm_30deg`**.

Training resizes images again inside the vision processor; parquet goals remain in **640×480 JPEG coordinates** (same convention as InternData-N1 reference).

---

## Known limitations

1. **`info.json` fps** is set to 30; rosbag sync is often ~15 Hz. Consider fixing metadata or downsampling frames in a future pass.
2. **End of episode / sharp turns**: fixed lookahead 200 may project off-screen → many `[-1,-1]` goals near boundaries.
3. **Re-run converter** after changing `internvla_labels.py` or `poses.json`; parquet is not updated automatically.
4. **`poses.json` must include** `world_x/y/z` and `camera_matrix` (re-run keyframe `finalize()` after precompute).

---

## Code review checklist

- [ ] `floor_calibration.json` present in `keyframe_root`
- [ ] `poses.json` has `world_x`, `world_y`, `world_z`, `camera_matrix` per frame
- [ ] `floor_trajectory.txt` in keyframe root (or GaussTrace copy) matches precompute
- [ ] `rosbag2lerobot.py --goal_lookahead` matches training expectation
- [ ] `visualize_pixel_goals.py`: red cross aligns with green path on `--draw_path`
- [ ] Parquet `goal.125cm_30deg` matches recomputed projection after `--overwrite`
