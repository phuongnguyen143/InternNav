# Odometry and floor trajectory processing

Offline tools that turn raw camera / LiDAR odometry into **floor embodiment trajectories** used by the keyframe extractor and LeRobot converter.

Downstream: [`../instruction_generator/README.md`](../instruction_generator/README.md) (keyframes + `poses.json`) → [`../dataset_converters/README.md`](../dataset_converters/README.md) (InternVLA-N1 parquet).

---

## Pipeline overview

```
                    ┌─────────────────────────────────────────┐
                    │  Raw inputs (per scene)                 │
                    │  • camera_odom txt (4×4 T_world_cam)    │
                    │  • optional: scene PCD (LiDAR / COLMAP) │
                    └─────────────────┬───────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
  precompute_floor_          project_slam_path.py          convert_poses_to_
  trajectory.py              --export-floor-trajectory     odom_droidw.py
  (PCD + camera odom)        (odom-only, no PCD)           (WildGS → odom txt)
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
                                      ▼
                    floor_calibration.json + floor_trajectory.txt
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  Optional: derive_floor_calibration │
                    │  (PCD-free, from trajectory only) │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    keyframe_extractor finalize → poses.json
```

| Output | Used for |
|--------|----------|
| `floor_trajectory.txt` | Floor-frame `(x, y, yaw)` + on-plane `world_x/y/z` → actions, pixel goals |
| `floor_calibration.json` | Floor plane equation; auto-derived when missing at convert time |

---

## Two odometry backends

Scene YAML under `scripts/utils/configs/scenes/` sets `odom_apply_body2optical` and default paths.

| Scene type | Example | Odometry source | Floor estimation | `odom_apply_body2optical` |
|------------|---------|-------------------|------------------|---------------------------|
| **LiDAR / BKHN** | `bkhn_round1` | `odometry_*_point2plane.txt` (ROS body frame) | PCD patch fit (`precompute_floor_trajectory.py`) | `true` |
| **Visual SLAM** | `office_round1` | DROID-W `odometry_camera.txt` (OpenCV optical c2w) | Leveled-camera floor contact (`project_slam_path.py --export-floor-trajectory`) | `false` |

**Camera pose** (from `camera_odom` txt) and **floor embodiment** (from `floor_trajectory.txt`) are intentionally separate:

- `camera_matrix` in `poses.json` → LeRobot camera extrinsics, pixel-goal projection camera
- `x, y, yaw` on the floor → discrete navigation actions, keyframe spacing

---

## Setup

```bash
conda activate internnav
cd vln/InternNav/scripts/process_odom
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
```

`precompute_floor_trajectory.py` requires **Open3D** (`pip install open3d`).

Set `VLN_DATA_ROOT` if your data is not under the default in `utils/configs/base.yaml` (`/home/lenguyen1/hoangpqn/vln/DATA`).

---

## Scripts

| File | Role |
|------|------|
| `extract_rgb_from_bag.sh` | Extract compressed RGB (+ synced depth) from ROS2 rosbag2 → `tmp/rgb_frames` + `frames.json` |
| `run_visual_odom_pipeline.sh` | One-shot: extract → DROID-W → `odometry_camera.txt` → `floor_trajectory.txt` |
| `precompute_floor_trajectory.py` | Estimate floor plane from PCD; project camera odom to floor trajectory |
| `project_slam_path.py` | Export floor trajectory from SLAM odom (no PCD); optional RGB path overlay |
| `derive_floor_calibration.py` | Rebuild `floor_calibration.json` from existing `floor_trajectory.txt` |
| `convert_poses_to_odom_droidw.py` | WildGS-SLAM `est_poses_full.txt` → timestamped 4×4 odom txt |
| `mapping/` | Vendored SLAM stacks (DROID-W, gluemap, vr_mapper / COLMAP) — see their own READMEs |

Shared logic lives in `scripts/utils/` (`floor_pose`, `trajectory_io`, `image_projector`, …).

---

## Quick start

### One-shot automation (RealSense / DROID-W path)

`run_visual_odom_pipeline.sh` chains extract, SLAM, odom convert, and floor export. Keyframes stay separate (`instruction_generator/run_extract_keyframe.sh`).

```bash
cd vln/InternNav/scripts/process_odom
./run_visual_odom_pipeline.sh \
  --bag /home/lenguyen1/Downloads/realsense_20260718_164645 \
  --extract-stride 3 \
  --droid-stride 1 \
  --write-scene-yaml \
  --gpu 0
```

| Flag | Role |
|------|------|
| `--extract-stride` | Fewer JPGs at bag extract (e.g. `3`) |
| `--droid-stride` | DROID-W + `convert_poses` stride (use `1` if extract already strided) |
| `--skip-extract` / `--skip-slam` / … | Resume when outputs exist |
| `--write-scene-yaml` | Writes `utils/configs/scenes/<scene-id>.yaml` for keyframes |
| `--droid-config` | Skip auto-generated DROID yaml |

BKHN / PCD floor (`precompute_floor_trajectory.py`) and live keyframe tmux are **not** included in this script.

### RealSense rosbag → DROID-W (WildGS-SLAM)

End-to-end flow for bags like `realsense_20260718_164645` (`/front/camera/...` topics). Uses **two conda envs**: `internnav` for bag extract, `droid-w` for SLAM.

#### 1. Extract RGB (+ depth) from rosbag2

The wrapper sources ROS Humble and calls `scripts/instruction_generator/extract_bag_frames.py`.

```bash
conda activate internnav   # OpenCV + scripts/utils; rosbag2_py from ROS
source /opt/ros/humble/setup.bash   # optional if extract_rgb_from_bag.sh sources it

cd vln/InternNav/scripts/process_odom
./extract_rgb_from_bag.sh \
  --bag /home/lenguyen1/Downloads/realsense_20260718_164645 \
  --stride 3
```

| Flag | Meaning |
|------|---------|
| `--bag` | Rosbag2 directory (`metadata.yaml` + `.db3` / `.mcap`) |
| `--output-dir` | Default: `extract_out/<bag_basename>/` |
| `--stride N` | Save every **Nth** synchronized RGB/depth pair (default `1` = all) |
| `--no-trim` | Full bag (default); use `--trim` for 20s head/tail skip |

Example output (stride 3 on a ~5k-frame bag → ~1718 JPGs):

- `extract_out/realsense_20260718_164645/tmp/rgb_frames/frame_XXXXXX.jpg`
- `extract_out/realsense_20260718_164645/tmp/depth_frames/frame_XXXXXX.png`
- `extract_out/realsense_20260718_164645/frames.json`

Run `./extract_rgb_from_bag.sh --help` for topics, storage, sync-slop, etc.

**Stride rule (do not double-subsample):**

| Where you subsample | DROID-W config `stride` |
|---------------------|-------------------------|
| Extract `--stride 3` | Set **`stride: 1`** |
| Extract default (all frames) | Use e.g. **`stride: 3`** in YAML |

#### 2. Point DROID-W at the extract folder

Edit or copy `mapping/DROID-W/configs/custom_realsense_20260718_164645.yaml`:

```yaml
data:
  input_folder: /path/to/process_odom/extract_out/realsense_20260718_164645/tmp/rgb_frames
  output: ./output/realsense_20260718_164645

stride: 1   # if extract used --stride 3
cam:
  H: 480
  W: 640
  # RealSense intrinsics …
```

#### 3. Run WildGS-SLAM

`run.py` must run in the **`droid-w`** env (not `internnav`). From `mapping/DROID-W`:

```bash
conda activate droid-w
cd vln/InternNav/scripts/process_odom/mapping/DROID-W

python run.py --config ./configs/custom_realsense_20260718_164645.yaml
```

On success you should see `INFO: <N> images got!`, tracking to 100%, then outputs under:

`./output/realsense_20260718_164645/realsense_20260718_164645/` (e.g. `video.npz`, trajectory under `traj/`).

Use a **free GPU** (one job per GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --config ./configs/custom_realsense_20260718_164645.yaml
```

#### 4. After SLAM → floor / InternNav

Convert poses if needed, then export floor trajectory (see [Office / DROID-W SLAM](#office--droid-w-slam-no-pcd) below):

```bash
conda activate internnav
cd vln/InternNav/scripts/process_odom
python convert_poses_to_odom_droidw.py \
  --poses ./mapping/DROID-W/output/.../traj/est_poses_full.txt \
  --frames-json ./extract_out/realsense_.../frames.json \
  --output ./extract_out/realsense_.../odometry_camera.txt \
  --stride 1
```

Use the same effective stride as during SLAM (usually `1` if extract already used `--stride`).

#### Troubleshooting (extract + SLAM)

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: munch` on `run.py` | `conda activate droid-w` before SLAM |
| `can't open file .../DROID-W/scripts/instruction_generator/extract_bag_frames.py` | Run extract from `process_odom` via `./extract_rgb_from_bag.sh`, or use `scripts/instruction_generator/extract_bag_frames.py` under `InternNav/scripts/` |
| `INFO: 859 images` but extract wrote ~1718 files | DROID-W `stride: 2` (or higher) on an already strided folder — set **`stride: 1`** |
| `CUDA out of memory` during tracking | Free GPU (`nvidia-smi`); keep `tracking.buffer: 350` (avoid huge values like 1500); lower `tracking.frontend.max_factors` / `window`; increase extract or YAML stride |
| Extract needs ROS | `source /opt/ros/humble/setup.bash` or let the shell wrapper source it |

DROID-W install and dependencies: [`mapping/DROID-W/README.md`](mapping/DROID-W/README.md).

### Extract only (reference)

Minimal extract without SLAM:

```bash
conda activate internnav
cd vln/InternNav/scripts/process_odom
./extract_rgb_from_bag.sh --bag /path/to/rosbag2_dir
```

### BKHN / LiDAR (PCD-based floor)

```bash
python precompute_floor_trajectory.py --scene bkhn_round1
```

Writes to the parent of `paths.floor_trajectory` in the scene config (e.g. `${data_root}/raw_rosbag/bkhn_round1/`):

- `floor_calibration.json`
- `floor_trajectory.txt`

Copy or symlink both into the keyframe `output_dir` before running extraction.

### Office / DROID-W SLAM (no PCD)

After DROID-W produces `odometry_camera.txt`:

```bash
# 1) Export floor trajectory + calibration from camera odom
python project_slam_path.py --scene office_round1 --export-floor-trajectory

# 2) Optional: debug overlay of future path on extracted RGB frames
python project_slam_path.py --scene office_round1 --visualize --project-images
```

If WildGS-SLAM poses need conversion first:

```bash
python convert_poses_to_odom_droidw.py \
  --poses /path/to/est_poses_full.txt \
  --frames-json /path/to/frames.json \
  --output /path/to/odometry_camera.txt \
  --stride 3
```

### Derive calibration only (no PCD, trajectory already exists)

```bash
python derive_floor_calibration.py \
  --floor_trajectory /path/to/floor_trajectory.txt \
  --output_dir /path/to/output_dir
```

---

## `floor_trajectory.txt` format

Two lines per sample:

```text
<timestamp>
<x> <y> <yaw> <legacy_z> <world_x> <world_y> <world_z>
```

| Field | Meaning |
|-------|---------|
| `x, y, yaw` | Floor-frame embodiment (X forward, Y left; positive yaw = turn left) |
| `legacy_z` | World Z of projected base point (compatibility) |
| `world_x/y/z` | 3D point on the floor plane in map/SLAM frame (pixel-goal targets) |

Older 3- or 4-field rows still parse; `world_*` is recomputed from `(x,y)` + calibration when missing.

---

## `precompute_floor_trajectory.py`

Estimates a local floor plane by voxel-downsampling the scene PCD and fitting patches, then projects each camera odom pose onto that plane.

| Flag | Default | Description |
|------|---------|-------------|
| `--scene` | — | Scene name (`utils/configs/scenes/<scene>.yaml`) |
| `--pcd` | from `paths.pcd` | Input point cloud |
| `--camera_odom` | from `paths.camera_odom` | Timestamped 4×4 camera poses |
| `--output_dir` | parent of `paths.floor_trajectory` | Output directory |
| `--voxel_size` | `0.1` | PCD voxel size (m) |
| `--patch_radius` / `--stride` | auto by cloud size | Local plane patch search |
| `--smooth` | `moving_average` | `none`, `moving_average`, or `bspline` |
| `--smooth_window` | `5` | Odd window ≥ 3 |
| `--smooth_s` | `1.0` | B-spline smoothing factor |
| `--draw_smooth` | off | Plot raw vs smoothed trajectory |

Large clouds automatically use coarser patch stride. Smoothing metadata is stored in `floor_calibration.json` under `trajectory_smooth`.

```bash
# B-spline smoothing example
python precompute_floor_trajectory.py --scene bkhn_round1 \
  --smooth bspline --smooth_s 2.0

# Disable smoothing
python precompute_floor_trajectory.py --scene bkhn_round1 --smooth none
```

---

## `project_slam_path.py`

Two modes:

1. **Export** (`--export-floor-trajectory [DIR]`): leveled-camera floor contact (undo mount pitch, offset on optical +Y); writes `floor_trajectory.txt` with raw offset `world_x/y/z` + `floor_calibration.json` (plane fitted from trajectory for downstream only). No PCD required.
2. **Project** (default when RGB inputs exist): Draw future path on RGB from exported `world_x/y/z` (same points as pixel goals downstream).

| Flag | Default | Description |
|------|---------|-------------|
| `--scene` | — | Load paths and defaults from scene YAML |
| `--odom` | `paths.camera_odom` | Input odometry txt |
| `--export-floor-trajectory` | parent of `paths.floor_trajectory` when `--scene` set | Export directory |
| `--visualize` | off | Save `floor_trajectory.png` + project RGB if frames exist |
| `--project-images` | off | Force RGB projection |
| `--frames-json` / `--rgb-dir` | from scene paths | Frame index + timestamp + RGB frames |
| `--config` | `paths.droid_cfg` | DROID-W `cfg.yaml` for intrinsics |
| `--slam-intrinsics` | off | Scale K to SLAM crop resolution |
| `--body2optical` | scene default | Apply ROS body→OpenCV optical on load |
| `--smooth` | `moving_average` | Floor trajectory smoothing on export |
| `--floor-smooth-window` | `5` | Smoothing window (export) |
| `--camera-pitch-deg` | `30` | Mount pitch to undo before floor contact (deg) |
| `--ground-offset-y` | `1.5` | Offset along optical +Y after leveling (m) |
| `--lookahead-m` / `--lookahead-s` | `10` / `10` | Path draw lookahead |
| `--stride` | `1` | Frame subsampling for projection |
| `--max-frames` | all | Limit projected frames (`-1` = all) |

**Note:** SLAM export and pixel goals both use `floor_trajectory.txt` `world_x/y/z` (BKHN-style downstream). Re-export after changing `--camera-pitch-deg` or `--ground-offset-y`.

---

## `convert_poses_to_odom_droidw.py`

Converts WildGS-SLAM TUM-style poses to the GaussTrace / BKHN odometry block format:

```text
<timestamp>
r00 r01 r02 tx
r10 r11 r12 ty
r20 r21 r22 tz
0   0   0   1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--poses` | repo default | `est_poses_full.txt` (`frame_idx tx ty tz qx qy qz qw`) |
| `--frames-json` | repo default | Per-frame ROS timestamps from bag extraction |
| `--output` | repo default | Output odom txt path |
| `--stride` | `3` | Must match WildGS-SLAM config stride |
| `--no-interpolate` | off | Only strided keyframes (skip SLERP fill) |

By default, intermediate frames are filled with linear translation + quaternion SLERP between SLAM keyframes.

---

## `derive_floor_calibration.py`

Rebuilds `floor_calibration.json` from `world_x/y/z` columns in an existing `floor_trajectory.txt` when you have trajectory but no PCD-based calibration (e.g. after SLAM export).

| Flag | Description |
|------|-------------|
| `--floor_trajectory` | Input trajectory txt |
| `--output_dir` | Where to write calibration |
| `--overwrite` | Replace existing file |

`rosbag2lerobot.py` can also auto-derive calibration at convert time if only the trajectory is present.

---

## Scene configuration

Example scene files: `scripts/utils/configs/scenes/bkhn_round1.yaml`, `office_round1.yaml`.

```yaml
scene: office_round1
odom_apply_body2optical: false   # DROID optical c2w

paths:
  bag: ${data_root}/raw_rosbag/office_round1
  camera_odom: ${data_root}/process_SLAM/office_round1/office/traj/odometry_camera.txt
  floor_trajectory: ${data_root}/process_SLAM/office_round1/office/traj_nav_test/floor_trajectory.txt
  output_dir: ${data_root}/process_keyframe/office_round1
  frames_json: ${data_root}/raw_img_extract/office_round1/frames.json
  rgb_dir: ${data_root}/raw_img_extract/office_round1/tmp/rgb_frames
  droid_cfg: ${data_root}/process_SLAM/office_round1/office/cfg.yaml
```

Add a new scene by copying a template, setting `odom_apply_body2optical`, and pointing `paths.*` at your data layout.

---

## `mapping/` (SLAM tooling)

Vendored third-party repos used to produce camera odometry — not part of the Python floor pipeline itself:

| Directory | Purpose |
|-----------|---------|
| `mapping/DROID-W/` | Visual SLAM → `odometry_camera.txt`, `cfg.yaml` |
| `mapping/gluemap/` | Structure-from-motion / mapping |
| `mapping/vr_mapper/` | COLMAP + glomap offline mapper |

Refer to each subdirectory's README for build and run instructions.

---

## Checklist before keyframe extraction

- [ ] `floor_trajectory.txt` exists in keyframe `output_dir` (or parent bag/SLAM dir, copied in)
- [ ] `floor_calibration.json` present (or derivable from trajectory)
- [ ] `camera_odom` txt timestamps align with the rosbag / `frames.json`
- [ ] Scene YAML `odom_apply_body2optical` matches how odom was recorded
- [ ] For BKHN: ran `precompute_floor_trajectory.py` on the scene PCD
- [ ] For office: ran `project_slam_path.py --export-floor-trajectory` after DROID-W

After keyframes: convert with [`../dataset_converters/README.md`](../dataset_converters/README.md).

---

## Rerun visualization

Visualize the **estimated floor plane**, **scene point cloud**, and **trajectories** with the [Rerun SDK](https://www.rerun.io/docs):

```bash
conda activate internnav
cd vln/InternNav/scripts
export PYTHONPATH="$(pwd):$PYTHONPATH"

# After precompute_floor_trajectory.py (BKHN / PCD scenes)
python process_odom/visualize_rerun.py --scene bkhn_round1

# Explicit paths
python process_odom/visualize_rerun.py \
  --floor-calibration DATA/raw_rosbag/bkhn_round1/floor_calibration.json \
  --pcd DATA/raw_rosbag/bkhn_round1/bkhn_round1_point2plane.pcd \
  --floor-trajectory DATA/raw_rosbag/bkhn_round1/floor_trajectory.txt \
  --camera-odom DATA/raw_rosbag/bkhn_round1/odometry_bkhn_round1_point2plane.txt

# Save .rrd (open later: rerun /path/to/recording.rrd)
python process_odom/visualize_rerun.py --scene bkhn_round1 --save /tmp/bkhn_floor.rrd
```

**Layers in the viewer:**
- `world/map` — downsampled point cloud (green ≈ near estimated plane)
- `world/floor_plane` — semi-transparent fitted plane quad
- `world/floor_path` — embodiment trajectory on the plane
- `world/camera_path` — camera odometry polyline
- `world/camera_axes` — RGB optical axes at subsampled poses (X red, Y green, Z yellow/forward)
- `world/camera_frustums` — wireframe camera frustums (OpenCV RDF, +Z forward)

Scrub a single moving camera with frustum: `--mode timeline --stride 5`

Tune pose density: `--camera-stride 30` (static) or `--camera-axis-len 0.5 --camera-frustum-depth 0.8`

Tune density with `--pcd-voxel 0.2` and `--pcd-max-points 250000`.

### LeRobot final dataset (RGB + pose + pixel goal)

Visualize exported parquet + per-frame RGB JPGs (e.g. `DATA/final/office_round1_ver2.0`):

```bash
python process_odom/visualize_rerun.py \
  --lerobot-root /home/lenguyen1/hoangpqn/vln/DATA/final/office_round1_ver2.0 \
  --setting 125cm_30deg \
  --episode 0 \
  --mode timeline \
  --stride 5
```

**Layers:**
- `world/episode_XXXXXX/camera_path` — 3D camera centres from `pose.{setting}`
- `world/episode_XXXXXX/floor_path` — ground-contact trajectory (SLAM trick from poses)
- `world/episode_XXXXXX/camera` — moving camera rig (timeline) with Pinhole + RGB
- `.../camera/rgb/goal` — pixel goal (red dot + Points2D)

Use the same `--setting` for `pose.*`, `goal.*`, and `observation.images.rgb.{setting}` folders. If goals are all `[-1,-1]`, the script prints which settings have valid goals.

```bash
# All episodes, save recording
python process_odom/visualize_rerun.py \
  --lerobot-root /path/to/final/office_round1_ver2.0 \
  --setting 125cm_30deg \
  --stride 10 \
  --save /tmp/office_lerobot.rrd
```

Local Rerun examples: `vln/rerun/examples/python/` (minimal, rgbd, lidar, ros_node).
