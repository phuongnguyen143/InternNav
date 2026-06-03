from pathlib import Path
import shutil
import numpy as np
import cv2
from dataclasses import dataclass, field

@dataclass
class KeyframeConfig:
    sharp_turn_thresh_deg: float = 25.0
    curvature_thresh_deg: float = 25.0
    curvature_window: int = 10
    max_dist_between_keyframes: float = 6.0
    min_dist_between_keyframes: float = 3.0
    merge_window_frames: int = 10

@dataclass
class KeyframeResult:
    frame_idx: int
    reason: str
    pose: dict

    delta_yaw_deg: float = 0.0
    accumulated_yaw_deg: float = 0.0
    dist_from_last: float = 0.0


def split_keyframes_into_episodes(keyframe_dir, output_dir=None, images_per_episode=40):
    """
    Split thousands of keyframe images into episodes.
    """
    keyframe_dir = Path(keyframe_dir)
    if output_dir is None:
        output_dir = keyframe_dir.parent / "episodes"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in keyframe_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    total_images = len(image_paths)

    print(f"Found {total_images} images")
    if total_images == 0:
        print("No images found")
        return

    num_episodes = 0
    for start_idx in range(0, total_images, images_per_episode):
        end_idx = min(start_idx + images_per_episode, total_images) # ensure we don't go out of bounds
        episode_images = image_paths[start_idx:end_idx]
        episode_dir = output_dir / (f"episode_{num_episodes:04d}")
        episode_dir.mkdir(parents=True, exist_ok=True)
        for local_idx, src_path in enumerate(episode_images):
            dst_path = episode_dir / (f"{local_idx:06d}{src_path.suffix.lower()}")
            shutil.copy2(src_path, dst_path)
        print(f"[Episode {num_episodes:04d}] {len(episode_images)} images")
        num_episodes += 1

def normalize_angle(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def delta_yaw_deg(a, b):
    return np.degrees(normalize_angle(b - a))


def euclidean_dist(a, b):
    return np.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2)


def merge_close_keyframes(keyframes, window):
    if len(keyframes) <= 2:
        return keyframes

    priority = {
        'start': 0,
        'end': 0,
        'sharp_turn': 1,
        'curvature': 2,
        'distance': 3,
    }

    merged = [keyframes[0]]
    for kf in keyframes[1:]:
        last = merged[-1]
        if (kf.frame_idx - last.frame_idx) <= window:
            if priority.get(kf.reason, 99) < priority.get(last.reason, 99):
                merged[-1] = kf
        else:
            merged.append(kf)

    return merged

def get_frame_from_video(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def extract_keyframes(poses, config: KeyframeConfig):
    if len(poses) < 2:
        return []

    keyframes = []
    keyframes.append(KeyframeResult(frame_idx=poses[0]['frame_idx'], reason='start', pose=poses[0]))
    last_kf_pose = poses[0]

    # precompute per-frame delta yaws
    delta_yaws = [0.0]
    for i in range(1, len(poses)):
        dyaw = delta_yaw_deg(poses[i - 1]['yaw'], poses[i]['yaw'])
        delta_yaws.append(dyaw)

    for i in range(1, len(poses) - 1):
        pose = poses[i]
        dist = euclidean_dist(pose, last_kf_pose)

        if dist < config.min_dist_between_keyframes:
            continue

        reasons = []
        abs_delta = abs(delta_yaws[i])

        # sharp turn
        if abs_delta >= config.sharp_turn_thresh_deg:
            reasons.append(('sharp_turn', abs_delta))

        # curvature over a local window
        window_start = max(0, i - config.curvature_window)
        accum = sum(abs(delta_yaws[j]) for j in range(window_start, i + 1))
        if accum >= config.curvature_thresh_deg:
            reasons.append(('curvature', accum))

        # max distance
        if dist >= config.max_dist_between_keyframes:
            reasons.append(('distance', dist))

        if reasons:
            priority_order = ['sharp_turn', 'curvature', 'distance']
            best_reason = next(
                (p for p in priority_order for r, _ in reasons if r == p),
                reasons[0][0],
            )

            kf = KeyframeResult(
                frame_idx=pose['frame_idx'],
                reason=best_reason,
                pose=pose,
                delta_yaw_deg=delta_yaws[i],
                accumulated_yaw_deg=accum,
                dist_from_last=dist,
            )
            keyframes.append(kf)
            last_kf_pose = pose
            print(
                f"[KF] frame={pose['frame_idx']} reason={best_reason} "
                f"dist={dist:.2f} dyaw={delta_yaws[i]:.1f} accum={accum:.1f}"
            )

    keyframes.append(KeyframeResult(frame_idx=poses[-1]['frame_idx'], reason='end', pose=poses[-1]))
    keyframes = merge_close_keyframes(keyframes, config.merge_window_frames)
    print(f"[KF] Total keyframes after merge: {len(keyframes)}")

    return keyframes

def rosbag2lerobot():
    pass
if __name__ == "__main__":
    split_keyframes_into_episodes("/home/lenguyen1/hoangpqn/vln/InternNav/scripts/instruction_generator/keyframe_output/keyframe_")