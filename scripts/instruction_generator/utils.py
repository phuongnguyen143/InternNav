from pathlib import Path
import shutil


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
def rosbag2lerobot():
    pass
if __name__ == "__main__":
    split_keyframes_into_episodes("/home/lenguyen1/hoangpqn/vln/InternNav/scripts/instruction_generator/keyframe_output/keyframe_")