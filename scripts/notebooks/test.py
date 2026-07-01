
from pathlib import Path
from PIL import Image

src_dir_rgb = Path("/home/phuongnh/khang/InternNav/assets/vr-office/observation.images.rgb.125cm_30deg")
dst_dir_rgb = Path("/home/phuongnh/khang/InternNav/assets/vr-office/rgb")
dst_dir_rgb.mkdir(exist_ok=True)

src_dir_depth = Path("/home/phuongnh/khang/InternNav/assets/vr-office/observation.images.depth.125cm_30deg")
dst_dir_depth = Path("/home/phuongnh/khang/InternNav/assets/vr-office/depth")
dst_dir_depth.mkdir(exist_ok=True)

for frame_num, rgb_file in enumerate(sorted(src_dir_rgb.glob("episode_000001_*.jpg")), start=1):
    frame_id = str(int(rgb_file.stem.split("_")[-1])).zfill(4)
    depth_file = src_dir_depth / f"{rgb_file.stem}.png"
    if not depth_file.exists():
        raise FileNotFoundError(f"Missing depth file for {rgb_file.name}: {depth_file}")

    if frame_num % 10 == 0 and frame_num > 0:
        out_name = f"debug_raw_{frame_id}_look_down.jpg"
    else:
        out_name = f"debug_raw_{frame_id}.jpg"

    Image.open(rgb_file).convert("RGB").save(
        dst_dir_rgb / out_name,
        quality=95,
    )
    Image.open(depth_file).convert("RGB").save(
        dst_dir_depth / out_name,
        quality=95,)