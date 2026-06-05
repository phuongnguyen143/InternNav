"""
=============================================================
TẠO CUSTOM EPISODE DATASET CHO INTERNNAV / HABITAT
=============================================================
Script này thực hiện toàn bộ Bước 2:
  - Load scene GLB của bạn vào Habitat Sim
  - Tự động tạo navmesh (lưới đường đi)
  - Tìm các điểm xuất phát và đích hợp lệ
  - Tạo đường đi tham khảo (reference_path) qua Shortest Path
  - Gán instruction (câu lệnh) do bạn tự viết
  - Xuất file JSON.gz đúng format VLN-CE mà InternNav đọc được

CÁCH DÙNG (zsh/bash):
  - Sau dấu \\ xuống dòng KHÔNG được có khoảng trắng hay comment trên cùng dòng,
    nếu không shell sẽ không nối dòng và các dòng --scene/--output chạy như lệnh riêng.

  python create_custom_dataset.py \\
      --scene path/to/your_scene.glb \\
      --output data/datasets/my_dataset/train/train.json.gz \\
      --instructions instructions.txt \\
      --num_episodes 10

  Hoặc một dòng:
  python create_custom_dataset.py --scene ... --output ... --instructions ...

REQUIREMENTS:
  conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
=============================================================
"""

import argparse
import gzip
import json
import math
import os
import random
import sys
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────
# KIỂM TRA habitat_sim đã được cài chưa
# ──────────────────────────────────────────────
try:
    import habitat_sim
    import numpy as np
    from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis

    print("[OK] habitat_sim imported successfully")
except ImportError:
    print("[ERROR] habitat_sim chưa được cài.")
    print("Chạy lệnh sau để cài:")
    print(
        "  conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat"
    )
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# PHẦN 1: KHỞI TẠO SIMULATOR
# ══════════════════════════════════════════════════════════════


def make_simulator(
    scene_glb_path: str, image_width=256, image_height=256
) -> habitat_sim.Simulator:
    """
    Tạo Habitat Simulator từ file GLB.
    Bao gồm camera RGB để kiểm tra scene nếu cần.
    """
    # -- Backend config (scene) --
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_glb_path
    backend_cfg.enable_physics = False  # không cần physics cho VLN
    backend_cfg.allow_sliding = True

    # -- Agent config (camera) --
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Camera RGB (tùy chọn, dùng để visualize nếu cần)
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [image_height, image_width]
    rgb_sensor.position = [0.0, 1.5, 0.0]  # camera ở độ cao 1.5m (ngang tầm mắt)
    rgb_sensor.hfov = math.degrees(math.pi / 2)  # 90 độ

    agent_cfg.sensor_specifications = [rgb_sensor]
    agent_cfg.height = 1.5  # chiều cao agent
    agent_cfg.radius = 0.18  # bán kính collision

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    print(f"[OK] Simulator loaded scene: {scene_glb_path}")
    return sim


# ══════════════════════════════════════════════════════════════
# PHẦN 2: XỬ LÝ NAVMESH
# ══════════════════════════════════════════════════════════════


def ensure_navmesh(sim: habitat_sim.Simulator, scene_glb_path: str) -> bool:
    """
    Kiểm tra navmesh đã tồn tại chưa.
    Nếu chưa → tự động tạo và lưu bên cạnh file GLB.
    Navmesh quyết định agent có thể đi đến đâu.
    """
    navmesh_path = scene_glb_path.replace(".glb", ".navmesh")

    if os.path.exists(navmesh_path):
        print(f"[OK] Navmesh đã tồn tại: {navmesh_path}")
        # Load navmesh từ file
        sim.pathfinder.load_nav_mesh(navmesh_path)
    else:
        print(f"[INFO] Chưa có navmesh, đang tạo mới...")
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()

        # Điều chỉnh theo kích thước scene của bạn
        navmesh_settings.agent_height = 1.5  # chiều cao agent (m)
        navmesh_settings.agent_radius = 0.18  # bán kính agent (m)
        navmesh_settings.agent_max_climb = 0.2  # bậc thang tối đa có thể leo (m)
        navmesh_settings.agent_max_slope = 45.0  # độ dốc tối đa (degrees)

        success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        if not success:
            print("[ERROR] Tạo navmesh thất bại! Kiểm tra lại file GLB.")
            return False

        sim.pathfinder.save_nav_mesh(navmesh_path)
        print(f"[OK] Navmesh đã được lưu: {navmesh_path}")

    # Kiểm tra navmesh có hợp lệ không
    if not sim.pathfinder.is_loaded:
        print("[ERROR] Navmesh không load được!")
        return False

    navigable_area = sim.pathfinder.navigable_area
    print(f"[INFO] Diện tích có thể đi được: {navigable_area:.2f} m²")
    if navigable_area < 1.0:
        print("[WARN] Diện tích navmesh quá nhỏ. Scene có thể không phù hợp.")

    return True


# ══════════════════════════════════════════════════════════════
# PHẦN 3: LẤY ĐIỂM START VÀ GOAL HỢP LỆ
# ══════════════════════════════════════════════════════════════


def get_random_navigable_pair(
    sim: habitat_sim.Simulator,
    min_distance: float = 3.0,
    max_distance: float = 15.0,
    max_tries: int = 1000,
) -> Optional[Tuple[List[float], List[float]]]:
    """
    Tìm một cặp (start, goal) ngẫu nhiên trên navmesh,
    cách nhau trong khoảng [min_distance, max_distance] mét.

    Trả về None nếu không tìm được sau max_tries lần thử.
    """
    for attempt in range(max_tries):
        # Lấy điểm ngẫu nhiên trên navmesh
        start = sim.pathfinder.get_random_navigable_point()
        goal = sim.pathfinder.get_random_navigable_point()

        if not sim.pathfinder.is_navigable(start) or not sim.pathfinder.is_navigable(
            goal
        ):
            continue

        # Tính khoảng cách Euclidean giữa start và goal
        dist = np.linalg.norm(np.array(start) - np.array(goal))

        if min_distance <= dist <= max_distance:
            # Kiểm tra xem có đường đi thực sự không (geodesic distance)
            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = goal
            found = sim.pathfinder.find_path(path)

            if found and path.geodesic_distance < float("inf"):
                return list(start), list(goal)

    print(f"[WARN] Không tìm được cặp start-goal sau {max_tries} lần thử.")
    print(f"       Thử giảm min_distance hoặc tăng max_distance.")
    return None


def get_reference_path(
    sim: habitat_sim.Simulator,
    start: List[float],
    goal: List[float],
) -> Tuple[List[List[float]], float]:
    """
    Tính đường đi ngắn nhất từ start → goal (dùng navmesh).
    Trả về: (danh sách waypoints, geodesic distance)
    """
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(start)
    path.requested_end = np.array(goal)

    found = sim.pathfinder.find_path(path)

    if not found or path.geodesic_distance == float("inf"):
        # Fallback: chỉ dùng [start, goal] nếu không tìm được path
        print(f"[WARN] Không tìm được shortest path, dùng [start, goal] trực tiếp.")
        return [start, goal], np.linalg.norm(np.array(start) - np.array(goal))

    # Chuyển waypoints thành list of list
    waypoints = [list(p) for p in path.points]

    # Đảm bảo bắt đầu bằng start và kết thúc bằng goal
    if waypoints[0] != start:
        waypoints.insert(0, start)
    if waypoints[-1] != goal:
        waypoints.append(goal)

    return waypoints, float(path.geodesic_distance)


def get_random_rotation() -> List[float]:
    """
    Chỉ xoay quanh trục Y (yaw) — KHÔNG xoay pitch/roll.
    Habitat convention: quaternion [x, y, z, w]
    Agent nhìn thẳng về phía trước = [0, 0, 0, 1]
    """
    import random, math

    yaw = random.uniform(0, 2 * math.pi)
    # Quaternion cho Y-axis rotation: [0, sin(yaw/2), 0, cos(yaw/2)]
    return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]


# ══════════════════════════════════════════════════════════════
# PHẦN 4: TẠO EPISODE
# ══════════════════════════════════════════════════════════════


def create_episode(
    episode_id: int,
    scene_id: str,  # đường dẫn tương đối đến GLB
    start: List[float],
    goal: List[float],
    reference_path: List[List[float]],
    instruction_text: str,
    geodesic_distance: float,
    goal_radius: float = 3.0,
    trajectory_id: Optional[int] = None,
) -> dict:
    """
    Tạo 1 episode theo đúng format VLN-CE mà InternNav đọc được.

    Format đầy đủ (dựa trên spec thực tế của R2R_VLNCE):
    {
        "episode_id": int,
        "trajectory_id": int,
        "scene_id": "path/to/scene.glb",
        "start_position": [x, y, z],
        "start_rotation": [x, y, z, w],   ← quaternion
        "goals": [{"position": [x,y,z], "radius": float}],
        "reference_path": [[x,y,z], ...],
        "instruction": {
            "instruction_text": "...",
            "instruction_tokens": []        ← để rỗng, InternNav không cần
        },
        "info": {"geodesic_distance": float}
    }
    """
    return {
        "episode_id": episode_id,
        "trajectory_id": trajectory_id if trajectory_id is not None else episode_id,
        "scene_id": scene_id,
        "start_position": start,
        "start_rotation": get_random_rotation(),
        "goals": [
            {
                "position": goal,
                "radius": goal_radius,
            }
        ],
        "reference_path": reference_path,
        "instruction": {
            "instruction_text": instruction_text,
            "instruction_tokens": [],  # bỏ trống — InternNav dùng text trực tiếp
        },
        "info": {
            "geodesic_distance": geodesic_distance,
        },
    }


# ══════════════════════════════════════════════════════════════
# PHẦN 5: BUILD TOÀN BỘ DATASET
# ══════════════════════════════════════════════════════════════


def build_dataset(
    sim: habitat_sim.Simulator,
    scene_id: str,
    instructions: List[str],
    num_episodes: int,
    min_dist: float = 3.0,
    max_dist: float = 15.0,
    goal_radius: float = 3.0,
) -> dict:
    """
    Tạo toàn bộ dataset với num_episodes episodes.
    Mỗi episode được ghép với 1 instruction từ danh sách (vòng lặp nếu cần).
    """
    episodes = []
    failed = 0

    print(f"\n[INFO] Đang tạo {num_episodes} episodes...")
    print(f"       Khoảng cách start-goal: {min_dist}m ~ {max_dist}m")
    print(f"       Goal radius: {goal_radius}m\n")

    for i in range(num_episodes):
        # Lấy instruction (vòng lặp nếu số instructions < num_episodes)
        instruction = instructions[i % len(instructions)].strip()
        if not instruction:
            instruction = f"Navigate to the destination. (episode {i})"

        # Tìm cặp start-goal hợp lệ
        result = get_random_navigable_pair(sim, min_dist, max_dist)
        if result is None:
            print(f"[SKIP] Episode {i}: không tìm được start-goal hợp lệ.")
            failed += 1
            continue

        start, goal = result

        # Tính reference path
        ref_path, geo_dist = get_reference_path(sim, start, goal)

        # Tạo episode
        ep = create_episode(
            episode_id=i,
            scene_id=scene_id,
            start=start,
            goal=goal,
            reference_path=ref_path,
            instruction_text=instruction,
            geodesic_distance=geo_dist,
            goal_radius=goal_radius,
        )
        episodes.append(ep)

        print(
            f"  Episode {i:3d} | dist={geo_dist:.2f}m | waypoints={len(ref_path)} | instruction='{instruction[:50]}...'"
        )

    print(f"\n[DONE] Tạo thành công: {len(episodes)} episodes, thất bại: {failed}")

    # ── instruction_vocab (để trống vì InternNav dùng LLM, không cần vocab) ──
    dataset = {
        "episodes": episodes,
        "instruction_vocab": {
            "word_list": [],
            "word2idx_dict": {},
            "itos": [],
            "stoi": {},
            "num_vocab": 0,
            "UNK_INDEX": 1,
            "PAD_INDEX": 0,
        },
    }
    return dataset


# ══════════════════════════════════════════════════════════════
# PHẦN 6: LƯU FILE JSON.GZ
# ══════════════════════════════════════════════════════════════


def _to_json_serializable(obj):
    """Chuyển numpy scalar/array → Python int/float/list để json.dumps không lỗi."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_serializable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_dataset(dataset: dict, output_path: str) -> None:
    """
    Lưu dataset dạng gzip JSON — đây là format Habitat yêu cầu.
    Tự động tạo thư mục cha nếu chưa tồn tại.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    json_str = json.dumps(_to_json_serializable(dataset), indent=2, ensure_ascii=False)

    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        f.write(json_str)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n[SAVED] Dataset đã lưu tại: {output_path}")
    print(f"        Kích thước file: {size_kb:.1f} KB")
    print(f"        Số episodes: {len(dataset['episodes'])}")


# ══════════════════════════════════════════════════════════════
# PHẦN 7: KIỂM TRA DATASET (VERIFY)
# ══════════════════════════════════════════════════════════════


def verify_dataset(output_path: str) -> None:
    """
    Đọc lại file vừa tạo và in thống kê để kiểm tra.
    """
    print("\n" + "=" * 50)
    print("KIỂM TRA DATASET VỪA TẠO")
    print("=" * 50)

    with gzip.open(output_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"]
    print(f"Số episodes: {len(episodes)}")

    if episodes:
        ep0 = episodes[0]
        print(f"\nEpisode đầu tiên (episode_id={ep0['episode_id']}):")
        print(f"  scene_id       : {ep0['scene_id']}")
        print(f"  start_position : {[f'{v:.3f}' for v in ep0['start_position']]}")
        print(f"  start_rotation : {[f'{v:.4f}' for v in ep0['start_rotation']]}")
        print(f"  goal position  : {[f'{v:.3f}' for v in ep0['goals'][0]['position']]}")
        print(f"  goal radius    : {ep0['goals'][0]['radius']} m")
        print(f"  geodesic_dist  : {ep0['info']['geodesic_distance']:.3f} m")
        print(f"  ref_path pts   : {len(ep0['reference_path'])} waypoints")
        print(f'  instruction    : "{ep0["instruction"]["instruction_text"]}"')

    # Thống kê khoảng cách
    dists = [ep["info"]["geodesic_distance"] for ep in episodes]
    if dists:
        print(f"\nThống kê geodesic distance:")
        print(f"  Min : {min(dists):.2f} m")
        print(f"  Max : {max(dists):.2f} m")
        print(f"  Avg : {sum(dists) / len(dists):.2f} m")

    print("\n[OK] Dataset hợp lệ và sẵn sàng dùng!")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tạo custom VLN-CE episode dataset từ scene GLB cho InternNav/Habitat"
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Đường dẫn đến file GLB của bạn. Ví dụ: data/scene_datasets/my_scene/my_scene.glb",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Đường dẫn output file JSON.gz. Ví dụ: data/datasets/my_dataset/train/train.json.gz",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="File TXT chứa các câu instruction (mỗi dòng 1 câu). "
        "Nếu không cung cấp, sẽ dùng instruction mẫu.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Số lượng episode muốn tạo (mặc định: 10)",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=3.0,
        help="Khoảng cách tối thiểu giữa start và goal (mét, mặc định: 3.0)",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=15.0,
        help="Khoảng cách tối đa giữa start và goal (mét, mặc định: 15.0)",
    )
    parser.add_argument(
        "--goal_radius",
        type=float,
        default=3.0,
        help="Bán kính vùng coi là 'đến đích' (mét, mặc định: 3.0)",
    )
    parser.add_argument(
        "--scene_id_override",
        default=None,
        help="Ghi đè scene_id trong JSON (mặc định dùng đường dẫn --scene). "
        "Hữu ích khi muốn dùng đường dẫn tương đối theo cấu trúc Habitat. "
        "Ví dụ: my_scene/my_scene.glb",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed để tái tạo kết quả (mặc định: 42)",
    )
    return parser.parse_args()


def load_instructions(instructions_file: Optional[str]) -> List[str]:
    """Load danh sách instruction từ file, hoặc dùng mẫu nếu không có file."""
    if instructions_file and os.path.exists(instructions_file):
        with open(instructions_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"[OK] Đã load {len(lines)} instructions từ: {instructions_file}")
        return lines
    else:
        # Instruction mẫu — thay bằng câu lệnh thực của bạn
        default_instructions = [
            "Walk forward through the hallway and turn left at the end.",
            "Go straight past the door and stop near the large window.",
            "Navigate to the kitchen area and stop in front of the counter.",
            "Follow the corridor and enter the second room on the right.",
            "Move towards the staircase and stop at the bottom step.",
            "Walk through the main entrance and proceed to the center of the room.",
            "Go past the sofa and stop near the bookshelf on the right wall.",
            "Navigate to the far end of the corridor and enter the last door.",
            "Walk diagonally towards the corner and stop by the plant.",
            "Proceed through the archway and stop in the open area ahead.",
        ]
        if instructions_file:
            print(
                f"[WARN] Không tìm thấy file '{instructions_file}', dùng instructions mẫu."
            )
        else:
            print("[INFO] Không có file instructions, dùng instructions mẫu.")
        return default_instructions


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Kiểm tra file GLB tồn tại
    if not os.path.exists(args.scene):
        print(f"[ERROR] Không tìm thấy file GLB: {args.scene}")
        sys.exit(1)

    print("=" * 60)
    print("CUSTOM EPISODE DATASET CREATOR FOR INTERNNAV/HABITAT")
    print("=" * 60)
    print(f"Scene      : {args.scene}")
    print(f"Output     : {args.output}")
    print(f"Num eps    : {args.num_episodes}")
    print(f"Dist range : {args.min_dist}m ~ {args.max_dist}m")
    print(f"Goal radius: {args.goal_radius}m")
    print(f"Seed       : {args.seed}")
    print("=" * 60 + "\n")

    # Bước 1: Load instructions
    instructions = load_instructions(args.instructions)

    # Bước 2: Khởi tạo simulator
    sim = make_simulator(args.scene)

    # Bước 3: Đảm bảo navmesh tồn tại
    if not ensure_navmesh(sim, args.scene):
        print("[ERROR] Không thể tạo navmesh. Thoát.")
        sim.close()
        sys.exit(1)

    # Bước 4: Xác định scene_id dùng trong JSON
    scene_id = args.scene_id_override if args.scene_id_override else args.scene

    # Bước 5: Build dataset
    dataset = build_dataset(
        sim=sim,
        scene_id=scene_id,
        instructions=instructions,
        num_episodes=args.num_episodes,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        goal_radius=args.goal_radius,
    )

    # Bước 6: Đóng simulator
    sim.close()
    print("\n[OK] Simulator closed.")

    # Bước 7: Lưu dataset
    save_dataset(dataset, args.output)

    # Bước 8: Verify
    verify_dataset(args.output)

    print("\n" + "=" * 60)
    print("HƯỚNG DẪN TIẾP THEO")
    print("=" * 60)
    print("1. Cập nhật config InternNav để trỏ vào dataset vừa tạo:")
    print(
        f'     "data_path": "{os.path.dirname(os.path.dirname(args.output))}/{{split}}/{{split}}.json.gz"'
    )
    print(f'     "scenes_dir": "<thư mục chứa scene GLB>"')
    print()
    print("2. Chạy evaluation:")
    print("     python scripts/eval/eval.py --config scripts/eval/configs/your_cfg.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
