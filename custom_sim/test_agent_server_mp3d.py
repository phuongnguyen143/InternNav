import argparse
import gzip
import json
import os
import time

import cv2
import imageio
import numpy as np
from enum import IntEnum
from habitat.config import read_write
from habitat_baselines.config.default import get_config as get_habitat_config

from internnav.configs.agent import AgentCfg
import internnav.habitat_extensions.vln.measures  # noqa: F401
from internnav.utils import AgentClient
import habitat
import habitat_sim

SERVER_HOST = "localhost"
SERVER_PORT = 8087

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Navigate using Habitat sim on MP3D scene."
    )
    parser.add_argument(
        "--habitat-config",
        type=str,
        default="scripts/eval/configs/vln_vinrobo.yaml",
        help="Habitat yaml config path for MP3D VLN episodes.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to InternVLA-N1 model checkpoint.",
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="Use habitat-sim scene directly (no dataset json required).",
    )
    parser.add_argument(
        "--scene", type=str, default=None, help="Scene path for --sim-only mode."
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="go to the target",
        help="Instruction text for --sim-only mode.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save rollout video after episode ends.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="logs/test_agent_server_mp3d_episode.mp4",
        help="Output mp4 path when --save-video is enabled.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=6,
        help="FPS for saved rollout video.",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help=(
            "episode_id string to run (e.g. '7'). "
            "Mutually exclusive with --episode-idx. "
            "If neither is set, a random episode is chosen."
        ),
    )
    parser.add_argument(
        "--stop-delay-seconds",
        type=float,
        default=5.0,
        help=(
            "After the first STOP (action 0), keep calling agent.step() for this many "
            "seconds without sending habitat STOP. Set to 0 to end immediately."
        ),
    )

    return parser.parse_args()


def _investigate_after_stop(args, agent, agent_obs, video_frames, step_idx):
    """Keep agent inference running after STOP without ending the Habitat episode."""
    delay = args.stop_delay_seconds
    if delay <= 0:
        return

    instruction = agent_obs.get("instruction", "")
    print(
        f"[step {step_idx + 1}] STOP received — continuing agent inference for "
        f"{delay:.1f}s (habitat STOP deferred)..."
    )
    deadline = time.time() + delay
    infer_count = 0
    while time.time() < deadline:
        out = agent.step([agent_obs])[0]
        action = out["action"][0]
        infer_count += 1
        print(f"[post-stop infer #{infer_count}] agent_action={action}")

        if args.save_video:
            rgb = np.asarray(agent_obs["rgb"], dtype=np.uint8)
            video_frames.append(_overlay_prompt_text(rgb, instruction))

    print(f"[post-stop infer] finished ({infer_count} extra agent.step calls)")


def build_agent_client(args):
    agent_cfg = AgentCfg(
        server_host=SERVER_HOST,
        server_port=SERVER_PORT,
        model_name="internvla_n1",
        ckpt_path="",
        model_settings={
            "policy_name": "InternVLAN1_Policy",
            "state_encoder": None,
            "env_num": 1,
            "sim_num": 1,
            "infer_mode": "partial_async",
            "model_path": args.model_path,
            "camera_intrinsic": [
                [585.0, 0.0, 320.0],
                [0.0, 585.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "width": 640,
            "height": 480,
            "hfov": 79,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            "num_frames": 32,
            "num_history": 8,
            "num_future_steps": 4,
            "device": "cuda:0",
            "predict_step_nums": 32,
            "continuous_traj": True,
            "vis_debug": False,
            "vis_debug_path": "./logs/vis_debug",
        },
    )
    return AgentClient(agent_cfg)


def normalize_obs(observations, instruction):
    rgb = np.asarray(observations["rgb"], dtype=np.uint8)
    depth = np.asarray(observations["depth"], dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[..., None]
    return {
        "rgb": rgb,
        "depth": depth,
        "instruction": instruction,
    }


def map_agent_action_to_habitat(action):
    return int(action)


def _overlay_prompt_text(frame: np.ndarray, prompt_text: str) -> np.ndarray:
    """Render a wrapped prompt caption near the bottom of a video frame."""
    out = np.asarray(frame, dtype=np.uint8).copy()
    h, w = out.shape[:2]

    prompt = "Prompt: " + " ".join(str(prompt_text).split())
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    margin_x = 10
    margin_bottom = 8
    pad = 6
    line_gap = 5

    max_text_width = max(40, w - 2 * (margin_x + pad))

    # Wrap text by pixel width, then cap to max lines to avoid covering too much image.
    words = prompt.split(" ")
    lines = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        (cand_w, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if cand_w <= max_text_width:
            cur = candidate
            continue
        if cur:
            lines.append(cur)
            cur = word
        else:
            # Handle a very long token: trim it to fit.
            clipped = word
            while clipped:
                (clip_w, _), _ = cv2.getTextSize(clipped, font, font_scale, thickness)
                if clip_w <= max_text_width:
                    break
                clipped = clipped[:-1]
            lines.append((clipped + "...") if clipped and clipped != word else word)
            cur = ""
    if cur:
        lines.append(cur)

    max_lines = 3
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if not lines[-1].endswith("..."):
            lines[-1] = lines[-1].rstrip(" .") + "..."

    (_, text_h), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_height = text_h + line_gap
    block_h = pad + len(lines) * line_height + baseline + pad
    y1 = h - margin_bottom
    y0 = max(0, y1 - block_h)
    x0 = margin_x
    x1 = w - margin_x

    cv2.rectangle(out, (x0, y0), (x1, y1), bg_color, thickness=-1)
    y = y0 + pad + text_h
    for line in lines:
        cv2.putText(
            out,
            line,
            (x0 + pad, y),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        y += line_height
    return out


def _load_episodes(dataset_path: str):
    """Load and return the full episode list from the (possibly gzipped) dataset."""
    if dataset_path.endswith(".gz"):
        opener = lambda: gzip.open(dataset_path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(dataset_path, "r", encoding="utf-8")

    with opener() as f:
        data = json.load(f)

    return data.get("episodes", []) if isinstance(data, dict) else data


def list_episodes(dataset_path: str):
    """Print a summary table of all episodes and exit."""
    episodes = _load_episodes(dataset_path)
    if not episodes:
        print("No episodes found in dataset.")
        return
    print(
        f"\n{'IDX':>4}  {'episode_id':>12}  {'scene_id':<40}  instruction (first 60 chars)"
    )
    print("-" * 120)
    for idx, ep in enumerate(episodes):
        ep_id = ep.get("episode_id", "?")
        scene = ep.get("scene_id", "?")
        instr = ep.get("instruction", {}).get("instruction_text", "")
        print(f"{idx:>4}  {str(ep_id):>12}  {scene:<40}  {instr[:60]}")
    print(f"\nTotal: {len(episodes)} episodes")


def _select_episode(episodes, episode_id=None):
    for episode in episodes:
        if str(episode.get("episode_id")) == str(episode_id):
            return episode
    return None


def _get_scene_id_from_episode(dataset_path: str, episode_id=None):
    episodes = _load_episodes(dataset_path)
    if not episodes:
        return None, None

    if episode_id is None:
        return None, None
    for episode in episodes:
        if str(episode.get("episode_id")) == str(episode_id):
            return episode, episode.get("scene_id")
    return None, None


def _resolve_scenes_dir(dataset_path: str, scenes_dir: str, episode_id=None):
    _, scene_id = _get_scene_id_from_episode(dataset_path, episode_id=episode_id)
    if not scene_id:
        return scenes_dir
    if os.path.isabs(scene_id):
        return scenes_dir

    candidates = [scenes_dir]
    parent = os.path.dirname(scenes_dir)
    grand_parent = os.path.dirname(parent)
    for cand in (parent, grand_parent):
        if cand and cand not in candidates:
            candidates.append(cand)

    for cand in candidates:
        scene_path = os.path.join(cand, scene_id)
        if os.path.exists(scene_path):
            if cand != scenes_dir:
                print(f"[auto-fix] scenes_dir adjusted: {scenes_dir} -> {cand}")
            print(f"[check] sample scene path: {scene_path}")
            return cand

    print(
        f"[warn] Could not validate scene_id '{scene_id}' under provided scenes_dir candidates."
    )
    print("[warn] Keep original scenes_dir and let Habitat report detailed path error.")
    return scenes_dir


# action codes mirroring the evaluator's action_code enum
LOOKDOWN = 5
LOOKUP = 4


def apply_lookdown_probe(env, observations):
    """
    Mirror habitat_vln_evaluator behavior exactly:
    - Step LOOKDOWN twice to get depth from floor-level view
    - Step LOOKUP twice to RESTORE camera back to horizontal
    - Return the restored (horizontal) observation
    """
    down_obs = env.step(LOOKDOWN)
    done = env.episode_over
    if done:
        env.step(LOOKUP)
        env.step(LOOKUP)
        return down_obs, done

    down_obs = env.step(LOOKDOWN)
    done = env.episode_over

    restored_obs = env.step(LOOKUP)
    if not env.episode_over:
        restored_obs = env.step(LOOKUP)
        done = env.episode_over

    return restored_obs, done


def _advance_env_to_episode(env, target_episode_id: str, max_resets: int = 500):
    """
    Reset the Habitat env repeatedly until current_episode.episode_id matches
    target_episode_id. Returns the observations at the target episode.

    Raises RuntimeError if the target is not reached within max_resets attempts.
    """
    for attempt in range(max_resets):
        observations = env.reset()
        current_id = str(env.current_episode.episode_id)
        if current_id == str(target_episode_id):
            print(
                f"[env] Reached target episode_id='{target_episode_id}' "
                f"after {attempt + 1} reset(s)."
            )
            return observations
    raise RuntimeError(
        f"Could not reach episode_id='{target_episode_id}' within {max_resets} resets. "
        f"Last episode_id was '{str(env.current_episode.episode_id)}'."
    )


def run_sim_only_loop(args, agent):
    if not args.scene:
        raise ValueError("--scene is required when --sim-only is enabled.")

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = args.scene

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 1.5, 0.0]

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_spec.resolution = [480, 640]
    depth_spec.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    action_name_map = {
        1: "move_forward",
        2: "turn_left",
        3: "turn_right",
        4: "look_up",
        5: "look_down",
    }

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    video_prompt = args.instruction

    with habitat_sim.Simulator(sim_cfg) as sim:
        sim.initialize_agent(0)
        print(f"Sim-only mode | scene={args.scene} | instruction={args.instruction}")
        video_frames = []
        step_idx = 0
        while step_idx < args.max_steps:
            sim_obs = sim.get_sensor_observations()
            rgb = np.asarray(sim_obs["rgb"], dtype=np.uint8)[..., :3]
            depth = np.asarray(sim_obs["depth"], dtype=np.float32)
            obs = {
                "rgb": rgb,
                "depth": depth[..., None],
                "instruction": args.instruction,
            }

            out = agent.step([obs])[0]
            action = out["action"][0]
            if action == 0:
                _investigate_after_stop(
                    args, agent, obs, video_frames, step_idx
                )
                print(f"[step {step_idx + 1}] agent_action=0 (STOP)")
                break
            if action == -1:
                sim.step("look_down")
                sim.step("look_down")
                sim.step("look_up")
                sim.step("look_up")
                if args.save_video:
                    frame = np.asarray(
                        sim.get_sensor_observations()["rgb"], dtype=np.uint8
                    )[..., :3]
                    video_frames.append(_overlay_prompt_text(frame, video_prompt))
                print(
                    f"[step {step_idx + 1}] agent_action=-1 lookdown x2 + lookup x2 (restored)"
                )
                continue

            action_name = action_name_map.get(int(action))
            if action_name is None:
                print(
                    f"[step {step_idx + 1}] unsupported action={action}, fallback turn_left"
                )
                action_name = "turn_left"
            sim.step(action_name)
            if args.save_video:
                frame = np.asarray(
                    sim.get_sensor_observations()["rgb"], dtype=np.uint8
                )[..., :3]
                video_frames.append(_overlay_prompt_text(frame, video_prompt))
            print(
                f"[step {step_idx + 1}] agent_action={action} sim_action={action_name}"
            )
            step_idx += 1

        if args.save_video and len(video_frames) > 0:
            video_dir = os.path.dirname(args.video_path)
            if video_dir:
                os.makedirs(video_dir, exist_ok=True)
            imageio.mimsave(args.video_path, video_frames, fps=args.video_fps)
            print(
                f"[video] Saved rollout video: {args.video_path} ({len(video_frames)} frames)"
            )


def main():
    args = parse_args()

    habitat_cfg = get_habitat_config(args.habitat_config)
    dataset_path = habitat_cfg.habitat.dataset.data_path.format(
        split=habitat_cfg.habitat.dataset.split
    )

    #build agent
    agent = build_agent_client(args)

    #sim mode
    if args.sim_only:
        run_sim_only_loop(args, agent)
        return
    
    #habitat mode
    all_episodes = _load_episodes(dataset_path)
    target_episode = _select_episode(all_episodes, episode_id=args.episode_id)
    print(f"Target episode: {target_episode}")
    if target_episode is None:
        raise ValueError(
            f"Could not find episode_id={args.episode_id} in dataset: {dataset_path}"
        )
    target_episode_id = str(target_episode.get("episode_id"))
    target_instruction = target_episode.get("instruction", {}).get(
        "instruction_text", "go to the target"
    )
    target_goals = target_episode.get("goals")

    print(f"\n{'=' * 60}")
    print(f"  Target episode_id : {target_episode_id}")
    print(f"  Scene             : {target_episode.get('scene_id')}")
    print(f"  Instruction       : {target_instruction}")
    print(f"{'=' * 60}\n")

    if not target_goals:
        raise ValueError(
            "Selected episode does not include `goals`. "
            "This usually happens on R2R `test` split annotations. "
            "Use a split with goals (e.g. val_seen/val_unseen/train) "
            "or a dataset export that contains goals."
        )

    with read_write(habitat_cfg):
        habitat_cfg.habitat.dataset.scenes_dir = _resolve_scenes_dir(
            dataset_path,
            habitat_cfg.habitat.dataset.scenes_dir,
            episode_id=target_episode_id,
        )
    print(f"[config] dataset_path={dataset_path}")
    print(f"[config] scenes_dir={habitat_cfg.habitat.dataset.scenes_dir}")

    with habitat.Env(config=habitat_cfg) as env:
        # Advance to the desired episode
        observations = _advance_env_to_episode(env, target_episode_id)

        episode = env.current_episode
        instruction = episode.instruction.instruction_text
        print(f"Episode {episode.episode_id} | instruction: {instruction}")
        video_frames = []

        if args.save_video:
            video_frames.append(
                _overlay_prompt_text(
                    np.asarray(observations["rgb"], dtype=np.uint8), instruction
                )
            )

        step_idx = 0
        while step_idx < args.max_steps:
            agent_obs = normalize_obs(observations, instruction)
            out = agent.step([agent_obs])[0]
            action = out["action"][0]

            if action == -1:
                observations, done = apply_lookdown_probe(env, observations)

                if args.save_video:
                    video_frames.append(
                        _overlay_prompt_text(
                            np.asarray(observations["rgb"], dtype=np.uint8), instruction
                        )
                    )

                metrics = env.get_metrics()
                print(
                    f"[step {step_idx + 1}] agent_action=-1 lookdown x2 + lookup x2 (restored) "
                    f"done={done} metrics={metrics}"
                )
                if done:
                    print("Episode finished.")
                    break
                continue

            if action == 0:
                _investigate_after_stop(
                    args, agent, agent_obs, video_frames, step_idx
                )

            habitat_action = map_agent_action_to_habitat(action)
            observations = env.step(habitat_action)

            if args.save_video:
                video_frames.append(
                    _overlay_prompt_text(
                        np.asarray(observations["rgb"], dtype=np.uint8), instruction
                    )
                )

            metrics = env.get_metrics()
            print(
                f"[step {step_idx + 1}] "
                f"agent_action={action} habitat_action={habitat_action} "
                f"done={env.episode_over} metrics={metrics}"
            )

            if env.episode_over:
                print("Episode finished.")
                break
            step_idx += 1

        if args.save_video and len(video_frames) > 0:
            video_dir = os.path.dirname(args.video_path)
            if video_dir:
                os.makedirs(video_dir, exist_ok=True)
            imageio.mimsave(args.video_path, video_frames, fps=args.video_fps)
            print(
                f"[video] Saved rollout video: {args.video_path} ({len(video_frames)} frames)"
            )


if __name__ == "__main__":
    main()
