from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1-DualVLN",  # path to model checkpoint
            "num_history": 8,
            "resize_w": 224,  # image resize width
            "resize_h": 224,  # image resize height
            "max_new_tokens": 1024,  # maximum number of tokens for generation
            "model_device_id": 1,  # VLM gpu. in vln_r2r.yaml, habitat use 0
            "vis_debug": True,  # save per-step RGB debug videos per episode
            "vis_debug_path": "./logs/habitat/vis_debug3",
            "extract_attention": False,  # save S2 attention heatmaps (requires eager attn)
            "attn_implementation": "sdpa",  # sdpa/eager 
            "attn_layers": [6,16,24],  
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
            'habitat_gpu_device_id': 0,  # sim renderer GPU (torchrun ranks auto-use local_rank if omitted)
            'max_episodes': 1,  # only run this many episodes (per rank)
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/test_dual_system3",  # fresh dir avoids progress.json resume
        "save_video": True,  # save top-down map video (all episodes when vis_debug is on)
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)
