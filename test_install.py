# Verify if the installation is working
# To verify the installation of InternNav, start the model server first.
# python scripts/eval/start_server.py --port 8087

from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient

cfg = AgentCfg(
    server_host='localhost',
    server_port=8087,
    model_name='internvla_n1',
    ckpt_path='',
    model_settings={
        'policy_name': "InternVLAN1_Policy", 
        'state_encoder': None,
        'env_num': 1,
        'sim_num': 1,
        'model_path': "checkpoints/InternVLA-N1-w-NavDP",
        'camera_intrinsic': [
            [585.0, 0.0, 320.0],
            [0.0, 585.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        'width': 640,
        'height': 480,
        'hfov': 79,
        'resize_w': 384,
        'resize_h': 384,
        'max_new_tokens': 1024,
        'num_frames': 32,
        'num_history': 8,
        'num_future_steps': 4,
        'device': 'cuda:0',
        'predict_step_nums': 32,
        'continuous_traj': True,
    }
)

agent = AgentClient(cfg)

from scripts.iros_challenge.onsite_competition.sdk.save_obs import load_obs_from_meta
rs_meta_path = './scripts/iros_challenge/onsite_competition/captures/rs_meta.json'

fake_obs_640 = load_obs_from_meta(rs_meta_path)
fake_obs_640['instruction'] = 'go to the direction with no clothes'
print(fake_obs_640['rgb'].shape, fake_obs_640['depth'].shape)


print("=================================================")
action = agent.step([fake_obs_640])#[0]['action'][0]
print(f"Action taken: {action}")