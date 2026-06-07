import os
import sys

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parent of the `diffusion_policy` package (third_party/diffusion-policy/diffusion_policy/...).
DIFFUSION_POLICY_ROOT = os.path.join(PROJECT_ROOT_PATH, "third_party", "diffusion-policy")
if os.path.isdir(os.path.join(DIFFUSION_POLICY_ROOT, "diffusion_policy")) and DIFFUSION_POLICY_ROOT not in sys.path:
    sys.path.insert(0, DIFFUSION_POLICY_ROOT)

print(f'PROJECT_ROOT_PATH:{PROJECT_ROOT_PATH}')
