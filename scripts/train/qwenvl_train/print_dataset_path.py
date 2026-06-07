#!/usr/bin/env python3
# Run:
#   export INTERNAV_R2R_DATA_PATH="${PWD}/data/InternData-N1/vln_ce/traj_data/r2r"
#   export VLN_DATASETS="r2r_125cm_0_30%10"
#   python scripts/train/qwenvl_train/print_dataset_path.py
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO_ROOT)

from internnav.dataset.internvla_n1_lerobot_dataset import data_list, debug_dataset_path


def main():
    vln_datasets = os.environ.get("VLN_DATASETS", "r2r_125cm_0_30%10")
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"VLN_DATASETS: {vln_datasets}")
    print(f"INTERNAV_R2R_DATA_PATH: {os.environ.get('INTERNAV_R2R_DATA_PATH', '(not set)')}")
    print(f"INTERNAV_RXR_DATA_PATH: {os.environ.get('INTERNAV_RXR_DATA_PATH', '(not set)')}")
    print(f"INTERNAV_SCALEVLN_DATA_PATH: {os.environ.get('INTERNAV_SCALEVLN_DATA_PATH', '(not set)')}")
    print()

    for cfg in data_list(vln_datasets.split(",")):
        setting = f"{cfg['height']}cm_{cfg['pitch_2']}deg"
        print(f"--- {cfg} ---")
        debug_dataset_path(cfg["data_path"], setting)


if __name__ == "__main__":
    main()
