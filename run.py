
import argparse
import json
import os

import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

from src import config
from src.PLGSLAM import PLGSLAM

def main():

    torch.manual_seed(20211202)
    np.random.seed(20211202)
    parser = argparse.ArgumentParser(
        description='Arguments for running PLGSLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/PLGSLAM.yaml')

    save_path = cfg["data"]["output"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'config.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))
    plgslam = PLGSLAM(cfg, args)
    plgslam.run()

if __name__ == '__main__':
    main()
