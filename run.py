import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from src import config
from src.PLGSLAM import PLGSLAM

def main():
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

    plgslam = PLGSLAM(cfg, args)

    plgslam.run()

if __name__ == '__main__':
    main()
