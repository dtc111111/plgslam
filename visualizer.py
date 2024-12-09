# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/visualizer.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import numpy as np
import torch

import argparse
import os
import time
import glob
import trimesh
from tqdm import tqdm

from src import config
from src.tools.visualizer_util import SLAMFrontend

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments to visualize the SLAM process.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `save_imgs.mp4` in output folder ')
    parser.add_argument('--top_view',
                        action='store_true', help='Setting the camera to top view. Otherwise, the camera is at the first frame\'s pose.')
    parser.add_argument('--no_gt_traj', action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/PLGSLAM.yaml')
    scale = cfg['scale']
    mesh_resolution = cfg['meshing']['resolution']
    if mesh_resolution <= 0.01:
        wait_time = 0.25
    else:
        wait_time = 0.1
    output = cfg['data']['output'] if args.output is None else args.output
    ckptsdir = f'{output}/ckpts'


    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            N = ckpt['idx']
            estimate_c2w_list[:, :3, 3] /= scale
            gt_c2w_list[:, :3, 3] /= scale
            estimate_c2w_list = estimate_c2w_list.cpu().numpy()
            gt_c2w_list = gt_c2w_list.cpu().numpy()

            ## Setting view point ##  f'/data0/sgl/ESLAM-main/output/Replica/room0_dis/0505_4_0526/mesh/final_mesh_eval_rec.ply'
            meshfile = sorted(glob.glob(f'{output}/mesh/*.ply'))[-1]
            if os.path.isfile(meshfile):
                if args.top_view:
                    # get the latest .ply file in the "mesh" folder and use it to set the view point
                    mesh = trimesh.load(meshfile, process=False)
                    to_origin, _ = trimesh.bounds.oriented_bounds(mesh, ordered=False)
                    init_pose = np.eye(4)
                    init_pose = np.linalg.inv(to_origin) @ init_pose
                else:
                    init_pose = gt_c2w_list[0].copy()

                frontend = SLAMFrontend(output, init_pose=init_pose, cam_scale=0.2,
                                        save_rendering=args.save_rendering, near=0,
                                        estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list)
                frontend.start()

                ## Visualize the trajectory ##
                for i in tqdm(range(0, N+1)):
                    meshfile = f'{output}/mesh/{i:05d}_mesh_culled.ply'
                    if os.path.isfile(meshfile):
                        frontend.update_mesh(meshfile)
                    frontend.update_pose(1, estimate_c2w_list[i], gt=False)
                    if not args.no_gt_traj:
                        frontend.update_pose(1, gt_c2w_list[i], gt=True)
                    if i % 10 == 0:
                        frontend.update_cam_trajectory(i, gt=False)
                        if not args.no_gt_traj:
                            frontend.update_cam_trajectory(i, gt=True)
                    time.sleep(wait_time)

                frontend.terminate()
                time.sleep(1)

                if args.save_rendering:
                    time.sleep(1)
                    os.system(f"/home/sgl/miniconda3/envs/eslam/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")
                    #/home/wjy/anaconda3/envs/eslam/bin/ffmpeg

    checkpoint_filename = 'checkpoint5577.pt'
    ckptsdir = f'{output}/{checkpoint_filename}'

    if os.path.exists(ckptsdir):

        gt_ckpt_path = '/data0/sgl/ESLAM-main_original_version/output/scannet/scans/scene0000_00/ckpts/05577.tar'  # 6001
        # scene0000_00/05577 scene0059_00/01806  scene0106_00/02323 scene0169_00/02033 scene0181_00/02348 scene0207_00/01987
        gt_ckpt = torch.load(gt_ckpt_path, map_location=torch.device('cpu'))
        ckpt = torch.load(ckptsdir)
        N = gt_ckpt['idx']
        gt_c2w_list = gt_ckpt['gt_c2w_list']
        estimate_c2w_dict = ckpt['pose']
        estimate_c2w_list = torch.stack([estimate_c2w_dict[i] for i in range(N + 1)], dim=0)



        estimate_c2w_list[:, :3, 3] /= scale
        gt_c2w_list[:, :3, 3] /= scale
        estimate_c2w_list = estimate_c2w_list.cpu().numpy()
        gt_c2w_list = gt_c2w_list.cpu().numpy()
        # print(gt_c2w_list.shape)
        #
        # trans = np.array([[-0.14031718, -0.875229, -0.46290958, 0.75258389],
        #                   [0.217254, -0.48335774, 0.84803655, 0.32966271],
        #                   [-0.96597712, 0.01842514, 0.2579704, 3.28585226],
        #                   [0., 0., 0., 1.]])
        #
        # gt_c2w_list = np.einsum('ij,kjl->kil', trans, gt_c2w_list)


        ## Setting view point ##
        meshfile = f'{output}/mesh_track5577.ply'
        if os.path.isfile(meshfile):
            if args.top_view:
                # get the latest .ply file in the "mesh" folder and use it to set the view point
                mesh = trimesh.load(meshfile, process=False)
                to_origin, _ = trimesh.bounds.oriented_bounds(mesh, ordered=False)
                init_pose = np.eye(4)
                init_pose = np.linalg.inv(to_origin) @ init_pose

            else:
                init_pose = gt_c2w_list[0].copy()

            frontend = SLAMFrontend(output, init_pose=init_pose, cam_scale=0.2,
                                    save_rendering=args.save_rendering, near=0,
                                    estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list)
            frontend.start()

            ## Visualize the trajectory ##
            for i in tqdm(range(0, N+1)):
                meshfile = f'{output}/mesh_track{i}.ply'
                if os.path.isfile(meshfile):
                    frontend.update_mesh(meshfile)
                frontend.update_pose(1, estimate_c2w_list[i], gt=False)
                if not args.no_gt_traj:
                    frontend.update_pose(1, gt_c2w_list[i], gt=True)
                if i % 10 == 0:
                    frontend.update_cam_trajectory(i, gt=False)
                    if not args.no_gt_traj:
                        frontend.update_cam_trajectory(i, gt=True)
                time.sleep(wait_time)

            frontend.terminate()
            time.sleep(1)

            if args.save_rendering:
                time.sleep(1)
                os.system(f"/home/sgl/miniconda3/envs/eslam/bin/ffmpeg -f image2 -r 90 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")
                #/home/wjy/anaconda3/envs/eslam/bin/ffmpeg