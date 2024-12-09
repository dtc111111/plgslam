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
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/Tracker.py
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

import torch
import copy
import os
import time

from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples)
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.utils.coordinates import coordinates
import torch.nn.functional as F
class Tracker(object):
    """
    Tracking main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        eslam (ESLAM): ESLAM object
    """
    def __init__(self, cfg, args, eslam):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']

        self.idx = eslam.idx
        self.bound = eslam.bound
        self.mesher = eslam.mesher
        self.output = eslam.output
        self.verbose = eslam.verbose
        self.renderer = eslam.renderer
        self.gt_c2w_list = eslam.gt_c2w_list
        self.mapping_idx = eslam.mapping_idx
        self.mapping_cnt = eslam.mapping_cnt
        # self.shared_decoders = eslam.shared_decoders

        self.estimate_c2w_list = eslam.estimate_c2w_list
        self.truncation = eslam.truncation

        # self.shared_planes_xy = eslam.shared_planes_xy
        # self.shared_planes_xz = eslam.shared_planes_xz
        # self.shared_planes_yz = eslam.shared_planes_yz
        #
        # self.shared_c_planes_xy = eslam.shared_c_planes_xy
        # self.shared_c_planes_xz = eslam.shared_c_planes_xz
        # self.shared_c_planes_yz = eslam.shared_c_planes_yz
    ##########################
        # self.shared_planes_xy_global = eslam.shared_planes_xy_global
        # self.shared_planes_xz_global = eslam.shared_planes_xz_global
        # self.shared_planes_yz_global = eslam.shared_planes_yz_global
        #
        # self.shared_c_planes_xy_global = eslam.shared_c_planes_xy_global
        # self.shared_c_planes_xz_global = eslam.shared_c_planes_xz_global
        # self.shared_c_planes_yz_global = eslam.shared_c_planes_yz_global
    ##########################
        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.w_depth = cfg['tracking']['w_depth']
        self.w_color = cfg['tracking']['w_color']
        self.w_smooth = cfg['mapping']['w_smooth']
        #self.sdf_weight = cfg['mapping']['sdf_weight']
        #self.fs_weight = cfg['mapping']['fs_weight']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=1, pin_memory=True, prefetch_factor=2)

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy

        # self.decoders = copy.deepcopy(self.shared_decoders)
        #
        # self.planes_xy = copy.deepcopy(self.shared_planes_xy)
        # self.planes_xz = copy.deepcopy(self.shared_planes_xz)
        # self.planes_yz = copy.deepcopy(self.shared_planes_yz)
        #
        # self.c_planes_xy = copy.deepcopy(self.shared_c_planes_xy)
        # self.c_planes_xz = copy.deepcopy(self.shared_c_planes_xz)
        # self.c_planes_yz = copy.deepcopy(self.shared_c_planes_yz)
    ##########################
        #init global planes
        # self.planes_xy_global = copy.deepcopy(self.shared_planes_xy_global)
        # self.planes_xz_global = copy.deepcopy(self.shared_planes_xz_global)
        # self.planes_yz_global = copy.deepcopy(self.shared_planes_yz_global)
        #
        # self.c_planes_xy_global = copy.deepcopy(self.shared_c_planes_xy_global)
        # self.c_planes_xz_global = copy.deepcopy(self.shared_c_planes_xz_global)
        # self.c_planes_yz_global = copy.deepcopy(self.shared_c_planes_yz_global)

        self.embedpos_fn = eslam.shared_embedpos_fn
        #self.embedpos_fn = eslam.embedpos_fn

        self.shared_all_planes_list = eslam.shared_all_planes_list
        self.shared_decoders_list = eslam.shared_decoders_list
        self.shared_cur_rf_id = eslam.shared_cur_rf_id
        self.pre_shared_cur_rf_id = self.shared_cur_rf_id[0].clone()
        self.update_para_flag = 1

        #self.all_planes_list = copy.deepcopy(self.shared_all_planes_list)
        self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz = copy.deepcopy(self.shared_all_planes_list[self.shared_cur_rf_id[0]])
        #self.planes_xy, self.planes_xz, self.planes_yz = copy.deepcopy(self.shared_all_planes_list[self.shared_cur_rf_id[0]])

        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = plane.to(self.device)
                #plane.share_memory_()
                planes[i] = plane

        for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
            for i, plane in enumerate(c_planes):
                plane = plane.to(self.device)
                #plane.share_memory_()
                c_planes[i] = plane
        #self.decoders_list = copy.deepcopy(self.shared_decoders_list)

        self.shared_decoders = eslam.shared_decoders_list[self.shared_cur_rf_id[0]]
        self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
     #  # ##########################
        for p in self.decoders.parameters():
            p.requires_grad_(False)


    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def get_masks(self, z_vals, target_d, truncation):
        '''
        Params:
            z_vals: torch.Tensor, (Bs, N_samples)
            target_d: torch.Tensor, (Bs,)
            truncation: float
        Return:
            front_mask: torch.Tensor, (Bs, N_samples)
            sdf_mask: torch.Tensor, (Bs, N_samples)
            fs_weight: float
            sdf_weight: float
        '''

        # before truncation
        # front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
        front_mask = torch.where(z_vals < (target_d - truncation).unsqueeze(1), torch.ones_like(z_vals),
                                 torch.zeros_like(z_vals))

        # after truncation
        back_mask = torch.where(z_vals > (target_d + truncation).unsqueeze(1), torch.ones_like(z_vals),
                                torch.zeros_like(z_vals))
        # valid mask
        depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d))
        # Valid sdf regionn
        # sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask.unsqueeze(1)

        num_fs_samples = torch.count_nonzero(front_mask)
        num_sdf_samples = torch.count_nonzero(sdf_mask)
        num_samples = num_sdf_samples + num_fs_samples
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        return front_mask, sdf_mask, fs_weight, sdf_weight

    def get_sdf_loss(self, z_vals, target_d, predicted_sdf):
        '''
        Params:
            z_vals: torch.Tensor, (Bs, N_samples)
            target_d: torch.Tensor, (Bs,)
            predicted_sdf: torch.Tensor, (Bs, N_samples)
            truncation: float
        Return:
            fs_loss: torch.Tensor, (1,)
            sdf_loss: torch.Tensor, (1,)
        '''
        front_mask, sdf_mask, fs_weight, sdf_weight = self.get_masks(z_vals, target_d, self.truncation)

        fs_loss = F.mse_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask) * fs_weight
        # sdf_loss = F.mse_loss((z_vals + predicted_sdf * self.truncation) * sdf_mask, target_d * sdf_mask) * sdf_weight

        sdf_loss = F.mse_loss((z_vals + predicted_sdf * self.truncation) * sdf_mask,
                              target_d.unsqueeze(1) * sdf_mask) * sdf_weight.unsqueeze(-1)
        return fs_loss, sdf_loss

    def smoothness_losses(self, all_planes, sample_points=256, voxel_size=0.1, margin=0.05,color=False):
        '''
        Smoothness loss of feature grid
        '''

        volume = self.bound[:, 1] - self.bound[:, 0]

        grid_size = (sample_points - 1) * voxel_size
        offset_max = self.bound[:, 1] - self.bound[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cuda:0', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1, 1, 1, 3)).to(volume)) * voxel_size + self.bound[:, 0] + offset

        if self.cfg['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])
        pts_tcnn = pts_tcnn.to(self.device)

        inputs_flat = torch.reshape(pts_tcnn, [-1, pts_tcnn.shape[-1]])
        embed_pos = self.embedpos_fn(inputs_flat)
        sdf = self.decoders.query_sdf(pts_tcnn, embed_pos, all_planes)
        #sdf = self.decoders.query_sdf_embed(pts_tcnn, embed=True)

        tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
        tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
        tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()

        smoothness_loss = (tv_x + tv_y + tv_z) / (sample_points ** 3)

        return smoothness_loss


    def optimize_tracking(self, cam_pose, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        ##########################
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        # all_planes_global = (
        # self.planes_xy_global, self.planes_xz_global, self.planes_yz_global, self.c_planes_xy_global,
        # self.c_planes_xz_global, self.c_planes_yz_global)
        ##########################
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(
            #t = (self.bound[self.shared_cur_rf_id[0]].unsqueeze(0).to(
                device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth
            inside_mask = inside_mask & (batch_gt_depth > 0)

        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]
#######################################
        depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d, batch_rays_o,
                                                                   self.device, self.truncation, gt_depth=batch_gt_depth)
##################################
        ## Filtering the rays for which the rendered depth error is greater than 10 times of the median depth error
        depth_error = (batch_gt_depth - depth.detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)

        #fs_loss, sdf_loss = self.get_sdf_loss(z_vals[depth_mask], batch_gt_depth[depth_mask], sdf[depth_mask])
        #loss = self.sdf_weight * sdf_loss
        #loss += self.fs_weight * fs_loss

        ## SDF losses
        loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

        ## Color Loss
        loss = loss + self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()

        ### Depth loss
        loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

        ## Smoothness loss
        loss = loss + self.w_smooth * self.smoothness_losses(all_planes, self.cfg['training']['smooth_pts'],
                                                             self.cfg['training']['smooth_vox'],
                                                             margin=self.cfg['training'][
                                                                 'smooth_margin'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def update_params_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        """

        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')

            #self.cur_rf_id = self.shared_cur_rf_id[0].clone()
            # self.decoders_list = copy.deepcopy(self.shared_decoders_list)
            # self.decoders = copy.deepcopy(self.shared_decoders)
            # for p in self.decoders.parameters():
            #     p.requires_grad_(False)
            #print(self.shared_cur_rf_id[0])

            #self.decoders = copy.deepcopy(self.shared_decoders_list[self.cur_rf_id])
            #self.decoders.load_state_dict(self.shared_decoders_list[self.shared_cur_rf_id[0]].state_dict())
###########################
            self.decoders.load_state_dict(self.shared_decoders_list[self.shared_cur_rf_id[0]].state_dict())
            ##########################
            #print(len(self.shared_decoders_list))
            #for p in self.decoders.parameters():
                #p.requires_grad_(False)
            #self.decoders = self.decoders.to(self.device)

            # self.decoders = copy.deepcopy(self.shared_decoders)
            # self.decoders = copy.deepcopy(self.shared_decoders_list[self.cur_rf_id])
            # self.decoders = self.decoders.to(self.device)
            # for p in self.decoders.parameters():
            #     p.requires_grad_(False)
            #
            # self.decoders.load_state_dict(self.decoders.state_dict())

            #shared_all_planes = self.shared_all_planes_list[self.shared_cur_rf_id[0]]
        ##########################
            shared_all_planes = self.shared_all_planes_list[self.shared_cur_rf_id[0]]
            self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz, self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz = shared_all_planes
            #self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz = shared_all_planes

            ##########################
            #print(len(self.shared_all_planes_list))
            # all_planes = self.shared_all_planes_list[self.cur_rf_id]
            # self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz = all_planes

            for planes, self_planes in zip(
                    [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
                    [self.planes_xy, self.planes_xz, self.planes_yz]):
                for i, plane in enumerate(planes):
                    self_planes[i] = plane.detach().to(self.device)

            # for c_planes, self_c_planes in zip(
            #         [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
            #         [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]):
            #     for i, c_plane in enumerate(c_planes):
            #         self_c_planes[i] = c_plane.detach().to(self.device)
            # list中的self.shared_planes_xy不会被移到gpu吗
         ##########################
            # # global planes
            # for planes_global, self_planes_global in zip(
            #         [self.shared_planes_xy_global, self.shared_planes_xz_global, self.shared_planes_yz_global],
            #         [self.planes_xy_global, self.planes_xz_global, self.planes_yz_global]):
            #     for i, plane_global in enumerate(planes_global):
            #         self_planes_global[i] = plane_global.detach()
            #
            # for c_planes_global, self_c_planes_global in zip(
            #         [self.shared_c_planes_xy_global, self.shared_c_planes_xz_global, self.shared_c_planes_yz_global],
            #         [self.c_planes_xy_global, self.c_planes_xz_global, self.c_planes_yz_global]):
            #     for i, c_plane_global in enumerate(c_planes_global):
            #         self_c_planes_global[i] = c_plane_global.detach()
         ##########################
            self.prev_mapping_idx = self.mapping_idx[0].clone()


    def run(self):
        """
            Runs the tracking thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        device = self.device
    ##########################
        #all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        # all_planes_global = (self.planes_xy_global, self.planes_xz_global, self.planes_yz_global, self.c_planes_xy_global,
        # self.c_planes_xz_global, self.c_planes_yz_global)
    ##########################
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            # initiate mapping every self.every_frame frames
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

         ##########################
                self.update_para_flag = 1

            if self.pre_shared_cur_rf_id != self.shared_cur_rf_id[0]:
                self.update_para_flag = 0

            if self.update_para_flag == 1:
                self.update_params_from_mapping()

            #self.update_params_from_mapping()
            self.pre_shared_cur_rf_id = self.shared_cur_rf_id[0].clone()

            all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
         ##########################
            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
            #if idx <= 1500 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
        ##########################
                    self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)
        ##########################
            else:
                if self.const_speed_assumption and idx - 2 >= 0:
                    ## Linear prediction for initialization
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_cam_pose(pre_poses)
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
                else:
                    ## Initialize with the last known pose
                    cam_pose = matrix_to_cam_pose(pre_c2w)

                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:, :4].clone())
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                     {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

                current_min_loss = torch.tensor(float('inf')).float().to(device)
                for cam_iter in range(self.num_cam_iters):
                    cam_pose = torch.cat([R, T], -1)
            ##########################
                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)
             ##########################
                    loss = self.optimize_tracking(cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose = cam_pose.clone().detach()

                c2w = cam_pose_to_matrix(candidate_cam_pose)

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            self.idx[0] = idx


