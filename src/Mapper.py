
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import os
import time
import random

from colorama import Fore, Style

from src.common import (get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix)
from src.utils.datasets import get_dataset, SeqSampler
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.tools.cull_mesh import cull_mesh
from src.utils.coordinates import coordinates
from src.common import normalize_3d_coordinate
from src import config
import copy
from src.keyframe import KeyFrameDatabase
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
import torch.nn.functional as F
torch.cuda.device_count()



class Mapper(object):
    """
    Mapping main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
    """

    def __init__(self, cfg, args, plgslam):

        self.cfg = cfg
        self.args = args

        self.idx = plgslam.idx
        self.truncation = plgslam.truncation
        self.bound = plgslam.bound
        self.logger = plgslam.logger
        self.mesher = plgslam.mesher
        self.output = plgslam.output
        self.verbose = plgslam.verbose
        self.renderer = plgslam.renderer
        self.mapping_idx = plgslam.mapping_idx
        self.mapping_cnt = plgslam.mapping_cnt

        self.estimate_c2w_list = plgslam.estimate_c2w_list
        self.mapping_first_frame = plgslam.mapping_first_frame

        self.scale = cfg['scale']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keyframe_device = cfg['keyframe_device']

        self.eval_rec = cfg['meshing']['eval_rec']
        self.joint_opt = False  # Even if joint_opt is enabled, it starts only when there are at least 4 keyframes
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr']  # The learning rate for camera poses during mapping
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.every_frame = cfg['mapping']['every_frame']
        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']
        self.w_sdf_center = cfg['mapping']['w_sdf_center']
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']
        self.w_depth = cfg['mapping']['w_depth']
        self.w_color = cfg['mapping']['w_color']
        self.w_smooth = cfg['mapping']['w_smooth']
        #self.sdf_weight = cfg['mapping']['sdf_weight']
        #self.fs_weight = cfg['mapping']['fs_weight']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']

        self.keyframe_dict = []
        # self.keyframe_loop_dict = []
        self.keyframe_list = []
        self.mapper_keyframe_dict = []
        self.shared_keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=True,
                                       prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'],
                                           inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = plgslam.H, plgslam.W, plgslam.fx, plgslam.fy, plgslam.cx, plgslam.cy
        self.embedpos_fn = plgslam.shared_embedpos_fn
        self.world2rf = torch.nn.ParameterList()  # 需要torch.nn.ParameterList()吗？
        # self.all_planes_list = []
        self.all_planes_list = plgslam.shared_all_planes_list

        # self.decoders_list = []
        self.decoders_list = plgslam.shared_decoders_list
        self.active_rf_ids = []
        self.cur_rf_id = plgslam.shared_cur_rf_id
        self.append_rf()
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.total_pixels = (self.H - self.crop_size * 2) * (self.W - self.crop_size * 2)
        self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])
        self.keyframeDatabase = self.create_kf_database(cfg)

    def create_kf_database(self, config):
        '''
        Create the keyframe database
        '''
        num_kf = int(self.n_img // self.cfg['mapping']['gb_keyframe'] + 1)
        print('#kf:', num_kf)
        print('#Pixels to save:', self.num_rays_to_save)
        return KeyFrameDatabase(config,
                                self.H,
                                self.W,
                                num_kf,
                                self.num_rays_to_save,
                                self.device)

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
        #front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
        front_mask = torch.where(z_vals < (target_d - truncation).unsqueeze(1), torch.ones_like(z_vals),
                                 torch.zeros_like(z_vals))

        # after truncation
        back_mask = torch.where(z_vals > (target_d + truncation).unsqueeze(1), torch.ones_like(z_vals), torch.zeros_like(z_vals))
        # valid mask
        depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d))
        # Valid sdf regionn
        #sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

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
        #sdf_loss = F.mse_loss((z_vals + predicted_sdf * self.truncation) * sdf_mask, target_d * sdf_mask) * sdf_weight

        sdf_loss = F.mse_loss((z_vals + predicted_sdf * self.truncation) * sdf_mask,
                              target_d.unsqueeze(1) * sdf_mask) * sdf_weight.unsqueeze(-1)

        return fs_loss, sdf_loss

    def smoothness_losses(self, all_planes, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
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
        # sdf = self.decoders.query_sdf_embed(pts_tcnn, embed=True)

        tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
        tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
        tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()

        smoothness_loss = (tv_x + tv_y + tv_z) / (sample_points ** 3)

        return smoothness_loss

    def init_all_planes(self, cfg):
        """
        Initialize the feature planes.

        Args:
            cfg (dict): parsed config dict.
        """
        self.coarse_planes_res = cfg['planes_res']['coarse']
        self.fine_planes_res = cfg['planes_res']['fine']

        self.coarse_c_planes_res = cfg['c_planes_res']['coarse']
        self.fine_c_planes_res = cfg['c_planes_res']['fine']

        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1] - self.bound[:, 0]
        # xyz_len = self.bound[self.cur_rf_id[0]][:, 1]-self.bound[self.cur_rf_id[0]][:, 0]
        # if planes_type == 'global_plane':
        #     planes_res = self.coarse_planes_res
        #     c_planes_res = self.coarse_c_planes_res
        #
        # elif planes_type == 'local_plane':
        #     planes_res = self.fine_planes_res
        #     c_planes_res = self.fine_c_planes_res
        ####### Initializing Planes ############
        planes_xy, planes_xz, planes_yz = [], [], []
        c_planes_xy, c_planes_xz, c_planes_yz = [], [], []
        planes_res = [self.coarse_planes_res, self.fine_planes_res]
        c_planes_res = [self.coarse_c_planes_res, self.fine_c_planes_res]

        planes_dim = c_dim
        #
        # grid_shape = list(map(int, (xyz_len / planes_res).tolist()))
        # grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
        # planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
        # planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
        # planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))
        #
        # grid_shape = list(map(int, (xyz_len / c_planes_res).tolist()))
        # grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
        # c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
        # c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
        # c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        for grid_res in planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        for grid_res in c_planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        for planes in [planes_xy, planes_xz, planes_yz]:
            for i, plane in enumerate(planes):
                plane = plane.to(self.device)
                # plane.share_memory_()
                planes[i] = plane

        for c_planes in [c_planes_xy, c_planes_xz, c_planes_yz]:
            for i, plane in enumerate(c_planes):
                plane = plane.to(self.device)
                # plane.share_memory_()
                c_planes[i] = plane

        return (planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz)

    def append_rf(self):
        cfg = self.cfg

        if len(self.decoders_list) > 0:
            # world2rf = -self.cur_t_c2w.clone().detach()
            world2rf = self.cur_t_c2w.clone().detach()
            # self.tensorfs[-1].to(torch.device("cpu"))
            # torch.cuda.empty_cache()
        else:
            # world2rf = torch.zeros(3, device=self.device)
            idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
            self.cur_t_c2w = gt_c2w[:3, 3]
            world2rf = self.cur_t_c2w.clone().detach().to(self.device)
            # self.cur_rf_id = 0
            # world2rf = -self.cur_t_c2w.clone().detach().to(self.device)

        model = config.get_model(cfg)
        model = model.to(self.device)
        model.bound = self.bound

        # model.bound = self.bound[self.cur_rf_id[0]]
        # model.bound = self.bound[self.cur_rf_id[0].clone()]

        decoder_on_cpu = model.to("cpu")

        self.decoders_list.append(decoder_on_cpu)

        all_planes = self.init_all_planes(cfg)

        all_planes_on_cpu = tuple([plane.to('cpu').detach() for plane in planes] for planes in all_planes)

        self.all_planes_list.append(all_planes_on_cpu)

        self.active_rf_ids.append(self.cur_rf_id[0].clone())
        self.world2rf.append(torch.nn.Parameter(world2rf.clone().detach()))
        # print('self.world2rf', self.world2rf)
        # world2rf_tensor = torch.cat([param.data.unsqueeze(0) for param in self.world2rf], dim=0)
        # print('world2rf_tensor', world2rf_tensor)
        # deactivate keyframe
        # keyframe_list = copy.deepcopy(self.keyframe_list)
        # print(self.keyframe_list)
        # self.keyframe_list = []
        # for idx in keyframe_list:
        #     if keyframe_list[-1]-idx < 2000:
        #         self.keyframe_list.append(idx)
        # print(self.keyframe_list)

        # self.decoders = self.decoders_list[self.cur_rf_id].to(self.device)
        # all_planes = self.all_planes_list[self.cur_rf_id]
        # self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz = all_planes

    def select_rf(self):
        cfg = self.cfg
        can_add_rf = False

        # dist_debug = torch.norm(self.cur_t_c2w + self.world2rf[self.cur_rf_id[0]])
        # dist_debug = torch.norm(self.cur_t_c2w - self.world2rf[self.cur_rf_id[0]], p=float('inf'))
        # dist_debug = torch.norm(self.cur_t_c2w - self.world2rf[self.cur_rf_id[0]])

        cur_dist = torch.max(torch.abs(self.cur_t_c2w - self.world2rf[self.cur_rf_id[0]]))

        # if torch.norm(self.cur_t_c2w - self.world2rf[self.cur_rf_id[0]], p=float('inf')) > cfg['mapping']['max_drift']:  # 超过当前场的范围
        # if torch.norm(self.cur_t_c2w + self.world2rf[self.cur_rf_id[0]]) > cfg['mapping']['max_drift']:  # 超过当前场的范围

        if cur_dist > cfg['mapping']['max_drift']:  # 超过当前场的范围

            print("progress!")
            # print('self.cur_t_c2w', self.cur_t_c2w)
            # print('self.world2rf[self.cur_rf_id[0]]', self.world2rf[self.cur_rf_id[0]])
            # print('dist_new_one', torch.norm(self.cur_t_c2w - self.world2rf[self.cur_rf_id[0]]))
            can_add_rf = True

            for rf_id in self.active_rf_ids:
                # dist = torch.norm(self.cur_t_c2w - self.world2rf[rf_id], p=float('inf'))

                # dist = torch.norm(self.cur_t_c2w - self.world2rf[rf_id])
                dist = torch.max(torch.abs(self.cur_t_c2w - self.world2rf[rf_id]))
                # dist = torch.norm(self.cur_t_c2w + self.world2rf[rf_id])
                print('dist_old_one:', dist)
                # if torch.norm(self.cur_t_c2w - self.world2rf[rf_id], p=float('inf')) <= cfg['mapping']['max_drift']:  # 在某个场的范围内
                if dist <= cfg['mapping']['max_drift']:
                    # if torch.norm(self.cur_t_c2w + self.world2rf[rf_id]) <= cfg['mapping']['max_drift']:  # 在某个场的范围内
                    print('old one!', rf_id)
                    self.cur_rf_id[0] = rf_id  # 激活并优化当前self.world2rf[i]对应的decoder
                    can_add_rf = False
                    break

        if can_add_rf:
            self.cur_rf_id[0] = len(self.active_rf_ids)
            print('add rf!', self.cur_rf_id[0])

        return can_add_rf

    def pose_distance_threshold(self, threshold):
        result = []
        for idx in self.keyframe_list:
            # dist = torch.norm(self.estimate_c2w_list[idx][:3, 3] - self.world2rf[self.cur_rf_id[0]], p=float('inf'))
            # dist = torch.norm(self.estimate_c2w_list[idx][:3, 3] - self.world2rf[self.cur_rf_id[0]])
            dist = torch.max(torch.abs(self.estimate_c2w_list[idx][:3, 3] - self.world2rf[self.cur_rf_id[0]]))
            result.append(dist <= threshold)
        return result

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, num_keyframes, num_samples=8, num_rays=50):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color: ground truth color image of the current frame.
            gt_depth: ground truth depth image of the current frame.
            c2w: camera to world matrix for target view (3x4 or 4x4 both fine).
            num_keyframes (int): number of overlapping keyframes to select.
            num_samples (int, optional): number of samples/points per ray. Defaults to 8.
            num_rays (int, optional): number of pixels to sparsely sample
                from each image. Defaults to 50.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """

        cfg = self.cfg
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy,
            c2w.unsqueeze(0), gt_depth.unsqueeze(0), gt_color.unsqueeze(0), device)

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth * 0.8
        far = gt_depth + 0.5
        z_vals = near * (1. - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [num_rays, num_samples, 3]
        pts = pts.reshape(1, -1, 3)

        result = self.pose_distance_threshold(cfg['mapping']['max_drift'])
        # print(result)

        keyframes_c2ws = torch.stack(
            [
                self.estimate_c2w_list[idx]
                for _, idx in enumerate(self.keyframe_list)
                if result[_]
            ],
            dim=0
        )
        # keyframes_c2ws = torch.stack([self.estimate_c2w_list[idx] for idx in keyframe_list], dim=0)

        w2cs = torch.inverse(keyframes_c2ws[:-2])  ## The last two keyframes are already included

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

        ## Considering only overlapped frames
        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)
        rnd_inds = torch.randperm(selected_keyframes.shape[0])
        selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

        selected_keyframes = list(selected_keyframes.cpu().numpy())

        return selected_keyframes


    def optimize_mapping(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict,
                         keyframe_list, cur_c2w, current_rays, poses_all):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if joint_opt enables).

        Args:
            iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): a list of dictionaries of keyframes info.
            keyframe_list (list): list of keyframes indices.
            cur_c2w (tensor): the estimated camera to world matrix of current frame.

        Returns:
            cur_c2w: return the updated cur_c2w, return the same input cur_c2w if no joint_opt
        """

        all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        # all_planes_global = (self.planes_xy_global, self.planes_xz_global, self.planes_yz_global, self.c_planes_xy_global,
        # self.c_planes_xz_global, self.c_planes_yz_global)

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = random_select(len(self.keyframe_dict) - 2, self.mapping_window_size - 1)
            elif self.keyframe_selection_method == 'overlap':
                optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w,
                                                                 self.mapping_window_size - 1)

        # add the last two keyframes and the current frame(use -1 to denote)
        if len(keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(keyframe_list) - 1] + [len(keyframe_list) - 2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]  ## -1 represents the current frame

        pixs_per_image = self.mapping_pixels // len(optimize_frame)

        decoders_para_list = []
        decoders_para_list += list(self.decoders.parameters())  # Need to append?

        planes_para = []
        c_planes_para = []

        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = nn.Parameter(plane)
                planes_para.append(plane)
                planes[i] = plane

        for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
            for i, c_plane in enumerate(c_planes):
                c_plane = nn.Parameter(c_plane)
                c_planes_para.append(c_plane)
                c_planes[i] = c_plane

        gt_depths = []
        gt_colors = []
        c2ws = []
        gt_c2ws = []
        for frame in optimize_frame:  # g et gt values
            # the oldest frame should be fixed to avoid drifting
            if frame != -1:
                gt_depths.append(keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(keyframe_dict[frame]['color'].to(device))
                c2ws.append(keyframe_dict[frame]['est_c2w'])
                gt_c2ws.append(keyframe_dict[frame]['gt_c2w'])
            else:
                gt_depths.append(cur_gt_depth)
                gt_colors.append(cur_gt_color)
                c2ws.append(cur_c2w)
                gt_c2ws.append(gt_cur_c2w)
        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        c2ws = torch.stack(c2ws, dim=0)

        if self.joint_opt:
            cam_poses = nn.Parameter(matrix_to_cam_pose(c2ws[1:]))

            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},  # Need to modify
                                          {'params': planes_para, 'lr': 0},
                                          {'params': c_planes_para, 'lr': 0},
                                          # {'params': planes_para_global, 'lr': 0},
                                          # {'params': c_planes_para_global, 'lr': 0},
                                          {'params': [cam_poses], 'lr': 0}])

        else:
            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                          {'params': planes_para, 'lr': 0},
                                          {'params': c_planes_para, 'lr': 0},
                                          # {'params': planes_para_global, 'lr': 0},
                                          # {'params': c_planes_para_global, 'lr': 0}
                                          ])

        optimizer.param_groups[0]['lr'] = cfg['mapping']['lr']['decoders_lr'] * lr_factor
        optimizer.param_groups[1]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor
        optimizer.param_groups[2]['lr'] = cfg['mapping']['lr']['c_planes_lr'] * lr_factor
        # optimizer.param_groups[3]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor
        # optimizer.param_groups[4]['lr'] = cfg['mapping']['lr']['c_planes_lr'] * lr_factor

        if self.joint_opt:
            optimizer.param_groups[3]['lr'] = self.joint_opt_cam_lr

        for joint_iter in range(iters):
            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes,
                                          self.decoders)

            if self.joint_opt:
                ## We fix the oldest c2w to avoid drifting
                c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(cam_poses)], dim=0)
            else:
                c2ws_ = c2ws

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, device)

            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(
                    # t = (self.bound[self.cur_rf_id[0]].unsqueeze(0).to(
                    device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]


            if idx % self.cfg['mapping']['gb_keyframe'] == 0:
                self.shared_keyframe_list.append(idx)
                self.mapper_keyframe_dict.append(
                    {'rays_o': batch_rays_o.to('cpu'), 'gt_depth': batch_gt_depth.to('cpu'),
                     'rays_d': batch_rays_d.to('cpu')})

            depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d,
                                                                       batch_rays_o, device,
                                                                       self.truncation, gt_depth=batch_gt_depth)

            depth_mask = (batch_gt_depth > 0)
            # print("depth_mask",depth_mask.shape)
            # print("batch_gt_depth", batch_gt_depth.shape,"sdf", sdf.shape,"z_vals", z_vals.shape)
            loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

            ## Color loss
            loss = loss + self.w_color * torch.square(batch_gt_color - color).mean()

            ### Depth loss
            loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

            ## Smoothness loss
            loss = loss + self.w_smooth * self.smoothness_losses(all_planes, self.cfg['training']['smooth_pts'],
            self.cfg['training']['smooth_vox'],
            margin = self.cfg['training'][
                'smooth_margin'])

            optimizer.zero_grad()
            loss.backward(retain_graph=False)  # retain_graph=True
            optimizer.step()

        if self.joint_opt:
            # put the updated camera poses back
            optimized_c2ws = cam_pose_to_matrix(cam_poses.detach())

            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]

        return cur_c2w



    def axis_angle_to_matrix(self, data):
        batch_dims = data.shape[:-1]

        theta = torch.norm(data, dim=-1, keepdim=True)
        omega = data / theta

        omega1 = omega[..., 0:1]
        omega2 = omega[..., 1:2]
        omega3 = omega[..., 2:3]
        zeros = torch.zeros_like(omega1)

        K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[..., None, :],
                          torch.concat([omega3, zeros, -omega1], dim=-1)[..., None, :],
                          torch.concat([-omega2, omega1, zeros], dim=-1)[..., None, :]], dim=-2)
        I = torch.eye(3).expand(*batch_dims, 3, 3).to(data)

        return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

    def matrix_from_tensor(self, rot, trans):
        """
        :param rot: axis-angle [bs, 3]
        :param trans: translation vector[bs, 3]
        :return: transformation matrix [b, 4, 4]
        """
        bs = rot.shape[0]
        T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
        R = self.axis_angle_to_matrix(rot)
        T[:, :3, :3] = R
        T[:, :3, 3] = trans
        return T

    def get_camera_rays(self, H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
        """Get ray origins, directions from a pinhole camera."""
        #  ----> i
        # |
        # |
        # X
        # j
        i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                              torch.arange(H, dtype=torch.float32), indexing='xy')

        # View direction (X, Y, Lambda) / lambda
        # Move to the center of the screen
        #  -------------
        # |      y      |
        # |      |      |
        # |      .-- x  |
        # |             |
        # |             |
        #  -------------

        if cx is None:
            cx, cy = 0.5 * W, 0.5 * H

        if fy is None:
            fy = fx
        if type is 'OpenGL':
            dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)
        elif type is 'OpenCV':
            dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)
        else:
            raise NotImplementedError()

        rays_d = dirs
        return rays_d

    def matrix_to_axis_angle(self, rot):
        """
        :param rot: [N, 3, 3]
        :return:
        """
        return quaternion_to_axis_angle(matrix_to_quaternion(rot))

    def run(self):
        """
            Runs the mapping thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        cfg = self.cfg
        batch = {}
        current_rays = torch.tensor(0.)
        poses_all = torch.tensor(0.)
        # all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        # all_planes_global = (self.planes_xy_global, self.planes_xz_global, self.planes_yz_global, self.c_planes_xy_global, self.c_planes_xz_global, self.c_planes_yz_global)

        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        data_iterator = iter(self.frame_loader)
        batch['rgb'] = gt_color.unsqueeze(0).to(self.device, non_blocking=True)
        batch['depth'] = gt_depth.unsqueeze(0).to(self.device, non_blocking=True)
        batch['frame_id'] = idx
        # print(batch['frame_id'])
        self.rays_d = self.get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy).unsqueeze(0)
        batch['direction'] = self.rays_d.to(self.device, non_blocking=True)

        ## Fixing the first camera pose
        self.estimate_c2w_list[0] = gt_c2w

        init_phase = True
        prev_idx = -1
        while True:
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img - 1:  ## Last input frame
                    break

                if idx % self.every_frame == 0 and idx != prev_idx:
                    break

                time.sleep(0.001)

            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = next(data_iterator)
            gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
            gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
            gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

            cur_c2w = self.estimate_c2w_list[idx]
            self.cur_t_c2w = cur_c2w[:3, 3]

            if not init_phase:
                lr_factor = cfg['mapping']['lr_factor']
                iters = cfg['mapping']['iters']
            else:
                lr_factor = cfg['mapping']['lr_first_factor']
                iters = cfg['mapping']['iters_first']

            ## Deciding if camera poses should be jointly optimized
            self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']

            can_add_rf = self.select_rf()  # get self.cur_rf_id
            if can_add_rf:
                self.append_rf()
                iters = cfg['mapping']['iters_first']

            self.decoders = self.decoders_list[self.cur_rf_id[0]].to(self.device)

            all_planes_on_cpu = self.all_planes_list[self.cur_rf_id[0]]
            all_planes = tuple(
                [plane.to(self.device).requires_grad_() for plane in planes] for planes in all_planes_on_cpu)
            self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz = all_planes

            cur_c2w = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w,
                                            self.keyframe_dict, self.keyframe_list, cur_c2w, current_rays, poses_all)

            decoder_on_cpu = self.decoders.to("cpu")
            self.decoders_list[self.cur_rf_id[0]] = decoder_on_cpu

            all_planes_on_cpu = tuple([plane.to('cpu').detach() for plane in planes] for planes in all_planes)

            self.all_planes_list[self.cur_rf_id[0]] = all_planes_on_cpu


            if self.joint_opt:
                self.estimate_c2w_list[idx] = cur_c2w

            # add new frame to keyframe set
            if idx % self.keyframe_every == 0:
                self.keyframe_list.append(idx)
                self.keyframe_dict.append({'gt_c2w': gt_c2w, 'idx': idx, 'color': gt_color.to(self.keyframe_device),
                                           'depth': gt_depth.to(self.keyframe_device), 'est_c2w': cur_c2w.clone()})

            init_phase = False
            self.mapping_first_frame[0] = 1  # mapping of first frame is done, can begin tracking

            if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img - 1:
                self.logger.log(idx, self.keyframe_list, self.keyframe_dict)

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'

                self.mesher.get_mesh(mesh_out_file, self.all_planes_list, self.decoders_list, self.world2rf,
                                     self.keyframe_dict, self.device)

                cull_mesh(mesh_out_file, self.cfg, self.args, self.device,
                          estimate_c2w_list=self.estimate_c2w_list[:idx + 1])

            if idx == self.n_img - 1:
                if self.eval_rec:
                    mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                else:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'

                self.mesher.get_mesh(mesh_out_file, self.all_planes_list, self.decoders_list, self.world2rf,
                                     self.keyframe_dict, self.device)

                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)

                break

            if idx == self.n_img - 1:
                break