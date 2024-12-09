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
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/utils/Mesher.py
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
import open3d as o3d
import skimage
import torch
import trimesh
from packaging import version
from src.utils.datasets import get_dataset
import datetime


class Mesher(object):
    """
    Mesher class.
    Args:
        cfg (dict): configuration dictionary.
        args (argparse.Namespace): arguments.
        eslam (ESLAM): ESLAM object.
        points_batch_size (int): number of points to be processed in each batch.
        ray_batch_size (int): number of rays to be processed in each batch.

    """

    def __init__(self, cfg, args, eslam, points_batch_size=500000, ray_batch_size=100000):
        self.points_batch_size = points_batch_size
        self.ray_batch_size = ray_batch_size
        self.renderer = eslam.renderer
        self.scale = cfg['scale']

        self.resolution = cfg['meshing']['resolution']
        self.level_set = cfg['meshing']['level_set']
        self.mesh_bound_scale = cfg['meshing']['mesh_bound_scale']

        self.bound = eslam.bound
        self.cur_rf_id = eslam.shared_cur_rf_id
        self.verbose = eslam.verbose

        self.marching_cubes_bound = torch.from_numpy(
            np.array(cfg['mapping']['marching_cubes_bound']) * self.scale)

        self.frame_reader = get_dataset(cfg, args, self.scale, device='cpu')
        self.n_img = len(self.frame_reader)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy
        self.embedpos_fn = eslam.embedpos_fn
        self.device = eslam.device

    def get_bound_from_frames(self, keyframe_dict, scale=1):
        """
        Get the scene bound (convex hull),
        using sparse estimated camera poses and corresponding depth images.

        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for keyframe in keyframe_dict:
            c2w = keyframe['est_c2w'].cpu().numpy()
            # convert to open3d camera pose
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])
            depth = keyframe['depth'].cpu().numpy()
            color = keyframe['color'].cpu().numpy()

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return return_mesh

    def eval_points(self, p, all_planes_list, decoders_list, world2rf):
        """
        Evaluates the TSDF and/or color value for the points.
        Args:
            p (torch.Tensor): points to be evaluated, shape (N, 3).
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
        Returns:
            ret (torch.Tensor): the evaluation result, shape (N, 4).
        """
        # print(p.shape)
        p_split = torch.split(p, self.points_batch_size)
        # bound = self.bound[self.cur_rf_id[0]]
        bound = self.bound
        rets = []

        # blended_ret = torch.Tensor()
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            # embed_pos = self.embedpos_fn(pi)
            # ret = decoders(pi, embed_pos, all_planes=all_planes)
            # ret[~mask, -1] = -1
            # rets.append(ret)

            ret_list = []
            for all_planes, decoders in zip(all_planes_list, decoders_list):
                embed_pos = self.embedpos_fn(pi)
                ret = decoders(pi, embed_pos, all_planes=all_planes)  # pi shape,4
                # print('ret', ret.shape)
                ret_list.append(ret)
                # print('ret', len(ret_list))
                # print('world2rf', len(world2rf))
            # print('start_rf_blend', datetime.datetime.now())

            blended_ret = self.blend_ret(pi, ret_list, world2rf)
            # blended_ret = self.rf_blend(pi, ret_list, world2rf)
            # print('finish_rf_blend', datetime.datetime.now())
            blended_ret[~mask, -1] = -1
            rets.append(blended_ret)

        ret = torch.cat(rets, dim=0)
        return ret

    def blend_ret(self, pnts, ret_list, world2rf_point, power=2):
        # 预先计算所有点与 p 之间的距离
        pnts_expanded = pnts.unsqueeze(1)
        # Shape: [len(pnts), 1, 3]
        world2rf_point_expanded = world2rf_point.unsqueeze(0)
        # Shape: [1, len(world2rf_point), 3]
        # distances = torch.norm(pnts_expanded - world2rf_point_expanded, dim=2)
        distances = torch.sum((pnts_expanded - world2rf_point_expanded) ** 2, dim=2)
        # Shape: [len(pnts), len(world2rf_point)]
        # 计算权重
        # weights = 1 / (distances ** power)
        weights = 1 / distances
        # Shape: [len(pnts), len(world2rf_point)]
        # 计算加权的值
        ret_list_stacked = torch.stack(ret_list)
        # Assuming shape: [len(world2rf_point), len(pnts), 4]
        ret_list_stacked_permute = ret_list_stacked.permute(1, 0, 2)
        # print('ret_list_stacked', ret_list_stacked1.shape)
        # print('weights', weights.unsqueeze(2).shape)
        weighted_values = weights.unsqueeze(2) * ret_list_stacked_permute  # [500000, 3, 1] [3, 500000, 4]
        # RuntimeError: The size of tensor a(3) must match the size of tensor b(500000) at non - singleton dimension 1
        # Broadcasting to shape: [len(pnts), len(world2rf_point), 4]
        # 计算总的加权值和总权重
        weight_sums = weighted_values.sum(dim=1)  # Sum over world2rf_point
        total_weights = weights.sum(dim=1).unsqueeze(1)  # Sum over world2rf_point and add a dimension
        # 计算权重平均
        weight_averages = weight_sums / total_weights
        # 结果合并
        blended_ret = weight_averages
        return blended_ret

    def rf_blend(self, pnts, ret_list, world2rf_point, power=2):
        '''
        pnts ([500000, 3])?
        ret_list (rfnums,500000, 4)  include ret for all rfs
        world2rf_point: rf location (x,y,z)
        '''
        # blended_value = 0
        # total_weight = 0

        # total_weight = torch.zeros((1))
        blended_ret = None
        for _, p in enumerate(pnts):
            # print('onepnt', datetime.datetime.now())
            # print('pntslen', len(pnts))#500000
            # print('pnts', pnts.shape)#pnts torch.Size([500000, 3])
            # print('p', p.shape)#p torch.Size([3])
            # print('ret_list', ret_list[0].shape) #ret_list torch.Size([500000, 4])
            # print('ret_listlen', len(ret_list[0])) #ret_listlen 500000
            # print('world2rf_point', world2rf_point.shape)
            weight_sum = torch.zeros(4, device=self.device)
            total_weight = 0
            for i, target_point in enumerate(world2rf_point):
                # print('i', i)
                # print('target_point', target_point)
                # print('ret_list[i][_]', ret_list[i][_])

                # squared_diff = (p - target_point) ** 2
                # distance = np.sqrt(np.sum(squared_diff.cpu().numpy()))
                # weight = 1 / (distance ** power)
                # weight *= distance ** power
                # blended_value += np.sum((ret_list[i][_] * weight).cpu().numpy())
                # total_weight += np.sum(weight)

                # distance = torch.sqrt(torch.sum((p - target_point) ** 2))
                # weight = 1 / (distance ** power)

                distance = torch.sum((p - target_point) ** 2)
                weight = 1 / distance
                weight_sum += ret_list[i][_] * weight  # color+sdf?
                total_weight += weight

            weight_average = weight_sum / total_weight  # current point color+sdf
            # blended_value = torch.tensor(blended_value).unsqueeze(0)
            weight_average = weight_average.unsqueeze(0)
            # print(blended_value.shape)
            if blended_ret is None:
                blended_ret = weight_average.clone()
            else:
                # print(blended_ret)
                # print(weight_average)
                blended_ret = torch.cat((blended_ret, weight_average.clone()), dim=0)

        return blended_ret

    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.marching_cubes_bound

        padding = 0.05

        nsteps_x = ((bound[0][1] - bound[0][0] + 2 * padding) / resolution).round().int().item()
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding, nsteps_x)

        nsteps_y = ((bound[1][1] - bound[1][0] + 2 * padding) / resolution).round().int().item()
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding, nsteps_y)

        nsteps_z = ((bound[2][1] - bound[2][0] + 2 * padding) / resolution).round().int().item()
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding, nsteps_z)

        x_t, y_t, z_t = torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(z).float()
        grid_x, grid_y, grid_z = torch.meshgrid(x_t, y_t, z_t, indexing='xy')
        grid_points_t = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)

        return {"grid_points": grid_points_t, "xyz": [x, y, z]}

    # def get_mesh(self, mesh_out_file, all_planes, decoders, keyframe_dict, device='cuda:0', color=True):

    #self.mesher.get_mesh(mesh_out_file, self.all_planes_list, self.decoders_list, self.world2rf,self.keyframe_dict, self.device)
    def get_mesh(self, mesh_out_file, all_planes_list, decoders_list, world2rf_paramlist, keyframe_dict,
                 device='cuda:0', color=True):
        """
        Get mesh from keyframes and feature planes and save to file.
        Args:
            mesh_out_file (str): output mesh file.
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            keyframe_dict (dict): keyframe dictionary.
            device (str): device to run the model.
            color (bool): whether to use color.
        Returns:
            None

        """
        print('getmesh', datetime.datetime.now())
        decoders_list = [decoders.to(self.device) for decoders in decoders_list]
        all_planes_list = [tuple([plane.to(self.device) for plane in planes] for planes in all_planes) for all_planes in
                           all_planes_list]

        # w2rf = [param.data.tolist() for param in world2rf]
        world2rf = torch.cat([param.data.unsqueeze(0) for param in world2rf_paramlist], dim=0)

        with torch.no_grad():
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            # print('points.shape', points.shape)
            mesh_bound = self.get_bound_from_frames(keyframe_dict, self.scale)

            z = []
            mask = []
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                mask.append(mesh_bound.contains(pnts.cpu().numpy()))
            mask = np.concatenate(mask, axis=0)

            print('start', datetime.datetime.now())
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                z.append(
                    self.eval_points(pnts.to(device), all_planes_list, decoders_list, world2rf).cpu().numpy()[:, -1])
            z = np.concatenate(z, axis=0)
            z[~mask] = -1
            print('sdffinish', datetime.datetime.now())

            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print('marching_cubes error. Possibly no surface extracted from the level set.')
                return

            # convert back to world coordinates
            vertices = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            if color:
                # color is extracted by passing the coordinates of mesh vertices through the network
                points = torch.from_numpy(vertices)
                z = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    z_color = self.eval_points(pnts.to(device).float(), all_planes_list, decoders_list, world2rf).cpu()[
                              ..., :3]
                    z.append(z_color)
                z = torch.cat(z, dim=0)
                vertex_colors = z.numpy()
            else:
                vertex_colors = None

            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh.export(mesh_out_file)
            if self.verbose:
                print('Saved mesh at', mesh_out_file)