

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate
import tinycudann as tcnn


class ColorNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=15,
                 hidden_dim_color=64, num_layers_color=3):
        super(ColorNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_color = num_layers_color

        self.model = self.get_model(config['decoder']['tcnn_network'])

    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                # dtype=torch.float
            )

        color_net = []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color

            if l == self.num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = self.hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))


class SDFNet(nn.Module):
    def __init__(self, config, input_ch=3, geo_feat_dim=15, hidden_dim=64, num_layers=2):
        super(SDFNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.model = self.get_model(tcnn_network=config['decoder']['tcnn_network'])

    def forward(self, x, return_geo=True):
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                # dtype=torch.float
            )
        else:
            sdf_net = []
            for l in range(self.num_layers):
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim

                if l == self.num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
                else:
                    out_dim = self.hidden_dim

                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))

class Decoders(nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
        learnable_beta: whether to learn beta

    """

    def __init__(self, config, input_ch=3, input_ch_pos=12, learnable_beta=True):
    #def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(Decoders, self).__init__()
        self.config = config
        self.color_net = ColorNet(config,
                                  input_ch=input_ch + input_ch_pos,
                                  #input_ch=input_ch_pos,
                                  geo_feat_dim=config['decoder']['geo_feat_dim'],
                                  hidden_dim_color=config['decoder']['hidden_dim_color'],
                                  num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                              input_ch=input_ch + input_ch_pos,
                              geo_feat_dim=config['decoder']['geo_feat_dim'],
                              hidden_dim=config['decoder']['hidden_dim'],
                              num_layers=config['decoder']['num_layers'])
        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

    # def __init__(self, config, input_ch=3, input_ch_pos=12, learnable_beta=True):
    #     super(Decoders, self).__init__()
    #     self.config = config
    #     self.color_net = ColorNet(config,
    #                               input_ch=input_ch_pos,
    #                               geo_feat_dim=config['decoder']['geo_feat_dim'],
    #                               hidden_dim_color=config['decoder']['hidden_dim_color'],
    #                               num_layers_color=config['decoder']['num_layers_color'])
    #     self.sdf_net = SDFNet(config,
    #                           input_ch=input_ch + input_ch_pos,
    #                           geo_feat_dim=config['decoder']['geo_feat_dim'],
    #                           hidden_dim=config['decoder']['hidden_dim'],
    #                           num_layers=config['decoder']['num_layers'])
    #     if learnable_beta:
    #         self.beta = nn.Parameter(10 * torch.ones(1))
    #     else:
    #         self.beta = 10


    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """
        vgrid = p_nor[None, :, None]

        feat = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            feat.append(xy + xz + yz)

        feat = torch.cat(feat, dim=-1)

        return feat


    def query_sdf(self, p, embed_pos, all_planes):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes

        feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz)

        h = self.sdf_net(torch.cat([feat, embed_pos], dim=-1), return_geo=True)
        sdf, geo_feat = h[..., :1], h[..., 1:]
        sdf = torch.reshape(sdf, list(p.shape[:-1]))

        return sdf

    def forward(self, p, embed_pos, all_planes):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            raw (tensor): raw SDF and RGB
        """
        p_shape = p.shape
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        #planes_xy, planes_xz, planes_yz = all_planes
        feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz)
        c_feat = self.sample_plane_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)

        # print(feat.shape)
        # print(embed_pos.shape)

        h = self.sdf_net(torch.cat([feat, embed_pos], dim=-1), return_geo=True)

        sdf, geo_feat = h[..., :1], h[..., 1:]

        rgb = self.color_net(torch.cat([embed_pos, c_feat, geo_feat], dim=-1))
        #rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))

        # planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        # feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz)
        # h = self.sdf_net(torch.cat([feat, embed_pos], dim=-1), return_geo=True)
        # sdf, geo_feat = h[..., :1], h[..., 1:]
        # rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        #sdf = torch.tanh(sdf)

        rgb = torch.sigmoid(rgb)
        raw = torch.cat([rgb, sdf], dim=-1)
        raw = raw.reshape(*p_shape[:-1], -1)
        return raw




