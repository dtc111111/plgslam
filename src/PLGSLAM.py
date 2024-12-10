import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp
from torch import nn

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer
from src.networks.encodings import get_encoder
#from src.networks.co_decoder import ColorSDFNet,ColorSDFNet_v2

torch.multiprocessing.set_sharing_strategy('file_system')

class PLGSLAM():
    """
    PLGSLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking processes.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args

        self.verbose = cfg['verbose']
        self.device = cfg['device']
        self.dataset = cfg['dataset']
        self.truncation = cfg['model']['truncation']

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()  #Update the camera intrinsics
        # model = config.get_model(cfg)   # Need to modify
        # self.shared_decoders = model  # Need to modify
        self.get_encoding(cfg)

        self.shared_embedpos_fn = self.embedpos_fn
        self.shared_input_ch_pos = self.input_ch_pos

        self.scale = cfg['scale']

        self.load_bound(cfg)
        # self.init_planes(cfg)
        # self.init_planes(cfg, 'global_plane')

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()

        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()

        manager = mp.Manager()
        self.shared_all_planes_list = manager.list()
        #self.all_planes_global_list = manager.list()
        #self.shared_decoders_list = []
        self.shared_decoders_list = manager.list()
        # #self.shared_all_planes_list.share_memory_()
        #self.shared_decoders_list = nn.ParameterList([])

        self.shared_cur_rf_id = torch.zeros((1)).int()
        self.shared_cur_rf_id.share_memory_()

        self.shared_embedpos_fn = self.shared_embedpos_fn.to(self.device)
        self.shared_embedpos_fn.share_memory()

        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def get_encoding(self, cfg):
        '''
        Get the encoding of the scene representation 获取场景表示的编码方式
        '''
        # Coordinate encoding 坐标编码
        self.embedpos_fn, self.input_ch_pos = get_encoder(cfg['pos']['enc'], n_bins=self.cfg['pos']['n_bins'])

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        print(
            f"INFO: The GT, generated and residual depth/color images can be found under " +
            f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """

        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float()
        bound_dividable = cfg['planes_res']['bound_dividable']
        # enlarge the bound a bit to allow it dividable by bound_dividable
        # for bound in self.bound:
        #     bound[:, 1] = (((bound[:, 1] - bound[:, 0]) /
        #                     bound_dividable).int() + 1) * bound_dividable + bound[:, 0]
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0]
        #self.shared_decoders.bound = self.bound


    def init_planes(self, cfg, planes_type):
        """
        Initialize the feature planes.

        Args:
            cfg (dict): parsed config dict.
        """
        # self.coarse_planes_res = cfg['planes_res']['coarse']
        # self.fine_planes_res = cfg['planes_res']['fine']
        # self.coarse_c_planes_res = cfg['c_planes_res']['coarse']
        # self.fine_c_planes_res = cfg['c_planes_res']['fine']
        #
        # if planes_type == 'global_plane':
        #     planes_res = self.coarse_planes_res
        #     c_planes_res = self.coarse_c_planes_res
        #
        # elif planes_type == 'local_plane':
        #     planes_res = self.fine_planes_res
        #     c_planes_res = self.fine_c_planes_res
        self.coarse_planes_res = cfg['planes_res']['coarse']
        self.fine_planes_res = cfg['planes_res']['fine']

        self.coarse_c_planes_res = cfg['c_planes_res']['coarse']
        self.fine_c_planes_res = cfg['c_planes_res']['fine']

        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        ####### Initializing Planes ############
        planes_xy, planes_xz, planes_yz = [], [], []
        c_planes_xy, c_planes_xz, c_planes_yz = [], [], []
        planes_res = [self.coarse_planes_res, self.fine_planes_res]
        c_planes_res = [self.coarse_c_planes_res, self.fine_c_planes_res]

        planes_dim = c_dim
        #
        # grid_shape = list(map(int, (xyz_len / planes_res).tolist()))
        # grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
        # # planes_xy = torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01)
        # # planes_xz = torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01)
        # # planes_yz = torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01)
        # planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
        # planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
        # planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))
        #
        # grid_shape = list(map(int, (xyz_len / c_planes_res).tolist()))
        # grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
        # # c_planes_xy = torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01)
        # # c_planes_xz = torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01)
        # # c_planes_yz = torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01)
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

        # self.shared_planes_xy_global = planes_xy
        # self.shared_planes_xz_global = planes_xz
        # self.shared_planes_yz_global = planes_yz
        #
        # self.shared_c_planes_xy_global = c_planes_xy
        # self.shared_c_planes_xz_global = c_planes_xz
        # self.shared_c_planes_yz_global = c_planes_yz

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while True:
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread.

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(0, 2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
