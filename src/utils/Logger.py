
import os
import torch

class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, plgslam):
        self.verbose = plgslam.verbose
        self.ckptsdir = plgslam.ckptsdir
        self.gt_c2w_list = plgslam.gt_c2w_list

        self.shared_decoders_list = plgslam.shared_decoders_list
        self.shared_all_planes_list = plgslam.shared_all_planes_list
        self.shared_cur_rf_id = plgslam.shared_cur_rf_id
        self.estimate_c2w_list = plgslam.estimate_c2w_list

    def log(self, idx, keyframe_list, keyframe_dict):
        self.shared_decoders = self.shared_decoders_list[self.shared_cur_rf_id[0]]
        self.all_planes = self.shared_all_planes_list[self.shared_cur_rf_id[0]]
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            #'keyframe_dict' : keyframe_dict,
            'all_planes': self.all_planes,
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
