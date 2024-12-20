

import torch
from src.common import get_rays, sample_pdf, normalize_3d_coordinate

class Renderer(object):
    """
    Renderer class for rendering depth and color.
    Args:
        cfg (dict): configuration.
        ray_batch_size (int): batch size for sampling rays.
    """
    def __init__(self, cfg, plgslam, ray_batch_size=10000):
        self.cfg = plgslam.cfg
        self.ray_batch_size = ray_batch_size

        self.perturb = cfg['rendering']['perturb']
        self.n_stratified = cfg['rendering']['n_stratified']
        self.n_importance = cfg['rendering']['n_importance']

        self.scale = cfg['scale']
        self.bound = plgslam.bound.to(plgslam.device, non_blocking=True)
        self.cur_rf_id = plgslam.shared_cur_rf_id

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = plgslam.H, plgslam.W, plgslam.fx, plgslam.fy, plgslam.cx, plgslam.cy
        self.embedpos_fn = plgslam.embedpos_fn

    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays.
        Returns:
            z_vals (tensor): perturbed depth values on the rays.
        """
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def render_batch_ray(self, all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=None):

        """
        Render depth and color for a batch of rays.
        Args:
            all_planes (Tuple): all feature planes.
            all_planes_global(Tuple): all global feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            rays_d (tensor): ray directions.
            rays_o (tensor): ray origins.
            device (torch.device): device to run on.
            truncation (float): truncation threshold.
            gt_depth (tensor): ground truth depth.
        Returns:
            depth_map (tensor): depth map.
            color_map (tensor): color map.
            volume_densities (tensor): volume densities for sampled points.
            z_vals (tensor): sampled depth values on the rays.

        """
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]

        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        ### pixels with gt depth:
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation)  + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        if self.perturb:
            z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero.float()
        #z_vals = z_vals.float()

        ### pixels without gt depth (importance sampling):
        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                #t = (self.bound[self.cur_rf_id[0]].unsqueeze(0) - det_rays_o) / det_rays_d  # (N, 3, 2)
                t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.perturb:
                    z_vals_uni = self.perturbation(z_vals_uni)
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]
                inputs_flat = torch.reshape(pts_uni, [-1, pts_uni.shape[-1]])
                embed_pos = self.embedpos_fn(inputs_flat)
            ##############################
                raw_uni = decoders(pts_uni, embed_pos, all_planes)
                sdf_uni = raw_uni[..., -1]
                #sdf_uni = decoders.get_raw_sdf(pts_uni_nor, embed_pos, all_planes)
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)
                weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]
                '''
                weights_uni = torch.sigmoid(sdf_uni / self.cfg['training']['trunc']) * torch.sigmoid(
                    -sdf_uni / self.cfg['training']['trunc'])

                signs = sdf_uni[:, 1:] * sdf_uni[:, :-1]
                mask = torch.where(signs < 0.0, torch.ones_like(signs),
                                   torch.zeros_like(signs))
                inds = torch.argmax(mask, axis=1)
                inds = inds[..., None]
                z_min = torch.gather(z_vals_uni, 1, inds)
                mask = torch.where(z_vals_uni < z_min + self.cfg['data']['sc_factor'] * self.cfg['training']['trunc'],
                                   torch.ones_like(z_vals_uni), torch.zeros_like(z_vals_uni))

                weights_uni = weights_uni * mask
                weights_uni = weights_uni / (torch.sum(weights_uni, axis=-1, keepdims=True) + 1e-8)
'''
                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]
        # mask_outbbox = ~(torch.max((torch.abs(world2rf - pts_uni))) > max_drift).any(
        #         dim=-1
        #         )
        # pts = pts[mask_outbbox]
        inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        embed_pos = self.embedpos_fn(inputs_flat)
        #raw = decoders(pts, embed_pos, all_planes)  #(4000,40,4) rgb+sdf
        raw = decoders(pts.to(torch.float32), embed_pos, all_planes)
        alpha = self.sdf2alpha(raw[..., -1], decoders.beta) # Need to modify
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                                , (1. - alpha + 1e-10)], -1), -1)[:, :-1]
        '''
        sdf = raw[..., -1]
        weights = torch.sigmoid(sdf / self.cfg['training']['trunc']) * torch.sigmoid(-sdf / self.cfg['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs),
                           torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds)
        mask = torch.where(z_vals < z_min + self.cfg['data']['sc_factor'] * self.cfg['training']['trunc'],
                           torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        weights = weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
'''
        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)

        return rendered_depth, rendered_rgb, raw[..., -1], z_vals

    def sdf2alpha(self, sdf, beta=10):
        """

        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def render_img(self, all_planes, decoders, c2w, truncation, device, gt_depth=None):
        """
        Renders out depth and color images.
        Args:
            all_planes (Tuple): feature planes
            all_planes_global(Tuple): all global feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            c2w (tensor, 4*4): camera pose.
            truncation (float): truncation distance.
            device (torch.device): device to run on.
            gt_depth (tensor, H*W): ground truth depth image.
        Returns:
            rendered_depth (tensor, H*W): rendered depth image.
            rendered_rgb (tensor, H*W*3): rendered color image.

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                depth, color, _, _ = ret
                depth_list.append(depth.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, color

