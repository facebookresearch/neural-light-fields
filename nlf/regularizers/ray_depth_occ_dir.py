import numpy as np
import torch
from torch import nn

from nlf.regularizers.base import BaseRegularizer
from nlf.nets import BaseNet
from nlf.embedding import (
    SequentialEmbedding,
    RayParam,
    WindowedPE
)
from utils.ray_utils import ray_param_dict, ray_param_pos_dict
from losses import loss_dict
from utils.ray_utils import dot, get_weight_map, weighted_stats


class RayDepthOccDirRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.range = cfg.range
        self.jitter = cfg.jitter

        ## Net
        self.ray_param_fn = ray_param_dict[cfg.param.fn]
        self.ray_param_pos_fn = ray_param_pos_dict[cfg.param.fn]

        self.num_features = cfg.num_features
        self.num_samples = cfg.num_samples
        self.n_ray_dims = cfg.param.n_dims

        self.pe = WindowedPE(
            3 * self.num_features + self.n_ray_dims,
            cfg.pe
        )

        self.net = BaseNet(
            self.pe.out_channels,
            1 + self.n_ray_dims * self.num_samples,
            cfg.net,
        )

        self.depth_model = nn.Sequential(
            self.pe, self.net
        )

    def get_scene_bounds(self):
        if 'near' in self.cfg:
            near = self.cfg.near
        else:
            near = self.get_dataset().near

        if 'far' in self.cfg:
            far = self.cfg.far
        else:
            far = self.get_dataset().far

        return (near, far)

    def get_depths(self):
        near, far = self.get_scene_bounds()

        ## Get depth values
        if self.cfg.use_disparity:
            inv_depths = torch.linspace(
                1.0 / far, 1.0 / near, self.num_features
            )
            depths = torch.flip(1.0 / inv_depths, (0,))
        else:
            depths = torch.linspace(
                near, far, self.num_features
            )

        return depths

    def forward(self, rays):
        ## Unified ray origins
        rays_o = self.ray_param_pos_fn(rays).unsqueeze(1)
        rays_d = rays[..., 3:6].unsqueeze(1)

        ## Ray features
        depths = self.get_depths().type_as(rays)
        depths = depths.view(1, -1, 1)

        points = rays_o + rays_d * depths
        points = points.view(points.shape[0], -1)

        near, far = self.get_scene_bounds()
        points = (points / (far - near)) * (2 ** 3)

        features = torch.cat([points, self.ray_param_fn(rays)], -1)

        ## Forward
        depth_outputs = self.depth_model(features)
        return depth_outputs

    def backproject(self, rays, depth):
        unified_ray_origins = self.ray_param_pos_fn(rays)
        backproj_points = unified_ray_origins + rays[..., 3:6] * depth

        return backproj_points

    def get_jittered_rays(self, backproj_points, jitter_rays):
        ## Jitter and project
        jitter_ray_origins = jitter_rays[..., :3]
        jitter_ray_dirs = torch.nn.functional.normalize(
            backproj_points - jitter_ray_origins, p=2, dim=-1
        )
        jitter_rays = torch.cat([jitter_ray_origins, jitter_ray_dirs], -1)

        return jitter_rays

    def get_jittered_depth(self, backproj_points, jitter_rays):
        jitter_ray_dirs = jitter_rays[..., 3:6]
        unified_jitter_ray_origins = self.ray_param_pos_fn(jitter_rays)

        diff = backproj_points - unified_jitter_ray_origins
        depth_proj = torch.linalg.norm(
            diff,
            dim=-1
        )[..., None] * (torch.sign(
            dot(diff, jitter_ray_dirs)
        )[..., None].detach())

        ## Get jittered depth
        sh = jitter_rays.shape

        jitter_depth_outputs = self(jitter_rays.view(-1, self.n_ray_dims))
        jitter_depth_outputs = jitter_depth_outputs.view(
            sh[0], sh[1], jitter_depth_outputs.shape[-1]
        )

        jitter_depth, jitter_occ_dir = \
            jitter_depth_outputs[..., :1], jitter_depth_outputs[..., 1:]
        jitter_occ_dir = jitter_occ_dir.view(
            sh[0], sh[1], self.num_samples, self.n_ray_dims
        )

        return depth_proj, jitter_depth, jitter_occ_dir

    def get_occ_weight_map(
        self,
        rays,
        jitter_rays,
        occ_dir,
    ):
        occ_weight_map = 1

        if self._do_loss('occ_loss'):
            unified_rays = self.ray_param_fn(rays)
            unified_jitter_rays = self.ray_param_fn(jitter_rays)

            ray_diff = unified_jitter_rays - unified_rays
            ray_diff = ray_diff.unsqueeze(-2)

            ray_dot = torch.clamp(dot(ray_diff, occ_dir), 0.0)
            occ_weight_map = torch.exp(-ray_dot.mean(-1))[..., None]

        return occ_weight_map

    def _loss(self, train_batch, batch_idx):
        system = self.get_system()
        dataset = self.get_dataset()
        batch = self.get_batch(train_batch, batch_idx, apply_ndc=dataset.use_ndc)

        #### Forward ####

        if dataset.use_ndc:
            rays, jitter_rays = batch['rays_no_ndc'], batch['jitter_rays_no_ndc']
        else:
            rays, jitter_rays = batch['rays'], batch['jitter_rays']

        if 'rgb' in batch:
            rgb = batch['rgb']
        else:
            rgb = system(rays, apply_ndc=dataset.use_ndc)['rgb']

        ## Get depth
        depth_outputs = self(rays)
        depth, occ_dir = depth_outputs[..., :1], depth_outputs[..., 1:]
        backproj_points = self.backproject(rays, depth)

        #### Losses ####

        all_losses = {
            loss: 0.0 for loss in self.loss_fns.keys()
        }

        ## Lookup loss helpers
        occ_dir = occ_dir.view(
            1, occ_dir.shape[0], self.num_samples, self.n_ray_dims
        )

        ## Color lookup loss
        if self._do_loss('color_lookup_loss'):
            lookup_rgb, lookup_rays, lookup_valid = self.get_dataset().lookup_points(
                backproj_points
            )

            occ_weight_map = self.get_occ_weight_map(
                rays.unsqueeze(0),
                lookup_rays,
                occ_dir,
            )

            lookup_weight_map = self.get_weight_map(
                rays.unsqueeze(0),
                lookup_rays,
                'lookup_weight_map'
            ) * occ_weight_map * lookup_valid

            all_losses['color_lookup_loss'] = self._loss_fn(
                'color_lookup_loss',
                rgb.unsqueeze(0) * lookup_weight_map,
                lookup_rgb * lookup_weight_map,
            )

        ## Consistency loss helpers
        occ_dir = occ_dir.view(
            occ_dir.shape[1], 1, self.num_samples, self.n_ray_dims
        )
        rgb = rgb.unsqueeze(-2)
        rays = rays.unsqueeze(-2)
        depth = depth.unsqueeze(-2)
        backproj_points = backproj_points.unsqueeze(-2)

        if self._do_loss('color_loss') \
            or self._do_loss('depth_loss') \
            or self._do_loss('occ_loss'):

            jitter_rays = self.get_jittered_rays(
                backproj_points, jitter_rays
            )

            occ_weight_map = self.get_occ_weight_map(
                rays,
                jitter_rays,
                occ_dir,
            )

            color_weight_map = self.get_weight_map(
                rays,
                jitter_rays,
                'color_weight_map'
            ) * occ_weight_map

            depth_weight_map = self.get_weight_map(
                rays,
                jitter_rays,
                'depth_weight_map'
            ) * occ_weight_map

        if self._do_loss('depth_loss') \
            or self._do_loss('occ_loss'):
            depth_proj, jitter_depth, jitter_occ_dir = self.get_jittered_depth(
                backproj_points, jitter_rays
            )

        ## Color consistency loss
        if self._do_loss('color_loss'):
            sh = jitter_rays.shape
            jitter_rgb = system(
                jitter_rays.view(-1, jitter_rays.shape[-1]), apply_ndc=dataset.use_ndc
            )['rgb']
            jitter_rgb = jitter_rgb.view(
                sh[0], sh[1], jitter_rgb.shape[-1]
            )

            all_losses['color_loss'] = self._loss_fn(
                'color_loss',
                (rgb * color_weight_map).permute(1, 0, 2),
                (jitter_rgb * color_weight_map).permute(1, 0, 2),
            )

        ## Depth consistency loss
        if self._do_loss('depth_loss'):
            all_losses['depth_loss'] = self._loss_fn(
                'depth_loss',
                depth_proj * depth_weight_map,
                jitter_depth * depth_weight_map,
            ) / self.get_dataset().far

        ## Occlusion loss
        if self._do_loss('occ_loss'):
            occ = torch.linalg.norm(
                occ_dir,
                dim=-1
            ).mean(-1)
            jitter_occ = torch.linalg.norm(
                jitter_occ_dir,
                dim=-1
            ).mean(-1)

            all_losses['occ_loss'] = self._loss_fn(
                'occ_loss',
                occ,
                torch.zeros_like(occ),
            ) * 0.5

            all_losses['occ_loss'] += self._loss_fn(
                'occ_loss',
                jitter_occ,
                torch.zeros_like(jitter_occ),
            ) * 0.5

        ## Total loss
        total_loss = 0.0

        for name in all_losses.keys():
            print(name + ':', all_losses[name])
            total_loss += all_losses[name]

        return total_loss

    def set_iter(self, i):
        super().set_iter(i)

        self.pe.set_iter(self.cur_iter)

    def validation(self, batch):
        system = self.get_system()
        dataset = self.get_dataset()
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        if dataset.use_ndc:
            rays = batch['rays_no_ndc'].squeeze()
        else:
            rays = batch['rays'].squeeze()

        # Depth
        depth = self(rays)
        depth, occ_dir = depth[..., :1], depth[..., 1:]

        # Depth view
        near, far = self.get_scene_bounds()
        depth = depth.view(H, W, 1).cpu()
        depth = depth.permute(2, 0, 1)

        disp = 1.0 / torch.abs(depth)
        disp[torch.abs(depth) < 1e-5] = 0

        depth = (depth - depth.min()) / (depth.max() - depth.min())
        disp = (disp - disp.min()) / (disp.max() - disp.min())

        # Occlusions
        occ_dir = occ_dir.view(
            occ_dir.shape[0], self.num_samples, self.n_ray_dims
        )
        occ = torch.linalg.norm(
            occ_dir,
            dim=-1
        ).mean(-1)
        occ = occ.view(H, W, 1).cpu()
        occ = occ.permute(2, 0, 1)
        occ = occ / occ.max()

        return {
            'depth': depth,
            'disp': disp,
            'occ': occ,
        }

    def validation_video(self, batch):
        # Outputs
        outputs = self.validation(batch)

        return {
            'videos/depth': outputs['depth'],
            'videos/disp': outputs['disp'],
            'videos/occ': outputs['occ'],
        }

    def validation_image(self, batch, batch_idx):
        # Outputs
        outputs = self.validation(batch)

        return {
            'images/depth': outputs['depth'],
            'images/disp': outputs['disp'],
            'images/occ': outputs['occ'],
        }
