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


class RayDepthRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.range = cfg.range
        self.jitter = cfg.jitter
        self.occlusion_aware = cfg.occlusion_aware

        ## Net
        self.ray_param_fn = ray_param_dict[cfg.param.fn]
        self.ray_param_pos_fn = ray_param_pos_dict[cfg.param.fn]
        self.n_ray_dims = cfg.param.n_dims

        self.pe = WindowedPE(
            self.n_ray_dims,
            cfg.pe
        )

        self.net = BaseNet(
            self.pe.out_channels,
            1,
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

    def forward(self, rays):
        features = self.ray_param_fn(rays)

        ## Forward
        return self.depth_model(features)

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

        jitter_depth = self(jitter_rays.view(-1, self.n_ray_dims))
        jitter_depth = jitter_depth.view(
            sh[0], sh[1], jitter_depth.shape[-1]
        )

        return depth_proj, jitter_depth

    def rgb_std(self, name):
        system = self.get_system()
        loss_cfg = self.cfg[name]

        if isinstance(loss_cfg.rgb_std, float):
            return loss_cfg.rgb_std
        elif loss_cfg.rgb_std.type == 'linear_decay':
            total_num_iters = loss_cfg.rgb_std.num_epochs \
                * len(system.trainer.datamodule.train_dataset) // system.trainer.datamodule.cur_batch_size

            t = self.cur_iter / total_num_iters

            return (1.0 - t) * loss_cfg.rgb_std.start + t * loss_cfg.rgb_std.end
        else:
            return loss_cfg.rgb_std

    def get_weight_map_rgb(
        self,
        rays,
        jitter_rays,
        rgb,
        jitter_rgb,
        name,
        isotropic=False
    ):
        # TODO / To Try:
        # 1) Make ray weights all ones
        # 2) Make RGB kernels isotropic
        # 4) Reduce kernel size, different scheduling arguments for losses

        # Ray weights
        if not isotropic:
            with torch.no_grad():
                ray_weights = get_weight_map(
                    rays,
                    jitter_rays,
                    self.cfg[name],
                    softmax=False,
                )
        else:
            ray_weights = torch.ones_like(jitter_rays[..., :1])

        # Color weights
        rgb_std = self.rgb_std(name)
        rgb_diff = rgb - jitter_rgb
        rgb_weights = torch.exp(
            0.5 * -torch.square(rgb_diff / rgb_std).sum(-1)
        )[..., None]

        constant = np.power(2 * np.pi * rgb_std * rgb_std, -3.0 / 2.0)
        return ray_weights * rgb_weights / constant

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
        depth = self(rays)
        backproj_points = self.backproject(rays, depth)

        #### Losses ####

        all_losses = {
            loss: 0.0 for loss in self.loss_fns.keys()
        }

        ## Color lookup loss
        if self._do_loss('color_lookup_loss'):
            lookup_rgb, lookup_rays, lookup_valid = self.get_dataset().lookup_points(
                backproj_points
            )

            if self.occlusion_aware:
                lookup_weight_map = self.get_weight_map_rgb(
                    rays.unsqueeze(0),
                    lookup_rays,
                    rgb.unsqueeze(0),
                    lookup_rgb,
                    'lookup_weight_map',
                    isotropic=False
                ) * lookup_valid

                lookup_probs = lookup_weight_map.mean(0)

                all_losses['color_lookup_loss'] = -self._loss_fn(
                    'color_lookup_loss',
                    lookup_probs,
                    torch.zeros_like(lookup_probs),
                )
            else:
                lookup_weight_map = self.get_weight_map(
                    rays.unsqueeze(0),
                    lookup_rays,
                    'lookup_weight_map'
                ) * lookup_valid

                all_losses['color_lookup_loss'] = self._loss_fn(
                    'color_lookup_loss',
                    rgb.unsqueeze(0) * lookup_weight_map,
                    lookup_rgb * lookup_weight_map,
                )

        ## Helpers
        rgb = rgb.unsqueeze(-2)
        rays = rays.unsqueeze(-2)
        depth = depth.unsqueeze(-2)
        backproj_points = backproj_points.unsqueeze(-2)

        if self._do_loss('color_loss') \
            or self._do_loss('depth_loss'):
            jitter_rays = self.get_jittered_rays(
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

            if self.occlusion_aware:
                color_weight_map = self.get_weight_map_rgb(
                    rays,
                    jitter_rays,
                    rgb,
                    jitter_rgb,
                    'color_weight_map',
                    isotropic=False
                ).permute(1, 0, 2)

                color_probs = color_weight_map.mean(0)

                all_losses['color_loss'] = -self._loss_fn(
                    'color_loss',
                    color_probs,
                    torch.zeros_like(color_probs),
                )
            else:
                color_weight_map = self.get_weight_map(
                    rays,
                    jitter_rays,
                    'color_weight_map'
                )

                all_losses['color_loss'] = self._loss_fn(
                    'color_loss',
                    (rgb * color_weight_map).permute(1, 0, 2),
                    (jitter_rgb * color_weight_map).permute(1, 0, 2),
                )

        ## Depth consistency loss
        if self._do_loss('depth_loss'):
            depth_proj, jitter_depth = self.get_jittered_depth(
                backproj_points, jitter_rays
            )

            depth_weight_map = self.get_weight_map(
                rays,
                jitter_rays,
                'depth_weight_map'
            )

            all_losses['depth_loss'] = self._loss_fn(
                'depth_loss',
                depth_proj * depth_weight_map,
                jitter_depth * depth_weight_map,
            ) / self.get_dataset().far

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
        depth = depth[..., :1]

        # Depth view
        near, far = self.get_scene_bounds()
        depth = depth.view(H, W, 1).cpu()
        depth = depth.permute(2, 0, 1)

        disp = 1.0 / torch.abs(depth)
        disp[torch.abs(depth) < 1e-5] = 0

        depth = (depth - depth.min()) / (depth.max() - depth.min())
        disp = (disp - disp.min()) / (disp.max() - disp.min())

        return {
            'depth': depth,
            'disp': disp,
        }

    def validation_video(self, batch):
        # Outputs
        outputs = self.validation(batch)

        return {
            'videos/depth': outputs['depth'],
            'videos/disp': outputs['disp'],
        }

    def validation_image(self, batch, batch_idx):
        # Outputs
        outputs = self.validation(batch)

        return {
            'images/depth': outputs['depth'],
            'images/disp': outputs['disp'],
        }
