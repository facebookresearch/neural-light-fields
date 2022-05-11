#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from nlf.regularizers.base import BaseRegularizer
from nlf.nets import BaseNet
from nlf.embedding import (
    WindowedPE
)
from utils.ray_utils import ray_param_dict, ray_param_pos_dict
from utils.ray_utils import dot, get_weight_map


class MultipleRayDepthRegularizer(BaseRegularizer):
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
        self.num_slices = cfg.num_slices
        self.num_filler = cfg.num_filler
        self.total_num_slices = self.num_slices + self.num_filler

        self.pe = WindowedPE(
            self.n_ray_dims,
            cfg.pe
        )

        self.net = BaseNet(
            self.pe.out_channels,
            self.num_slices,
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
                1.0 / far, 1.0 / near, self.num_slices
            )
            depths = torch.flip(1.0 / inv_depths, (0,))
        else:
            depths = torch.linspace(
                near, far, self.num_slices
            )

        return depths

    def forward(self, rays):
        ## Forward
        features = self.ray_param_fn(rays)
        depth_offsets = self.depth_model(features)

        ## Add offsets
        depth_centers = self.get_depths().type_as(rays).unsqueeze(0)
        depths = depth_centers + depth_offsets

        return torch.cat(
            [
                depths,
                depths.new_zeros(depths.shape[0], self.num_filler)
            ],
            -1
        )

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

    def get_jittered_depth(self, backproj_points, jitter_rays, jitter_idx):
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

        jitter_depth = self(jitter_rays.view(-1, jitter_rays.shape[-1]))
        jitter_depth = jitter_depth.view(
            sh[0], sh[1], self.total_num_slices, 1
        )
        jitter_depth = torch.gather(jitter_depth, -2, jitter_idx)
        jitter_depth = jitter_depth.view(sh[0], sh[1], 1)

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

    def get_weight_map_rgba(
        self,
        rays,
        jitter_rays,
        rgba,
        jitter_rgba,
        name,
        isotropic=False
    ):
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
        rgba_diff = rgba - jitter_rgba
        rgba_weights = torch.exp(
            0.5 * -torch.square(rgba_diff / rgb_std).sum(-1)
        )[..., None]

        constant = np.power(2 * np.pi * rgb_std * rgb_std, -4.0 / 2.0)
        return ray_weights * rgba_weights / constant

    def get_weight_map_rgb(
        self,
        rays,
        jitter_rays,
        rgb,
        jitter_rgb,
        name,
        isotropic=False
    ):
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

        ray_outputs = system(
            rays, apply_ndc=dataset.use_ndc, include_all=True
        )
        rgba = torch.cat(
            [ray_outputs['all_rgb'], ray_outputs['all_alpha'].unsqueeze(-1)],
            dim=-1
        )

        ## Backproject
        depth = self(rays)
        depth = depth.view(
            -1, self.total_num_slices, 1
        )

        ## Select
        idx = torch.argsort(
            torch.rand(*depth.shape).type_as(depth),
            dim=-2
        )[..., 0:1, :].long()

        rgba_idx = idx.repeat(
            1, 1, rgba.shape[-1]
        )
        jitter_idx = idx.unsqueeze(1).repeat(
            1, self.jitter.bundle_size, 1, 1
        )
        jitter_rgba_idx = jitter_idx.repeat(
            1, 1, 1, 4
        )

        depth = torch.gather(depth, -2, idx).reshape(-1,  1)
        rgba = torch.gather(rgba, -2, rgba_idx).reshape(-1, rgba.shape[-1])

        ## Backproject
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
                    rgba[..., :3].unsqueeze(0),
                    lookup_rgb,
                    'lookup_weight_map'
                ) * lookup_valid

                all_losses['color_lookup_loss'] = -self._loss_fn(
                    'color_lookup_loss',
                    lookup_weight_map,
                    torch.zeros_like(lookup_weight_map),
                )
            else:
                lookup_weight_map = self.get_weight_map(
                    rays.unsqueeze(0),
                    lookup_rays,
                    'lookup_weight_map'
                ) * lookup_valid

                all_losses['color_lookup_loss'] = self._loss_fn(
                    'color_lookup_loss',
                    rgba[..., :3].unsqueeze(0) * lookup_weight_map,
                    lookup_rgb * lookup_weight_map,
                )

        ## Consistency loss helpers
        rgba = rgba.unsqueeze(-2)
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

            jitter_ray_outputs = system(
                jitter_rays.view(-1, jitter_rays.shape[-1]),
                apply_ndc=dataset.use_ndc, include_all=True
            )
            jitter_rgba = torch.cat(
                [jitter_ray_outputs['all_rgb'], jitter_ray_outputs['all_alpha'].unsqueeze(-1)],
                dim=-1
            )
            jitter_rgba = jitter_rgba.view(
                sh[0], sh[1], self.total_num_slices, jitter_rgba.shape[-1]
            )
            jitter_rgba = torch.gather(
                jitter_rgba, -2, jitter_rgba_idx
            ).reshape(sh[0], sh[1], jitter_rgba.shape[-1])

            if self.occlusion_aware:
                color_weight_map = self.get_weight_map_rgba(
                    rays,
                    jitter_rays,
                    rgba,
                    jitter_rgba,
                    'color_weight_map'
                ).permute(1, 0, 2)

                all_losses['color_loss'] = -self._loss_fn(
                    'color_loss',
                    color_weight_map,
                    torch.zeros_like(color_weight_map),
                )
            else:
                color_weight_map = self.get_weight_map(
                    rays,
                    jitter_rays,
                    'color_weight_map'
                )

                all_losses['color_loss'] = self._loss_fn(
                    'color_loss',
                    (rgba * color_weight_map).permute(1, 0, 2),
                    (jitter_rgba * color_weight_map).permute(1, 0, 2),
                )

        ## Depth consistency loss
        if self._do_loss('depth_loss'):
            depth_proj, jitter_depth = self.get_jittered_depth(
                backproj_points, jitter_rays, jitter_idx
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
        all_depth = self(rays)
        all_depth = all_depth.view(
            -1, self.total_num_slices, 1
        )

        # Outputs
        outputs = {}

        for i in range(all_depth.shape[1]):
            near, far = self.get_scene_bounds()
            depth = all_depth[..., i, :]

            disp = 1.0 / torch.abs(depth)
            disp[torch.abs(depth) < 1e-5] = 0

            depth = (depth - depth.min()) / (depth.max() - depth.min())
            disp = (disp - disp.min()) / (disp.max() - disp.min())

            outputs[f'depth{i}'] = depth.reshape(
                H, W, 1
            ).cpu().permute(2, 0, 1)

            outputs[f'disp{i}'] = disp.reshape(
                H, W, 1
            ).cpu().permute(2, 0, 1)

        return outputs

    def validation_video(self, batch):
        temp_outputs = self.validation(
            batch
        )
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'videos/{key}'] = temp_outputs[key]

        return outputs

    def validation_image(self, batch, batch_idx):
        temp_outputs = self.validation(
            batch
        )
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'images/{key}'] = temp_outputs[key]

        return outputs
