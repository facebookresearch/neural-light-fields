#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from nlf.regularizers.base import BaseRegularizer
from nlf.nets import BaseNet
from nlf.embedding import (
    WindowedPE
)
from utils.ray_utils import ray_param_dict, ray_param_pos_dict
from utils.ray_utils import get_weight_map


class RayDepthRegularizer(BaseRegularizer):
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

        #self.pe = WindowedPE(
        #    3 * self.num_features + cfg.param.n_dims,
        #    cfg.pe
        #)
        self.pe = WindowedPE(
            cfg.param.n_dims,
            cfg.pe
        )
        self.net = BaseNet(
            self.pe.out_channels,
            self.num_features,
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

        #features = torch.cat([points, self.ray_param_fn(rays)], -1)
        features = self.ray_param_fn(rays)

        ## Run net
        depth_probs = self.depth_model(features)
        return depth_probs

    def select_depths(self, depth_probs, weighted=False):
        batch_size = depth_probs.shape[0]

        depths = self.get_depths().type_as(depth_probs)
        depths = depths.view(1, -1).repeat(
            batch_size, 1
        )

        if weighted:
            cdf = torch.cumsum(depth_probs, -1)
            u = torch.rand(batch_size, self.num_samples).type_as(depth_probs)
            idx = torch.searchsorted(cdf, u, right=True)
        else:
            idx = torch.argsort(
                torch.rand(batch_size, self.num_features).type_as(depth_probs),
                -1
            )[..., :self.num_samples]

        depths = torch.gather(depths, -1, idx.long())
        weights = torch.gather(depth_probs, -1, idx.long())
        weights = weights * (self.num_features / self.num_samples)

        print(idx[0], depths[0], weights[0])

        return depths, weights

    def average_depth(self, depth_probs):
        depths = self.get_depths().type_as(depth_probs)
        return (depth_probs * depths.view(1, -1)).sum(-1)

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

    def get_weight_map(
        self,
        rays,
        jitter_rays,
        name,
    ):
        return get_weight_map(
            rays,
            jitter_rays,
            self.cfg[name],
            softmax=False,
        )

    def loss(self, train_batch, batch_idx):
        if self.cur_iter < 0:
            return 0.0

        dataset = self.get_dataset()
        batch = self.get_batch(train_batch, batch_idx)

        ## Forward
        if dataset.use_ndc:
            rays, jitter_rays = batch['rays_no_ndc'], batch['jitter_rays']
        else:
            rays, jitter_rays = batch['rays'], batch['jitter_rays']

        if 'rgb' in batch:
            rgb = batch['rgb']
        else:
            rgb = self._color(rays)['rgb']

        ## Get probabilities
        depth_probs = self(rays)
        depths, weights = self.select_depths(depth_probs)

        ## Reshape
        rays = rays.unsqueeze(1).repeat(
            1, self.num_samples, 1
        ).view(-1, rays.shape[-1])
        jitter_rays = jitter_rays.unsqueeze(1).repeat(
            1, self.num_samples, 1
        ).view(-1, jitter_rays.shape[-1])
        rgb = rgb.unsqueeze(1).repeat(
            1, self.num_samples, 1
        ).view(-1, rgb.shape[-1])

        depths = depths.view(-1, 1)
        depth_weights = weights.view(-1, 1)

        ## Backproject
        backproj_points = self.backproject(rays, depths)

        #### Helper vars ####

        ## Lookup loss helpers
        if self._do_loss('color_lookup_loss') \
            or self._do_loss('embedding_lookup_loss'):

            rgb_lookup, lookup_rays, lookup_valid = self.get_dataset().lookup_points(
                backproj_points
            )

            lookup_weight_map = self.get_weight_map(
                rays.unsqueeze(0),
                lookup_rays,
                'lookup_weight_map'
            ) * lookup_valid * depth_weights.unsqueeze(0)

        ## Consistency loss helpers
        if self._do_loss('color_loss') \
            or self._do_loss('embedding_loss'):
            jitter_rays = self.get_jittered_rays(
                backproj_points, jitter_rays
            )

            weight_map = self.get_weight_map(
                rays,
                jitter_rays,
                'weight_map'
            ) * depth_weights

        #### Losses ####

        all_losses = {
            loss: 0.0 for loss in self.loss_fns.keys()
        }

        ## Color lookup loss
        if self._do_loss('color_lookup_loss'):
            all_losses['color_lookup_loss'] = self._loss_fn(
                'color_lookup_loss',
                rgb.unsqueeze(0) * lookup_weight_map,
                rgb_lookup * lookup_weight_map,
            )

        ## Embedding lookup loss
        if self._do_loss('embedding_lookup_loss'):
            idx = torch.argsort(torch.rand(*lookup_weight_map.shape).type_as(rays), dim=0)
            lookup_rays = torch.gather(lookup_rays, 0, idx.long())[0]
            lookup_weight_map_one = torch.gather(lookup_weight_map, 0, idx.long())[0]

            embed_lookup_rays = self.embed(lookup_rays)
            embed_rays = self.embed(rays)

            all_losses['embedding_lookup_loss'] = self._loss_fn(
                'embedding_lookup_loss',
                embed_lookup_rays * lookup_weight_map_one,
                embed_rays * lookup_weight_map_one,
            )

        ## Color consistency loss
        if self._do_loss('color_loss'):
            jitter_rgb = self._color(jitter_rays)['rgb']

            all_losses['color_loss'] = self._loss_fn(
                'color_loss',
                rgb * weight_map,
                jitter_rgb * weight_map,
            )

        ## Embedding consistency loss
        if self._do_loss('embedding_loss'):
            embed_jitter_rays = self.embed(jitter_rays)
            embed_rays = self.embed(rays)

            all_losses['embedding_loss'] = self._loss_fn(
                'embedding_loss',
                embed_jitter_rays * weight_map,
                embed_rays * weight_map,
            )

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
        depth_probs = self(rays)
        depth = self.average_depth(depth_probs)

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
