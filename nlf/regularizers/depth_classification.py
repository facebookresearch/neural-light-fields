#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import numpy as np

from nlf.regularizers.base import BaseRegularizer
from nlf.nets import BaseNet
from nlf.embedding import (
    RayParam,
    WindowedPE
)
from utils.ray_utils import ray_param_dict, ray_param_pos_dict
from losses import loss_dict
from utils.ray_utils import dot, get_weight_map, weighted_stats


class DepthClassificationRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.color_loss_fn = loss_dict[self.cfg.color_loss.type]()
        self.depth_loss_fn = loss_dict[self.cfg.depth_loss.type]()

        self.lookup_loss_fn = loss_dict[self.cfg.lookup_loss.type]()
        self.lookup_warmup_iters = self.cfg.lookup_loss.warmup_iters

        self.range = cfg.range
        self.jitter = cfg.jitter

        self.cfg.lookup_loss.angle_std = float(np.radians(cfg.lookup_loss.angle_std))
        self.cfg.color_loss.angle_std = float(np.radians(cfg.color_loss.angle_std))
        self.cfg.depth_loss.angle_std = float(np.radians(cfg.depth_loss.angle_std))

        ## Depth slices
        self.near = self.get_dataset().near
        self.far = self.get_dataset().far
        #self.near = self.cfg.near
        #self.far = self.cfg.far
        self.num_slices = cfg.num_slices

        if cfg.use_disparity:
            inv_depths = torch.linspace(
                1.0 / self.far, 1.0 / self.near, self.num_slices
            )
            self.depths = torch.flip(1.0 / inv_depths, (0,))
        else:
            self.depths = torch.linspace(
                self.near, self.far, self.num_slices
            )

        self.depths += self.cfg.offset

        ## Net
        self.ray_param_fn = ray_param_dict[cfg.param.fn]
        self.ray_param_pos_fn = ray_param_pos_dict[cfg.param.fn]

        if not self.cfg.use_color_embedding:
            self.param = RayParam(
                cfg.param.fn,
                in_channels=6,
                out_channels=cfg.param.n_dims
            )

            self.pe = WindowedPE(
                cfg.param.n_dims,
                cfg.pe
            )

            self.embedding = nn.Sequential(
                self.param, self.pe
            )

            self.net = BaseNet(
                self.pe.out_channels + self.num_slices * 3,
                self.num_slices,
                cfg.net,
            )
        else:
            self.net = BaseNet(
                system.embeddings[0].out_channels,
                self.num_slices,
                cfg.net,
            )

    def forward(self, rays):
        ## Lookup
        rgb_lookup, lookup_weights, cost_volume = self.get_ray_outputs(
            rays
        )

        ## Run MLP
        if not self.cfg.use_color_embedding:
            rays_embedded = self.embedding(rays)
            net_input = torch.cat([rays_embedded, cost_volume], -1)
            net_output = self.net(net_input)
        else:
            system = self.get_system()
            embed_rays = system.embeddings[0](rays)
            net_output = self.net(embed_rays)

        depth_probs = torch.nn.functional.softmax(
            net_output, dim=-1
        )

        ## Return
        return {
            'depth_probs': depth_probs,
            'rgb_lookup': rgb_lookup,
            'lookup_weights': lookup_weights,
            'cost_volume': cost_volume,
        }

    def get_depth(self, depth_probs):
        depths = self.depths[None].type_as(depth_probs)

        return (depth_probs * depths).sum(-1).unsqueeze(-1)

    def get_depth_from_cost_volume(self, cost_volume):
        cost_volume = torch.argmin(cost_volume, dim=-1)
        cost_volume = torch.nn.functional.one_hot(cost_volume, self.num_slices)
        cost_volume = torch.where(
            torch.isnan(cost_volume),
            torch.zeros_like(cost_volume),
            cost_volume
        )

        return self.get_depth(cost_volume)

    def get_ray_outputs(self, rays):
        batch_size = rays.shape[0]

        depths = self.depths[..., None].type_as(rays)
        depths = depths.repeat(1, batch_size).view(-1, 1)
        rays = rays.repeat(self.num_slices, 1)

        ## Backproject
        unified_ray_origins = self.ray_param_pos_fn(rays)
        backproj_points = unified_ray_origins + rays[..., 3:6] * depths
        backproj_points = backproj_points.view(-1, 3)

        ## Lookup
        rgb_lookup, lookup_weights = self.get_dataset().lookup_points(
            rays, backproj_points, self.cfg.lookup_loss
        )
        rgb_lookup = rgb_lookup.view(
            -1, self.num_slices, batch_size, 3
        )
        lookup_weights = lookup_weights.view(
            -1, self.num_slices, batch_size, 1
        )

        ## Calculate cost volume
        rgb_mean, rgb_std = weighted_stats(rgb_lookup, lookup_weights)
        cost_volume = rgb_std.sum(-1)
        cost_volume = cost_volume.view(
            self.num_slices, batch_size
        ).detach().permute(1, 0)

        rgb_lookup = rgb_lookup.permute(2, 0, 1, 3)
        lookup_weights = lookup_weights.permute(2, 0, 1, 3)

        return rgb_lookup, lookup_weights, cost_volume

    def get_jittered_outputs(self, rays, jitter_rays, depth):
        ## Backproject
        unified_ray_origins = self.ray_param_pos_fn(rays)
        backproj_points = unified_ray_origins + rays[..., 3:6] * depth

        ## Jitter and project
        jitter_ray_origins = jitter_rays[..., :3]
        jitter_ray_dirs = torch.nn.functional.normalize(
            backproj_points - jitter_ray_origins, p=2, dim=-1
        )
        jitter_rays = torch.cat([jitter_ray_origins, jitter_ray_dirs], -1)
        unified_jitter_ray_origins = self.ray_param_pos_fn(jitter_rays)

        diff = backproj_points - unified_jitter_ray_origins
        depth_proj = torch.linalg.norm(
            diff,
            dim=-1
        )[..., None] * (torch.sign(
            dot(diff, jitter_ray_dirs)
        )[..., None].detach())

        ## Jittered lookup
        rgb_lookup, lookup_weights, cost_volume = self.get_ray_outputs(
            jitter_rays
        )

        ## Get jittered depth
        jitter_depth_probs = self(rays)['depth_probs']
        jitter_depth = self.get_depth(jitter_depth_probs)

        return jitter_rays, depth_proj, jitter_depth, backproj_points

    def get_weight_maps(
        self,
        rays,
        jitter_rays
    ):
        ## Color and depth
        color_weight_map = get_weight_map(
            rays,
            jitter_rays,
            self.cfg.color_loss
        )

        depth_weight_map = get_weight_map(
            rays,
            jitter_rays,
            self.cfg.depth_loss
        ) / self.far

        return color_weight_map, depth_weight_map

    def loss(self, batch, batch_idx):
        system = self.get_system()

        ## Get rays
        train_batch = batch
        batch = self.get_batch(batch_idx)
        rays = batch['rays'].type_as(train_batch['rays'])
        jitter_rays = batch['jitter_rays'].type_as(train_batch['rays'])

        ## Color
        rgb = system(rays)['rgb']

        ## Run forward
        outputs = self(rays)
        depth_probs = outputs['depth_probs']
        rgb_lookup = outputs['rgb_lookup']
        lookup_weights = outputs['lookup_weights']

        ## Consistency
        if self.cfg.color_loss.weight > 0 \
            or self.cfg.depth_loss.weight > 0:

            depth = self.get_depth(depth_probs)

            ## Jittered outputs
            jitter_rays, depth_proj, jitter_depth, backproj_points = \
                self.get_jittered_outputs(
                    rays, jitter_rays, depth
                )

            ## Weight maps
            color_weight_map, depth_weight_map = self.get_weight_maps(
                rays,
                jitter_rays,
            )

            ## Color consistency
            jitter_rgb = system(jitter_rays)['rgb']

            color_loss = self.color_loss_fn(
                rgb * color_weight_map,
                jitter_rgb * color_weight_map
            ) * self.cfg.color_loss.weight

            ## Depth consistency
            depth_loss = self.depth_loss_fn(
                depth_proj * depth_weight_map,
                jitter_depth * depth_weight_map
            ) * self.cfg.depth_loss.weight

        else:
            color_loss = 0.0
            depth_loss = 0.0

        ## Lookup loss
        if self.cur_iter >= self.lookup_warmup_iters \
            and self.cfg.lookup_loss.weight > 0:

            depth_probs = depth_probs.view(
                -1, 1, self.num_slices, 1
            )

            lookup_loss = self.lookup_loss_fn(
                (rgb[:, None, None, :] * depth_probs) * lookup_weights,
                (rgb_lookup * depth_probs) * lookup_weights,
            ) * self.cfg.lookup_loss.weight * self.num_slices * rgb_lookup.shape[0]
        else:
            lookup_loss = 0.0

        return lookup_loss + color_loss + depth_loss

    def set_iter(self, i):
        super().set_iter(i)

        if not self.cfg.use_color_embedding:
            self.pe.set_iter(i)

    def validation(self, rays):
        system = self.get_system()
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        ## Run forward
        outputs = self.run_chunked(rays)

        ## Depth
        depth_probs = outputs['depth_probs']
        depth = self.get_depth(depth_probs)
        #cost_volume = outputs['cost_volume']
        #depth = self.get_depth_from_cost_volume(cost_volume)

        # Lookup colors
        unified_ray_origins = self.ray_param_pos_fn(rays)
        backproj_points = unified_ray_origins + depth * rays[..., 3:6]

        rgb_lookup, lookup_weights = self.get_dataset().lookup_points(
            rays, backproj_points, self.cfg.lookup_loss
        )
        rgb_lookup_mean, rgb_lookup_var = weighted_stats(
            rgb_lookup, lookup_weights
        )

        rgb_lookup0 = rgb_lookup[0].view(H, W, 3).cpu()
        rgb_lookup0 = rgb_lookup0.permute(2, 0, 1)

        rgb_lookup_mean = rgb_lookup_mean.view(H, W, 3).cpu()
        rgb_lookup_mean = rgb_lookup_mean.permute(2, 0, 1)

        rgb_lookup_var = rgb_lookup_var.view(H, W, 3).cpu()
        rgb_lookup_var = rgb_lookup_var.permute(2, 0, 1)

        # Depth view
        near, far = self.get_dataset().near, self.get_dataset.far()
        depth = depth.view(H, W, 1).cpu()
        depth = depth.permute(2, 0, 1)
        disp = ((1 / depth) - 1 / near) * (1 / near - 1 / far)
        disp[depth == 0] = 0
        depth = (depth - near) / (far - near)

        return {
            'depth': depth,
            'rgb_lookup0': rgb_lookup0,
            'rgb_lookup_mean': rgb_lookup_mean,
            'rgb_lookup_var': rgb_lookup_var,
        }

    def validation_video(self, rays):
        # Outputs
        outputs = self.validation(torch.squeeze(rays))

        return {
            'videos/depth': outputs['depth'],
            'ignore/rgb_lookup0': outputs['rgb_lookup0'],
            'ignore/rgb_lookup_mean': outputs['rgb_lookup_mean'],
            'ignore/rgb_lookup_var': outputs['rgb_lookup_var'],
        }

    def validation_image(self, batch, batch_idx):
        # Outputs
        outputs = self.validation(torch.squeeze(batch['rays']))

        return {
            'depth/pred': outputs['depth'],
            'ignore/rgb_lookup0': outputs['rgb_lookup0'],
            'ignore/rgb_lookup_mean': outputs['rgb_lookup_mean'],
            'ignore/rgb_lookup_var': outputs['rgb_lookup_var'],
        }
