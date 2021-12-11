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


class InverseRayDepthRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.range = cfg.range
        self.num_samples = cfg.num_samples

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

    def select_depths(self, rays):
        near, far = self.get_scene_bounds()
        u = torch.rand(
            rays.shape[0], self.num_samples, device=rays.device
        )

        if self.cfg.use_disparity:
            s = 1.0 / far
            t = 1.0 / near

            depths = 1.0 / ((1 - u) * s + u * t)
        else:
            s = near
            t = far

            depths = (1 - u) * s + u * t

        return depths

    def backproject(self, rays, depth):
        ray_origins = rays[..., :3]
        backproj_points = ray_origins + rays[..., 3:6] * depth

        return backproj_points

    def get_weight_map(
        self,
        rays,
        jitter_rays,
        rgb,
        jitter_rgb,
        name,
    ):
        dataset = self.get_dataset()

        ray_weights = get_weight_map(
            rays,
            jitter_rays,
            self.cfg[name],
            softmax=False,
        )

        rgb_std = dataset.rgb_std.type_as(rgb) * self.cfg[name].rgb_std
        rgb_diff = rgb - jitter_rgb

        rgb_weights = torch.exp(
            -torch.square(rgb_diff / rgb_std).sum(-1)
        )[..., None]

        constant =  1 / torch.sqrt(torch.square(rgb_std).sum())
        return ray_weights * rgb_weights / constant

    def _loss(self, train_batch, batch_idx):
        system = self.get_system()
        dataset = self.get_dataset()
        batch = self.get_batch(train_batch, batch_idx, apply_ndc=True)

        ## Forward
        if dataset.use_ndc:
            rays = batch['rays_no_ndc']
        else:
            rays = batch['rays']

        if 'rgb' in batch:
            rgb = batch['rgb']
        else:
            rgb = system(rays, apply_ndc=dataset.use_ndc)['rgb']

        depths = self.select_depths(rays)

        ## Reshape
        rays = rays.unsqueeze(1).repeat(
            1, self.num_samples, 1
        ).view(-1, rays.shape[-1])
        rgb = rgb.unsqueeze(1).repeat(
            1, self.num_samples, 1
        ).view(-1, rgb.shape[-1])
        depths = depths.view(-1, 1)

        ## Backproject
        backproj_points = self.backproject(rays, depths)

        #### Helper vars ####

        ## Lookup loss helpers
        if self._do_loss('embedding_lookup_loss'):
            lookup_rgb, lookup_rays, lookup_valid = self.get_dataset().lookup_points(
                backproj_points
            )

            lookup_weight_map = self.get_weight_map(
                rays.unsqueeze(0),
                lookup_rays,
                rgb.unsqueeze(0),
                lookup_rgb,
                'lookup_weight_map',
            ) * lookup_valid

        #### Losses ####

        all_losses = {
            loss: 0.0 for loss in self.loss_fns.keys()
        }

        ## Embedding lookup loss
        if self._do_loss('embedding_lookup_loss'):
            idx = torch.argsort(
                torch.rand(*lookup_weight_map.shape).type_as(rays),
                dim=0
            ).long()

            lookup_rays = torch.gather(
                lookup_rays, 0, idx.repeat(1, 1, rays.shape[-1])
            )[0]
            lookup_rgb = torch.gather(
                lookup_rgb, 0, idx.repeat(1, 1, rgb.shape[-1])
            )[0]
            lookup_weight_map = torch.gather(
                lookup_weight_map, 0, idx
            )[0]

            lookup_embed = system.embed_params(lookup_rays, apply_ndc=True)['params']
            embed = system.embed_params(rays, apply_ndc=True)['params']

            lookup_embed = lookup_embed.view(lookup_embed.shape[0], -1)
            embed = embed.view(embed.shape[0], -1)

            all_losses['embedding_lookup_loss'] = self._loss_fn(
                'embedding_lookup_loss',
                lookup_embed * lookup_weight_map,
                embed * lookup_weight_map,
            )

        ## Total loss
        total_loss = 0.0

        for name in all_losses.keys():
            print(name + ':', all_losses[name])
            total_loss += all_losses[name]

        return total_loss
