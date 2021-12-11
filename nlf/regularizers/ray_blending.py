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


class RayDepthBlendingRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.color_loss_fn = loss_dict[self.cfg.color_loss.type]()
        self.depth_loss_fn = loss_dict[self.cfg.depth_loss.type]()

        self.range = cfg.range
        self.jitter = cfg.jitter

        self.cfg.color_loss.angle_std = float(np.radians(cfg.color_loss.angle_std))
        self.cfg.depth_loss.angle_std = float(np.radians(cfg.depth_loss.angle_std))

        self.warmup_iters = cfg.warmup_iters

        ## Net
        self.ray_param_fn = ray_param_dict[cfg.param.fn]
        self.ray_param_pos_fn = ray_param_pos_dict[cfg.param.fn]
        self.num_views = self.cfg.dataset.num_views

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
            self.net = BaseNet(
                self.pe.out_channels,
                self.num_views + 1,
                cfg.net,
            )

            self.depth_model = nn.Sequential(
                self.param, self.pe, self.net
            )
        else:
            self.net = BaseNet(
                system.embeddings[0].out_channels,
                self.num_views + 1,
                cfg.net,
            )

    def forward(self, rays):
        if not self.cfg.use_color_embedding:
            depth_outputs = self.depth_model(rays)
        else:
            system = self.get_system()
            embed_rays = system.embeddings[0](rays)
            depth_outputs = self.net(embed_rays)

        return depth_outputs

    def get_ray_color(self, rays, depth_outputs):
        depths = depth_outputs[..., self.num_views:self.num_views+1].repeat(
            1, self.num_views
        ).permute(1, 0).reshape(
            self.num_views, -1, 1
        )
        weights = depth_outputs[..., -self.num_views:].permute(1, 0)
        rays = rays.unsqueeze(0).repeat(self.num_views, 1, 1)

        ## Backproject
        unified_ray_origins = self.ray_param_pos_fn(rays)
        backproj_points = unified_ray_origins + rays[..., 3:6] * depths

        ## Lookup
        rgb_lookup, lookup_rays, lookup_weights = self.get_dataset().lookup_points_single(
            rays, backproj_points, self.cfg.lookup
        )

        weights = weights * lookup_weights
        weights = weights / (weights.sum(0) + 1e-5)

        return (rgb_lookup * weights).sum(0)

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

        ## Get jittered depth
        jitter_depth_outputs = self(jitter_rays)
        jitter_depth = jitter_depth_outputs[..., self.num_views:self.num_views+1]
        jitter_rgb_blend = self.get_ray_color(jitter_rays, jitter_depth_outputs)

        return jitter_rays, jitter_rgb_blend, depth_proj, jitter_depth, backproj_points

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
        ) / self.get_dataset().far

        return color_weight_map, depth_weight_map

    def loss(self, train_batch, batch_idx):
        if self.cur_iter < 0:
            return 0.0

        system = self.get_system()
        batch = self.get_batch(train_batch, batch_idx)

        ## Forward
        rays, jitter_rays = batch['rays'], batch['jitter_rays']
        rgb = system(rays)['rgb']

        ## Get depth outputs
        depth_outputs = self(rays)
        depth = depth_outputs[..., self.num_views:self.num_views+1]
        rgb_blend = self.get_ray_color(rays, depth_outputs)

        ## Jittered outputs
        jitter_rays, jitter_rgb_blend, depth_proj, jitter_depth, backproj_points = \
            self.get_jittered_outputs(
                rays, jitter_rays, depth
            )
        jitter_rgb = system(jitter_rays)['rgb']

        ## Weight maps
        color_weight_map, depth_weight_map = self.get_weight_maps(
            rays,
            jitter_rays
        )

        if True:
            color_loss = self.color_loss_fn(
                rgb * color_weight_map,
                rgb_blend * color_weight_map
            ) * self.cfg.color_loss.weight

            return color_loss
        else:
            ## Color consistency
            color_loss = self.color_loss_fn(
                rgb * color_weight_map,
                jitter_rgb * color_weight_map
            ) * self.cfg.color_loss.weight

            ## Depth consistency
            depth_loss = self.depth_loss_fn(
                depth_proj * depth_weight_map,
                jitter_depth * depth_weight_map
            ) * self.cfg.depth_loss.weight

            return color_loss + depth_loss

    def set_iter(self, i):
        super().set_iter(i)

        if not self.cfg.use_color_embedding:
            self.pe.set_iter(self.cur_iter)

    def validation(self, rays):
        system = self.get_system()
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        ## Depth outputs
        depth_outputs = self(rays)
        depth = depth_outputs[..., self.num_views:self.num_views+1]
        rgb_blend = self.get_ray_color(rays, depth_outputs)

        rgb_blend = rgb_blend.view(H, W, 3).cpu()
        rgb_blend = rgb_blend.permute(2, 0, 1)

        # Depth view
        near, far = self.get_dataset().near, self.get_dataset().far
        depth = depth.view(H, W, 1).cpu()
        depth = depth.permute(2, 0, 1)

        disp = 1.0 / torch.abs(depth)
        disp[torch.abs(depth) < 1e-5] = 0

        depth = (depth - depth.min()) / (depth.max() - depth.min())
        disp = (disp - disp.min()) / (disp.max() - disp.min())

        return {
            'depth': depth,
            'rgb_blend': rgb_blend,
        }

    def validation_video(self, rays):
        # Outputs
        outputs = self.validation(torch.squeeze(rays))

        return {
            'videos/depth': outputs['depth'],
            'ignore/rgb_blend': outputs['rgb_blend']
        }

    def validation_image(self, batch, batch_idx):
        # Outputs
        outputs = self.validation(torch.squeeze(batch['rays']))

        return {
            'depth/pred': outputs['depth'],
            'ignore/rgb_blend': outputs['rgb_blend'],
        }
