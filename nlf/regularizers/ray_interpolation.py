import numpy as np
import torch
from torch import nn

from nlf.regularizers.base import BaseRegularizer
from nlf.nets import net_dict
from nlf.embedding import (
    SequentialEmbedding,
    RayParam,
    WindowedPE
)
from utils.ray_utils import ray_param_dict, ray_param_pos_dict
from losses import loss_dict
from utils.ray_utils import dot, get_weight_map, weighted_stats


class RayInterpolationRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        ## Blending net
        self.num_views = cfg.dataset.num_views
        self.num_points = cfg.points_per_view * self.num_views
        self.embedding_channels = system.cfg.model.embedding_net.out_channels

        if True:
            self.blending_net = net_dict[cfg.blending_net.type](
                self.num_points + 3 * self.num_points,
                3,
                cfg.blending_net,
            )
        else:
            self.blending_net = net_dict[cfg.blending_net.type](
                self.num_points,
                self.num_points,
                cfg.blending_net,
            )

        self.cosine_sim_fn = torch.nn.CosineSimilarity(dim=-1)

    def get_depths(self):
        near, far = self.get_dataset().near, self.get_dataset().far

        if self.cfg.use_disparity:
            inv_depths = torch.linspace(
                1.0 / far, 1.0 / near, self.num_points
            )
            depths = torch.flip(1.0 / inv_depths, (0,))
        else:
            depths = torch.linspace(
                near, far, self.num_points
            )

        depths = depths[torch.randperm(depths.shape[0])]

        return depths

    def forward(self, rays):
        ## Backproject
        depths = self.get_depths().type_as(rays)
        sorted_idx = torch.argsort(depths, dim=-1).to(rays.device).long()
        depths = depths.view( -1, 1, 1)

        rays = rays.unsqueeze(0).repeat(
            self.num_points, 1, 1
        )

        backproj_points = rays[..., :3] + rays[..., 3:6] * depths

        ## Lookup
        rays = rays.view(self.num_views, -1, rays.shape[-1])
        backproj_points = backproj_points.view(self.num_views, -1, backproj_points.shape[-1])

        rgb_lookup, lookup_rays, _ = self.get_dataset().lookup_points_single(
            backproj_points
        )

        ## Get embeddings, similarities
        embed_rays = self.embed(rays).view(
            self.num_points, -1, self.embedding_channels
        ).permute(1, 0, 2)

        embed_lookup_rays = self.embed(lookup_rays).view(
            self.num_points, -1, self.embedding_channels
        ).permute(1, 0, 2)

        cosine_sim = self.cosine_sim_fn(embed_rays, embed_lookup_rays)

        if True:
            rgb_lookup = rgb_lookup.reshape(self.num_points, -1, 3)
            sorted_idx_rgb = sorted_idx[..., None, None].repeat(
                1, rgb_lookup.shape[1], rgb_lookup.shape[2]
            )
            rgb_lookup = torch.gather(rgb_lookup, 0, sorted_idx_rgb)
            rgb_lookup = rgb_lookup.permute(1, 0, 2).reshape(-1, self.num_points * 3)

            sorted_idx_sim = sorted_idx[None].repeat(
                cosine_sim.shape[0], 1
            )
            cosine_sim = torch.gather(cosine_sim, -1, sorted_idx_sim)
            blend_inps = torch.cat([cosine_sim, rgb_lookup], -1)

            rgb_interp = self.blending_net(blend_inps)
        else:
            rgb_lookup = rgb_lookup.reshape(self.num_points, -1, 3).permute(
                1, 0, 2
            )

            weights = torch.nn.functional.softmax(cosine_sim, dim=-1)
            rgb_interp = (rgb_lookup * weights.unsqueeze(-1)).sum(1)

        return {
            'rgb_interp': rgb_interp
        }

    def loss(self, train_batch, batch_idx):
        if self.cur_iter < 0:
            return 0.0

        dataset = self.get_dataset()
        batch = self.get_batch(train_batch, batch_idx)

        ## Forward
        if dataset.use_ndc:
            rays = batch['rays_no_ndc']
        else:
            rays = batch['rays']

        rgb = batch['rgb']

        #### Losses ####

        all_losses = {
            loss: 0.0 for loss in self.loss_fns.keys()
        }

        ## Color loss
        outputs = self.run_chunked(rays)

        if self._do_loss('color_loss'):
            all_losses['color_loss'] = self._loss_fn(
                'color_loss',
                outputs['rgb_interp'],
                rgb
            )

        ## Total loss
        total_loss = 0.0

        for name in all_losses.keys():
            print(name + ':', all_losses[name])
            total_loss += all_losses[name]

        return total_loss

    def set_iter(self, i):
        super().set_iter(i)

    def validation(self, batch):
        system = self.get_system()
        dataset = self.get_dataset()
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        if dataset.use_ndc:
            rays = batch['rays_no_ndc'].squeeze()
        else:
            rays = batch['rays'].squeeze()

        outputs = self.run_chunked(rays)

        rgb_interp = outputs['rgb_interp']
        rgb_interp = rgb_interp.view(H, W, 3).cpu()
        rgb_interp = rgb_interp.permute(2, 0, 1)

        return {
            'rgb_interp': rgb_interp
        }

    def validation_video(self, batch):
        # Outputs
        outputs = self.validation(batch)

        return {
            'videos/rgb_interp': outputs['rgb_interp']
        }

    def validation_image(self, batch, batch_idx):
        # Outputs
        outputs = self.validation(batch)

        return {
            'ignore/rgb_interp': outputs['rgb_interp'],
        }
