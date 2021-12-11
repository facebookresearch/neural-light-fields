import numpy as np
import torch
from torch import nn
from losses import loss_dict

from nlf.regularizers.base import BaseRegularizer

class RayBundleRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.range = cfg.range
        self.jitter = cfg.jitter

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

        #### Losses ####

        all_losses = {
            loss: 0.0 for loss in self.loss_fns.keys()
        }

        ## Compute weight
        rgb = rgb.unsqueeze(-2)
        rays = rays.unsqueeze(-2)

        if self._do_loss('color_loss'):
            color_weight_map = self.get_weight_map(
                rays,
                jitter_rays,
                'color_weight_map'
            )

        ## Color consistency loss
        if self._do_loss('color_loss'):
            sh = jitter_rays.shape
            jitter_rgb = system(jitter_rays.view(-1, jitter_rays.shape[-1]), apply_ndc=dataset.use_ndc)['rgb']
            jitter_rgb = jitter_rgb.view(sh[0], sh[1], jitter_rgb.shape[-1])

            all_losses['color_loss'] = self._loss_fn(
                'color_loss',
                (rgb * color_weight_map).permute(1, 0, 2),
                (jitter_rgb * color_weight_map).permute(1, 0, 2),
            )

        ## Total loss
        total_loss = 0.0

        for name in all_losses.keys():
            print(name + ':', all_losses[name])
            total_loss += all_losses[name]

        return total_loss
