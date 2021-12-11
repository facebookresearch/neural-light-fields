import numpy as np
import torch
from torch import nn
from losses import loss_dict

from nlf.regularizers.base import BaseRegularizer
from utils.ray_utils import dot

from kornia.filters import gaussian_blur2d


class TeacherRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.loss_fn = loss_dict[self.cfg.loss.type]()
        self.use_inp_freq = 'inf'

    def _loss(self, train_batch, batch_idx):
        system = self.get_system()

        if self.loss_weight() < self.cfg.weight.stop_weight:
            return 0.0

        # Get inputs
        dataset = self.get_dataset()
        batch = dataset.get_batch(batch_idx, self.batch_size)

        rays = batch['rays'].type_as(train_batch['rays'])
        rgb = batch['rgb'].type_as(train_batch['rgb'])

        # Loss
        pred_rgb = system(rays)['rgb']

        return self.loss_fn(
            pred_rgb,
            rgb
        )


class BlurryTeacherRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.loss_fn = loss_dict[self.cfg.loss.type]()
        self.use_inp_freq = 'inf'

        self.patch_width = self.cfg.dataset.patch_width
        self.blur_radius = self.cfg.blur_radius
        self.batch_size = self.patch_width * self.patch_width

    def _loss(self, train_batch, batch_idx):
        system = self.get_system()

        if self.loss_weight() < self.cfg.weight.stop_weight:
            return 0.0

        # Get inputs
        dataset = self.get_dataset()
        batch = dataset.get_batch(batch_idx, self.batch_size)

        rays = batch['rays'].type_as(train_batch['rays'])
        rgb = batch['rgb'].type_as(train_batch['rgb'])

        # Run forward and blur
        pred_rgb = system(rays)['rgb']
        pred_rgb = pred_rgb.view(-1, self.patch_width, self.patch_width, 3).permute(0, 3, 1, 2)
        rgb = rgb.view(-1, self.patch_width, self.patch_width, 3).permute(0, 3, 1, 2)

        if self.blur_radius > 0:
            blur_rgb = gaussian_blur2d(
                pred_rgb,
                (self.blur_radius * 2 + 1, self.blur_radius * 2 + 1),
                (self.blur_radius / 3.0, self.blur_radius / 3.0)
            )
            blur_rgb = blur_rgb[
                ...,
                self.blur_radius:-self.blur_radius,
                self.blur_radius:-self.blur_radius
            ]
            rgb = rgb[
                ...,
                self.blur_radius:-self.blur_radius,
                self.blur_radius:-self.blur_radius
            ]
        else:
            blur_rgb = pred_rgb

        # Loss
        return self.loss_fn(
            blur_rgb,
            rgb
        )
