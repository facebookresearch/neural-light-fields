#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from datasets.base import BaseDataset
from utils.ray_utils import (
    get_lightfield_rays
)


class LightfieldDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        ## Dataset cfg
        self.cfg = cfg
        self.split = split if 'split' not in cfg.dataset else cfg.dataset.split
        self.dataset_cfg = cfg.dataset[self.split] if self.split in cfg.dataset else cfg.dataset

        ## Param

        # Lightfield params
        self.rows = self.dataset_cfg.lightfield.rows
        self.cols = self.dataset_cfg.lightfield.cols
        self.step = self.dataset_cfg.lightfield.step

        self.st_scale = self.dataset_cfg.lightfield.st_scale if 'st_scale' in self.dataset_cfg.lightfield else 1.0
        self.num_images = (self.rows // self.step + 1) * (self.cols // self.step + 1)

        self.near = 0
        self.far = 1

        # Validation and testing
        self.val_all = self.dataset_cfg.val_all if 'val_all' in self.dataset_cfg else False
        self.val_pairs = self.dataset_cfg.val_pairs if 'val_pairs' in self.dataset_cfg else []

        if len(self.val_pairs) > 0:
            self.val_pairs = list(zip(self.val_pairs[::2], self.val_pairs[1::2]))
            self.num_test_images = len(self.val_pairs)
        elif self.val_all:
            self.num_test_images = self.rows * self.cols
        else:
            self.num_test_images = self.rows * self.cols - self.num_images

        # Render params
        self.disp_row = self.dataset_cfg.lightfield.disp_row
        self.supersample = self.dataset_cfg.lightfield.supersample

        self.render_spiral = self.dataset_cfg.render_params.spiral if 'spiral' in self.dataset_cfg.render_params else False
        self.render_far = self.dataset_cfg.render_params.far if 'far' in self.dataset_cfg.render_params else False

        self.spiral_rad = self.dataset_cfg.render_params.spiral_rad if 'spiral_rad' in self.dataset_cfg.render_params else 0.5
        self.uv_downscale = self.dataset_cfg.render_params.uv_downscale if 'uv_downscale' in self.dataset_cfg.render_params else 0.0

        if 'vis_st_scale' in self.dataset_cfg.lightfield:
            self.vis_st_scale = self.dataset_cfg.lightfield.vis_st_scale \
                if self.dataset_cfg.lightfield.vis_st_scale is not None else self.st_scale
        else:
            self.vis_st_scale = self.st_scale

        super().__init__(cfg, split, **kwargs)

        self.poses = [self.get_coord(st_idx) for st_idx in self.all_st_idx]

    def read_meta(self):
        pass

    def prepare_train_data(self):
        self.all_rays = []
        self.all_rgb = []
        self.all_st_idx = []

        for t_idx in range(0, self.rows, self.step):
            for s_idx in range(0, self.cols, self.step):
                if (s_idx, t_idx) in self.val_pairs:
                    continue

                # Rays
                self.all_rays += [self.get_rays(s_idx, t_idx)]

                # Color
                self.all_rgb += [self.get_rgb(s_idx, t_idx)]

                # Pose
                self.all_st_idx.append((s_idx, t_idx))

        self.all_rays = torch.cat(self.all_rays, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)
        self.all_weights = self.get_weights()
        self.all_inputs = torch.cat(
            [self.all_rays, self.all_rgb, self.all_weights], -1
        )

    def prepare_val_data(self):
        self.prepare_test_data()

    def prepare_test_data(self):
        self.all_st_idx = []

        for t_idx in range(self.rows):
            for s_idx in range(self.cols):
                if len(self.val_pairs) == 0:
                    if (t_idx % self.step) == 0 and (s_idx % self.step) == 0 \
                        and not (self.val_all and self.split == 'val'):
                        continue
                elif (s_idx, t_idx) not in self.val_pairs:
                    continue

                self.all_st_idx.append((s_idx, t_idx))

    def prepare_render_data(self):
        if not self.render_spiral:
            self.all_st_idx = []
            t_idx = self.disp_row

            for s_idx in range(self.cols * self.supersample):
                self.all_st_idx.append((s_idx / self.supersample, t_idx))
        else:
            N = 120
            rots = 2
            scale = self.spiral_rad

            self.all_st_idx = []

            for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
                s = (np.cos(theta) * scale + 1) / 2.0 * (self.cols - 1)
                t = -np.sin(theta) * scale / 2.0 * (self.rows - 1) + ((self.rows - 1) - self.disp_row)

                self.all_st_idx.append((s, t))

    def get_coord(self, st_idx):
        s = (st_idx[0] / (self.cols - 1)) * 2 - 1 \
            if self.cols > 1 else 0
        t = -(((st_idx[1] / (self.rows - 1)) * 2 - 1) \
            if self.rows > 1 else 0)

        return (s, t)

    def get_rays(self, s_idx, t_idx):
        if self.split == 'render':
            st_scale = self.vis_st_scale
        else:
            st_scale = self.st_scale

        s, t = self.get_coord((s_idx, t_idx))

        if self.render_spiral or self.render_far:
            return get_lightfield_rays(
                self.img_wh[0], self.img_wh[1], s, t, self.aspect,
                st_scale=st_scale,
                use_inf=True, center_u=-s*self.uv_downscale, center_v=-t*self.uv_downscale
            )
        else:
            return get_lightfield_rays(
                self.img_wh[0], self.img_wh[1], s, t, self.aspect,
                st_scale=st_scale
            )

    def get_rgb(self, s_idx, t_idx):
        pass

    def get_closest_rgb(self, query_st):
        W = self.img_wh[0]
        H = self.img_wh[1]

        images = self.all_rgb.view(self.num_images, H, W, -1)
        dists = np.linalg.norm(
            np.array(self.poses) - np.array(query_st)[None], axis=-1
        )
        return images[list(np.argsort(dists))[0]]

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'val':
            return min(self.val_num, self.num_test_images)
        elif self.split == 'render':
            if not self.render_spiral:
                return self.supersample * self.cols
            else:
                return 120
        else:
            return self.num_test_images

    def __getitem__(self, idx):
        if self.split == 'render':
            s_idx, t_idx = self.all_st_idx[idx]

            batch = {
                'rays': LightfieldDataset.get_rays(self, s_idx, t_idx),
                'pose': self.poses[idx],
                'idx': idx,
                's_idx': s_idx,
                't_idx': t_idx,
            }

        elif self.split == 'val' or self.split == 'test':
            s_idx, t_idx = self.all_st_idx[idx]

            batch = {
                'rays': self.get_rays(s_idx, t_idx),
                'rgb': self.get_rgb(s_idx, t_idx),
                'idx': idx,
                's_idx': s_idx,
                't_idx': t_idx,
            }
        else:
            batch = {
                'inputs': self.all_inputs[idx],
            }

        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        return batch
