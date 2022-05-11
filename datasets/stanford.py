#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np

from PIL import Image

from datasets.lightfield import LightfieldDataset

from utils.ray_utils import (
    get_lightfield_rays
)


class StanfordLightfieldDataset(LightfieldDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        self.use_file_coords = cfg.dataset.lightfield.use_file_coords if 'use_file_coords' in cfg.dataset.lightfield else False

        super().__init__(cfg, split, **kwargs)

        if self.split == 'train':
            self.poses = []

            for (s_idx, t_idx) in self.all_st_idx:
                idx = t_idx * self.cols + s_idx
                coord = self.normalize_coord(self.camera_coords[idx])
                self.poses.append(coord)

    def read_meta(self):
        self.image_paths = sorted(
            self.pmgr.ls(self.root_dir)
        )

        if self.img_wh is None:
            image_path = self.image_paths[0]

            with self.pmgr.open(
                os.path.join(self.root_dir, image_path),
                'rb'
            ) as im_file:
                img = np.array(Image.open(im_file).convert('RGB'))

            self._img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

        self.camera_coords = []

        if self.use_file_coords:
            for image_path in self.image_paths:
                if self.dataset_cfg.collection in ['beans', 'tarot', 'tarot_small', 'knights']:
                    yx = image_path.split('_')[-2:]
                    y = float(yx[0])
                    x = float(yx[1].split('.png')[0])
                else:
                    yx = image_path.split('_')[-3:-1]
                    y, x = -float(yx[0]), float(yx[1])

                self.camera_coords.append((x, y))

    def get_camera_range(self):
        xs = [coord[0] for coord in self.camera_coords]
        ys = [coord[1] for coord in self.camera_coords]

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)

        return (min_x, max_x), (min_y, max_y)

    def normalize_coord(self, coord):
        x_range, y_range = self.get_camera_range()
        aspect = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0])

        norm_x = ((coord[0] - x_range[0]) / (x_range[1] - x_range[0])) * 2 - 1
        norm_y = (((coord[1] - y_range[0]) / (y_range[1] - y_range[0])) * 2 - 1) / aspect

        return (norm_x, norm_y)

    def get_rays(self, s_idx, t_idx):
        if not self.use_file_coords:
            return super().get_rays(s_idx, t_idx)

        idx = t_idx * self.cols + s_idx
        coord = self.normalize_coord(self.camera_coords[idx])

        if self.split == 'render':
            st_scale = self.vis_st_scale
        else:
            st_scale = self.st_scale

        return get_lightfield_rays(
            self.img_wh[0], self.img_wh[1],
            coord[0], coord[1],
            self.aspect,
            st_scale=st_scale
        )

    def get_rgb(self, s_idx, t_idx):
        idx = t_idx * self.cols + s_idx
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, image_path),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGB')

        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img
