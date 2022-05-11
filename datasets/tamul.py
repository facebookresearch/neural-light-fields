#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np

from PIL import Image

from datasets.lightfield import LightfieldDataset


class TamulLightfieldDataset(LightfieldDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

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
