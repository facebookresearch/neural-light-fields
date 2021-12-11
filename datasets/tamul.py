#!/usr/bin/env python3

import os
import numpy as np

import cv2
from PIL import Image

from datasets.lightfield import LightfieldDataset

from utils.ray_utils import (
    get_lightfield_rays
)


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

    def get_rgb(self, s_idx, t_idx):
        idx = t_idx * self.cols + s_idx
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, image_path),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGB')

        img = cv2.resize(img, self.img_wh, cv2.INTER_AREA)
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img
