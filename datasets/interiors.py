#!/usr/bin/env python3

import os
import cv2
from PIL import Image

from datasets.lightfield import LightfieldDataset

class InteriorsLightfieldDataset(LightfieldDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        self.image_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'images/'))
        )

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f"{image_path}.png"),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGBA')

        img = cv2.resize(img, self.img_wh, cv2.INTER_AREA)
        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img
