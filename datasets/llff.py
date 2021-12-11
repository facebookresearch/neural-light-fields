#!/usr/bin/env python3

import torch

import numpy as np
import os

from PIL import Image
import imageio
import cv2

from datasets.base import BaseDataset
from utils.pose_utils import (
    correct_poses_bounds,
    create_spiral_poses,
)
from utils.ray_utils import (
    get_rays,
    get_ray_directions_K,
    get_ndc_rays_fx_fy,
    ray_param_dict,
)


class LLFFDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses_bounds.npy'), 'rb'
        ) as f:
            poses_bounds = np.load(f)

        self.image_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'images/'))
        )

        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        self.bounds = poses_bounds[:, -2:]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]
        self.cx, self.cy = W / 2.0, H / 2.0

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Step 2: correct poses, bounds
        self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
            poses, self.bounds
        )
        self.near = self.bounds.min()
        self.far = self.bounds.max()

        # Step 3: Ray directions for all pixels
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K
        )

        # Step 4: Holdout validation images
        if self.val_set != "":
            val_indices = [int(i) for i in self.val_set.split(",")]
        else:
            self.val_skip = min(
                len(self.image_paths), self.val_skip
            )
            val_indices = list(range(0, len(self.image_paths), self.val_skip))

        train_indices = [i for i in range(len(self.image_paths)) if i not in val_indices]

        if self.val_all:
            val_indices = [i for i in train_indices]

        if self.split == 'val' or self.split == 'test':
            self.image_paths = [self.image_paths[i] for i in val_indices]
            self.poses = self.poses[val_indices]
        elif self.split == 'train':
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.poses = self.poses[train_indices]

    def get_intrinsics(self):
        return self.K

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], 1.0, rays
        )

    def get_rays(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)
        return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        # Colors
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, 'images', image_path),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGB')

        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img


class DenseLLFFDataset(LLFFDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        ## Bounds
        with self.pmgr.open(
            os.path.join(self.root_dir, 'bounds.npy'), 'rb'
        ) as f:
            bounds = np.load(f)

        self.bounds = bounds[:, -2:]

        ## Poses
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses.npy'), 'rb'
        ) as f:
            poses = np.load(f)

        ## Image paths
        self.image_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'images/'))
        )

        ## Skip
        row_skip = self.dataset_cfg.train_row_skip
        col_skip = self.dataset_cfg.train_col_skip

        poses_skipped = []
        image_paths_skipped = []

        for row in range(self.dataset_cfg.num_rows):
            for col in range(self.dataset_cfg.num_cols):
                idx = row * self.dataset_cfg.num_cols + col

                if self.split == 'train' and (
                    (row % row_skip) != 0 or (col % col_skip) != 0 or (idx % self.val_skip) == 0
                    ):
                    continue

                if (self.split == 'val' or self.split == 'test') and (
                    ((row % row_skip) == 0 and (col % col_skip) == 0) and (idx % self.val_skip) != 0
                    ):
                    continue

                poses_skipped.append(poses[idx])
                image_paths_skipped.append(self.image_paths[idx])

        poses = np.stack(poses_skipped, axis=0)
        self.poses = poses.reshape(-1, 3, 5)
        self.image_paths = image_paths_skipped

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]
        self.cx, self.cy = W / 2.0, H / 2.0

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Step 2: correct poses, bounds
        self.near = self.bounds.min()
        self.far = self.bounds.max()

        # Step 3: Ray directions for all pixels
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K
        )
