#!/usr/bin/env python3

import os
import json
import numpy as np
import torch

import cv2
from PIL import Image

from datasets.base import BaseDataset
from datasets.lightfield import LightfieldDataset

from utils.ray_utils import (
    get_rays,
    get_ndc_rays,
    get_ray_directions,
)

class BlenderLightfieldDataset(LightfieldDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
    ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        # Read meta
        transforms_path = os.path.join(self.root_dir, 'transforms.json')

        with self.pmgr.open(transforms_path, 'r') as f:
            self.meta = json.load(f)

        # Image paths and pose
        self.image_paths = []
        self.poses = []

        for frame in self.meta['frames']:
            # Image path
            image_path = frame['file_path'].split('/')[-1]
            self.image_paths += [image_path]

            # Pose
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]

    def get_rgb(self, s_idx, t_idx):
        idx = t_idx * self.cols + s_idx
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}.png'),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGBA')

        img = cv2.resize(img, self.img_wh, cv2.INTER_AREA)
        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img


class BlenderDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        if self.split == 'render':
            self.read_meta_for_split('test')
        else:
            self.read_meta_for_split(self.split)

    def read_meta_for_split(self, split):
        with self.pmgr.open(
            os.path.join(self.root_dir, f'transforms_{split}.json'),
            'r'
        ) as f:
            self.meta = json.load(f)

        if split == 'val':
            self.meta['frames'] = self.meta['frames'][:self.val_num]

        w, h = self.img_wh

        self.focal = 0.5 * 800 / np.tan(
            0.5 * self.meta['camera_angle_x']
        )
        self.focal *= self.img_wh[0] / 800

        # Bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # Ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)

        # Image paths and pose
        self.image_paths = []
        self.poses = []

        for frame in self.meta['frames']:
            # Image path
            self.image_paths += [frame['file_path']]

            # Pose
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]

        self.poses = np.stack(self.poses, axis=0)

    def prepare_render_data(self):
        self.prepare_test_data()

    def get_rays(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)
        return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}.png'),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGBA')

        img = cv2.resize(img, self.img_wh, cv2.INTER_AREA)
        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def get_intrinsics(self):
        K = np.eye(3)
        K[0, 0] = self.focal
        K[0, 2] = self.img_wh[0] / 2
        K[1, 1] = self.focal
        K[1, 2] = self.img_wh[1] / 2

        return K


class DenseBlenderDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        with self.pmgr.open(
            os.path.join(self.root_dir, 'transforms.json'),
            'r'
        ) as f:
            self.meta = json.load(f)
            self.meta['frames'] = self.meta['frames'][:self.cfg.dataset.size]

        if self.split == 'train':
            self.meta['frames'] = self.meta['frames'][::self.cfg.dataset.train_skip]

        w, h = self.img_wh

        self.focal = 0.5 * 800 / np.tan(
            0.5 * self.meta['camera_angle_x']
        )
        self.focal *= self.img_wh[0] / 800

        # Bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # Ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)

        # Image paths and pose
        self.image_paths = []
        self.poses = []

        for frame in self.meta['frames']:
            # Image path
            image_path = frame['file_path'].split('/')[-1]
            self.image_paths += [image_path]

            # Pose
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]

    def prepare_render_data(self):
        self.prepare_test_data()

    def get_rays(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)
        return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}.png'),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGBA')

        img = cv2.resize(img, self.img_wh, cv2.INTER_AREA)
        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def get_intrinsics(self):
        K = np.eye(3)
        K[0, 0] = self.focal
        K[0, 2] = self.img_wh[0] / 2
        K[1, 1] = self.focal
        K[1, 2] = self.img_wh[1] / 2

        return K
