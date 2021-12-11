#!/usr/bin/env python3

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

from iopath.common.file_io import (
    PathManager,
    NativePathHandler
)

from utils.pose_utils import (
    interpolate_poses,
    create_spiral_poses,
)

import copy
# @manual //github/third-party/omry/omegaconf:omegaconf
from omegaconf import OmegaConf


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
    ):

        # Path manager
        self.pmgr = PathManager()
        self.pmgr.register_handler(NativePathHandler())

        # Copy train dataset config
        if 'train_dataset' in kwargs:
            base_dataset_cfg = copy.deepcopy(kwargs['train_dataset'].cfg.dataset)
            OmegaConf.set_struct(base_dataset_cfg, False)

            for key in cfg.dataset.keys():
                base_dataset_cfg.__dict__[key] = cfg.dataset[key]
                setattr(base_dataset_cfg, key, cfg.dataset[key])

            cfg.dataset = base_dataset_cfg

        ## Dataset cfg
        self.cfg = cfg
        self.split = split if 'split' not in cfg.dataset else cfg.dataset.split
        self.dataset_cfg = cfg.dataset[self.split] if self.split in cfg.dataset else cfg.dataset

        # Basic dataset params
        self.root_dir = os.path.expanduser(self.dataset_cfg.root_dir)
        self._img_wh = tuple(self.dataset_cfg.img_wh)
        self.img_wh = self._img_wh
        self.aspect = (float(self.img_wh[0]) / self.img_wh[1])
        self.use_ndc = self.dataset_cfg.use_ndc if 'use_ndc' in self.dataset_cfg else False

        # Validation
        self.val_num = self.dataset_cfg.val_num
        self.val_skip = self.dataset_cfg.val_skip
        self.val_set = self.dataset_cfg.val_set if 'val_set' in self.dataset_cfg else ""
        self.val_crop = self.dataset_cfg.val_crop if 'val_crop' in self.dataset_cfg else 1.0
        self.val_all = self.dataset_cfg.val_all if 'val_all' in self.dataset_cfg else False

        # Rendering
        self.render_interpolate = self.dataset_cfg.render_params.interpolate
        self.render_supersample = self.dataset_cfg.render_params.supersample
        self.render_max_frames = self.dataset_cfg.render_params.max_frames if 'max_frames' in self.dataset_cfg.render_params else 0
        self.render_crop = self.dataset_cfg.render_params.crop

        # Crop
        self.precrop_iters = self.dataset_cfg.precrop_iters if 'precrop_iters' in self.dataset_cfg else 0
        self.use_crop = self.precrop_iters > 0
        self.cur_iter = 0
        self.precrop_frac = self.dataset_cfg.precrop_frac if 'precrop_fac' in self.dataset_cfg else 0.5

        # Patch loading
        self.use_patches = self.dataset_cfg.use_ndc if 'use_patches' in self.dataset_cfg else False
        self.blur_radius = self.dataset_cfg.blur_radius if 'blur_radius' in self.dataset_cfg else 0

        # Setup data
        self.define_transforms()
        self.prepare_data()

    def read_meta(self):
        pass

    def calculate_scene_bounds(self):
        pass

    def prepare_train_data(self):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_rays = []
        self.all_rgb = []

        if self.use_ndc:
            self.all_ndc_rays = []

        for idx in range(len(self.image_paths)):
            # Rays
            self.all_rays += [self.get_rays(idx)]

            if self.use_ndc:
                self.all_ndc_rays += [self.to_ndc(self.all_rays[-1])]

            # Color
            self.all_rgb += [self.get_rgb(idx)]

        self.all_rays = torch.cat(self.all_rays, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)

        if self.use_ndc:
            self.all_ndc_rays = torch.cat(self.all_ndc_rays, 0)

        if self.use_patches or self.use_crop:
            self._all_rays = torch.clone(self.all_rays)
            self._all_rgb = torch.clone(self.all_rgb)

            if self.use_ndc:
                self._all_ndc_rays = torch.clone(self.all_ndc_rays)

    def prepare_val_data(self):
        self.prepare_test_data()

    def prepare_test_data(self):
        pass

    def prepare_render_data(self):
        if not self.render_interpolate:
            close_depth, inf_depth = self.bounds.min()*.9, self.bounds.max()*5.
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focus_depth = mean_dz

            radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
            self.poses = create_spiral_poses(self.poses, radii, focus_depth)
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)

    def prepare_data(self):
        self.read_meta()
        self.calculate_scene_bounds()

        if self.split == 'train':
            self.prepare_train_data()
        elif self.split == 'val':
            self.prepare_val_data()
        elif self.split == 'test':
            self.prepare_test_data()
        elif self.split == 'render':
            self.prepare_render_data()

    def define_transforms(self):
        if self.blur_radius > 0:
            self.transform = T.Compose([
                T.ToTensor(),
                T.GaussianBlur(
                    (self.blur_radius * 2 + 1, self.blur_radius * 2 + 1),
                    self.blur_radius / 3.0
                )
            ])
        else:
            self.transform = T.ToTensor()

    def get_rays(self, idx):
        pass

    def get_rgb(self, idx):
        pass

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'val':
            return min(self.val_num, len(self.poses))
        elif self.split == 'render':
            if self.render_max_frames > 0:
                return  min(self.render_max_frames, len(self.poses))
            else:
                return len(self.poses)
        else:
            return len(self.poses)

    def __getitem__(self, idx):
        if self.split == 'render':
            batch = {
                'rays': self.get_rays(idx),
                'idx': idx
            }

            if self.use_ndc:
                batch['rays_no_ndc'] = batch['rays']
                batch['rays'] = self.to_ndc(batch['rays'])

        elif self.split == 'val' or self.split == 'test':
            batch = {
                'rays': self.get_rays(idx),
                'rgb': self.get_rgb(idx),
                'idx': idx
            }

            if self.use_ndc:
                batch['rays_no_ndc'] = batch['rays']
                batch['rays'] = self.to_ndc(batch['rays'])
        else:
            batch = {
                'rays': self.all_rays[idx],
                'rgb': self.all_rgb[idx],
                'idx': idx
            }

            if self.use_ndc:
                batch['rays_no_ndc'] = batch['rays']
                batch['rays'] = self.all_ndc_rays[idx]


        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        return batch

    def get_batch(self, batch_idx, batch_size, jitter=None):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size

        return self[batch_start:batch_end]

    def crop_all_tensors(
        self,
        t,
        W,
        H,
        dW,
        dH,
    ):
        t = t.view(self.num_images, H, W, -1)

        t = t[
            :,
            (H//2 - dH):(H // 2 + dH + 1),
            (W//2 - dW):(W // 2 + dW + 1),
        ]

        return t.reshape(-1, t.shape[-1])

    def crop_one_tensor(
        self,
        t,
        W,
        H,
        dW,
        dH,
    ):
        t = t.view(1, H, W, -1)

        t = t[
            :,
            (H//2 - dH):(H // 2 + dH + 1),
            (W//2 - dW):(W // 2 + dW + 1),
        ]

        H, W = t.shape[1], t.shape[2]

        return W, H, t.reshape(-1, t.shape[-1])

    def crop(self):
        if self.use_crop and self.cur_iter < self.precrop_iters:
            W = self.img_wh[0]
            H = self.img_wh[1]
            dW = int(W//2 * self.precrop_frac)
            dH = int(H//2 * self.precrop_frac)

            self.all_rays = self.crop_all_tensors(self._all_rays, W, H, dW, dH)
            self.all_rgb = self.crop_all_tensors(self._all_rgb, W, H, dW, dH)

            if self.use_ndc:
                self.all_ndc_rays = self.crop_all_tensors(self._all_ndc_rays, W, H, dW, dH)

    def scale(self, scale):
        self.img_wh = (self._img_wh[0] // scale, self._img_wh[1] // scale)
        self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

        self.define_transforms()
        self.prepare_data()

    def crop_batch(self, batch):
        W = self.img_wh[0]
        H = self.img_wh[1]

        if self.split == 'val' or self.split == 'test':
            crop = self.val_crop
        elif self.split == 'render':
            crop = self.render_crop
        else:
            crop = 1.0

        if crop < 1.0:
            dW = int(W//2 * crop)
            dH = int(H//2 * crop)

            for k in batch.keys():
                if torch.is_tensor(batch[k]):
                    temp_W, temp_H, batch[k] = self.crop_one_tensor(batch[k], W, H, dW, dH)

            W, H = temp_W, temp_H

        return W, H, batch

    def shuffle(self):
        if not self.use_patches:
            perm = torch.tensor(
                np.random.permutation(len(self))
            )
            self.all_rays = self.all_rays[perm]
            self.all_rgb = self.all_rgb[perm]

            if self.use_ndc:
                self.all_ndc_rays = self.all_ndc_rays[perm]
        else:
            self.shuffle_patches()

    def patchify_tensor(
        self,
        t,
        width,
        height,
        patch_offset,
        patch_width,
    ):
        c = t.shape[-1]
        t = t.view(self.num_images, height, width, c)

        # Remove boundaries
        p = self.blur_radius

        if p > 0:
            t = t[:, p:-p, p:-p]

        # Patch offset
        t = t[:, patch_offset:, patch_offset:]

        # Crop to multiple of patch width
        round_height = (t.shape[1] // patch_width) * patch_width
        round_width = (t.shape[2] // patch_width) * patch_width
        t = t[:, :round_height, :round_width]

        t = t.reshape(
            t.shape[0],
            round_height // patch_width,
            patch_width,
            round_width // patch_width,
            patch_width,
            c
        ).permute(0, 1, 3, 2, 4, 5)

        return t.reshape(-1, patch_width * patch_width, c)

    def shuffle_patches(self):
        print("Shuffle patches")

        # Patchify
        patch_width = self.dataset_cfg.patch_width
        width, height = self.img_wh[0], self.img_wh[1]
        patch_offset = int(np.random.uniform() * patch_width)

        self.all_rays = self.patchify_tensor(
            self._all_rays,
            width,
            height,
            patch_offset,
            patch_width
        )

        self.all_rgb = self.patchify_tensor(
            self._all_rgb,
            width,
            height,
            patch_offset,
            patch_width
        )

        if self.use_ndc:
            self.all_ndc_rays = self.patchify_tensor(
                self._all_ndc_rays,
                width,
                height,
                patch_offset,
                patch_width
            )

        # Shuffle
        perm = torch.tensor(
            np.random.permutation(self.all_rays.shape[0])
        )

        self.all_rays = self.all_rays[perm].reshape(-1, self.all_rays.shape[-1])
        self.all_rgb = self.all_rgb[perm].reshape(-1, self.all_rgb.shape[-1])

        if self.use_ndc:
            self.all_ndc_rays = self.all_ndc_rays[perm].reshape(-1, self.all_ndc_rays.shape[-1])

    def get_intrinsics_screen_space(self):
        K = np.copy(self.get_intrinsics())
        K[0, 2] = (K[0, 2] - self.img_wh[0] / 2)
        K[1, 2] = (K[1, 2] - self.img_wh[1] / 2)
        K[0, :] = 2 * K[0, :] / self.img_wh[0]
        K[1, :] = -2 * K[1, :] / self.img_wh[1]
        return K

    def get_intrinsics(self):
        pass

    def to_ndc(self, x):
        return x
