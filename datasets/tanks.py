#!/usr/bin/env python3

import torch

import numpy as np
import os

import cv2
from PIL import Image

from datasets.base import BaseDataset
from utils.pose_utils import (
    p34_to_44,
    correct_poses_bounds,
    create_spiral_poses,
    create_spherical_poses,
    get_bounding_sphere,
    get_bounding_box,
)
from utils.ray_utils import (
    get_rays,
    get_ndc_rays_fx_fy,
    get_ray_directions_K
)

class TanksDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_cam_file(self, filename):
        with self.pmgr.open(filename, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]

        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(
            ' '.join(lines[1:5]), dtype=np.float32, sep=' '
        )
        extrinsics = extrinsics.reshape((4, 4))
        pose = np.linalg.inv(extrinsics)[:3]
        pose[..., 1] *= -1
        pose[..., 2] *= -1

        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(
            ' '.join(lines[7:10]), dtype=np.float32, sep=' '
        )
        intrinsics = intrinsics.reshape((3, 3))

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])
        return intrinsics, pose, depth_min, depth_max

    def read_meta(self):
        # Get paths
        self.image_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'images/'))
        )

        self.cam_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'cams/'))
        )

        if self.split in ['train', 'val']:
            assert len(self.cam_paths) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        # Original width, height
        W, H = 1920, 1080

        # Step 1: Poses, intrinsics, directions, bounds
        poses = []
        intrinsics = []
        bounds = []
        directions = []

        for cam_path in self.cam_paths:
            K, pose, depth_min, depth_max = self.read_cam_file(
                os.path.join(self.root_dir, 'cams', cam_path)
            )
            K[0, :] *= self.img_wh[0] / float(W)
            K[1, :] *= self.img_wh[1] / float(H)
            dirs = get_ray_directions_K(
                self.img_wh[1], self.img_wh[0], K
            )

            poses.append(pose)
            intrinsics.append(K)
            directions.append(dirs)

            if depth_min > 0:
                bounds.append(depth_min)
            if depth_max > 0:
                bounds.append(depth_max)

        self.poses = np.stack(poses, 0)
        self.intrinsics = np.stack(intrinsics, 0)
        self.bounds = np.stack(bounds, 0)
        self.directions = torch.stack(directions, 0)

        # Step 2: correct poses, bounds
        self.poses, self.pose_avg, self.bounds = correct_poses_bounds(
            self.poses, self.bounds, flip=False
        )

        self.near = self.bounds.min()
        self.far = self.bounds.max()

    def prepare_test_data(self):
        mid_pose = self.poses.shape[0] // 4
        pose_center = self.poses[mid_pose:mid_pose+1, :, :]

        if not self.spherical_poses or True:
            focus_depth = 20
            radii = np.percentile(np.abs(self.poses[..., 3]), 10, axis=0)
            self.poses = create_spiral_poses(radii, focus_depth)
            self.poses = pose_center @ p34_to_44(self.poses)
        else:
            self.poses = create_spherical_poses(self.bounds.max())

    def calculate_bounding_shapes(self):
        self.bs_radius = get_bounding_sphere(self.poses) * 2
        self.bbox = get_bounding_box(self.poses)

    def get_rays(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions[idx], c2w)

        # Convert to NDC
        if not self.spherical_poses:
            K = torch.FloatTensor(self.intrinsics[idx])
            rays_o, rays_d = get_ndc_rays_fx_fy(
                self.img_wh[1], self.img_wh[0],
                K[0, 0], K[1, 1], 1.0, rays_o, rays_d
            )

        return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        # Colors
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, 'images', image_path),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGB')

        img = cv2.resize(img, self.img_wh, cv2.INTER_AREA)
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img
