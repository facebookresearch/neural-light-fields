#!/usr/bin/env python3

import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_max # @manual

from utils.ray_utils import (
    intersect_axis_plane,
    intersect_sphere,
    ray_param_dict,
    ray_param_pos_dict,
)

from third_party.intersect import (
    svo_ray_intersect,
)

from third_party.intersect.utils import (
    trilinear_interp,
    build_easy_octree,
    vertices_from_points,
    offset_points,
    offset_points_faces,
    splitting_points,
)


class Subdivision(nn.Module):
    def __init__(
        self,
        system,
        cfg
        ):
        super().__init__()

        self.cfg = cfg

        ## (Hack) Prevent from storing system variables
        self.systems = [system]

        self.update_every = cfg.update_every if 'update_every' in cfg \
            else float('inf')

        if self.update_every == 'inf':
            self.update_every = float('inf')

    def get_system(self):
        return self.systems[0]

    def get_dataset(self):
        return self.systems[0].trainer.datamodule.train_dataset

    def intersect(self, rays):
        pass

    def get_codes(self, pts, dirs):
        return torch.zeros(pts.shape[:-1] + (0,)).type_as(pts)

    def forward(self, rays):
        isect_pts, isect_depths = self.intersect(rays)
        isect_dirs = rays[..., 3:6].unsqueeze(1).repeat(
            1, isect_pts.shape[1], 1
        )

        isect_codes = self.get_codes(isect_pts, isect_dirs)

        return torch.cat(
            [isect_pts, isect_dirs, isect_pts, isect_codes], -1
        ), isect_depths

    def validation(self, rays, results):
        system = self.get_system()
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        depth = results['depth'].view(H, W, 1).cpu().numpy()
        depth = depth.transpose(2, 0, 1)

        disp = 1 / depth
        disp[depth == 0] = 0

        disp = (disp - disp.min()) / (disp.max() - disp.min())
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        accum = results['accum'].view(H, W, 1).cpu().numpy()
        accum = accum.transpose(2, 0, 1)

        return {
            'depth': depth,
            'disp': disp,
            'accum': accum,
        }

    def validation_video(self, rays, results):
        outputs = self.validation(rays, results)

        return {
            'videos/subdivision_depth': outputs['depth'],
            'videos/subdivision_disp': outputs['disp'],
            'videos/subdivision_accum': outputs['accum'],
        }

    def validation_image(self, batch, batch_idx, results):
        outputs = self.validation(batch['rays'], results)

        return {
            'images/subdivision_depth': outputs['depth'],
            'images/subdivision_disp': outputs['disp'],
            'images/subdivision_accum': outputs['accum'],
        }

    def update(self):
        pass


class DepthSubdivision(Subdivision):
    def __init__(
        self,
        system,
        cfg
        ):
        super().__init__(system, cfg)

        self.near = cfg.near
        self.far = cfg.far
        self.voxel_size = cfg.voxel_size
        self.num_slices = int(np.round((self.far - self.near) / self.voxel_size))
        self.max_hits = self.num_slices

        # Depths
        self.depths = torch.linspace(
            self.near, self.far - self.voxel_size, self.num_slices
        )

    def forward(self, rays):
        with torch.no_grad():
            isect_pts, isect_depths, isect_idx = self.intersect(rays)

        isect_rays, isect_centers = self.process_intersect(
            rays, isect_pts, isect_idx
        )

        return torch.cat(
            [isect_rays, isect_centers], -1
        ), isect_depths, isect_idx, isect_idx.eq(-1)

    def process_intersect(self, rays, pts, idx):
        # Reparametrize rays
        rays = rays[..., :6].unsqueeze(1).repeat(1, self.num_slices, 1)
        rays[..., 2] = rays[..., 2] - self.depths[None].to(rays)
        centers = torch.ones_like(rays[..., 0:1]) * self.depths[None, ..., None].to(rays)

        return rays, centers

    def intersect(self, rays):
        rays = rays[..., :6].unsqueeze(1).repeat(1, self.num_slices, 1)

        depths = self.depths.view(1, self.num_slices).to(rays)
        depths = torch.tile(depths, (rays.shape[0], 1))

        isect_pts, isect_depth = intersect_axis_plane(
            rays, depths, -1
        )
        isect_depth = isect_depth[..., -1]
        isect_idx = torch.ones_like(isect_depth).long()

        return isect_pts, isect_depth, isect_idx


class RadialSubdivision(Subdivision):
    def __init__(
        self,
        system,
        cfg
        ):
        super().__init__(system, cfg)

        self.start_radius = cfg.start_radius
        self.end_radius = cfg.end_radius
        self.num_slices = cfg.num_slices
        self.latent_dim = cfg.latent_dim

        # Depths
        if cfg.use_disparity:
            inv_radii = torch.linspace(
                1.0 / self.end_radius, 1.0 / self.start_radius, self.num_slices
            )
            self.radii = 1.0 / inv_radii
        else:
            self.radii = torch.linspace(
                self.start_radius, self.end_radius, self.num_slices
            )
            self.radii = torch.flip(self.radii, (0,))

        self.radii = self.radii + cfg.offset

        # Latent codes
        latent_codes_init = torch.randn(
            (self.num_slices, self.latent_dim)
        ) * 0.01 / np.sqrt(self.latent_dim)

        self.latent_codes = nn.Parameter(
            latent_codes_init, requires_grad=True
        )

    def get_codes(self, pts, dirs):
        isect_codes = self.latent_codes[None].repeat(
            pts.shape[0], 1, 1
        ).to(pts.device)
        isect_codes = nn.functional.normalize(isect_codes, dim=-1)
        return isect_codes

    def intersect(self, rays):
        rays = rays[..., :6].unsqueeze(1)
        rays = torch.tile(rays, (1, self.num_slices, 1))

        radii = self.radii.view(1, self.num_slices).to(rays.device)
        radii = torch.tile(radii, (rays.shape[0], 1))

        isect_pts = intersect_sphere(rays, radii)
        isect_depth = torch.norm(
            rays[..., :3] - isect_pts, dim=-1
        )

        # Sort
        sort_idx = torch.argsort(isect_depth, dim=-1)
        sort_idx_pts = torch.stack(
            [sort_idx, sort_idx, sort_idx], dim=1
        )

        isect_depth = torch.gather(isect_depth, -1, sort_idx)

        isect_pts = isect_pts.permute(0, 2, 1)
        isect_pts = torch.gather(isect_pts, -1, sort_idx_pts)
        isect_pts = isect_pts.permute(0, 2, 1)

        return isect_pts, isect_depth


def voxels_from_bb(min_point, max_point, voxel_size):
    steps = ((max_point - min_point) / voxel_size).round().astype('int64') + 1
    x, y, z = [
        c.reshape(-1).astype('float32') for c in np.meshgrid(
            np.arange(steps[0]),
            np.arange(steps[1]),
            np.arange(steps[2])
        )
    ]
    x = x * voxel_size + min_point[0]
    y = y * voxel_size + min_point[1]
    z = z * voxel_size + min_point[2]

    return np.stack([x, y, z]).T.astype('float32')


class VoxelSubdivision(Subdivision):
    def __init__(
        self,
        system,
        cfg
        ):
        super().__init__(system, cfg)

        self.num_update_iters = 0
        self.split_every = cfg.split_every if 'split_every' in cfg else 'inf'

        ## Other parameters
        self.forward_facing = False if 'forward_facing' not in cfg else cfg.forward_facing
        self.no_voxel = False if 'no_voxel' not in cfg else cfg.no_voxel
        self.latent_dim = cfg.latent_dim

        # Setup
        if 'bootstrap' in cfg:
            with system.pmgr.open(cfg.bootstrap.path_prefix + '_density.npy', 'rb') as f:
                density_grid = torch.Tensor(
                    np.load(f)
                ).reshape(-1, 1)

            with system.pmgr.open(cfg.bootstrap.path_prefix + '_points.npy', 'rb') as f:
                points = torch.Tensor(
                    np.load(f)
                ).reshape(-1, 3)

                points = points[
                    (density_grid > self.threshold()).repeat(1, 3)
                ].reshape(-1, 3)

            with system.pmgr.open(cfg.bootstrap.path_prefix + '_voxel_size.npy', 'rb') as f:
                voxel_size = np.load(f) + 1e-8
        else:
            min_point = np.array(cfg.min_point)
            max_point = np.array(cfg.max_point)

            voxel_size = cfg.voxel_size + 1e-8
            points = torch.Tensor(
                voxels_from_bb(min_point, max_point, voxel_size)
            )

        # Build
        voxel_centers, voxel_children = build_easy_octree(
            points, voxel_size / 2
        )
        voxel_centers = voxel_centers[None]
        voxel_children = voxel_children[None]

        voxel_keys, voxel_vertices, num_keys = vertices_from_points(
            points, voxel_size / 2
        )
        num_voxels = voxel_vertices.shape[0]

        ## Intersection
        if 'bootstrap' not in cfg:
            if self.forward_facing:
                self.max_hits = int(np.round((max_point[2] - min_point[2]) / voxel_size)) + 1
            else:
                self.max_hits = int(np.round((max_point[2] - min_point[2]) / voxel_size)) * 2 + 1
        else:
            self.max_hits = cfg.max_hits + 1

        ## Register
        self.register_buffer('voxel_size', torch.scalar_tensor(voxel_size))
        self.register_buffer('num_keys', torch.scalar_tensor(num_keys).long())
        self.register_buffer('num_voxels', torch.scalar_tensor(num_voxels).long())

        self.register_buffer('points', points)
        self.register_buffer('voxel_centers', voxel_centers)
        self.register_buffer('voxel_children', voxel_children)
        self.register_buffer('voxel_keys', voxel_keys)
        self.register_buffer('voxel_vertices', voxel_vertices)

    def forward(self, rays):
        with torch.no_grad():
            isect_pts, isect_depths, isect_idx = self.intersect(rays)

        isect_dirs = rays[..., 3:6].unsqueeze(1).repeat(
            1, isect_pts.shape[1], 1
        )
        isect_rays, isect_centers = self.process_intersect(
            rays, isect_pts, isect_dirs, isect_idx
        )

        return torch.cat(
            [isect_rays, isect_centers], -1
        ), isect_depths, isect_idx, isect_idx.eq(-1)

    def process_intersect(self, rays, pts, dirs, idx):
        # Mask out invalid
        mask = idx.eq(-1)

        # Get codes
        voxel_centers = self.voxel_centers.type_as(pts)

        idx = torch.where(
            mask,
            torch.zeros_like(idx),
            idx
        )
        centers = F.embedding(idx, voxel_centers[0])
        idx = torch.where(
            mask,
            -torch.ones_like(idx),
            idx
        )

        # Reparametrize rays
        if self.no_voxel:
            centers[..., :2] = 0

        voxel_pts = rays[..., None, 0:3] - centers

        if self.no_voxel:
            centers = centers[..., -1:]

        pts = torch.where(
            mask.unsqueeze(-1).repeat(1, 1, 3),
            pts,
            voxel_pts
        )
        rays = torch.cat([pts, dirs], -1)

        return rays, centers

    def intersect(self, rays):
        rays_o = rays[..., :3].contiguous().view(1, -1, 3)
        rays_d = rays[..., 3:6].contiguous().view(1, -1, 3)

        ## Intersect voxel grid
        voxel_centers = self.voxel_centers.type_as(rays).contiguous()
        voxel_children = self.voxel_children.to(rays.device).contiguous()

        isect_idx, min_depth, max_depth = svo_ray_intersect(
            self.voxel_size, self.max_hits,
            voxel_centers, voxel_children,
            rays_o, rays_d, self.forward_facing
        )

        ## Sort the depths
        min_depth.masked_fill_(isect_idx.eq(-1), 10000.0)
        max_depth.masked_fill_(isect_idx.eq(-1), 10000.0)

        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        isect_depths = min_depth

        isect_idx = isect_idx.gather(-1, sorted_idx).long()
        isect_idx[..., -1] = -1 # Background ray

        ## Get outputs
        isect_depths = isect_depths.view(-1, isect_depths.shape[-1])[..., :self.max_hits]

        isect_pts = rays[..., None, :3] + isect_depths[..., None] * rays[..., None, 3:6]
        isect_pts = isect_pts.view(-1, isect_pts.shape[-2], 3)

        isect_idx = isect_idx[..., :self.max_hits].view(-1, isect_depths.shape[-1])

        ## Return
        return isect_pts, isect_depths, isect_idx

    def threshold(self):
        if isinstance(self.cfg.threshold, float):
            return self.cfg.threshold
        elif self.cfg.threshold.type == 'linear':
            t = self.num_update_iters / float(self.cfg.max_update_iters)
            return (1.0 - t) * self.cfg.threshold.start + t * self.cfg.threshold.end
        else:
            return self.cfg.threshold

    def prepare_update(self):
        self.voxel_scores = torch.zeros(self.points.shape[:-1]).cuda()

    def post_update(self):
        self.num_update_iters += 1

        if self.num_update_iters > self.cfg.max_update_iters:
            return

        ## Prune
        keep = (self.voxel_scores > self.threshold())
        keep_sum = keep.sum().detach().cpu()

        if keep_sum > 0:
            num_points = self.points.shape[0]
            self.points = self.points[keep.unsqueeze(-1).repeat(1, 3)].view(-1, 3)

            self.voxel_centers, self.voxel_children = build_easy_octree(
                self.points, self.voxel_size / 2
            )
            self.voxel_centers = self.voxel_centers[None]
            self.voxel_children = self.voxel_children[None]

            self.voxel_keys, self.voxel_vertices, self.num_keys = vertices_from_points(
                self.points, self.voxel_size / 2
            )
            self.num_keys = torch.scalar_tensor(self.num_keys).long()
            self.num_voxels = torch.scalar_tensor(self.voxel_vertices.shape[0]).long()
            print("Keep:", keep_sum, "Remove:", num_points - keep_sum)

        ## Split
        do_split = self.split_every != 'inf' \
            and (self.num_update_iters % self.split_every == 0)

        if do_split:
            self.voxel_size = torch.scalar_tensor(self.voxel_size / 2)

            new_points, new_vertices, new_codes, new_keys = splitting_points(
                self.points.cuda(),
                self.voxel_vertices.cuda(),
                None,
                self.voxel_size
            )

            ## New octree
            self.voxel_centers, self.voxel_children = build_easy_octree(
                self.points, self.voxel_size
            )
            self.voxel_centers = self.voxel_centers[None]
            self.voxel_children = self.voxel_children[None]

            self.voxel_keys, self.voxel_vertices, self.num_keys = vertices_from_points(
                self.points, self.voxel_size / 2
            )
            self.num_keys = torch.scalar_tensor(self.num_keys).long()
            self.num_voxels = torch.scalar_tensor(self.voxel_vertices.shape[0]).long()

        print("Number of remaining points:", self.points.shape[0])
        print("New voxel size:", self.voxel_size)

    def update(self):
        system = self.get_system()
        batch_size = system.trainer.datamodule.cur_batch_size
        dataset = self.get_dataset()
        num_batches = len(dataset) // batch_size

        self.prepare_update()

        for idx in range(num_batches + 1):
            batch_start = idx * batch_size
            batch_end = (idx + 1) * batch_size
            batch = dataset[batch_start:batch_end]

            self.update_iter(batch, idx)

        self.post_update()

    def update_iter(self, batch, batch_idx):
        system = self.get_system()
        batch_size = system.trainer.datamodule.cur_batch_size
        print("Update iter:", batch_idx)

        with torch.no_grad():
            # Get outputs
            outputs = system(batch['rays'].cuda(), include_all=True)

            weights = outputs['all_weights'].view(-1)
            isect_idx = outputs['all_indices'].view(-1)

            # Mask
            mask = isect_idx.eq(-1)
            isect_idx[mask] = 0
            weights[mask] = 0

            # Record alphas
            print(self.voxel_scores.shape, isect_idx.shape, weights.shape)
            scatter_max(weights.view(-1), isect_idx.view(-1), dim=-1, out=self.voxel_scores)


class LatentVoxelSubdivision(VoxelSubdivision):
    def __init__(
        self,
        system,
        cfg
        ):

        super().__init__(system, cfg)

        self.is_coarse = cfg.is_coarse if 'is_coarse' in cfg else True

        # Voxel features
        if self.latent_dim > 0:
            self.voxel_codes = nn.Embedding(
                self.num_voxels, self.latent_dim, padding_idx=None
            )
            nn.init.normal_(self.voxel_codes.weight, mean=0, std=self.latent_dim ** -0.5)

    def forward(self, rays):
        with torch.no_grad():
            isect_pts, isect_depths, isect_idx = self.intersect(rays)

        isect_dirs = rays[..., 3:6].unsqueeze(1).repeat(
            1, isect_pts.shape[1], 1
        )
        isect_rays, isect_centers, isect_codes = self.process_intersect(
            rays, isect_pts, isect_dirs, isect_idx
        )

        if self.is_coarse:
            isect_pts = isect_centers

        return torch.cat(
            [isect_rays, isect_pts, isect_codes], -1
        ), isect_depths, isect_idx, isect_idx.eq(-1)

    def get_voxel_codes(self, pts, idx, mask):
        if self.latent_dim > 0:
            voxel_codes = self.voxel_codes.weight.type_as(pts)
            voxel_codes = F.embedding(idx, voxel_codes).view(
                pts.shape[0], pts.shape[1], self.latent_dim
            )
            voxel_codes = F.normalize(voxel_codes, dim=-1)

            voxel_codes = torch.where(
                mask.unsqueeze(-1).repeat(1, 1, voxel_codes.shape[-1]),
                torch.zeros_like(voxel_codes),
                voxel_codes
            )
        else:
            voxel_codes = torch.zeros_like(pts[..., 0:0])

        return voxel_codes

    def process_intersect(self, rays, pts, dirs, idx):
        rays, centers = super().process_intersect(rays, pts, dirs, idx)

        # Mask
        mask = idx.eq(-1)

        # Get codes
        idx = torch.where(
            mask,
            torch.zeros_like(idx),
            idx
        )
        voxel_codes = self.get_voxel_codes(pts, idx, mask)
        idx = torch.where(
            mask,
            -torch.ones_like(idx),
            idx
        )

        return rays, centers, voxel_codes

    def post_update(self):
        super().post_update()

        ## Set new voxel embeddings
        if self.latent_dim > 0:
            self.voxel_codes = nn.Embedding(
                self.num_voxels, self.latent_dim, padding_idx=None
            )
            nn.init.normal_(self.voxel_codes.weight, mean=0, std=self.latent_dim ** -0.5)


subdivision_dict = {
    'depth': DepthSubdivision,
    'radial': RadialSubdivision,
    'voxel': VoxelSubdivision,
    'latent_voxel': LatentVoxelSubdivision,
}
