#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from nlf.activations import get_activation

def dot(a, b, axis=-1):
    return torch.sum(a * b, dim=axis)

def intersect_sphere(rays, radius):
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    o = rays_o
    d = rays_d

    dot_o_o = dot(o, o)
    dot_d_d = dot(d, d)
    dot_o_d = dot(o, d)

    a = dot_d_d
    b = 2 * dot_o_d
    c = dot_o_o - radius * radius
    disc = b * b - 4 * a * c

    t1 = (-b + torch.sqrt(disc + 1e-8)) / (2 * a)
    t2 = (-b - torch.sqrt(disc + 1e-8)) / (2 * a)

    t1 = torch.where(
        disc < 0,
        torch.zeros_like(t1),
        t1
    )
    t2 = torch.where(
        disc < 0,
        torch.zeros_like(t2),
        t2
    )
    t2 = torch.where(
        t2 < 0,
        t1,
        t2
    )

    return rays_o + t2[..., None] * rays_d

def intersect_sphere_twice(rays, radius):
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    o = rays_o
    d = rays_d

    dot_o_o = dot(o, o)
    dot_d_d = dot(d, d)
    dot_o_d = dot(o, d)

    a = dot_d_d
    b = 2 * dot_o_d
    c = dot_o_o - radius * radius
    disc = b * b - 4 * a * c

    t1 = (-b + torch.sqrt(disc + 1e-5)) / (2 * a)
    t2 = (-b - torch.sqrt(disc + 1e-5)) / (2 * a)

    t1 = torch.where(
        disc < 0,
        torch.zeros_like(t1),
        t1
    )
    t2 = torch.where(
        disc < 0,
        torch.zeros_like(t2),
        t2
    )
    p1 = rays_o + t1[..., None] * rays_d
    p2 = rays_o + t2[..., None] * rays_d

    return torch.cat([p1, p2], dim=-1)

def intersect_axis_plane(rays, val, dim, exclude=False, normalize=False):
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    t = (val - rays_o[..., dim]) / rays_d[..., dim]
    loc = rays_o + t[..., None] * rays_d

    t = torch.where(
        torch.isnan(t),
        torch.zeros_like(t),
        t
    )

    if exclude:
        loc = torch.cat(
            [
                loc[..., :dim],
                loc[..., dim+1:],
            ],
            dim=-1
        )

        t = torch.stack([t, t], dim=-1)
    else:
        t = torch.stack([t, t, t], dim=-1)

    if normalize:
        loc = loc / torch.maximum(torch.abs(val), torch.ones_like(val)).unsqueeze(-1)

    return loc, t

def intersect_plane(rays, normal, distance, exclude_dim=-1):
    dim = exclude_dim

    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    o_dot_n = dot(rays_o, normal)
    d_dot_n = dot(rays_d, normal)
    t = (distance - o_dot_n) / (d_dot_n)
    loc = rays_o + t[..., None] * rays_d

    t = torch.where(
        torch.abs(d_dot_n) < 1e-5,
        torch.zeros_like(t),
        t
    )

    if exclude_dim != -1:
        loc = torch.cat(
            [
                loc[..., :dim],
                loc[..., dim+1:],
            ],
            dim=-1
        )

        t = torch.stack([t, t], dim=-1)
    else:
        t = torch.stack([t, t, t], dim=-1)

    return loc, t

def planar_intersect(cfg):
    use_norm = cfg.use_norm if 'use_norm' in cfg else False
    out_channels = cfg.out_channels_per_z

    near = cfg.near if 'near' in cfg else -1.0
    far = cfg.far if 'far' in cfg else 0.0
    initial_z = cfg.initial_z if 'initial_z' in cfg else 0.0

    if 'voxel_size' in cfg and cfg.voxel_size is not None:
        near = cfg.near if 'near' in cfg else -0.5 * cfg.voxel_size
        far = cfg.far if 'far' in cfg else 0.5 * cfg.voxel_size

    z_diff = far - near

    def intersect_fn(rays, param_rays, z_vals):
        z_vals = z_vals * z_diff + initial_z

        if use_norm:
            r = (z_vals - near) / z_diff
            fac = 1.0 / torch.sqrt(((-r + 1) * (-r + 1) + r * r))
            fac = fac[..., None].repeat(1, 1, out_channels).view(fac.shape[0], -1)

        if out_channels == 3:
            points, _ = intersect_axis_plane(rays[..., None, :], z_vals, 2, exclude=False)
        else:
            points, _ = intersect_axis_plane(rays[..., None, :], z_vals, 2, exclude=True)

        if use_norm:
            return points.view(points.shape[0], -1) * fac
        else:
            return points.view(points.shape[0], -1)

    return intersect_fn

def planar_intersect_normal(cfg):
    out_channels = cfg.out_channels_per_z
    no_degen = cfg.no_degen if 'no_degen' in cfg else True

    near = cfg.near if 'near' in cfg else -1.0
    far = cfg.far if 'far' in cfg else 0.0
    initial_distance = cfg.initial_distance if 'initial_distance' in cfg else 0.0

    if 'voxel_size' in cfg and cfg.voxel_size is not None:
        near = cfg.near if 'near' in cfg else -0.5 * cfg.voxel_size
        far = cfg.far if 'far' in cfg else 0.5 * cfg.voxel_size

    distance_scale = (far - near)

    def intersect_fn(rays, param_rays, normal):
        if no_degen:
            normal = normal.view(normal.shape[0], -1, 3)
            normal = torch.cat([normal[..., :2], torch.ones_like(normal[..., :1]), normal[..., -1:]], -1)
        else:
            normal = normal.view(normal.shape[0], -1, 4)

        distance = normal[..., -1] * distance_scale + initial_distance
        normal = normal[..., :3]

        if out_channels == 3:
            points, _ = intersect_plane(rays[..., None, :], normal, distance)
        else:
            points, _ = intersect_plane(rays[..., None, :], normal, distance, exclude_dim=2)

        return points.view(points.shape[0], -1)

    return intersect_fn

def planar_intersect_matrix(cfg):
    tform_act = get_activation(cfg.tform_activation)

    near = cfg.near if 'near' in cfg else -1.0
    far = cfg.far if 'far' in cfg else 0.0
    initial_z = cfg.initial_z if 'initial_z' in cfg else 0.0

    if 'voxel_size' in cfg and cfg.voxel_size is not None:
        near = cfg.near if 'near' in cfg else -0.5 * cfg.voxel_size
        far = cfg.far if 'far' in cfg else 0.5 * cfg.voxel_size

    z_diff = far - near

    def intersect_fn(rays, param_rays, z_vals):
        z_vals = z_vals * z_diff + (initial_z - near)
        r = z_vals / z_diff

        zero = torch.zeros_like(r)
        m = torch.stack(
            [
                -r + 1, zero, r, zero,
                zero, -r + 1, zero, r,
            ],
            -1
        ).view(z_vals.shape[0], -1)
        m = tform_act(m).view(-1, z_vals.shape[-1] * 2, 4)
        return (m @ param_rays.unsqueeze(-1)).squeeze(-1)

    return intersect_fn

def spherical_intersect(cfg):
    use_norm = cfg.use_norm if 'use_norm' in cfg else False
    initial_radius = cfg.initial_radius
    out_channels = cfg.out_channels_per_z
    voxel_size = cfg.voxel_size if 'voxel_size' in cfg else 1.0

    def intersect_fn(rays, param_rays, z_vals):
        with torch.no_grad():
            rays_o, rays_d = rays[..., :3], rays[..., 3:6]
            rays_d = torch.nn.functional.normalize(rays_d, p=2, dim=-1)
            m = torch.cross(rays_o, rays_d, dim=-1)
            m = torch.cross(rays_d, m, dim=-1)
            min_radius = torch.linalg.norm(m, ord=2, dim=-1, keepdim=True) + 1e-5

        z_vals = (z_vals + initial_radius) * voxel_size
        z_vals = torch.where(
            z_vals < min_radius,
            min_radius,
            z_vals
        )
        points = intersect_sphere_twice(rays[..., None, :], z_vals)
        points = torch.where(
            (z_vals < min_radius)[..., None].repeat(1, 1, 6),
            (torch.nn.functional.normalize(m, p=2, dim=-1)[..., None, :] * z_vals[..., None]).repeat(1, 1, 2),
            points
        )

        return points.view(points.shape[0], -1)

    return intersect_fn

def triplane_intersect(cfg):
    out_channels = cfg.out_channels_per_z

    min_point = cfg.min_point
    max_point = cfg.max_point
    voxel_size = cfg.voxel_size if 'voxel_size' in cfg else 1.0

    def intersect_fn(rays, param_rays, z_vals):
        z_vals = z_vals * voxel_size

        num_points = z_vals.shape[-1] // 3
        x, y, z = torch.split(z_vals, num_points, -1)

        points_x, _ = intersect_axis_plane(rays[..., None, :], x, 0, exclude=False, normalize=True)
        points_y, _ = intersect_axis_plane(rays[..., None, :], y, 1, exclude=False, normalize=True)
        points_z, _ = intersect_axis_plane(rays[..., None, :], z, 2, exclude=False, normalize=True)

        points_x = torch.clip(points_x, min_point[0], max_point[0])
        points_y = torch.clip(points_y, min_point[1], max_point[1])
        points_z = torch.clip(points_z, min_point[2], max_point[2])

        return torch.cat([points_x, points_y, points_z], -1)

    return intersect_fn

intersect_dict = {
    'plane_point': planar_intersect,
    'plane_normal': planar_intersect_normal,
    'plane_matrix': planar_intersect_matrix,
    'sphere_point': spherical_intersect,
    'triplane_point': triplane_intersect,
}
