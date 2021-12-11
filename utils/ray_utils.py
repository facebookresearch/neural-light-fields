import torch
import numpy as np

from kornia import create_meshgrid

def get_lightfield_rays(
    U, V, s, t, aspect, st_scale=1.0, uv_scale=1.0, near=-1, far=0,
    use_inf=False, center_u=0.0, center_v=0.0,
    ):
    u = torch.linspace(-1, 1, U, dtype=torch.float32)
    v = torch.linspace(-1, 1, V, dtype=torch.float32) / aspect

    vu = list(torch.meshgrid([v, u]))
    u = vu[1] * uv_scale
    v = vu[0] * uv_scale
    s = torch.ones_like(vu[1]) * s * st_scale
    t = torch.ones_like(vu[0]) * t * st_scale

    if use_inf:
        rays = torch.stack(
            [
                s,
                t,
                near * torch.ones_like(s),
                u + center_u - s,
                v + center_v - t,
                (far - near) * torch.ones_like(s),
            ],
            axis=-1
        ).view(-1, 6)
    else:
        rays = torch.stack(
            [
                s,
                t,
                near * torch.ones_like(s),
                u - s,
                v - t,
                (far - near) * torch.ones_like(s),
            ],
            axis=-1
        ).view(-1, 6)

    return torch.cat(
        [
            rays[..., 0:3],
            torch.nn.functional.normalize(rays[..., 3:6], p=2, dim=-1)
        ],
        -1
    )

def get_epi_rays_horz(
    U, v, S, t, aspect, st_scale=1.0, uv_scale=1.0, near=-1, far=0,
    use_inf=False, center_u=0.0, center_v=0.0,
    ):
    u = torch.linspace(-1, 1, U, dtype=torch.float32)
    s = torch.linspace(-1, 1, S, dtype=torch.float32) / aspect

    su = list(torch.meshgrid([s, u]))
    u = su[1] * uv_scale
    v = torch.ones_like(su[0]) * v * uv_scale
    s = su[0] * st_scale
    t = torch.ones_like(su[0]) * t * st_scale

    if use_inf:
        rays = torch.stack(
            [
                s,
                t,
                near * torch.ones_like(s),
                u + center_u - s,
                v + center_v - t,
                (far - near) * torch.ones_like(s),
            ],
            axis=-1
        ).view(-1, 6)
    else:
        rays = torch.stack(
            [
                s,
                t,
                near * torch.ones_like(s),
                u - s,
                v - t,
                (far - near) * torch.ones_like(s),
            ],
            axis=-1
        ).view(-1, 6)

    return torch.cat(
        [
            rays[..., 0:3],
            torch.nn.functional.normalize(rays[..., 3:6], p=2, dim=-1)
        ],
        -1
    )

def reparam_inf(rays, focal, z, S_ext, T_ext):
    ST_ext = torch.tensor([S_ext, T_ext])[None].to(rays.device)

    uv = rays[..., :2]
    st = rays[..., -2:]
    st = torch.cat([-st[..., :1], st[..., -1:]], axis=-1)

    uv_p = (focal / z) * ST_ext * st + uv
    st_p = st

    return torch.cat([uv_p, st_p], axis=-1)

def get_ray_directions(H, W, focal):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)

    directions = \
        torch.stack(
            [
                (i - W / 2 + 0.5) / focal, -(j - H / 2 + 0.5) / focal, -torch.ones_like(i)
            ],
            -1
            )

    return directions

def get_ray_directions_cx_cy(H, W, focal, cx, cy):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)

    directions = \
        torch.stack(
            [
                (i - cx + 0.5) / focal, -(j - cy + 0.5) / focal, -torch.ones_like(i)
            ],
            -1
            )

    return directions

def get_ray_directions_K(H, W, K, centered_pixels=False):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)

    if centered_pixels:
        directions = \
            torch.stack(
                [
                    (i - K[0, 2] + 0.5) / K[0, 0], -(j - K[1, 2] + 0.5) / K[1, 1], -torch.ones_like(i)
                ],
                -1
                )
    else:
        directions = \
            torch.stack(
                [
                    (i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -torch.ones_like(i)
                ],
                -1
                )

    return directions

def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = torch.nn.functional.normalize(rays_d, p=2, dim=-1)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ndc_rays(H, W, focal, near, rays):
    rays_o, rays_d = rays[..., 0:3], rays[..., 3:6]

    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]

    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    rays_d = torch.nn.functional.normalize(rays_d, p=2, dim=-1)

    return torch.cat([rays_o, rays_d], -1)

def get_ndc_rays_fx_fy(H, W, fx, fy, near, rays):
    rays_o, rays_d = rays[..., 0:3], rays[..., 3:6]

    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]

    # Projection
    o0 = -1./(W/(2.*fx)) * ox_oz
    o1 = -1./(H/(2.*fy)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*fx)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*fy)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    rays_d = torch.nn.functional.normalize(rays_d, p=2, dim=-1)

    return torch.cat([rays_o, rays_d], -1)

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
    p1 = rays_o + t1[..., None] * rays_d
    p2 = rays_o + t2[..., None] * rays_d

    return torch.cat([p1, p2], dim=-1)

def intersect_axis_plane(rays, val, dim, exclude=False):
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

    return loc, t

def identity(cfg):
    def param(x):
        return x[..., :6]

    return param

def position(cfg):
    def pos(rays):
        return rays[..., :3]

    return pos

def two_plane_param(cfg):
    near = cfg.near if 'near' in cfg else -1.0
    far = cfg.far if 'far' in cfg else 0.0

    pre_mult = 1.0
    post_mult = 1.0

    if 'voxel_size' in cfg:
        near = cfg.near if 'near' in cfg else -0.5
        far = cfg.far if 'far' in cfg else 0.5

        pre_mult = 1.0 / cfg.voxel_size
        post_mult = cfg.voxel_size

    if 'multiplier' in cfg:
        post_mult *= cfg.multiplier

    def param(rays):
        rays = rays * pre_mult

        isect_pts_1, _ = intersect_axis_plane(
            rays, near, 2, exclude=True
        )

        isect_pts_2, _ = intersect_axis_plane(
            rays, far, 2, exclude=True
        )

        param_rays = torch.cat([isect_pts_1, isect_pts_2], dim=-1)
        param_rays = param_rays * post_mult

        return param_rays

    return param

def two_plane_param_no_voxel(cfg):
    near = cfg.near if 'near' in cfg else -0.5
    far = cfg.far if 'far' in cfg else 0.5

    pre_mult = 1.0
    post_mult = 1.0

    def param(rays):
        rays = rays * pre_mult

        isect_pts_1, _ = intersect_axis_plane(
            rays, near, 2, exclude=True
        )

        isect_pts_2, _ = intersect_axis_plane(
            rays, far, 2, exclude=True
        )

        param_rays = torch.cat([isect_pts_1, isect_pts_2], dim=-1)
        param_rays = param_rays * post_mult

        return param_rays

    return param

def two_plane_with_z_param(cfg):
    near = cfg.near if 'near' in cfg else -1.0
    far = cfg.far if 'far' in cfg else 0.0

    pre_mult = 1.0
    post_mult = 1.0

    if 'voxel_size' in cfg:
        near = cfg.near if 'near' in cfg else -0.5
        far = cfg.far if 'far' in cfg else 0.5

        pre_mult = 1.0 / cfg.voxel_size
        post_mult = cfg.voxel_size

    if 'multiplier' in cfg:
        post_mult *= cfg.multiplier

    def param(rays):
        rays = rays * pre_mult

        isect_pts_1, _ = intersect_axis_plane(
            rays, near, 2, exclude=False
        )

        isect_pts_2, _ = intersect_axis_plane(
            rays, far, 2, exclude=False
        )

        param_rays = torch.cat([isect_pts_1, isect_pts_2], dim=-1)
        param_rays = param_rays * post_mult

        return param_rays

    return param

def two_plane_pos(cfg):
    near = cfg.near if 'near' in cfg else -1.0
    far = cfg.far if 'far' in cfg else 0.0

    pre_mult = 1.0
    post_mult = 1.0

    if 'voxel_size' in cfg:
        near = cfg.near if 'near' in cfg else -0.5
        far = cfg.far if 'far' in cfg else 0.5

        pre_mult = 1.0 / cfg.voxel_size
        post_mult = cfg.voxel_size

    if 'multiplier' in cfg:
        post_mult *= cfg.multiplier

    def pos(rays):
        rays = rays * pre_mult

        isect_pts, _ = intersect_axis_plane(
            rays, near, 2, exclude=False
        )

        return isect_pts * post_mult

    return pos

def pluecker(cfg):
    def param(rays):
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        rays_d = torch.nn.functional.normalize(rays_d, p=2, dim=-1)

        m = torch.cross(rays_o, rays_d, dim=-1)

        return torch.cat([rays_d, m], dim=-1)

    return param

def pluecker_pos(cfg):
    def pos(rays):
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        rays_d = torch.nn.functional.normalize(rays_d, p=2, dim=-1)

        m = torch.cross(rays_o, rays_d, dim=-1)
        rays_o = torch.cross(rays_d, m, dim=-1)

        return rays_o

    return pos

def spherical_param(cfg):
    def param(rays):
        isect_pts = intersect_sphere(
            rays,
            cfg.bs_radius
        ) / cfg.bs_radius

        return torch.cat([isect_pts, rays[..., 3:6]], dim=-1)

    return param

def two_sphere_pos(cfg):
    def pos(rays):
        isect_pts = intersect_sphere_twice(
            rays,
            cfg.bs_radius
        )

        return isect_pts[..., 3:6]

    return pos

def two_sphere_param(cfg):
    def param(rays):
        isect_pts = intersect_sphere_twice(
            rays,
            cfg.bs_radius
        ) / cfg.bs_radius

        return isect_pts

    return param

def cube_param(cfg):
    def param(rays):
        # Placeholders
        _, pos_t = intersect_axis_plane(
            rays, cfg.bbox[0], 0, True
        )
        pos_t = torch.ones_like(pos_t) * float("inf")
        neg_t = -torch.ones_like(pos_t) * float("inf")

        for i in range(6):
            _, cur_t = intersect_axis_plane(
                rays, cfg.bbox[i], i % 3, True
            )

            is_best_pos = (cur_t > torch.zeros_like(cur_t)) \
                & (cur_t < pos_t)
            is_best_neg = (cur_t < torch.zeros_like(cur_t)) \
                & (cur_t > neg_t)

            pos_t[is_best_pos] = cur_t[is_best_pos]
            neg_t[is_best_neg] = cur_t[is_best_neg]

        # Get intersections
        all_ints = []

        for i in range(6):
            cur_int, cur_t = intersect_axis_plane(
                rays, cfg.bbox[i], i % 3, True
            )

            is_best_pos = (cur_t > torch.zeros_like(cur_t)) \
                & (cur_t < (pos_t + 1e-5))
            is_best_neg = (cur_t < torch.zeros_like(cur_t)) \
                & (cur_t > (neg_t - 1e-5))

            cur_int[~is_best_pos & ~is_best_neg] = 0
            all_ints.append(cur_int)

        return torch.cat(all_ints, dim=-1)

    return param

def get_stats(rays):
    return (rays.mean(0), rays.std(0))

def get_weight_map(
    rays,
    jitter_rays,
    cfg,
    weights=None,
    softmax=True
):
    ray_dim = rays.shape[-1] // 2

    # Angles
    angles = torch.acos(
        torch.clip(
            dot(rays[..., ray_dim:], jitter_rays[..., ray_dim:]),
            -1 + 1e-8, 1 - 1e-8
        )
    ).detach()

    # Distances
    dists = torch.linalg.norm(
        rays[..., :ray_dim] - jitter_rays[..., :ray_dim],
        dim=-1
    ).detach()

    # Weights
    if weights is None:
        weights = torch.zeros_like(angles)

    if softmax:
        weights = torch.nn.functional.softmax(
            0.5 * -(torch.square(angles / cfg.angle_std) + torch.square(dists / cfg.dist_std)) + weights, dim=0
        )[..., None]
    else:
        #print("Angle:", angles.max(), angles.mean(), cfg.angle_std)
        #print("Dist:", dists.max(), dists.mean(), cfg.dist_std)

        weights = torch.exp(
            0.5 * -(torch.square(angles / cfg.angle_std) + torch.square(dists / cfg.dist_std)) + weights
        )[..., None]

    # Normalization constant
    constant = np.power(2 * np.pi * cfg.angle_std * cfg.angle_std, -1.0 / 2.0) \
        * np.power(2 * np.pi * cfg.dist_std * cfg.dist_std, -1.0 / 2.0)

    return weights / constant


def weighted_stats(rgb, weights):
    weights_sum = weights.sum(0)
    rgb_mean = ((rgb * weights).sum(0) / weights_sum)
    rgb_mean = torch.where(
        weights_sum == 0,
        torch.zeros_like(rgb_mean),
        rgb_mean
    )

    diff = rgb - rgb_mean.unsqueeze(0)
    rgb_var = (diff * diff * weights).sum(0) / weights_sum
    rgb_var = torch.where(
        weights_sum == 0,
        torch.zeros_like(rgb_var),
        rgb_var
    )

    return rgb_mean, rgb_var


ray_param_dict = {
    'identity': identity,
    'pluecker': pluecker,
    'position': position,
    'spherical': spherical_param,
    'two_plane': two_plane_param,
    'two_plane_no_voxel': two_plane_param_no_voxel,
    'two_plane_with_z': two_plane_with_z_param,
    'two_sphere': two_sphere_param,
    'cube': cube_param,
}

ray_param_pos_dict = {
    'pluecker': pluecker_pos,
    'two_sphere': two_sphere_pos,
    'two_plane': two_plane_pos,
}
