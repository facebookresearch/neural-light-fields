import numpy as np
import os, imageio

from load_llff import _minify, spherify_poses, normalize, ptstocam, viewmatrix, poses_avg, recenter_poses, render_path_spiral, render_path_lightfield


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if width is not None and height is not None:
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif factor is not None and factor != 1:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        width = int(np.round(sh[1] / factor))
        height = int(np.round(sh[0] / factor))
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(np.round(sh[1] / factor))
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(np.round(sh[0] / factor))
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    sh = imageio.imread(imgfiles[0]).shape
    assert sh[1] == width
    assert sh[0] == height

    # compose camera parameters

    imgnames = [f for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    collection = basedir.split('/')[-1]
    st_coords = []

    for name in imgnames:
        if collection in ['beans', 'tarot', 'tarot_small', 'knights']:
            yx = name.split('_')[-2:]
            y = float(yx[0])
            x = float(yx[1].split('.png')[0])
        else:
            yx = name.split('_')[-3:-1]
            y, x = float(yx[0]), float(yx[1])

        st_coords.append([x, y])

    st_coords = np.array(st_coords)

    # normalize st coordinates - camera positions will span [-1, 1]^2 on the x-y plane
    st_min = np.min(st_coords, axis=0)
    st_max = np.max(st_coords, axis=0)
    st_coords = (st_coords - st_min) / (st_max - st_min) * 2 - 1
    st_aspect = (st_max[0] - st_min[0]) / (st_max[1] - st_min[1])
    st_coords[:, 1] /= st_aspect

    st_scale = 0.25
    st_plane_location = -1
    st_coords *= st_scale

    # extrinsics (camera to world) - all cameras share the same rotation but differ in centers
    poses = np.tile(np.eye(3, 4)[..., None], [1, 1, len(imgnames)])
    poses[:, 1:3, :] *= -1
    poses[:2, 3, :] = st_coords.T
    poses[2, 3, :] = st_plane_location

    # intrinsics - all cameras share the same focal length but varying, non-center principal points
    Ks = np.tile(np.eye(3)[..., None], [1, 1, len(imgnames)])
    focal = 1
    pixel_scale = width / 2
    Ks[0, 0, :] = focal * pixel_scale
    Ks[1, 1, :] = focal * pixel_scale
    Ks[0, 2, :] = st_coords.T[0] * focal * pixel_scale + width / 2
    Ks[1, 2, :] = st_coords.T[1] * focal * pixel_scale + height / 2

    # depth bounds
    if collection in ['knights']:
        near = 0.25
    else:
        near = 0.5

    far = 2
    bds = np.tile(np.array([near, far])[..., None], [1, len(imgnames)])

    # make the poses array what's expected by the loader - note this won't be
    # correct intrinsics due to the non-center principal points
    hwf = np.zeros([3, 1, len(imgnames)])
    hwf[:2, 0, :] = np.array([height, width]).reshape([2, 1])
    hwf[2, 0, :] = focal * pixel_scale
    poses = np.hstack([poses, hwf])

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs, Ks, st_aspect, width, height


def load_stanford_data(basedir, factor=None, width=None, height=None, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, args=None):
    poses, bds, imgs, Ks, st_aspect, width, height = _load_data(basedir, factor=factor, width=width, height=height) # downsample images either by a factor or a resolution
    print('Loaded', basedir, bds.min(), bds.max())

    # Move variable dim to axis 0
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    Ks = np.moveaxis(Ks, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc

    if recenter:
        poses, central_pose = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # intrinsics - all cameras share the same focal length but varying, non-center principal points
        if args.render_spiral:
            N = 120
            rows = 17
            cols = 17
            disp_row = 8

            render_poses = np.tile(np.eye(3, 4)[None], [N, 1, 1])
            render_poses[:, 1:3, :] *= -1
            render_poses[:, 2, 3] = -1

            render_Ks = np.tile(np.eye(3)[None], [N, 1, 1]).astype(np.float32)
            focal = 1
            pixel_scale = width / 2
            render_Ks[:, 0, 0] = focal * pixel_scale
            render_Ks[:, 1, 1] = focal * pixel_scale

            rots = 2
            st_scale = 0.25
            spiral_scale = args.spiral_scale

            for i, theta in enumerate(np.linspace(0., 2. * np.pi * rots, N+1)[:-1]):
                s_idx = (np.cos(theta) * spiral_scale + 1) / 2.0 * (cols - 1)
                t_idx = (-np.sin(theta) * spiral_scale + 1) / 2.0 * (rows - 1)

                s = ((s_idx / (cols - 1)) * 2 - 1) * st_scale
                t = ((t_idx / (rows - 1)) * 2 - 1) * st_scale / st_aspect

                render_Ks[i, 0, 2] = s * focal * pixel_scale + width / 2
                render_Ks[i, 1, 2] = t * focal * pixel_scale + height / 2
                render_poses[i, 0, 3] = s
                render_poses[i, 1, 3] = t
        else:
            supersample = 4
            rows = 17
            cols = 17
            N = supersample * cols

            st_scale = 0.25

            render_poses = np.tile(np.eye(3, 4)[None], [N, 1, 1])
            render_poses[:, 1:3, :] *= -1
            render_poses[:, 2, 3] = -1

            render_Ks = np.tile(np.eye(3)[None], [N, 1, 1]).astype(np.float32)
            focal = 1
            pixel_scale = width / 2
            render_Ks[:, 0, 0] = focal * pixel_scale
            render_Ks[:, 1, 1] = focal * pixel_scale

            t_idx = args.disp_row

            for i in range(N):
                s_idx = i / supersample
                s = ((s_idx / (cols - 1)) * 2 - 1) * st_scale
                t = ((t_idx / (rows - 1)) * 2 - 1) * st_scale / st_aspect

                render_Ks[i, 0, 2] = s * focal * pixel_scale + width / 2
                render_Ks[i, 1, 2] = t * focal * pixel_scale + height / 2
                render_poses[i, 0, 3] = s
                render_poses[i, 1, 3] = t

    render_poses[:,:3,3] *= sc

    bottom = np.reshape([0,0,0,1.], [1,4])
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [render_poses.shape[0],1,1])

    render_poses = np.concatenate([render_poses[:,:3,:4], bottom], -2)
    render_poses = np.linalg.inv(central_pose) @ render_poses
    render_poses = render_poses[:,:3,:4]


    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test, Ks, render_Ks
