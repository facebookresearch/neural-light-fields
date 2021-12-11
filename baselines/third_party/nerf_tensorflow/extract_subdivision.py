import os, sys
import tensorflow as tf

import numpy as np
import imageio
import pprint

import matplotlib.pyplot as plt

from load_llff import load_llff_data
from load_shiny import load_shiny_data
from load_tanks import load_tanks_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, load_blender_lightfield_data

import run_nerf
import run_nerf_helpers

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

## Load weights

def load_data(args):
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, K = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Intrinsics')
        print(hwf)
        print(K)
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'shiny':
        images, poses, bds, render_poses, i_test, K = load_shiny_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Intrinsics')
        print(hwf)
        print(K)
        print('Loaded shiny', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'tanks':
        images, poses, bds, render_poses, i_test, K = load_tanks_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Intrinsics')
        print(hwf)
        print(K)
        print('Loaded shiny', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        near = tf.reduce_min(bds) * .9
        far = tf.reduce_max(bds) * 1
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, K, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'blender_lightfield':
        images, poses, render_poses, hwf, K, i_split = load_blender_lightfield_data(
            args.datadir, args.half_res, args)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    return near, far, hwf, K, poses

def extract_subdivision(basedir, expname, N=16, N_rand=16, render_mesh=False):
    config = os.path.join(basedir, expname, 'config.txt')
    print('Args:')
    print(open(config, 'r').read())

    parser = run_nerf.config_parser()
    ft_str = ''
    ft_str = '--ft_path {}'.format(os.path.join(basedir, expname, 'model_200000.npy'))
    args = parser.parse_args('--config {} '.format(config) + ft_str)

    # Load data
    near, far, hwf, K, poses = load_data(args)

    # Create nerf model
    _, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)

    bds_dict = {
        'near' : tf.cast(near, tf.float32),
        'far' : tf.cast(far, tf.float32),
    }
    render_kwargs_test.update(bds_dict)

    print('Render kwargs:')
    pprint.pprint(render_kwargs_test)

    net_fn = render_kwargs_test['network_query_fn']
    print(net_fn)

    # Render an overhead view to check model was loaded correctly
    c2w = poses[poses.shape[0] // 2, :3, :4].astype(np.float32)
    H, W, focal = hwf[0], hwf[1], hwf[2]
    down = 8

    K = np.copy(K)
    K[0, :] /= down
    K[1, :] /= down

    ## Render test
    test = run_nerf.render(H // down, W // down, focal / down, K, c2w=c2w, **render_kwargs_test)
    img = np.clip(test[0],0,1)

    plt.figure()
    plt.imshow(img)
    plt.savefig('~/local/test/img_0.png')
    plt.close()

    ## Query densely
    if args.no_ndc or args.dataset_type != 'llff':
        t = np.linspace(-1.2, 1.2, N+1)
    else:
        t = np.linspace(-1, 1, N+1)

    step = t[1] - t[0]
    base_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)

    if N_rand > 0:
        base_pts = base_pts[None]
        eps = np.random.uniform(0, 1, size=((N_rand,) + base_pts.shape[1:])).astype(np.float32)
        query_pts = (base_pts - 0.5 * step) + eps * step
        print(eps.shape)
    else:
        query_pts = base_pts

    sh = query_pts.shape
    flat = query_pts.reshape([-1,3])

    def fn(i0, i1):
        return net_fn(
            flat[i0:i1,None,:],
            viewdirs=np.zeros_like(flat[i0:i1]),
            network_fn=render_kwargs_test['network_fine']
        )

    chunk = 1024*64
    raw = np.concatenate(
        [fn(i, i+chunk).numpy() for i in range(0, flat.shape[0], chunk)],
        0
    )
    raw = np.reshape(raw, list(sh[:-1]) + [-1])

    if N_rand > 0:
        raw = raw.mean(0)

    sigma = np.maximum(raw[...,-1], 0.)

    print(raw.shape)

    plt.figure()
    plt.hist(np.maximum(0,sigma.ravel()), log=True)
    plt.savefig('~/local/test/img_1.png')
    plt.close()

    ## Save
    np.save('~/local/bootstrap/' + expname + '_density.npy', raw[..., -1])
    np.save('~/local/bootstrap/' + expname + '_points.npy', base_pts)
    np.save('~/local/bootstrap/' + expname + '_voxel_size.npy', step)

    ## Marching cubes

    import mcubes

    threshold = 50.
    print('fraction occupied', np.mean(sigma > threshold))
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    print('done', vertices.shape, triangles.shape)

    if render_mesh:
        import trimesh

        mesh = trimesh.Trimesh(vertices / N - .5, triangles)
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        import pyrender

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        camera_pose = np.eye(4)
        camera_pose[:3, :4] = poses[poses.shape[0] // 2, :3, :4].astype(np.float32)
        nc = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(nc)

        # Set up the light -- a point light in the same spot as the camera
        light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
        nl = pyrender.Node(light=light, matrix=camera_pose)
        scene.add_node(nl)

        # Render the scene
        r = pyrender.OffscreenRenderer(640, 480)
        color, depth = r.render(scene)

        plt.figure()
        plt.imshow(color)
        plt.savefig('~/local/test/img_2.png')
        plt.close()

        plt.figure()
        plt.imshow(depth)
        plt.savefig('~/local/test/img_3.png')
        plt.close()

if __name__ == '__main__':
    basedir = sys.argv[1]
    expname = sys.argv[2]
    N = int(sys.argv[3])
    N_rand = int(sys.argv[4])

    extract_subdivision(basedir, expname, N=N, N_rand=N_rand, render_mesh=False)
