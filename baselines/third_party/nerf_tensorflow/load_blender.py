import os
import tensorflow as tf
import numpy as np
import imageio
import json




trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w



def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)

    if half_res:
        imgs = tf.image.resize(imgs, [400, 400], method='area').numpy()
        H = H//2
        W = W//2
        focal = focal/2.

    hwf = [H, W, focal]

    K = np.eye(3)
    K[0, 0] = hwf[2]
    K[0, 2] = hwf[1] / 2

    K[1, 1] = hwf[2]
    K[1, 2] = hwf[0] / 2

    Ks = np.tile(K[None, ...], (poses.shape[0], 1, 1)).astype(np.float32)

    return imgs, poses, render_poses, hwf, Ks, i_split


def load_blender_lightfield_data(basedir, half_res=False, args=None):
    splits = ['train', 'val', 'test']
    metas = {}

    for i, s in enumerate(splits):
        with open(os.path.join(basedir, 'transforms.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        for s_idx in range(0, args.lf_cols):
            for t_idx in range(0, args.lf_rows):
                i = t_idx * args.lf_cols + s_idx
                frame = meta['frames'][i]

                if s == 'train' and (
                    (s_idx % args.lf_step != 0) or \
                    (t_idx % args.lf_step != 0) \
                    ):
                    continue

                fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)

    if half_res:
        imgs = tf.image.resize(imgs, [400, 400], method='area').numpy()
        H = H//2
        W = W//2
        focal = focal/2.

    hwf = [H, W, focal]

    K = np.eye(3)
    K[0, 0] = hwf[2]
    K[0, 2] = hwf[1] / 2

    K[1, 1] = hwf[2]
    K[1, 2] = hwf[0] / 2

    Ks = np.tile(K[None, ...], (poses.shape[0], 1, 1)).astype(np.float32)

    return imgs, poses, render_poses, hwf, Ks, i_split
