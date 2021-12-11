import numpy as np
import tensorflow as tf

import lpips_tf
import elpips

import os, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import glob

import cv2
import imageio
from PIL import Image, ImageCms
from skimage import exposure

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def compute_psnr(image0, image1):
    return peak_signal_noise_ratio(image1, image0, data_range=1.0)

def compute_ssim(image0, image1):
    return structural_similarity(image1, image0, win_size=11, multichannel=True, gaussian_weights=True)

def compute_lpips(image0_ph, image1_ph, distance_t, image0, image1, sess):
    return sess.run(distance_t, feed_dict={image0_ph: image0, image1_ph: image1})

def imread(f):
    if f.endswith('png'):
        img = imageio.imread(f, ignoregamma=True)
    else:
        img = imageio.imread(f)

    return img

def temp(gamma_img_file, img_file):
    gamma_img = Image.open(gamma_img_file)
    gamma_icc = gamma_img.info['icc_profile']
    gamma_prf = ImageCms.ImageCmsProfile(io.BytesIO(gamma_icc))

    img = Image.open(img_file)
    color_layout = "RGB"

    if np.array(img).shape[-1] == 4:
        color_layout = "RGBA"

    tform = ImageCms.buildTransform(
        ImageCms.createProfile("sRGB"),
        gamma_prf,
        color_layout,
        color_layout
    )

    return tform.apply(img)

def correct_gamma(img_file):
    img = Image.open(img_file)

    img = imread(img_file) / 255.0
    img = exposure.adjust_gamma(img, 1.15)

    return img

def get_files(args):
    if args.mode == 'xfields':
        rows = 17
        cols = 17
        step = 4

        gt_files = sorted(os.listdir(args.gt_dir))
        pred_files = sorted(os.listdir(args.pred_dir))

        new_gt_files = []
        new_pred_files = []

        for t_idx in range(0, rows, 1):
            for s_idx in range(0, cols, 1):
                if ((t_idx % step) == 0) and ((s_idx % step) == 0):
                    continue

                idx = t_idx * cols + s_idx
                new_gt_files.append(gt_files[idx])
                new_pred_files.append(pred_files[idx])

        gt_files = new_gt_files
        pred_files = new_pred_files
    else:
        all_files = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')))
        gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*gt*.png')))
        pred_files = sorted(list(set(all_files) - set(gt_files)))

    return gt_files, pred_files

def main(args):
    gt_files, pred_files = get_files(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # Read and gamma correct
    lpips_vals = []
    ssim_vals = []
    psnr_vals = []

    # E-LPIPS setup
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)

    with tf.Session(config=config) as session:
        if args.use_elpips:
            metric = elpips.Metric(elpips.elpips_vgg(batch_size=1))
            distance_t = metric.forward(image0_ph, image1_ph)
        else:
            distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')

        # Run
        for i, (gt_file, pred_file) in enumerate(zip(gt_files, pred_files)):
            print(f'Reading {gt_file} {pred_file}')
            # Read and gamma correct
            if args.gamma_correct:
                print('Correcting gamma')

                gt_img = correct_gamma(
                    os.path.join(args.gt_dir, gt_file),
                )

                pred_img = correct_gamma(
                    os.path.join(args.pred_dir, pred_file),
                )
            else:
                gt_img = np.array(Image.open(os.path.join(args.gt_dir, gt_file)).convert('RGB')) / 255.0
                pred_img = np.array(Image.open(os.path.join(args.pred_dir, pred_file)).convert('RGB')) / 255.0

            # Resize
            if gt_img.shape[0] != pred_img.shape[0] or gt_img.shape[1] != pred_img.shape[1]:
                gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]), cv2.INTER_LANCZOS4)

            # Eval
            lpips_vals.append(
                compute_lpips(image0_ph, image1_ph, distance_t, gt_img[None], pred_img[None], session)
            )
            ssim_vals.append(compute_ssim(gt_img, pred_img))
            psnr_vals.append(compute_psnr(gt_img, pred_img))

            print()
            print(f'Image {i}')
            print()
            print( "PSNR:",  psnr_vals[-1] )
            print( "SSIM:",  ssim_vals[-1] )
            print( "LPIPS:", lpips_vals[-1] )
            print()

            print( "Running Mean/Std PSNR:",  np.mean(psnr_vals ), np.std(psnr_vals ) )
            print( "Running Mean/Std SSIM:",  np.mean(ssim_vals ), np.std(ssim_vals ) )
            print( "Running Mean/Std LPIPS:", np.mean(lpips_vals), np.std(lpips_vals) )
            print()

            # Write out image files
            imageio.imwrite(
                os.path.join(args.out_dir, f'{i:04d}_gt.png'),
                np.uint8(gt_img * 255)
            )
            imageio.imwrite(
                os.path.join(args.out_dir, f'{i:04d}_pred.png'),
                np.uint8(pred_img * 255)
            )

    print("Total Mean LPIPS:", np.mean(lpips_vals))
    print("Total Mean SSIM:", np.mean(ssim_vals))
    print("Total Mean PSNR:", np.mean(psnr_vals))

    with open(
        os.path.join(args.out_file),
        'w'
    ) as f:
        f.write(f"LPIPS Mean, Std: {np.mean(lpips_vals)} +- {np.std(lpips_vals)}\n")
        f.write(f"SSIM Mean, Std: {np.mean(ssim_vals)} +- {np.std(ssim_vals)}\n")
        f.write(f"PSNR Mean, Std: {np.mean(psnr_vals)} +- {np.std(psnr_vals)}\n")

    suffix = args.out_dir.split('/')[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default='xfields', help='')
    parser.add_argument("--gt_dir", type=str, default='gt', help='')
    parser.add_argument("--pred_dir", type=str, default='pred', help='')
    parser.add_argument("--out_dir", type=str, default='eval', help='')
    parser.add_argument("--out_file", type=str, default='metrics.txt', help='')
    parser.add_argument("--gamma_correct", action='store_true', help='')
    parser.add_argument("--use_elpips", action='store_true', help='')

    args = parser.parse_args()

    main(args)
