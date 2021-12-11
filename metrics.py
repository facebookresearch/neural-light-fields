import torch
import numpy as np

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2

    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)

    return value

def psnr(image0, image1):
    return peak_signal_noise_ratio(np.array(image1), np.array(image0), data_range=1.0)

def ssim(image0, image1):
    return structural_similarity(np.array(image1), np.array(image0), win_size=11, multichannel=True, gaussian_weights=True)

def get_mean_outputs(outputs, cpu=False):
    # Stack
    stacked = {}

    for x in outputs:
        for key, val in x.items():
            if key not in stacked:
                stacked[key] = []

            stacked[key].append(val)

    # Mean
    mean = {}

    for key in stacked:
        if cpu:
            mean_val = np.stack(stacked[key]).mean()
        else:
            mean_val = torch.stack(stacked[key]).mean()

        mean[key] = mean_val

    return mean

from kornia.losses import ssim_loss as dssim

def psnr_gpu(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim_gpu(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 11, reduction=reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]
