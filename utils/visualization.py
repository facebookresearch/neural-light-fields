#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def get_warp_dimensions(embedding, W, H, k=3):
    embedding_std = np.array(torch.std(embedding, 0, True).cpu())
    return list(np.argsort(-embedding_std, axis=-1)[:k])

def visualize_warp(embedding, warp_dims, W, H, k=3):
    warp_vis = np.array(embedding[..., warp_dims].cpu())

    warp_vis = np.abs(warp_vis) / np.max(np.abs(warp_vis))
    warp_vis = np.clip(warp_vis, 0, 1)
    warp_vis = np.transpose(
        np.reshape(warp_vis, (H, W, k)), (2, 0, 1)
    )

    return warp_vis
