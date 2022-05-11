#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .depth_classification import DepthClassificationRegularizer
from .fourier import FourierRegularizer
from .inverse_ray_depth import InverseRayDepthRegularizer
from .multiple_ray_depth import MultipleRayDepthRegularizer
from .ray_blending import RayDepthBlendingRegularizer
from .ray_bundle import RayBundleRegularizer
from .ray_depth import RayDepthRegularizer
from .ray_depth_occ_dir import RayDepthOccDirRegularizer
from .ray_interpolation import RayInterpolationRegularizer
from .teacher import TeacherRegularizer, BlurryTeacherRegularizer
from .voxel_sparsity import VoxelSparsityRegularizer
from .warp import WarpRegularizer, WarpLevelSetRegularizer

regularizer_dict = {
    'ray_depth': RayDepthRegularizer,
    'ray_depth_occ_dir': RayDepthOccDirRegularizer,
    'multiple_ray_depth': MultipleRayDepthRegularizer,
    'inverse_ray_depth': InverseRayDepthRegularizer,
    'ray_depth_blending': RayDepthBlendingRegularizer,
    'ray_interpolation': RayInterpolationRegularizer,
    'depth_classification': DepthClassificationRegularizer,
    'fourier': FourierRegularizer,
    'ray_bundle': RayBundleRegularizer,
    'teacher': TeacherRegularizer,
    'blurry_teacher': BlurryTeacherRegularizer,
    'voxel_sparsity': VoxelSparsityRegularizer,
    'warp': WarpRegularizer,
    'warp_level': WarpLevelSetRegularizer,
}
