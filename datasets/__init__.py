#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .blender import BlenderLightfieldDataset, BlenderDataset, DenseBlenderDataset
from .fourier import FourierDataset, FourierLightfieldDataset
from .llff import LLFFDataset, DenseLLFFDataset
from .random import (
    RandomRayDataset,
    RandomViewSubsetDataset,
    RandomRayLightfieldDataset,
)
from .shiny import ShinyDataset, DenseShinyDataset
from .stanford import StanfordLightfieldDataset
from .tamul import TamulLightfieldDataset
from .tanks import TanksDataset

dataset_dict = {
    'fourier': FourierDataset,
    'fourier_lightfield': FourierLightfieldDataset,
    'random_ray': RandomRayDataset,
    'random_lightfield': RandomRayLightfieldDataset,
    'random_view': RandomViewSubsetDataset,
    'blender': BlenderDataset,
    'dense_blender': DenseBlenderDataset,
    'llff': LLFFDataset,
    'dense_llff': DenseLLFFDataset,
    'dense_shiny': DenseShinyDataset,
    'shiny': ShinyDataset,
    'tanks': TanksDataset,
    'blender_lightfield': BlenderLightfieldDataset,
    'stanford': StanfordLightfieldDataset,
    'tamul_lightfield': TamulLightfieldDataset,
}
