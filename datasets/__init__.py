#!/usr/bin/env python3

from .blender import BlenderLightfieldDataset, BlenderDataset, DenseBlenderDataset
from .stanford import StanfordLightfieldDataset
from .tamul import TamulLightfieldDataset

from .llff import LLFFDataset, DenseLLFFDataset
from .shiny import ShinyDataset, DenseShinyDataset
from .tanks import TanksDataset

from .random import RandomRayDataset, RandomViewSubsetDataset, RandomRayLightfieldDataset
from .fourier import FourierDataset, FourierLightfieldDataset

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
    'stanford_lightfield': StanfordLightfieldDataset,
    'tamul_lightfield': TamulLightfieldDataset,
}
