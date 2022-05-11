#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["PATH"] += os.pathsep + os.path.join(os.path.expanduser('~'), 'anaconda3/envs/neural-light-fields/pkgs/cuda-toolkit')
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

# build clib
_ext_src_root = "third_party/intersect"
_ext_sources = glob.glob("{}/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/*.h".format(_ext_src_root))

setup(
    name='third_party.intersect',
    ext_modules=[
        CUDAExtension(
            name='third_party.intersect.cuda_lib',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
