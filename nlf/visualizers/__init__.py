#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .closest_view import ClosestViewVisualizer
from .epipolar import EPIVisualizer
from .focus import FocusVisualizer

visualizer_dict = {
    'closest_view': ClosestViewVisualizer,
    'epipolar': EPIVisualizer,
    'focus': FocusVisualizer,
}
