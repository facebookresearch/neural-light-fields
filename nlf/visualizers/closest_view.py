#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np # noqa

from nlf.visualizers.base import BaseVisualizer


class ClosestViewVisualizer(BaseVisualizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

    def validation(self, batch):
        system = self.get_system()

        pose = batch['pose']
        rgb = system.trainer.datamodule.train_dataset.get_closest_rgb(pose).cpu()
        rgb = rgb.permute(2, 0, 1)

        return {
            'rgb': rgb
        }

    def validation_video(self, batch):
        temp_outputs = self.validation(batch)
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'videos/closest_{key}'] = temp_outputs[key]

        return outputs

    def validation_image(self, batch, batch_idx):
        return {}
