#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn


class BaseVisualizer(nn.Module):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__()

        self.cfg = cfg

        ## (Hack) Prevent from storing system variables
        self.systems = [system]

    def get_system(self):
        return self.systems[0]

    def validation_video(self, batch):
        return {}

    def validation_image(self, batch, batch_idx):
        return {}
