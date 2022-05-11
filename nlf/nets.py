#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from nlf.embedding import WindowedPE
from nlf.activations import get_activation


class BaseNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
    ):
        super().__init__()

        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = cfg.skips if 'skips' in cfg else []
        self.activation = cfg.activation if 'activation' in cfg else 'sigmoid'
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'

        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels, self.W)
            else:
                layer = nn.Linear(self.W, self.W)

            layer = nn.Sequential(layer, get_activation(self.layer_activation))
            setattr(self, f'encoding{i+1}', layer)

        self.encoding_final = nn.Linear(self.W, self.W)

        # Output
        self.out_layer = nn.Sequential(
            nn.Linear(self.W, self.out_channels),
            get_activation(self.activation)
        )

    def forward(self, x, sigma_only=False):
        input_x = x

        for i in range(self.D):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = getattr(self, f'encoding{i+1}')(x)

        encoding_final = self.encoding_final(x)
        return self.out_layer(encoding_final)

    def set_iter(self, i):
        pass


class NeRFNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
    ):
        super().__init__()

        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = cfg.skips if 'skips' in cfg else []
        self.activation = cfg.activation if 'activation' in cfg else 'sigmoid'

        self.pos_pe = WindowedPE(
            3,
            cfg.pos_pe
        )
        self.dir_pe = WindowedPE(
            3,
            cfg.dir_pe
        )

        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.pos_pe.out_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.pos_pe.out_channels, self.W)
            else:
                layer = nn.Linear(self.W, self.W)

            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f'encoding{i+1}', layer)

        self.encoding_dir = nn.Sequential(
            nn.Linear(self.W + self.dir_pe.out_channels, self.W // 2),
            nn.ReLU(True)
        )

        self.encoding_final = nn.Linear(self.W, self.W)

        # Output
        self.sigma = nn.Sequential(
            nn.Linear(self.W, 1),
            nn.Sigmoid(),
        )

        self.rgb = nn.Sequential(
            nn.Linear(self.W // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, sigma_only=False):
        input_pos, input_dir = torch.split(x, [3, 3], -1)

        input_pos = self.pos_pe(input_pos)
        input_dir = self.dir_pe(input_dir)

        pos = input_pos

        for i in range(self.D):
            if i in self.skips:
                pos = torch.cat([input_pos, pos], -1)

            pos = getattr(self, f'encoding{i+1}')(pos)

        encoding_final = self.encoding_final(x)
        return self.out_layer(encoding_final)

    def set_iter(self, i):
        pass

net_dict = {
    'base': BaseNet,
    'nerf': NeRFNet,
}
