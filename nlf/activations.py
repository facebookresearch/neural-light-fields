#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn


class LeakyReLU(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        if 'a' in cfg and not isinstance(cfg, str):
            self.a = cfg.a
        else:
            self.a = 0.01

        self.act = nn.LeakyReLU(self.a, True)

    def forward(self, x):
        return self.act(x)


class ReLU(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(x)


class Abs(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class Sigmoid(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(x)


class Softmax(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.Softmax()

    def forward(self, x):
        return self.act(x)


class Tanh(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(x)


class Identity(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.act = nn.Identity()

    def forward(self, x):
        return self.act(x)


class L1Norm(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=1, dim=-1) * x.shape[-1]


class Probs(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(torch.abs(x), p=1, dim=-1)


class RowL2Norm(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        if 'param_channels' in cfg and not isinstance(cfg, str):
            self.param_channels = cfg.param_channels
        else:
            self.param_channels = 4

        if 'fac' in cfg and not isinstance(cfg, str):
            self.fac = cfg.fac
        else:
            self.fac = 1.0

    def forward(self, x):
        batch_size = x.shape[0]
        total_channels = x.shape[-1]

        if total_channels > 0:
            x = x.view(-1, total_channels // self.param_channels, self.param_channels)
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x.view(batch_size, total_channels) * self.fac


class RowLInfNorm(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        if 'param_channels' in cfg and not isinstance(cfg, str):
            self.param_channels = cfg.param_channels
        else:
            self.param_channels = 4

        if 'fac' in cfg and not isinstance(cfg, str):
            self.fac = cfg.fac
        else:
            self.fac = 1.0

    def forward(self, x):
        batch_size = x.shape[0]
        total_channels = x.shape[-1]

        if total_channels > 0:
            x = x.view(-1, total_channels // self.param_channels, self.param_channels)
            x = torch.nn.functional.normalize(x, p=float('inf'), dim=-1)

        return x.view(batch_size, total_channels) * self.fac


class RowL1Norm(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        if 'param_channels' in cfg and not isinstance(cfg, str):
            self.param_channels = cfg.param_channels
        else:
            self.param_channels = 4

        if 'fac' in cfg and not isinstance(cfg, str):
            self.fac = cfg.fac
        else:
            self.fac = 1.0

    def forward(self, x):
        batch_size = x.shape[0]
        total_channels = x.shape[-1]

        if total_channels > 0:
            x = x.view(-1, total_channels // self.param_channels, self.param_channels)
            x = torch.nn.functional.normalize(x, p=1, dim=-1)

        return x.view(batch_size, total_channels) * self.fac


class L2Norm(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        if 'fac' in cfg and not isinstance(cfg, str):
            self.fac = np.sqrt(cfg.fac)
        else:
            self.fac = None

    def forward(self, x):
        if self.fac is not None:
            return torch.nn.functional.normalize(x, p=2, dim=-1) * self.fac
        else:
            return torch.nn.functional.normalize(x, p=2, dim=-1) * np.sqrt(x.shape[-1])


class Zero(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class Alpha(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

    def forward(self, x):
        return 1.0 - torch.exp(-torch.relu(x))


class Gaussian(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        if 'sigma' in cfg and not isinstance(cfg, str):
            self.sigma = cfg.sigma
        else:
            self.sigma = 0.05

    def forward(self, x):
        return torch.exp(-0.5 * torch.square(x / self.sigma))


activation_map = {
    'alpha': Alpha,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'tanh': Tanh,
    'identity': Identity,
    'probs': Probs,
    'l1_norm': L1Norm,
    'l2_norm': L2Norm,
    'row_l1_norm': RowL1Norm,
    'row_l2_norm': RowL2Norm,
    'row_linf_norm': RowLInfNorm,
    'zero': Zero,
    'gaussian': Gaussian,
    'leaky_relu': LeakyReLU,
    'relu': ReLU,
    'abs': Abs,
}

def get_activation(cfg):
    if isinstance(cfg, str):
        return activation_map[cfg](cfg)
    else:
        return activation_map[cfg.type](cfg)
