#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import nn

from utils.ray_utils import (
    ray_param_dict,
)

from utils.intersect_utils import (
    intersect_dict,
)

from nlf.activations import get_activation


class IdentityEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        *args,
        **kwargs
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

    def forward(self, x):
        return x

    def set_iter(self, i):
        self.cur_iter = i


class RayParam(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
    ):

        super().__init__()

        self.ray_param_fn = ray_param_dict[cfg.fn](cfg)
        self.in_channels = in_channels
        self.out_channels = cfg.n_dims

        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.ray_param_fn(x)

    def set_iter(self, i):
        self.cur_iter = i


class WindowedPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_freqs = cfg.n_freqs
        self.cur_iter = 0
        self.wait_iters = cfg.wait_iters
        self.max_freq_iter = float(cfg.max_freq_iter)
        self.exclude_identity = cfg.exclude_identity \
            if 'exclude_identity' in cfg \
            else False

        self.funcs = [torch.sin, torch.cos]
        self.freq_multiplier = cfg.freq_multiplier if 'freq_multiplier' in cfg else 2.0
        self.freq_bands = self.freq_multiplier ** torch.linspace(1, cfg.n_freqs, cfg.n_freqs)

        self.in_channels = in_channels
        if self.exclude_identity:
            self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs)
        else:
            self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs + 1)

        self.dummy_layer = nn.Linear(1, 1)

    def weight(self, j):
        if self.max_freq_iter == 0:
            return 1.0
        elif self.cur_iter < self.wait_iters:
            return 0.0
        elif self.cur_iter > self.max_freq_iter:
            return 1.0

        cur_iter = (self.cur_iter - self.wait_iters)
        alpha = (cur_iter / self.max_freq_iter) * self.n_freqs
        return (1.0 - np.cos(np.pi * np.clip(alpha - j, 0.0, 1.0))) / 2

    def forward(self, x, **render_kwargs):
        out = []

        if not self.exclude_identity:
            out += [x]

        for j, freq in enumerate(self.freq_bands):
            for func in self.funcs:
                out += [self.weight(j) * func(freq * x)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


class WeightedWindowedPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_freqs = cfg.n_freqs
        self.cur_iter = 0
        self.wait_iters = cfg.wait_iters
        self.max_freq_iter = float(cfg.max_freq_iter)
        self.exclude_identity = cfg.exclude_identity \
            if 'exclude_identity' in cfg \
            else False

        self.funcs = [torch.sin, torch.cos]
        self.freq_multiplier = cfg.freq_multiplier if 'freq_multiplier' in cfg else 2.0
        self.freq_bands = self.freq_multiplier ** torch.linspace(1, cfg.n_freqs, cfg.n_freqs)

        self.in_channels = in_channels
        if self.exclude_identity:
            self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs)
        else:
            self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs + 1)

        self.dummy_layer = nn.Linear(1, 1)

    def weight(self, j):
        if self.max_freq_iter == 0:
            return 1.0
        elif self.cur_iter < self.wait_iters:
            return 0.0
        elif self.cur_iter > self.max_freq_iter:
            return 1.0

        cur_iter = (self.cur_iter - self.wait_iters)
        alpha = (cur_iter / self.max_freq_iter) * self.n_freqs
        return (1.0 - np.cos(np.pi * np.clip(alpha - j, 0.0, 1.0))) / 2

    def forward(self, x, **render_kwargs):
        out = []

        if not self.exclude_identity:
            out += [x]

        if 'pe_weight' in render_kwargs:
            pe_weight = render_kwargs['pe_weight']
            use_pe_weight = True
        else:
            use_pe_weight = False

        for j, freq in enumerate(self.freq_bands):
            for func in self.funcs:
                if use_pe_weight:
                    out += [self.weight(j) * pe_weight[j] * func(freq * x)]
                else:
                    out += [self.weight(j) * func(freq * x)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


class SelectPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.select_channels = cfg.select_channels
        self.discard = cfg.discard if 'discard' in cfg else False

        self.pe = WindowedPE(
            self.select_channels,
            cfg
        )

        if self.discard:
            self.out_channels = self.pe.out_channels
        else:
            self.out_channels = (self.in_channels - self.select_channels) \
                + self.pe.out_channels

    def forward(self, x):
        out_x = self.pe(x[..., :self.select_channels])

        if not self.discard:
            out_x = torch.cat([ out_x, x[..., self.select_channels:] ], -1)

        return out_x

    def set_iter(self, i):
        self.pe.set_iter(i)


class RandomPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_bands = cfg.n_bands
        self.sigma = cfg.sigma
        self.funcs = [torch.sin, torch.cos]

        self.in_channels = in_channels
        self.out_channels = len(self.funcs) * cfg.n_bands

        self.embedding_matrix = (torch.randn(
            (self.n_bands, self.in_channels)
        ) * self.sigma).cuda()

    def forward(self, x):
        # Convert to correct device
        embedding_matrix = self.embedding_matrix.type_as(x)

        out = []
        raw = (embedding_matrix @ x.permute(1, 0)).permute(1, 0)

        for func in self.funcs:
            out += [func(raw)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


class LearnablePE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_bands = cfg.n_bands
        self.sigma = cfg.sigma
        self.funcs = [torch.sin, torch.cos]

        self.in_channels = in_channels
        self.out_channels = len(self.funcs) * cfg.n_bands
        self.embedding_layer = nn.Linear(in_channels, cfg.n_bands)

        self.embedding_matrix = (torch.randn(
            (self.n_bands, self.in_channels)
        ) * self.sigma).cuda()
        self.embedding_matrix = nn.Parameter(
            self.embedding_matrix, requires_grad=True
        )

        self.embedding_bias = (torch.randn(
            (1, self.n_bands)
        ) * self.sigma).cuda()
        self.embedding_bias = nn.Parameter(
            self.embedding_bias, requires_grad=True
        )

    def forward(self, x):
        # Convert to correct device
        embedding_matrix = self.embedding_matrix.type_as(x)
        embedding_bias = self.embedding_bias.type_as(x)

        out = []
        raw = (embedding_matrix @ x.permute(1, 0)).permute(1, 0) + embedding_bias

        for func in self.funcs:
            out += [func(raw)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


pe_dict = {
    'windowed': WindowedPE,
    'weighted_windowed': WeightedWindowedPE,
    'random': RandomPE,
    'learnable': LearnablePE,
    'select': SelectPE,
}


class HomogenousEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = cfg.out_channels if cfg.out_channels is not None else in_channels
        self.homogenous_layer = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        return self.homogenous_layer(x)

    def set_iter(self, i):
        self.cur_iter = i


class FeatureEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):

        super().__init__()

        self.dummy_layer = nn.Linear(1, 1)
        self.in_channels = in_channels
        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.skips = cfg.skips if 'skips' in cfg else []
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'
        self.activation = cfg.activation if 'activation' in cfg else 'identity'

        # Outputs
        if self.D == 0:
            self.out_channels = in_channels
        else:
            self.out_channels = cfg.out_channels

        # Hidden
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(in_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + in_channels, self.W)
            elif i == self.D - 1:
                layer = nn.Linear(self.W, self.out_channels)
            else:
                layer = nn.Linear(self.W, self.W)

            if i < self.D - 1:
                layer = nn.Sequential(layer, get_activation(self.layer_activation))

            setattr(self, f"transform{i+1}", layer)

        # Out
        self.out_layer = get_activation(self.activation)

    def forward(self, x, **render_kwargs):
        input_x = x

        # Transform
        for i in range(self.D):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = getattr(self, f"transform{i+1}")(x)

        return self.out_layer(x)

    def set_iter(self, i):
        self.cur_iter = i


class LocalAffineEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):

        super().__init__()

        # Param
        self.param_channels = in_channels if cfg.param_channels == 'all' else cfg.param_channels

        # MLP
        self.dummy_layer = nn.Linear(1, 1)

        self.in_channels = in_channels
        self.latent_channels = self.in_channels - self.param_channels

        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.skips = cfg.skips if 'skips' in cfg else []
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'

        # Global
        self.is_global = cfg.is_global if 'is_global' in cfg else False

        if self.is_global:
            self.in_channels = self.latent_channels

        # PE
        self.pe = pe_dict[cfg.pe.type](
            self.param_channels,
            cfg.pe
        )

        # Extra
        self.extra_in_channels = cfg.extra_in_channels if 'extra_in_channels' in cfg else 0
        self.extra_out_channels = cfg.extra_out_channels if 'extra_out_channels' in cfg else 0
        self.extra_tform_size = self.extra_in_channels * self.extra_out_channels

        if self.extra_tform_size == 0:
            self.extra_in_channels = 0
            self.extra_out_channels = 0

        self.extra_tform_activation = cfg.extra_tform_activation if 'extra_tform_activation' in cfg else 'identity'
        self.extra_bias_activation = cfg.extra_bias_activation if 'extra_bias_activation' in cfg else 'zero'
        self.extra_activation = cfg.extra_activation if 'extra_activation' in cfg else 'identity'

        # Tform
        self.tform_in_channels = self.param_channels
        self.tform_out_channels = cfg.tform_out_channels
        self.tform_size = self.tform_in_channels * self.tform_out_channels

        self.tform_activation = cfg.tform_activation if 'tform_activation' in cfg else 'identity'
        self.bias_activation = cfg.bias_activation if 'bias_activation' in cfg else 'zero'
        self.activation = cfg.activation if 'activation' in cfg else 'identity'

        # Extra outputs
        if self.D == 0:
            self.extra_out_channels = self.extra_in_channels

        # Tform outputs
        self.total_pred_channels = self.tform_size + self.extra_tform_size
        self.out_channels_after_tform = cfg.tform_out_channels

        if self.bias_activation != 'zero':
            self.total_pred_channels += self.tform_out_channels

        if self.extra_bias_activation != 'zero':
            self.total_pred_channels += self.extra_out_channels

        if self.D == 0:
            self.out_channels_after_tform = self.param_channels
        
        # Outputs
        self.out_channels = self.extra_out_channels + self.out_channels_after_tform

        # Hidden
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.pe.out_channels + self.latent_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.pe.out_channels + self.latent_channels, self.W)
            elif i == self.D - 1:
                layer = nn.Linear(
                    self.W,
                    self.total_pred_channels
                )
            else:
                layer = nn.Linear(self.W, self.W)

            if i < self.D - 1:
                layer = nn.Sequential(layer, get_activation(self.layer_activation))

            setattr(self, f"transform{i+1}", layer)

        # Out
        self.out_extra_tform_layer = get_activation(self.extra_tform_activation)
        self.out_extra_bias_layer = get_activation(self.extra_bias_activation)

        self.out_tform_layer = get_activation(self.tform_activation)
        self.out_bias_layer = get_activation(self.bias_activation)

        self.out_extra_layer = get_activation(self.extra_activation)
        self.out_layer = get_activation(self.activation)

    def embed_params(self, x):
        _, _, tform_flat, _ = self._embed_params(x)

        return tform_flat

    def _embed_params(self, x):
        # Apply PE
        x = torch.cat(
            [self.pe(x[..., :self.param_channels]), x[..., self.param_channels:]],
            dim=-1
        )

        # MLP
        input_x = x

        for i in range(self.D):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = getattr(self, f"transform{i+1}")(x)

        # Outputs
        extra_tform_flat = self.out_extra_tform_layer(
            x[..., :self.extra_tform_size]
        )
        x = x[..., self.extra_tform_size:]

        if self.extra_bias_activation != 'zero':
            extra_bias = self.out_extra_bias_layer(
                x[..., :self.extra_out_channels]
            )
            x = x[..., self.extra_out_channels:]
        else:
            extra_bias = None

        if self.bias_activation == 'zero':
            tform_flat = self.out_tform_layer(
                x
            )
            bias = None
        else:
            tform_flat = self.out_tform_layer(
                x[..., :-self.out_channels_after_tform]
            )
            bias = self.out_bias_layer(
                x[..., -self.out_channels_after_tform:]
            )

        return extra_tform_flat, extra_bias, tform_flat, bias

    def forward(self, x, **render_kwargs):
        if self.is_global:
            extra_tform, extra_bias, tform, bias = self._embed_params(
                x[..., -self.in_channels:]
            )
        else:
            extra_tform, extra_bias, tform, bias = self._embed_params(
                x
            )

        # Extra channel transform
        extra_x = x[..., :self.extra_in_channels]

        if self.D != 0 and self.extra_tform_size > 0:
            extra_tform = extra_tform.view(
                -1, self.extra_out_channels, self.extra_in_channels
            )
            extra_x = self.out_extra_layer(
                (extra_tform @ extra_x.unsqueeze(-1)).squeeze(-1)
            )

            if extra_bias is not None:
                extra_x = extra_x + extra_bias

        # Transform
        x = x[..., :self.param_channels]

        if self.D != 0:
            tform = tform.view(
                -1, self.out_channels_after_tform, self.param_channels
            )
            x = self.out_layer(
                (tform @ x.unsqueeze(-1)).squeeze(-1)
            )

            if bias is not None:
                x = x + bias
        
        # Return
        x = torch.cat([extra_x, x], -1)

        if 'embed_params' in render_kwargs and render_kwargs['embed_params']:
            return torch.cat([tform.view(tform.shape[0], -1), bias], -1), x
        else:
            return x

    def set_iter(self, i):
        self.cur_iter = i
        self.pe.set_iter(i)


class EpipolarEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):

        super().__init__()

        # Param
        self.param = RayParam(cfg.param, in_channels=6)
        self.param_channels = self.param.out_channels

        # MLP
        self.dummy_layer = nn.Linear(1, 1)

        self.in_channels = in_channels
        self.latent_channels = self.in_channels - 6

        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.skips = cfg.skips if 'skips' in cfg else []
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'

        # Global
        self.is_global = cfg.is_global if 'is_global' in cfg else False

        if self.is_global:
            self.in_channels = self.latent_channels

        # PE
        self.pe = pe_dict[cfg.pe.type](
            self.param_channels,
            cfg.pe
        )

        # Extra
        self.extra_in_channels = cfg.extra_in_channels if 'extra_in_channels' in cfg else 0
        self.extra_out_channels = cfg.extra_out_channels if 'extra_out_channels' in cfg else cfg.extra_in_channels
        self.extra_tform_size = self.extra_in_channels * self.extra_out_channels

        if self.extra_tform_size == 0:
            self.extra_in_channels = 0
            self.extra_out_channels = 0

        self.extra_tform_activation = cfg.extra_tform_activation if 'extra_tform_activation' in cfg else 'identity'
        self.extra_bias_activation = cfg.extra_bias_activation if 'extra_bias_activation' in cfg else 'zero'
        self.extra_activation = cfg.extra_activation if 'extra_activation' in cfg else 'identity'
        
        # Intersect
        self.intersect_fn = intersect_dict[cfg.intersect.type](
            cfg.intersect
        )

        self.z_channels = cfg.z_channels
        self.preds_per_z = cfg.preds_per_z if 'preds_per_z' in cfg else 1
        self.out_channels_per_z = cfg.intersect.out_channels_per_z

        self.z_activation = cfg.z_activation if 'z_activation' in cfg else 'identity'

        # Tform
        self.tform_in_channels = cfg.tform_in_channels if 'tform_in_channels' in cfg else cfg.intersect.out_channels_per_z
        self.tform_out_channels = cfg.tform_out_channels if 'tform_out_channels' in cfg else cfg.intersect.out_channels_per_z
        self.tform_size = self.tform_in_channels * self.tform_out_channels
        self.tform_multiplier = (self.out_channels_per_z // self.tform_in_channels)

        self.tform_activation = cfg.tform_activation if 'tform_activation' in cfg else 'identity'
        self.bias_activation = cfg.bias_activation if 'bias_activation' in cfg else 'identity'
        self.activation = cfg.activation if 'activation' in cfg else 'identity'

        # Extra outputs
        if self.D == 0:
            self.extra_out_channels = self.extra_in_channels

        # Tform outputs
        if self.D == 0:
            self.out_channels_after_tform = self.param_channels

        # Outputs
        self.total_pred_channels = self.z_channels * self.preds_per_z + self.z_channels * self.tform_multiplier * self.tform_size + self.extra_tform_size 
        self.out_channels_after_tform = cfg.z_channels * self.tform_multiplier * self.tform_out_channels

        if self.bias_activation != 'zero':
            self.total_pred_channels += self.z_channels * self.tform_multiplier * self.tform_out_channels

        if self.extra_bias_activation != 'zero':
            self.total_pred_channels += self.extra_out_channels
        
        self.out_channels = self.extra_out_channels + self.out_channels_after_tform

        # Hidden
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.pe.out_channels + self.latent_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.pe.out_channels + self.latent_channels, self.W)
            elif i == self.D - 1:
                layer = nn.Linear(
                    self.W,
                    self.total_pred_channels
                )
            else:
                layer = nn.Linear(self.W, self.W)

            if i < self.D - 1:
                layer = nn.Sequential(layer, get_activation(self.layer_activation))

            setattr(self, f"transform{i+1}", layer)

        # Out
        self.out_z_layer = get_activation(self.z_activation)

        self.out_extra_tform_layer = get_activation(self.extra_tform_activation)
        self.out_extra_bias_layer = get_activation(self.extra_bias_activation)

        self.out_tform_layer = get_activation(self.tform_activation)
        self.out_bias_layer = get_activation(self.bias_activation)

        self.out_extra_layer = get_activation(self.extra_activation)
        self.out_layer = get_activation(self.activation)

    def embed_params(self, x):
        z_vals, _, _, _, _ = self._embed_params(x, apply_param=True)

        return z_vals

    def _embed_params(self, x, apply_param=False):
        # Apply parameterization
        if apply_param:
            x = torch.cat(
                [self.param(x[..., :6]), x[..., 6:]],
                dim=-1
            )

        # Apply positional encoding
        x = torch.cat(
            [self.pe(x[..., :self.param_channels]), x[..., self.param_channels:]],
            dim=-1
        )

        # MLP
        input_x = x

        for i in range(self.D):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = getattr(self, f"transform{i+1}")(x)

        # Outputs
        z_vals = self.out_z_layer(x[..., :self.z_channels * self.preds_per_z])
        x = x[..., self.z_channels * self.preds_per_z:]

        extra_tform_flat = self.out_extra_tform_layer(
            x[..., :self.extra_tform_size]
        )
        x = x[..., self.extra_tform_size:]

        if self.extra_bias_activation != 'zero':
            extra_bias = self.out_extra_bias_layer(
                x[..., :self.extra_out_channels]
            )
            x = x[..., self.extra_out_channels:]
        else:
            extra_bias = None

        if self.bias_activation == 'zero':
            tform_flat = self.out_tform_layer(
                x
            )
            bias = None
        else:
            tform_flat = self.out_tform_layer(
                x[..., :-self.out_channels_after_tform]
            )
            bias = self.out_bias_layer(
                x[..., -self.out_channels_after_tform:]
            )

        return z_vals, extra_tform_flat, extra_bias, tform_flat, bias

    def forward(self, x, **render_kwargs):
        # Apply parameterization
        param_x = torch.cat(
            [self.param(x[..., :6]), x[..., 6:]],
            dim=-1
        )

        # Get z value (disparity, radial depth, etc.), and transforms
        if self.is_global:
            z_vals, extra_tform, extra_bias, tform, bias = self._embed_params(
                param_x[..., self.param_channels:]
            )
        else:
            z_vals, extra_tform, extra_bias, tform, bias = self._embed_params(
                param_x
            )

        # Extra channel transform
        batch_size = x.shape[0]
        extra_x = x[..., :self.extra_in_channels]

        if self.D != 0 and self.extra_tform_size > 0:
            extra_tform = extra_tform.view(
                -1, self.extra_out_channels, self.extra_in_channels
            )
            extra_x = extra_x.view(-1, self.extra_in_channels)
            extra_x = self.out_extra_layer(
                (extra_tform @ extra_x.unsqueeze(-1)).squeeze(-1)
            )

            if extra_bias is not None:
                extra_x = extra_x.view(batch_size, -1) + extra_bias
            else:
                extra_x = extra_x.view(batch_size, -1)

        # Intersect and transform
        x = self.intersect_fn(
            x[..., :6], param_x[..., :self.param_channels], z_vals
        )

        if self.D != 0 and self.tform_size > 0:
            tform = tform.view(
                -1, self.tform_out_channels, self.tform_in_channels
            )
            x = x.view(-1, self.tform_in_channels)
            x = self.out_layer(
                (tform @ x.unsqueeze(-1)).squeeze(-1)
            )

            if bias is not None:
                x = x.view(batch_size, -1) + bias
            else:
                x = x.view(batch_size, -1)

        # Return
        x = torch.cat([extra_x, x], -1)
        return x

    def set_iter(self, i):
        self.cur_iter = i
        self.pe.set_iter(i)


embedding_dict = {
    'identity': IdentityEmbedding,
    'feature': FeatureEmbedding,
    'local_affine': LocalAffineEmbedding,
    'epipolar': EpipolarEmbedding,
    'global_warp': HomogenousEmbedding,
}
