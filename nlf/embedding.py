#!/usr/bin/env python3

import torch
import numpy as np
from torch import nn

from utils.ray_utils import (
    ray_param_dict,
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
        self.freq_bands = 2 ** torch.linspace(0, cfg.n_freqs - 1, cfg.n_freqs)

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
    'random': RandomPE,
    'learnable': LearnablePE,
    'select': SelectPE,
}

class SequentialEmbedding(nn.Module):
    def __init__(
        self,
        *nets
    ):
        super().__init__()

        self.nets = nets
        self.model = nn.Sequential(
            *nets
        )
        self.out_channels = nets[-1].out_channels

    def forward(self, x):
        return self.model(x)

    def set_iter(self, i):
        for net in self.nets:
            net.set_iter(i)


class ConcatEmbedding(nn.Module):
    def __init__(
        self,
        *nets
    ):
        super().__init__()

        self.nets = nets
        self.out_channels = 0

        for i, net in enumerate(nets):
            setattr(self, f"net{i+1}", net)
            self.out_channels += net.out_channels

    def forward(self, x):
        xs = []

        for i in range(len(self.nets)):
            xs.append(getattr(self, f"net{i+1}")(x))

        return torch.cat(xs, -1)

    def set_iter(self, i):
        for i in range(len(self.nets)):
            getattr(self, f"net{i+1}").set_iter(i)


class SplitEmbedding(nn.Module):
    def __init__(
        self,
        splits=None,
        dim=-1,
        *nets,
    ):
        super().__init__()

        self.splits = splits
        self.dim = dim
        self.nets = nets
        self.out_channels = 0

        for i, net in enumerate(nets):
            setattr(self, f"net{i+1}", net)
            self.out_channels += net.out_channels

    def forward(self, x):
        xs = torch.split(x, self.splits, dim=self.dim)
        outs = []

        for i, x in enumerate(xs):
            outs.append(getattr(self, f"net{i+1}")(x))

        return torch.cat(outs, -1)

    def set_iter(self, i):
        pass


class BranchEmbedding(nn.Module):
    def __init__(
        self,
        net_1,
        net_2,
    ):
        super().__init__()

        self.net_1 = net_1
        self.net_2 = net_2
        self.out_channels = self.net_1.out_channels + self.net_2.out_channels

    def forward(self, x):
        x1 = self.net_1(x)
        x2 = self.net_2(x)
        return torch.cat([x1, x2], -1)

    def set_iter(self, i):
        self.net_1.set_iter(i)
        self.net_2.set_iter(i)


class SkipEmbedding(nn.Module):
    def __init__(
        self,
        net_1,
        net_2,
    ):
        super().__init__()

        self.net_1 = net_1
        self.net_2 = net_2
        self.out_channels = self.net_2.out_channels

    def forward(self, x):
        net_input = x
        x = self.net_1(x)
        x = torch.cat([x, net_input], -1)
        return self.net_2(x)

    def set_iter(self, i):
        self.net_1.set_iter(i)
        self.net_2.set_iter(i)


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


class WarpEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):

        super().__init__()

        self.D = cfg.depth
        self.W = cfg.hidden_channels
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'

        self.in_channels = in_channels
        if self.D == 0:
            self.out_channels = in_channels
        else:
            self.out_channels = cfg.out_channels
        self.skips = cfg.skips if 'skips' in cfg else []

        # Dummy
        self.dummy_layer = nn.Linear(1, 1)

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


class LocalWarpEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):

        super().__init__()

        self.D = cfg.depth
        self.W = cfg.hidden_channels
        self.tform_activation = cfg.tform_activation if 'tform_activation' in cfg else 'identity'
        self.bias_activation = cfg.bias_activation if 'bias_activation' in cfg else 'identity'
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'
        self.is_global = cfg.is_global if 'is_global' in cfg else False
        self.param_channels = in_channels if cfg.param_channels == 'all' else cfg.param_channels
        self.multiplier = cfg.multiplier if 'multiplier' in cfg else 1.0

        if self.is_global:
            self.in_channels = in_channels - self.param_channels
        else:
            self.in_channels = in_channels

        # PE
        self.pe = pe_dict[cfg.pe.type](
            self.in_channels,
            cfg.pe
        )

        if self.D == 0:
            self.out_channels = self.pe.out_channels
        else:
            self.out_channels = cfg.out_channels

        self.skips = cfg.skips if 'skips' in cfg else []

        # Dummy
        self.dummy_layer = nn.Linear(1, 1)

        # Hidden
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.pe.out_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.pe.out_channels, self.W)
            elif i == self.D - 1:
                layer = nn.Linear(
                    self.W,
                    self.out_channels * (self.param_channels + 1)
                )
            else:
                layer = nn.Linear(self.W, self.W)

            if i < self.D - 1:
                layer = nn.Sequential(layer, get_activation(self.layer_activation))

            setattr(self, f"transform{i+1}", layer)

        # Out
        self.out_tform_layer = get_activation(self.tform_activation)
        self.out_bias_layer = get_activation(self.bias_activation)
        self.out_layer = get_activation(self.activation)

    def embed_params(self, x):
        x = self.pe(x)
        input_x = x

        # Transform
        for i in range(self.D):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = getattr(self, f"transform{i+1}")(x)

        # Outputs
        tform_flat = self.out_tform_layer(x[..., :-self.out_channels])
        bias = self.out_bias_layer(x[..., -self.out_channels:])

        return torch.cat([tform_flat, bias], -1)

    def forward(self, x, **render_kwargs):
        if self.D == 0:
            return x

        if self.is_global:
            raw = self.embed_params(x[..., -self.in_channels:])
        else:
            raw = self.embed_params(x)

        tform = raw[..., :-self.out_channels].view(
            -1, self.out_channels, self.param_channels
        )
        bias = raw[..., -self.out_channels:]

        param_x = x[..., :self.param_channels]

        embed_x = self.out_layer(
            (tform @ param_x.unsqueeze(-1)).squeeze(-1)
        ) * self.multiplier + bias

        if 'embed_params' in render_kwargs and render_kwargs['embed_params']:
            return raw, embed_x
        else:
            return embed_x

    def set_iter(self, i):
        self.cur_iter = i
        self.pe.set_iter(i)


class HybridWarpEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):

        super().__init__()

        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.tform_activation = cfg.tform_activation if 'tform_activation' in cfg else 'identity'
        self.bias_activation = cfg.bias_activation if 'bias_activation' in cfg else 'identity'
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'
        self.tform_out_channels = cfg.tform_out_channels if 'tform_out_channels' in cfg else 0

        self.res_iters = cfg.res_iters if 'res_iters' in cfg else 0
        self.res_out_channels = cfg.res_out_channels if 'res_out_channels' in cfg else 0
        self.res_activation = cfg.res_activation if 'res_activation' in cfg else 'identity'

        self.is_global = cfg.is_global if 'is_global' in cfg else False
        self.param_channels = in_channels if cfg.param_channels == 'all' else cfg.param_channels

        if self.is_global:
            self.in_channels = in_channels - self.param_channels
        else:
            self.in_channels = in_channels

        # PE
        self.pe = pe_dict[cfg.pe.type](
            self.in_channels,
            cfg.pe
        )

        if self.D == 0:
            self.out_channels = self.pe.out_channels
        else:
            self.out_channels = self.tform_out_channels + self.res_out_channels

        self.skips = cfg.skips if 'skips' in cfg else []

        # Dummy
        self.dummy_layer = nn.Linear(1, 1)

        # Hidden
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.pe.out_channels, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.pe.out_channels, self.W)
            elif i == self.D - 1:
                layer = nn.Linear(
                    self.W,
                    self.tform_out_channels * (self.param_channels + 1) + self.res_out_channels
                )
            else:
                layer = nn.Linear(self.W, self.W)

            if i < self.D - 1:
                layer = nn.Sequential(layer, get_activation(self.layer_activation))

            setattr(self, f"transform{i+1}", layer)

        # Out
        self.out_tform_layer = get_activation(self.tform_activation)
        self.out_bias_layer = get_activation(self.bias_activation)
        self.out_res_layer = get_activation(self.res_activation)
        self.out_layer = get_activation(self.activation)

    def embed_params(self, x, include_res=False):
        x = self.pe(x)
        input_x = x

        # Transform
        for i in range(self.D):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = getattr(self, f"transform{i+1}")(x)

        # Outputs
        tform_flat = self.out_tform_layer(x[..., :-(self.tform_out_channels + self.res_out_channels)])
        bias = self.out_bias_layer(x[..., -(self.tform_out_channels + self.res_out_channels):-self.res_out_channels])
        res = self.out_res_layer(x[..., -self.res_out_channels:])

        return torch.cat([tform_flat, bias, res], -1)

    def forward(self, x):
        if self.D == 0:
            return x

        if self.is_global:
            raw = self.embed_params(x[..., -self.in_channels:])
        else:
            raw = self.embed_params(x)

        tform = raw[..., :-(self.tform_out_channels + self.res_out_channels)].view(
            -1, self.tform_out_channels, self.param_channels
        )
        bias = raw[..., -(self.tform_out_channels + self.res_out_channels):-self.res_out_channels]
        res = raw[..., -self.res_out_channels:]

        param_x = x[..., :self.param_channels]
        tform_x = self.out_layer(
            (tform @ param_x.unsqueeze(-1)).squeeze(-1) + bias
        )

        return torch.cat([tform_x, res], -1)

    def set_iter(self, i):
        self.cur_iter = i
        self.pe.set_iter(i)


embedding_dict = {
    'identity': IdentityEmbedding,
    'warp': WarpEmbedding,
    'local_warp': LocalWarpEmbedding,
    'hybrid_warp': HybridWarpEmbedding,
    'global_warp': HomogenousEmbedding,
}
