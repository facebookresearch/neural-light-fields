#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from collections import defaultdict


class Render(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        if 'net_chunk' in kwargs:
            self.net_chunk = kwargs['net_chunk']
        else:
            self.net_chunk = 32768

    def _run(self, x, fn, **render_kwargs):
        x = x.view(-1, x.shape[-1])

        # Chunked inference
        B = x.shape[0]
        out_chunks = []

        for i in range(0, B, self.net_chunk):
            out_chunks += [fn(x[i:i+self.net_chunk], **render_kwargs)]

        out = torch.cat(out_chunks, 0)
        return out.view(-1, out.shape[-1])

    def forward_all(self, rays, **render_kwargs):
        # Get outputs
        results = {}
        results['value'] = self._run(
            rays,
            self.model,
            **render_kwargs
        )

        return results


class RenderValue(Render):
    def __init__(
        self,
        model,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model

    def forward(self, rays, **render_kwargs):
        # Get outputs
        results = {}
        results['value'] = self._run(
            rays,
            self.model,
            **render_kwargs
        )

        return results


class RenderLightfield(Render):
    def __init__(
        self,
        model,
        *args,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model

    def embed_params(self, rays, **render_kwargs):
        raw = self._run(
            rays,
            self.model.embed_params,
            **render_kwargs
        )

        return {'params': raw}

    def embed(self, rays, **render_kwargs):
        raw = self._run(
            rays,
            self.model.embed,
            **render_kwargs
        )

        return {'embedding': raw}

    def forward(self, rays, **render_kwargs):
        # Get outputs
        results = {}
        results['rgb'] = self._run(
            rays,
            self.model,
            **render_kwargs
        )

        return results


class RenderMultipleLightfield(RenderLightfield):
    def __init__(
        self,
        model,
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)

    def forward(self, rays, **render_kwargs):
        # Get outputs
        results = {}

        if 'include_all' in render_kwargs and render_kwargs['include_all']:
            outputs = self._run(
                rays,
                self.model,
                **render_kwargs
            )

            outputs = outputs.view(
                rays.shape[0], -1, 4
            )

            results['all_rgb'] = outputs[..., :3]
            results['all_alpha'] = outputs[..., -1]
        else:
            results['rgb'] = self._run(
                rays,
                self.model,
                **render_kwargs
            )

        return results


class RenderSubdividedLightfield(Render):
    def __init__(
        self,
        model,
        subdivision,
        *args,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model
        self.subdivision = subdivision
        self.use_bg = False if 'use_bg' not in self.subdivision.cfg else self.subdivision.cfg.use_bg
        self.white_background = False if 'white_background' not in self.subdivision.cfg else self.subdivision.cfg.white_background
        self.forward_facing = True if 'forward_facing' not in self.subdivision.cfg else self.subdivision.cfg.forward_facing

    def _run_fg_bg(self, isect_inps, isect_bg, fn, **render_kwargs):
        batch_size, num_slices = isect_inps.shape[0], isect_inps.shape[1]

        # Get foreground, background inputs
        isect_inps_bg = isect_inps[..., -1, :]

        fg_mask = ~isect_bg.unsqueeze(-1).repeat(
            1, 1, isect_inps.shape[-1]
        )
        isect_inps_fg = isect_inps[fg_mask].view(
            -1, isect_inps.shape[-1]
        )

        # Evaluate background
        if self.white_background:
            raw_bg = 1.0
        else:
            raw_bg = 0.0

        # Evaluate foreground
        raw_fg = self._run(isect_inps_fg, fn, bg=1, **render_kwargs).view(
            isect_inps_fg.shape[0], -1
        )

        # Add fg to raw outputs
        raw = torch.zeros(
            (batch_size, num_slices, raw_fg.shape[-1]), device=isect_inps.device
        )
        fg_mask = ~isect_bg.unsqueeze(-1).repeat(
            1, 1, raw.shape[-1]
        )
        raw[fg_mask] = raw_fg.view(-1)

        # Add bg to raw outputs
        raw = raw.view(batch_size, num_slices, raw.shape[-1])
        raw[..., -1, :] = raw_bg

        return raw

    def embed_params(self, rays, **render_kwargs):
        if 'isect_inps' in render_kwargs and 'isect_mask' in render_kwargs:
            isect_inps = render_kwargs['isect_inps']
            isect_mask = render_kwargs['isect_mask']

            del render_kwargs['isect_inps']
            del render_kwargs['isect_mask']
        else:
            isect_inps, isect_depths, isect_idx, isect_mask = self.subdivision(rays)

        raw = self._run_fg_bg(
            isect_inps, isect_mask, self.model.embed_params, **render_kwargs
        )

        return {'params': raw, 'isect_inps': isect_inps, 'isect_mask': isect_mask}

    def embed(self, rays, **render_kwargs):
        isect_inps, isect_depths, isect_idx, isect_mask = self.subdivision(rays)
        raw = self._run_fg_bg(
            isect_inps, isect_mask, self.model.embed, **render_kwargs
        )

        return {'embedding': raw}

    def embed_vis(self, rays, **render_kwargs):
        isect_inps, isect_depths, isect_idx, isect_mask = self.subdivision(rays)

        raw_embed = self._run_fg_bg(
            isect_inps, isect_mask, self.model.embed, **render_kwargs
        )

        raw_color = self._run_fg_bg(isect_inps, isect_mask, self.model)

        embed_final, _, _, _ = over_composite_all(
            raw_embed,
            raw_color[..., -1],
            isect_depths,
            ~isect_mask
        )

        # Return
        results = {}
        results['embedding'] = embed_final

        return results

    def embed_params_vis(self, rays, **render_kwargs):
        isect_inps, isect_depths, isect_idx, isect_mask = self.subdivision(rays)

        raw_embed_params = self._run_fg_bg(
            isect_inps, isect_mask, self.model.embed_params, **render_kwargs
        )

        raw_color = self._run_fg_bg(isect_inps, isect_mask, self.model)

        embed_params_final, _, _, _ = over_composite_all(
            raw_embed_params,
            raw_color[..., -1],
            isect_depths,
            ~isect_mask
        )

        # Return
        results = {}
        results['params'] = embed_params_final

        return results

    def forward_all(self, rays, **render_kwargs):
        if 'isect_inps' in render_kwargs and 'isect_mask' in render_kwargs:
            isect_inps = render_kwargs['isect_inps']
            isect_mask = render_kwargs['isect_mask']

            del render_kwargs['isect_inps']
            del render_kwargs['isect_mask']
        else:
            isect_inps, isect_depths, isect_idx, isect_mask = self.subdivision(rays)

        raw = self._run_fg_bg(
            isect_inps, isect_mask, self.model, **render_kwargs
        )

        return {'value': raw, 'isect_inps': isect_inps, 'isect_mask': isect_mask}

    def forward(self, rays, **render_kwargs):
        isect_inps, isect_depths, isect_idx, isect_mask = self.subdivision(rays)
        raw = self._run_fg_bg(isect_inps, isect_mask, self.model)

        # Over-composite
        rgb, alphas = raw[..., :3], raw[..., -1]
        isect_mask[..., -1] = 0

        rgb_final, depth_final, accum, weights = over_composite_all(
            rgb,
            alphas,
            isect_depths,
            ~isect_mask
        )

        # Return
        results = {}
        results['rgb'] = rgb_final
        results['depth'] = depth_final
        results['accum'] = accum

        if 'include_all' in render_kwargs and render_kwargs['include_all']:
            results['all_rgb'] = rgb
            results['all_alpha'] = alphas
            results['all_depths'] = isect_depths
            results['all_weights'] = weights
            results['all_indices'] = isect_idx

        return results


def blend_one(rgb, weights):
    rgb_final = torch.sum(weights.unsqueeze(-1) * rgb, -2)

    return rgb_final


def over_composite_one(rgb, alphas):
    alphas_shifted = torch.cat(
        [
            torch.ones_like(alphas[:, :1]),
            1 - alphas + 1e-8
        ],
        -1
    )
    weights = alphas * torch.cumprod(
        alphas_shifted, -1
    )[:, :-1]
    accum = weights[:, :-1].sum(-1)

    rgb_final = torch.sum(weights.unsqueeze(-1) * rgb, -2)

    return rgb_final, accum, weights


def over_composite_all(rgb, alphas, depths, is_valid):
    alphas = torch.where(is_valid, alphas, torch.zeros_like(alphas))
    alphas_shifted = torch.cat(
        [
            torch.ones_like(alphas[:, :1]),
            1 - alphas + 1e-8
        ],
        -1
    )
    weights = alphas * torch.cumprod(
        alphas_shifted, -1
    )[:, :-1]
    accum = weights[:, :-1].sum(-1)

    rgb_final = torch.sum(weights.unsqueeze(-1) * rgb, -2)
    depth_final = torch.sum(weights[:, :-1] * depths[:, :-1], -1) / (accum + 1e-8)

    return rgb_final, depth_final, accum, weights


render_fn_dict = {
    'lightfield': RenderLightfield,
    'subdivided': RenderSubdividedLightfield,
    'multiple': RenderMultipleLightfield,
}


def render_chunked(
    rays,
    render_fn,
    render_kwargs,
    chunk,
    ):
    B = rays.shape[0]
    results = defaultdict(list)
    chunk_args = getattr(render_kwargs, 'chunk_args', None)

    for i in range(0, B, chunk):
        if chunk_args is None:
            chunk_render_kwargs = render_kwargs
        else:
            chunk_render_kwargs = {}

            for k in render_kwargs.keys():
                if k in chunk_args:
                    chunk_render_kwargs[k] = {}

                    for j in render_kwargs[k]:
                        chunk_render_kwargs[k][j] = render_kwargs[k][j][i:i+chunk]
                else:
                    chunk_render_kwargs[k] = render_kwargs[k]

        rendered_ray_chunks = \
            render_fn(
                rays[i:i+chunk],
                **chunk_render_kwargs
            )

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    return results
