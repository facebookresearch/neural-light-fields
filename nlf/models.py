#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import torch.nn.functional as F
from torch import nn

from nlf.embedding import (
    RayParam,
    pe_dict,
    embedding_dict,
)

from nlf.nets import (
    net_dict
)

from utils.visualization import (
    get_warp_dimensions,
    visualize_warp
)

from nlf.rendering import ( # noqa
    over_composite_one,
    blend_one,
)

from nlf.activations import get_activation


class BaseLightfieldModel(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg
        self.pes = []
        self.embeddings = []
        self.models = []

        if 'latent_dim' in kwargs:
            self.latent_dim = kwargs['latent_dim']
        else:
            self.latent_dim = 0

        if 'is_subdivided' in kwargs:
            self.is_subdivided = kwargs['is_subdivided']
        else:
            self.is_subdivided = False

        if 'num_outputs' in kwargs:
            self.num_outputs = kwargs['num_outputs']
        else:
            self.num_outputs = cfg.num_outputs if 'num_outputs' in cfg else 3

        ## Latent embedding
        if self.is_subdivided:
            self.latent_pe = pe_dict[cfg.latent_pe.type](
                self.latent_dim + 3,
                cfg.latent_pe
            )

            self.pes += [self.latent_pe]

            self.latent_out_channels = self.latent_pe.out_channels
        else:
            self.latent_out_channels = 0

        ## Ray embedding

        # Initial embedding
        self.param = RayParam(
            cfg.param,
            in_channels=6,
        )

        self.embedding_pe = pe_dict[cfg.embedding_pe.type](
            cfg.param.n_dims,
            cfg.embedding_pe
        )

        self.pes += [self.embedding_pe]

        self.embedding_net = embedding_dict[cfg.embedding_net.type](
            self.embedding_pe.out_channels + self.latent_out_channels,
            cfg.embedding_net,
        )

        self.embeddings += [self.embedding_net]

    def set_iter(self, i):
        for emb in self.embeddings:
            emb.set_iter(i)

    def latent_head(self, rays, **render_kwargs):
        if 'no_param' in render_kwargs and render_kwargs['no_param']:
            param_rays = self.embedding_pe(rays[..., :self.cfg.param.n_dims])
        else:
            param_rays = self.embedding_pe(self.param(rays[..., :6]))

        if self.is_subdivided:
            if 'no_param' in render_kwargs and render_kwargs['no_param']:
                latent_codes = self.latent_pe(rays[..., self.cfg.param.n_dims:])
            else:
                latent_codes = self.latent_pe(rays[..., 6:])

            latent_rays = torch.cat([param_rays, latent_codes], -1)
        else:
            latent_rays = param_rays

        return latent_rays

    def embed_params(self, rays, **render_kwargs):
        latent_rays = self.latent_head(rays, **render_kwargs)
        embed_op = getattr(self.embedding_net, "embed_params", None)

        if callable(embed_op):
            return self.embedding_net.embed_params(latent_rays, **render_kwargs)
        else:
            return self.embedding_net(latent_rays, **render_kwargs)

    def embed(self, rays, **render_kwargs):
        latent_rays = self.latent_head(rays, **render_kwargs)
        return self.embedding_net(latent_rays, **render_kwargs)

    def validation(self, system, rays, idx):
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        # Add latent dims
        if self.is_subdivided:
            latent_codes = torch.zeros(
                rays.shape[:-1] + (self.latent_dim + 3,), device=rays.device
            )

            rays = torch.cat([rays, latent_codes], -1)

        # Warp embedding
        if not self.is_subdivided:
            embedding = system.embed(rays)['embedding']
            params = system.embed_params(rays)['params']
        elif system.render_fn.subdivision.max_hits < 4:
            embedding = system.render_fn.embed_vis(rays)['embedding']
            params = system.render_fn.embed_params_vis(rays)['params']

        if not self.is_subdivided or system.render_fn.subdivision.max_hits < 4:
            if idx == 0:
                self.embedding_warp_dims = get_warp_dimensions(
                    embedding, W, H, k=min(embedding.shape[-1], 3)
                )

                self.params_warp_dims = get_warp_dimensions(
                    params, W, H, k=min(params.shape[-1], 3)
                )

            embedding = visualize_warp(
                embedding, self.embedding_warp_dims, W, H, k=min(embedding.shape[-1], 3)
            )

            params = visualize_warp(
                params, self.params_warp_dims, W, H, k=min(embedding.shape[-1], 3)
            )

            return {
                'warp': embedding,
                'tform': params,
            }
        else:
            return {}

    def validation_video(self, system, rays, idx):
        temp_outputs = self.validation(
            system, rays, idx
        )
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'videos/{key}'] = temp_outputs[key]

        return outputs

    def validation_image(self, system, rays, idx):
        temp_outputs = self.validation(
            system, rays, idx
        )
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'images/{key}'] = temp_outputs[key]

        return outputs


class LightfieldModel(BaseLightfieldModel):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg, **kwargs)

        self.use_latent_color = cfg.use_latent_color if 'use_latent_color' in cfg else False

        ## Positional encoding and skip connection
        self.color_pe = pe_dict[cfg.color_pe.type](
            self.embedding_net.out_channels,
            cfg.color_pe
        )
        self.pes += [self.color_pe]

        color_model_in_channels = self.color_pe.out_channels

        ## Subdivision
        if self.is_subdivided and self.use_latent_color:
            color_model_in_channels += self.latent_pe.out_channels

        ## Color network
        self.color_model = net_dict[cfg.color_net.type](
            color_model_in_channels,
            self.num_outputs,
            cfg.color_net,
        )

        self.models += [self.color_model]

        ## Add pes to embeddings
        self.embeddings += self.pes

    def forward(self, rays, **render_kwargs):
        if 'embed_params' in render_kwargs and render_kwargs['embed_params']:
            embed_params, embed_rays = self.embed(rays, **render_kwargs)
            embed_rays = self.color_pe(embed_rays, **render_kwargs)
        else:
            embed_rays = self.color_pe(self.embed(rays, **render_kwargs), **render_kwargs)

        if self.is_subdivided and self.use_latent_color:
            if 'no_param' in render_kwargs and render_kwargs['no_param']:
                latent_codes = self.latent_pe(rays[..., self.cfg.param.n_dims:])
            else:
                latent_codes = self.latent_pe(rays[..., 6:])

            embed_rays = torch.cat([embed_rays, latent_codes], -1)

        outputs = self.color_model(embed_rays)

        if 'embed_params' in render_kwargs and render_kwargs['embed_params']:
            return torch.cat([embed_params, outputs], -1)
        else:
            return outputs


class BasisModel(BaseLightfieldModel):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg, **kwargs)

        self.use_latent_coeff = cfg.use_latent_coeff if 'use_latent_coeff' in cfg else False
        self.use_latent_basis = cfg.use_latent_basis if 'use_latent_basis' in cfg else False

        self.basis_channels = cfg.basis_channels
        self.num_basis = cfg.num_basis

        ## Positional encoding and skip connection
        self.coeff_pe = pe_dict[cfg.coeff_pe.type](
            self.embedding_net.out_channels - self.basis_channels,
            cfg.coeff_pe
        )
        self.pes += [self.coeff_pe]
        coeff_model_in_channels = self.coeff_pe.out_channels

        self.basis_pe = pe_dict[cfg.basis_pe.type](
            self.basis_channels,
            cfg.basis_pe
        )
        self.pes += [self.basis_pe]
        basis_model_in_channels = self.basis_pe.out_channels

        ## Subdivision
        if self.is_subdivided and self.use_latent_coeff:
            coeff_model_in_channels += self.latent_pe.out_channels

        if self.is_subdivided and self.use_latent_basis:
            basis_model_in_channels += self.latent_pe.out_channels

        ## Coefficient network
        self.coeff_model = net_dict[cfg.coeff_net.type](
            coeff_model_in_channels,
            (self.num_basis + 1) * 3,
            cfg.coeff_net,
        )
        self.models += [self.coeff_model]

        self.basis_model = net_dict[cfg.basis_net.type](
            basis_model_in_channels,
            self.num_basis * 3,
            cfg.basis_net,
        )
        self.models += [self.basis_model]

        self.activation = cfg.activation if 'activation' in cfg else 'sigmoid'
        self.out_layer = get_activation(self.activation)

        ## Add pes to embeddings
        self.embeddings += self.pes

    def forward(self, rays, **render_kwargs):
        # Get embedded rays
        if 'embed_params' in render_kwargs and render_kwargs['embed_params']:
            embed_params, embed_rays = self.embed(rays, **render_kwargs)
        else:
            embed_rays = self.embed(rays, **render_kwargs)

        # Get latent codes
        if self.is_subdivided and (self.use_latent_coeff or self.use_latent_basis):
            if 'no_param' in render_kwargs and render_kwargs['no_param']:
                latent_codes = self.latent_pe(rays[..., self.cfg.param.n_dims:])
            else:
                latent_codes = self.latent_pe(rays[..., 6:])
        
        # Get coefficient inputs
        coeff_rays = self.coeff_pe(embed_rays[..., self.basis_channels:], **render_kwargs)

        if self.is_subdivided and self.use_latent_coeff:
            coeff_rays = torch.cat([coeff_rays, latent_codes], -1)

        # Get basis inputs
        basis_rays = self.basis_pe(embed_rays[..., :self.basis_channels], **render_kwargs)

        if self.is_subdivided and self.use_latent_basis:
            basis_rays = torch.cat([basis_rays, latent_codes], -1)

        # Get outputs
        coeffs = self.coeff_model(coeff_rays)
        base_color = coeffs[..., -3:]
        coeffs = coeffs[..., :-3].view(rays.shape[0], -1, 3)
        basis = self.basis_model(basis_rays).view(rays.shape[0], -1, 3)

        outputs = self.out_layer((coeffs * basis).mean(1) + base_color)

        # Return
        if 'embed_params' in render_kwargs and render_kwargs['embed_params']:
            return torch.cat([embed_params, outputs], -1)
        else:
            return outputs


class MultipleLightfieldModel(BaseLightfieldModel):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg, **kwargs)

        self.num_slices = cfg.num_slices
        self.compressed = cfg.compressed

        ## Positional encoding and skip connection
        if self.compressed:
            self.color_pe = pe_dict[cfg.color_pe.type](
                self.embedding_net.out_channels,
                cfg.color_pe
            )
        else:
            self.color_pe = pe_dict[cfg.color_pe.type](
                self.embedding_net.out_channels // self.num_slices,
                cfg.color_pe
            )

        self.pes += [self.color_pe]

        ## Color network
        if self.compressed:
            self.color_model = net_dict[cfg.color_net.type](
                self.color_pe.out_channels,
                4 * self.num_slices,
                cfg.color_net,
            )
        else:
            self.color_model = net_dict[cfg.color_net.type](
                self.color_pe.out_channels,
                4,
                cfg.color_net,
            )

        self.models += [self.color_model]

        ## Add pes to embeddings
        self.embeddings += self.pes

    def _forward_compressed(self, rays, **render_kwargs):
        embed_rays = self.embed(rays)

        outputs = self.color_model(self.color_pe(embed_rays, **render_kwargs))
        outputs = outputs.view(
            -1, self.num_slices, 4
        )

        raw_rgbs, raw_alphas = outputs[..., :3], outputs[..., -1]
        rgbs = F.sigmoid(raw_rgbs)
        alphas = F.sigmoid(raw_alphas)
        #alphas = F.softmax(raw_alphas, dim=-1)

        if 'include_all' in render_kwargs and render_kwargs['include_all']:
            return torch.cat([rgbs, alphas.unsqueeze(-1)], -1)

        rgb, _, _ = over_composite_one(rgbs, alphas)
        #rgb = blend_one(rgbs, alphas)
        return rgb

    def _forward(self, rays, **render_kwargs):
        embed_rays = self.embed(rays)
        embed_rays = embed_rays.view(
            -1, embed_rays.shape[-1] // self.num_slices
        )

        outputs = self.color_model(self.color_pe(embed_rays, **render_kwargs))
        outputs = outputs.view(
            -1, self.num_slices, 4
        )

        raw_rgbs, raw_alphas = outputs[..., :3], outputs[..., -1]
        rgbs = F.sigmoid(raw_rgbs)
        alphas = F.sigmoid(raw_alphas)
        #alphas = F.softmax(raw_alphas, dim=-1)

        if 'include_all' in render_kwargs and render_kwargs['include_all']:
            return torch.cat([rgbs, alphas.unsqueeze(-1)], -1)

        rgb, _, _ = over_composite_one(rgbs, alphas)
        #rgb = blend_one(rgbs, alphas)
        return rgb

    def forward(self, rays, **render_kwargs):
        if self.compressed:
            return self._forward_compressed(rays, **render_kwargs)
        else:
            return self._forward(rays, **render_kwargs)

    def validation(self, system, rays, idx):
        outputs = super().validation(system, rays, idx)

        W = system.cur_wh[0]
        H = system.cur_wh[1]

        # RGBs, Alphas
        temp_outputs = system(rays, include_all=True)
        rgbs, alphas = temp_outputs['all_rgb'], temp_outputs['all_alpha']

        # Outputs
        outputs = {}

        for i in range(1):
            outputs[f'rgbs{i}'] = rgbs[..., i, :].reshape(
                H, W, 3
            ).cpu().permute(2, 0, 1)
            outputs[f'alphas{i}'] = alphas[..., i].reshape(
                H, W, 1
            ).cpu().permute(2, 0, 1)

        return outputs


ray_model_dict = {
    'lightfield': LightfieldModel,
    'multiple_lightfield': MultipleLightfieldModel,
}


class GeometryModel(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg

        if 'latent_dim' in kwargs:
            self.latent_dim = kwargs['latent_dim']
        else:
            self.latent_dim = 0

        if 'is_subdivided' in kwargs:
            self.is_subdivided = kwargs['is_subdivided']
        else:
            self.is_subdivided = False

        if 'num_outputs' in kwargs:
            self.num_outputs = kwargs['num_outputs']
        else:
            self.num_outputs = cfg.num_outputs if 'num_outputs' in cfg else 1

        self.pes = []
        self.embeddings = []
        self.models = []

        ## Position embedding
        self.geom_pe = pe_dict[cfg.geom_pe.type](
            3,
            cfg.geom_pe
        )
        self.pes += [self.geom_pe]

        ## Latent embedding
        if self.is_subdivided:
            self.latent_pe = pe_dict[cfg.latent_pe.type](
                self.latent_dim + 3,
                cfg.latent_pe
            )
            self.pes += [self.latent_pe]

            latent_out_channels = self.latent_pe.out_channels
        else:
            latent_out_channels = 0

        ## Geometry network
        self.geom_model = net_dict[cfg.geom_net.type](
            self.geom_pe.out_channels + latent_out_channels,
            self.num_outputs,
            cfg.geom_net,
        )
        self.models += [self.geom_model]

    def latent_head(self, pos):
        pos_embed = self.geom_pe(pos[..., :3])

        if self.is_subdivided:
            latent_embed = self.latent_pe(pos)
            latent_inp = torch.cat([pos_embed, latent_embed], -1)
        else:
            latent_inp = pos_embed

        return latent_inp

    def forward(self, rays):
        latent_inp = self.latent_head(rays[..., 6:])
        return self.geom_model(latent_inp)


pos_model_dict = {
    'geometry': GeometryModel,
    'lightfield': LightfieldModel,
}


def fuse_outputs_default(ray_outputs, pos_outputs):
    if pos_outputs.shape[-1] == 1:
        return torch.cat([ray_outputs, pos_outputs], -1)
    else:
        rgb = (ray_outputs + pos_outputs[..., :3]) / 2
        alpha = pos_outputs[..., -1:]

        return torch.cat([rgb, alpha], -1)


fuse_fn_dict = {
    'default': fuse_outputs_default,
}


class SubdividedModel(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.cfg = cfg

        if 'no_voxel' in self.cfg.subdivision and self.cfg.subdivision.no_voxel:
            self.latent_dim = self.cfg.subdivision.latent_dim - 2
        else:
            self.latent_dim = self.cfg.subdivision.latent_dim

        self.pes = []
        self.embeddings = []
        self.models = []

        ## Models
        self.use_pos = ('pos' in cfg)

        if self.use_pos:
            self.ray_model = ray_model_dict[cfg.ray.type](
                cfg.ray,
                latent_dim=self.latent_dim,
                is_subdivided=True,
                num_outputs=3
            )

            self.pos_model = pos_model_dict[cfg.pos.type](
                cfg.pos,
                latent_dim=self.latent_dim,
                is_subdivided=True,
                num_outputs=1
            )

            self.pes += self.ray_model.pes + self.pos_model.pes
            self.embeddings += self.ray_model.embeddings + self.pos_model.embeddings
            self.models += self.ray_model.models + self.pos_model.models
        else:
            self.ray_model = ray_model_dict[cfg.ray.type](
                cfg.ray,
                latent_dim=self.latent_dim,
                is_subdivided=True,
                num_outputs=4
            )

            self.pes += self.ray_model.pes
            self.embeddings += self.ray_model.embeddings
            self.models += self.ray_model.models

        ## Fuse
        self.fuse_outputs = fuse_fn_dict[cfg.fuse.type]

    def set_iter(self, i):
        for pe in self.pes:
            pe.set_iter(i)

    def embed_params(self, rays, bg, **render_kwargs):
        return self.ray_model.embed_params(rays, **render_kwargs)

    def embed(self, rays, bg, **render_kwargs):
        return self.ray_model.embed(rays, **render_kwargs)

    def forward(self, rays, bg, **render_kwargs):
        # Get and fuse outputs
        ray_outputs = self.ray_model(rays, **render_kwargs)

        if self.use_pos:
            pos_outputs = self.pos_model(rays)
            embed_params = ray_outputs[..., :-3]
        else:
            embed_params = ray_outputs[..., :-4]
            pos_outputs = ray_outputs[..., -1:]
            ray_outputs = ray_outputs[..., -4:-1]

        outputs = self.fuse_outputs(ray_outputs, pos_outputs)

        if 'embed_params' in render_kwargs and render_kwargs['embed_params']:
            return torch.cat([embed_params, outputs], -1)
        else:
            return outputs

    def validation_video(self, system, rays, idx):
        return self.ray_model.validation_video(system, rays, idx)

    def validation_image(self, system, rays, idx):
        return self.ray_model.validation_image(system, rays, idx)


class LatentSubdividedModel(SubdividedModel):
    def __init__(
        self,
        cfg,
    ):
        super().__init__(cfg)

        ## Latent model
        self.latent_model = pos_model_dict[cfg.latent.type](
            cfg.latent,
            latent_dim=0,
            is_subdivided=True,
            num_outputs=self.latent_dim,
        )

        self.pes += self.latent_model.pes
        self.embeddings += self.latent_model.embeddings
        self.models += self.latent_model.models

    def forward(self, rays, bg):
        # Latent codes
        latent_codes = self.latent_model(rays) * bg

        # Ray outputs
        ray_inputs = torch.cat([rays, latent_codes], -1)

        # Get and fuse outputs
        ray_outputs = self.ray_model(ray_inputs)

        if self.use_pos:
            pos_outputs = self.pos_model(ray_inputs)
        else:
            pos_outputs = ray_outputs[..., -1:]
            ray_outputs = ray_outputs[..., :3]

        return self.fuse_outputs(ray_outputs, pos_outputs)


model_dict = {
    'lightfield': LightfieldModel,
    'basis': BasisModel,
    'multiple_lightfield': MultipleLightfieldModel,
    'subdivided_lightfield': SubdividedModel,
    'latent_subdivided_lightfield': LatentSubdividedModel,
}
