#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from nlf.visualizers.base import BaseVisualizer
from utils.ray_utils import (
    get_epi_rays,
)

from utils.visualization import (
    get_warp_dimensions,
    visualize_warp
)


class EPIVisualizer(BaseVisualizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.v = cfg.v if 'v' in cfg else None
        self.t = cfg.v if 't' in cfg else None
        self.H = cfg.H if 'H' in cfg else None

        self.near = cfg.near if 'near' in cfg else -1.0
        self.far = cfg.far if 'far' in cfg else 0.0

        if 'st_scale' in cfg and cfg.st_scale is not None:
            self.st_scale = cfg.st_scale
        elif 'lightfield' in system.cfg.dataset and 'st_scale' in system.cfg.dataset.lightfield:
            self.st_scale = system.cfg.dataset.lightfield.st_scale
        else:
            self.st_scale = 1.0

        if 'uv_scale' in cfg and cfg.uv_scale is not None:
            self.uv_scale = cfg.uv_scale
        else:
            self.uv_scale = 1.0

    def validation(self, batch, batch_idx):
        if batch_idx > 0:
            return

        system = self.get_system()
        dataset = system.trainer.datamodule.train_dataset
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        # Coordinates
        if self.t is not None:
            t = self.t
        else:
            t = 0

        if self.v is not None:
            v = self.v
        else:
            v = 0

        if self.H is not None:
            H = self.H

        ## Forward
        outputs = {}

        # Generate EPI rays
        rays = get_epi_rays(
            W, v, H, t, dataset.aspect,
            st_scale=self.st_scale,
            uv_scale=self.uv_scale,
            near=self.near, far=self.far
        ).type_as(batch['rays'])

        # RGB
        rgb = system(rays)['rgb']
        rgb = rgb.view(H, W, 3).cpu()
        rgb = rgb.permute(2, 0, 1)

        outputs['rgb'] = rgb

        # Warp
        if not system.is_subdivided:
            embedding = system.embed(rays)['embedding']
            params = system.embed_params(rays)['params']
        elif system.render_fn.subdivision.max_hits < 4:
            embedding = system.render_fn.embed_vis(rays)['embedding']
            params = system.render_fn.embed_params_vis(rays)['params']

        if not system.is_subdivided or system.render_fn.subdivision.max_hits < 4:
            warp_dims = get_warp_dimensions(
                embedding, W, H, k=min(embedding.shape[-1], 3)
            )
            warp = visualize_warp(
                embedding, warp_dims, W, H, k=min(embedding.shape[-1], 3)
            )
            outputs['warp'] = warp

            warp_dims = get_warp_dimensions(
                params, W, H, k=min(embedding.shape[-1], 3)
            )
            warp = visualize_warp(
                params, warp_dims, W, H, k=min(embedding.shape[-1], 3)
            )
            outputs['tform'] = warp

        return outputs

    def validation_image(self, batch, batch_idx):
        if batch_idx > 0:
            return {}

        # Outputs
        temp_outputs = self.validation(batch, batch_idx)
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'images/epi_{key}'] = temp_outputs[key]

        return outputs
