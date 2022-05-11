#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

from torch.utils.data import DataLoader

import numpy as np
import torch

import imageio # noqa
from PIL import Image

from iopath.common.file_io import PathManager, NativePathHandler

from pytorch_lightning import (
    LightningModule,
    LightningDataModule,
    Trainer
)

from datasets import dataset_dict
from nlf.subdivision import subdivision_dict
from nlf.rendering import (
    render_chunked,
    render_fn_dict
)

from losses import loss_dict
from metrics import ( # noqa
    psnr_gpu,
    ssim_gpu,
    psnr,
    ssim,
    get_mean_outputs
)

from utils.config_utils import replace_config
from utils import (
    to8b,
    get_optimizer,
    get_scheduler,
    weight_init_dict
)

from nlf.models import model_dict
from nlf.regularizers import regularizer_dict
from nlf.visualizers import visualizer_dict


class LightfieldTrainer(Trainer):
    def __init__(
        self,
        cfg,
        **kwargs,
        ):
        super().__init__(
            gpus=cfg.training.num_gpus,
            max_epochs=cfg.training.num_epochs,
            log_every_n_steps=cfg.training.flush_logs,
            flush_logs_every_n_steps=cfg.training.flush_logs,
            **kwargs
        )

    def save_checkpoint(self, *args, **kwargs):
        if not self.is_global_zero:
            return

        super().save_checkpoint(*args, **kwargs)


class LightfieldDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.test_every = cfg.training.test_every
        self.current_epoch = 0
        self.is_testing = False

        # Multiscale
        self.test_only = self.cfg.params.test_only if ('test_only' in cfg.params) else False

        self.multiscale_training = self.cfg.training.multiscale if ('multiscale' in cfg.training) else False
        self.scales = self.cfg.training.scales if ('scales' in cfg.training) else []
        self.scale_epochs = self.cfg.training.scale_epochs if ('scale_epochs' in cfg.training) else []
        self.scale_batch_sizes = self.cfg.training.scale_batch_sizes if ('scale_batch_sizes' in cfg.training) else []

        self.cur_scale = 1.0
        self.cur_batch_size = self.cfg.training.batch_size

    def get_cur_scale(self, epoch):
        if not self.multiscale_training:
            return 1.0

        scale = 1.0
        batch_size = self.cfg.training.batch_size

        for s, b, i in zip(self.scales, self.scale_batch_sizes, self.scale_epochs):
            if epoch >= i:
                scale = s
                batch_size = b

        return scale, batch_size

    def prepare_data(self):
        ## Train, val, test datasets
        dataset_cl = dataset_dict[self.cfg.dataset.train.name] \
            if 'train' in self.cfg.dataset else dataset_dict[self.cfg.dataset.name]
        self.train_dataset = dataset_cl(self.cfg, split='train')
        dataset_cl = dataset_dict[self.cfg.dataset.val.name] \
            if 'val' in self.cfg.dataset else dataset_dict[self.cfg.dataset.name]
        self.val_dataset = dataset_cl(self.cfg, split='val')
        dataset_cl = dataset_dict[self.cfg.dataset.test.name] \
            if 'test' in self.cfg.dataset else dataset_dict[self.cfg.dataset.name]
        self.test_dataset = dataset_cl(self.cfg, split='test')
        dataset_cl = dataset_dict[self.cfg.dataset.render.name] \
            if 'render' in self.cfg.dataset else dataset_dict[self.cfg.dataset.name]
        self.render_dataset = dataset_cl(self.cfg, split='render')

        ## Regularizer datasets
        self.regularizer_datasets = {}

        for key in self.cfg.regularizers.keys():
            cfg = self.cfg.regularizers[key]

            if cfg is not None and 'dataset' in cfg:
                dataset_cl = dataset_dict[cfg.dataset.name]
                self.regularizer_datasets[cfg.type] = dataset_cl(
                    cfg, train_dataset=self.train_dataset
                )

        self.update_data()

    def setup(self, stage):
        pass

    def update_data(self):
        # Set iter
        self.train_dataset.cur_iter = self.current_epoch

        # Resize
        reset_dataloaders = False

        if self.multiscale_training:
            cur_scale, cur_batch = self.get_cur_scale(self.current_epoch)

            if cur_scale != self.cur_scale or cur_batch != self.cur_batch_size:
                print(f"Scaling dataset to scale {cur_scale} batch_size: {cur_batch}")
                self.cur_scale = cur_scale
                self.cur_batch_size = cur_batch

                self.train_dataset.scale(self.cur_scale)
                self.val_dataset.scale(self.cur_scale)
                self.test_dataset.scale(self.cur_scale)
                self.render_dataset.scale(self.cur_scale)
                reset_dataloaders = True

        # Crop
        self.train_dataset.crop()

        # Shuffle
        for dataset in self.regularizer_datasets.values():
            dataset.shuffle()

        return reset_dataloaders

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            batch_size=self.cur_batch_size,
            pin_memory=True
        )

    def val_dataloader(self):
        if ((self.current_epoch + 1) % self.test_every == 0) or self.test_only:
            print("Testing")
            dataset = self.test_dataset
            self.is_testing = True
        else:
            print("Validating")
            self.is_testing = False
            dataset = self.val_dataset

        return DataLoader(
            dataset,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            batch_size=1,
            pin_memory=True
        )


class LightfieldSystem(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.models = []
        self.embeddings = []

        # Path manager
        self.pmgr = PathManager()
        self.pmgr.register_handler(NativePathHandler())

        # Training and loss
        self.automatic_optimization = False
        self.training_started = False
        self.multiscale_training = self.cfg.training.multiscale if ('multiscale' in cfg.training) else False

        self.render_only = self.cfg.params.render_only if ('render_only' in cfg.params) else False
        self.test_only = self.cfg.params.test_only if ('test_only' in cfg.params) else False

        # Loss
        self.loss = loss_dict[self.cfg.training.loss.type]()

        # Subdivision
        self.is_subdivided = ('subdivision' in cfg.model) and (cfg.model.subdivision.type is not None)
        self.has_subdivision_geometry = (self.cfg.model.render.type == "subdivided")

        if self.is_subdivided:
            self.subdivision = subdivision_dict[
                self.cfg.model.subdivision.type
            ](
                self,
                self.cfg.model.subdivision,
            )

            replace_config(self.cfg, voxel_size=float(self.subdivision.voxel_size.cpu()))
            self.models += [self.subdivision]
        else:
            self.subdivision = None

        # Model
        full_model = model_dict[self.cfg.model.type](
            self.cfg.model,
        )
        self.models += full_model.models
        self.embeddings += full_model.embeddings

        # Render function
        self.render_fn = render_fn_dict[
            self.cfg.model.render.type
        ](
            full_model,
            self.subdivision,
            net_chunk=self.cfg.training.net_chunk,
        )

        # Regularizers
        self.regularizers = []
        self.regularizer_configs = []

        for key in self.cfg.regularizers.keys():
            cfg = self.cfg.regularizers[key]
            reg = regularizer_dict[cfg.type](
                self,
                cfg
            )

            self.regularizer_configs.append(cfg)
            self.regularizers.append(reg)
            setattr(self, f"reg_{cfg.type}", reg)

        # Visualizers
        self.visualizers = []

        for key in self.cfg.visualizers.keys():
            cfg = self.cfg.visualizers[key]
            vis = visualizer_dict[cfg.type](
                self,
                cfg
            )

            self.visualizers.append(vis)

        # Weight initialization
        self.apply(
            weight_init_dict[self.cfg.training.weight_init.type](self.cfg.training.weight_init)
        )

    def load_state_dict(self, state_dict, strict=False):
        new_state_dict = {}

        for key in state_dict.keys():
            if 'subdivision' in key:
                setattr(self.subdivision, key.split('subdivision.')[-1], state_dict[key])
            else:
                new_state_dict[key] = state_dict[key]

        super().load_state_dict(new_state_dict, strict=False)

    def to_ndc(self, rays):
        return self.trainer.datamodule.train_dataset.to_ndc(rays)

    def embed_params(self, rays, **render_kwargs):
        if 'apply_ndc' in render_kwargs:
            if render_kwargs['apply_ndc']:
                rays = self.to_ndc(rays)

            del render_kwargs['apply_ndc']

        return self.run_chunked(
            rays, self.render_fn.embed_params, **render_kwargs
        )

    def embed(self, rays, **render_kwargs):
        if 'apply_ndc' in render_kwargs:
            if render_kwargs['apply_ndc']:
                rays = self.to_ndc(rays)

            del render_kwargs['apply_ndc']

        return self.run_chunked(
            rays, self.render_fn.embed, **render_kwargs
        )

    def forward_all(self, rays, **render_kwargs):
        if 'apply_ndc' in render_kwargs:
            if render_kwargs['apply_ndc']:
                rays = self.to_ndc(rays)

            del render_kwargs['apply_ndc']

        return self.run_chunked(
            rays, self.render_fn.forward_all, **render_kwargs
        )

    def forward(self, rays, **render_kwargs):
        if 'apply_ndc' in render_kwargs:
            if render_kwargs['apply_ndc']:
                rays = self.to_ndc(rays)

            del render_kwargs['apply_ndc']

        return self.run_chunked(
            rays, self.render_fn, **render_kwargs
        )

    def run_chunked(self, rays, fn, **render_kwargs):
        return render_chunked(
            rays,
            fn,
            render_kwargs,
            chunk=self.cfg.training.ray_chunk
        )

    def configure_optimizers(self):
        ## Optimizers
        self.model_optimizer = get_optimizer(
            self.cfg.training.color, self.models
        )
        self.embedding_optimizer = get_optimizer(
            self.cfg.training.embedding, self.embeddings
        )

        optimizers = [self.model_optimizer, self.embedding_optimizer]

        ## Schedulers
        model_scheduler = get_scheduler(
            self.cfg.training.color, self.model_optimizer
        )
        embedding_scheduler = get_scheduler(
            self.cfg.training.embedding, self.embedding_optimizer
        )

        schedulers = [model_scheduler, embedding_scheduler]
        self.scheduler_configs = [self.cfg.training.color, self.cfg.training.embedding]

        ## Regularizers
        for i, cfg in enumerate(self.regularizer_configs):
            if 'optimizer' in cfg:
                reg_optim = get_optimizer(
                    cfg.optimizer, [self.regularizers[i]]
                )
                optimizers.append(reg_optim)
                reg_sched = get_scheduler(
                    cfg.optimizer, reg_optim
                )
                schedulers.append(reg_sched)
                self.scheduler_configs.append(cfg.optimizer)

        return optimizers, schedulers

    def get_train_iter(self, epoch, batch_idx, val=False):
        if self.render_only or self.test_only:
            epoch = 10000000
        
        if self.cfg.params.load_from_weights:
            epoch += self.cfg.params.start_epoch

        train_iter = (
            len(self.trainer.datamodule.train_dataset) // self.cfg.training.batch_size
        ) * epoch + batch_idx * self.cfg.training.num_gpus

        if not val:
            train_iter += self.global_rank

        return train_iter

    def training_step(self, batch, batch_idx):
        if self.render_only or self.test_only:
            return {}

        ## Flag indicating the training has started
        self.training_started = True

        ## Tell model what training iter it is
        train_iter = self.get_train_iter(self.current_epoch, batch_idx)
        self.render_fn.model.set_iter(train_iter)

        ## Inputs and outputs
        outputs = {}
        batch = self.trainer.datamodule.train_dataset.format_batch(batch)
        rays, rgb, = batch['rays'], batch['rgb']

        ## Image loss
        results = self(rays)
        image_loss = self.loss(results['rgb'], rgb)
        loss = image_loss

        if self.cfg.params.print_loss:
            print("Image loss, iter:", loss, train_iter)

        outputs['train/psnr'] = psnr_gpu(results['rgb'], rgb).detach()

        ## Regularization losses
        reg_loss = 0.0
        opt_idx = 2

        optimizers = self.optimizers(use_pl_optimizer=True)
        loss_optimizers = [optimizers[0], optimizers[1]]
        reg_loss_optimizers = []

        for reg, cfg in zip(self.regularizers, self.regularizer_configs):
            reg.batch_size = self.trainer.datamodule.cur_batch_size
            reg.set_iter(train_iter)
            cur_loss = reg.loss(batch, batch_idx) * reg.loss_weight()
            reg_loss += cur_loss

            if not reg.warming_up():
                loss += cur_loss

            if 'optimizer' in cfg:
                if reg.warming_up() and reg.cur_iter > 0:
                    reg_loss_optimizers.append(optimizers[opt_idx])
                else:
                    loss_optimizers.append(optimizers[opt_idx])

                opt_idx += 1

        ## Gradient descent step
        if len(reg_loss_optimizers) > 0:
            for opt in reg_loss_optimizers: opt.zero_grad()
            self.manual_backward(reg_loss)
            for opt in reg_loss_optimizers: opt.step()

        for opt in loss_optimizers: opt.zero_grad()
        self.manual_backward(loss)
        for opt in optimizers: opt.step()

        ## Return
        outputs['train/loss'] = loss.detach()

        return outputs

    def training_epoch_end(self, outputs):
        if ((self.current_epoch + 1) % self.cfg.training.log_every) == 0:
            # Log
            mean = get_mean_outputs(outputs)

            for key, val in mean.items():
                self.log(key, val, on_epoch=True, on_step=False, sync_dist=True)
                print(f"{key}: {val}")

        # Schedulers
        if self.training_started:
            schedulers = self.lr_schedulers()

            for sched in schedulers:
                sched.step()

        # Dataset update
        self.trainer.datamodule.current_epoch = self.current_epoch + 1

        reset_val = (
            (self.current_epoch + 2) % self.trainer.datamodule.test_every == 0 \
                or (self.current_epoch + 1) % self.trainer.datamodule.test_every == 0 \
                or (self.current_epoch) % self.trainer.datamodule.test_every == 0
        ) or self.test_only
        resized = False

        if ((self.current_epoch + 1) % self.cfg.training.update_data_every) == 0:
            print("Updating data")
            resized = self.trainer.datamodule.update_data()

        if resized:
            print("Resized data")
            self.trainer.reset_train_dataloader(self)

        if reset_val or resized:
            print("Re-setting dataloaders")
            self.trainer.reset_val_dataloader(self)

        # Subdivisions
        if self.is_subdivided and self.trainer.is_global_zero:
            perform_update = self.subdivision.update_every != float("inf") \
                and ((self.current_epoch + 1) % self.subdivision.update_every) == 0 \
                and self.training_started

            if perform_update:
                self.subdivision.update()

    def validation_video(self, batch, batch_idx): # noqa
        if not self.trainer.is_global_zero:
            return

        # Setup
        all_videos = {}

        # Render
        all_videos['videos/rgb'] = []

        def _add_outputs(outputs):
            for key in outputs:
                if key not in all_videos:
                    all_videos[key] = []

                all_videos[key].append(np.array(outputs[key]))

        all_times = []

        for idx in range(len(self.trainer.datamodule.render_dataset)):
            # Image
            cur_batch = self.trainer.datamodule.render_dataset[idx]
            W, H = cur_batch['W'], cur_batch['H']
            self.cur_wh = [int(W), int(H)]

            for k in cur_batch:
                if isinstance(cur_batch[k], torch.Tensor):
                    cur_batch[k] = cur_batch[k].cuda()

            start_time = time.time()
            cur_results = self(cur_batch['rays'])
            torch.cuda.synchronize()
            all_times.append(time.time() - start_time)
            print(idx, all_times[-1])

            cur_img = cur_results['rgb'].view(H, W, 3).cpu().numpy()

            cur_img = cur_img.transpose(2, 0, 1)
            all_videos['videos/rgb'].append(cur_img)

            # Model outputs
            outputs = self.render_fn.model.validation_video(
                self, cur_batch['rays'], idx
            )
            _add_outputs(outputs)

            # Subdivision outputs
            if self.is_subdivided:
                outputs = self.subdivision.validation_video(
                    cur_batch['rays'], cur_results
                )
                _add_outputs(outputs)

            # Regularizer outputs
            for reg in self.regularizers:
                outputs = reg.validation_video(cur_batch)
                _add_outputs(outputs)

            # Visualizer outputs
            for vis in self.visualizers:
                outputs = vis.validation_video(cur_batch)
                _add_outputs(outputs)

        print("Average time:", np.mean(all_times[1:-1]))

        # Log all videos
        for key in all_videos:
            cur_vid = np.stack(all_videos[key], axis=0)

            if 'ignore' in key:
                continue

            if self.cfg.training.num_gpus <= 1 and self.cfg.params.log_videos:
                self.logger.experiment.add_video(
                    key,
                    cur_vid[None],
                    self.global_step,
                    fps=24
                )

        # Save outputs
        if self.cfg.params.save_results:
            epoch = str(self.current_epoch + 1)

            if self.render_only:
                save_video_dir = self.cfg.params.save_video_dir.replace('val_videos', 'render')
            else:
                save_video_dir = os.path.join(self.cfg.params.save_video_dir, epoch)

            for key in all_videos:
                cur_vid = all_videos[key]
                vid_suffix = key.split('/')[-1]

                self.pmgr.mkdirs(os.path.join(save_video_dir, vid_suffix))

                for i in range(len(cur_vid)):
                    cur_im = np.squeeze(cur_vid[i])

                    with self.pmgr.open(
                        os.path.join(save_video_dir, vid_suffix, f'{i:04d}.png'),
                        'wb'
                    ) as f:
                        if len(cur_im.shape) == 3:
                            Image.fromarray(to8b(cur_im.transpose(1, 2, 0))).save(f)
                        else:
                            Image.fromarray(to8b(cur_im)).save(f)

    def validation_image(self, batch, batch_idx): # noqa
        batch_idx = batch_idx * self.cfg.training.num_gpus + self.global_rank

        # Forward
        rays, rgb, = batch['rays'], batch['rgb']
        rays = rays.view(-1, 6)
        rgb = rgb.view(-1, 3)
        results = self(rays)

        # Setup
        W, H = batch['W'], batch['H']
        self.cur_wh = [int(W), int(H)]
        all_images = {}

        # Logging
        img = results['rgb'].view(H, W, 3).cpu().numpy()
        img = img.transpose(2, 0, 1)
        img_gt = rgb.view(H, W, 3).cpu().numpy()
        img_gt = img_gt.transpose(2, 0, 1)
        stack = np.concatenate([img_gt, img], -1)

        all_images['ignore/pred'] = img
        all_images['ignore/gt'] = img_gt
        all_images['images/gt_pred'] = stack

        def _add_outputs(outputs):
            for key in outputs:
                if key not in all_images:
                    all_images[key] = np.array(outputs[key])

        # Model images
        if not self.trainer.datamodule.is_testing:
            outputs = self.render_fn.model.validation_image(
                self, rays, batch_idx
            )
            _add_outputs(outputs)

            # Subdivision images
            if self.is_subdivided:
                outputs = self.subdivision.validation_image(
                    batch, batch_idx, results
                )
                _add_outputs(outputs)

            # Regularizer images
            for reg in self.regularizers:
                outputs = reg.validation_image(batch, batch_idx)
                _add_outputs(outputs)

            # Visualizer images
            for vis in self.visualizers:
                outputs = vis.validation_image(batch, batch_idx)
                _add_outputs(outputs)

        # Log all images
        for key in all_images:
            if 'ignore' in key:
                continue

            if self.cfg.training.num_gpus <= 1 and self.cfg.params.log_images:
                self.logger.experiment.add_images(
                    f'{key}_{batch_idx}',
                    all_images[key][None],
                    self.global_step,
                )

        # Save outputs
        if self.cfg.params.save_results:
            epoch = str(self.current_epoch + 1)

            if self.test_only:
                save_image_dir = self.cfg.params.save_image_dir.replace('val_images', 'testset')
            else:
                save_image_dir = os.path.join(self.cfg.params.save_image_dir, epoch)

            for key in all_images:
                im_suffix = key.split('/')[0]
                im_name = key.split('/')[-1]

                self.pmgr.mkdirs(os.path.join(save_image_dir, im_suffix))

                with self.pmgr.open(
                    os.path.join(save_image_dir, im_suffix, f'{batch_idx:04d}_{im_name}.png'),
                    'wb'
                ) as f:
                    all_images[key] = np.squeeze(all_images[key])

                    if len(all_images[key].shape) == 3:
                        Image.fromarray(to8b(all_images[key].transpose(1, 2, 0))).save(f)
                    else:
                        Image.fromarray(to8b(all_images[key])).save(f)

        return {
            'val/loss': self.loss(results['rgb'], rgb).detach().cpu().numpy(),
            'val/psnr': psnr(img.transpose(1, 2, 0), img_gt.transpose(1, 2, 0)),
            'val/ssim': ssim(img.transpose(1, 2, 0), img_gt.transpose(1, 2, 0)),
        }

    def validation_step(self, batch, batch_idx):
        train_iter = self.get_train_iter(self.current_epoch + 1, 0, True)
        self.render_fn.model.set_iter(train_iter)

        for reg in self.regularizers:
            reg.set_iter(train_iter)

        # Render video
        try:
            if batch_idx == 0 and \
                ((self.current_epoch + 1) % self.cfg.training.render_every == 0 or self.render_only):
                self.validation_video(batch, batch_idx)

            # Render image
            log = self.validation_image(batch, batch_idx)

            if self.render_only or self.test_only:
                exit(0)
        except OSError as error:
            print(error)

        # Return
        return log

    def validation_epoch_end(self, outputs):
        # Log
        mean = get_mean_outputs(outputs, cpu=True)
        epoch = str(self.current_epoch + 1)
        self.pmgr.mkdirs(os.path.join(self.cfg.params.save_image_dir, epoch))

        with self.pmgr.open(
            os.path.join(self.cfg.params.save_image_dir, epoch, 'metrics.txt'),
            'w'
        ) as f:
            for key, val in mean.items():
                self.log(key, val, on_epoch=True, on_step=False, sync_dist=True)
                print(f"{key}: {val}")
                f.write(f'{key}: {float(val)}\n')

        return {}
