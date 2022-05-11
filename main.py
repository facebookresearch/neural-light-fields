#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from uuid import uuid4

import hydra
from omegaconf import DictConfig, OmegaConf # @manual //github/third-party/omry/omegaconf:omegaconf

from iopath.common.file_io import PathManager, NativePathHandler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.distributed.launcher import (
    LaunchConfig,
    elastic_launch as launch
)

from nlf import (
    LightfieldTrainer,
    LightfieldDataModule,
    LightfieldSystem
)


def run(cfg: DictConfig, log_dir: str, model_dir: str, workflow_id: str) -> None:
    # Print
    print(OmegaConf.to_yaml(cfg))
    cfg = cfg.experiment

    # Seed
    if 'seed' in cfg.params \
        and not isinstance(cfg.params.seed, str) \
        and cfg.params.seed is not None:

        seed_everything(cfg.params.seed)

    # PathManager
    pmgr = PathManager()
    pmgr.register_handler(NativePathHandler())

    # CWD paths
    dir_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))
    os.chdir(dir_path)

    # Logging and saving
    if log_dir is None or log_dir == "":
        log_dir = os.path.expanduser(cfg.params.log_dir)

    pmgr.mkdirs(log_dir)

    if cfg.params.save_results:
        cfg.params.save_video_dir = os.path.join(
            log_dir,
            cfg.params.save_video_dir
        )
        cfg.params.save_image_dir = os.path.join(
            log_dir,
            cfg.params.save_image_dir
        )
        pmgr.mkdirs(cfg.params.save_video_dir)
        pmgr.mkdirs(cfg.params.save_image_dir)

    logger = TensorBoardLogger(save_dir=log_dir, name=cfg.params.name)

    # Setup system and datamodule
    dm = LightfieldDataModule(cfg)
    dm.prepare_data()

    # Checkpointing
    if model_dir is not None and model_dir != "":
        ckpt_dirpath = model_dir
    else:
        ckpt_dirpath = os.path.join(
            os.path.expanduser(cfg.params.ckpt_dir), cfg.params.name
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dirpath,
        filename='{epoch:d}',
        monitor='val/loss',
        mode='min',
        save_top_k=-1,
        save_last=True,
        every_n_val_epochs=cfg.training.ckpt_every
    )

    weights_checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        dirpath=ckpt_dirpath,
        filename='{epoch:d}-weights',
        monitor='val/loss',
        mode='min',
        save_top_k=-1,
        save_last=True,
        every_n_val_epochs=cfg.training.ckpt_every
    )
    weights_checkpoint_callback.CHECKPOINT_NAME_LAST = 'last-weights'

    # Trainer
    if cfg.params.load_from_weights:
        last_ckpt_path = f'{ckpt_dirpath}/last-weights.ckpt'
    else:
        last_ckpt_path = f'{ckpt_dirpath}/last.ckpt'

    if not pmgr.exists(last_ckpt_path):
        last_ckpt_path = None

    if cfg.params.render_only:
        cfg.training.render_every = 1
        cfg.training.val_every = 1

    if cfg.params.test_only:
        cfg.training.test_every = 1
        cfg.training.val_every = 1

    trainer = LightfieldTrainer(
        cfg,
        callbacks=[checkpoint_callback, weights_checkpoint_callback],
        resume_from_checkpoint=last_ckpt_path if not cfg.params.load_from_weights else None,
        logger=logger if cfg.params.tensorboard else None,
        weights_summary=None,
        progress_bar_refresh_rate=1,
        strategy='ddp' if cfg.training.num_gpus > 1 else None,
        check_val_every_n_epoch=cfg.training.val_every,
        benchmark=True,
        profiler=None,
        reload_dataloaders_every_n_epochs=cfg.training.reload_data_every,
    )

    if last_ckpt_path is not None and cfg.params.load_from_weights:
        system = LightfieldSystem.load_from_checkpoint(last_ckpt_path, cfg=cfg)
    else:
        system = LightfieldSystem(cfg)

    trainer.fit(system, datamodule=dm)


def elastic_run(cfg: DictConfig):
    if cfg.experiment.training.num_gpus > 1:
        lc = LaunchConfig(
            # Assuming devgpu testing, min = max nodes = 1
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=cfg.experiment.training.num_gpus,
            rdzv_backend="zeus",
            # run_id just has to be globally unique
            run_id=f"nlf_{uuid4()}",
            # for fault tolerance; for testing set it to 0 (no fault tolerance)
            max_restarts=0,
            start_method="spawn",
        )
        # The "run" function is called inside the elastic_launch
        ret = launch(lc, run)(cfg, "", "", "")
        print(f"Rank 0 results = {ret[0]}")
    else:
        run(cfg, "", "", "")


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    elastic_run(cfg)


if __name__ == '__main__':
    main()
