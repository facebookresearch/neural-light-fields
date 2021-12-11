#!/usr/bin/env python3

# @manual //github/third-party/omry/omegaconf:omegaconf
from omegaconf import DictConfig

def format_config(cfg: DictConfig):
    format_config_helper(cfg, cfg)

def format_config_helper(cfg, master_config: DictConfig):
    if isinstance(cfg, DictConfig):
        for key, _ in cfg.items():
            if isinstance(cfg[key], str):
                cfg[key] = cfg[key].format(config=master_config)
            else:
                format_config_helper(cfg[key], master_config)
