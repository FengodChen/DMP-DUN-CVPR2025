"""api func"""
import importlib
from typing import Type
from base import Base as BaseClass

def api(cfg_name, gpu_num, img_channels, cs_ratio, if_train=True, log_timestamp=None, log_rollback_epoch=None, **kwargs):
    cfg_gen = importlib.import_module(f"configs.{cfg_name}", __package__).cfg_gen
    cfg:dict = cfg_gen(img_channels, cs_ratio, name=cfg_name, **kwargs)

    base_class:Type[BaseClass] = cfg["Base"]
    base = base_class(gpu_num=gpu_num, log_timestamp=log_timestamp, log_rollback_epoch=log_rollback_epoch, cfg=cfg, **(cfg["base_config"]))

    if if_train:
        base.set_dataset(**(cfg["dataset_train_config"]))
        base.set_dataset(**(cfg["dataset_val_config"]))
        base.set_dataloader(**(cfg["dataloader_train_config"]))
        base.set_dataloader(**(cfg["dataloader_val_config"]))
        base.train(**(cfg["train_config"]))

    return base
