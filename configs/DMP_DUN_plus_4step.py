from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from imageCS_utils.base_utils.Logger import MetricsLogger
from imageCS_utils.metrics.psnr import psnr
from imageCS_utils.metrics.msssim.ssim_skimage import ssim

from models.DMP_DUN_plus import Diffusion, Loss
from trainer import Trainer
from base import Base
from dataset import LabDataset

def cfg_gen(img_channels, cs_ratio, name):
    save_path = f"save/{name}/C{img_channels}R{cs_ratio}"

    cs_ratio = float(cs_ratio)
    phi_size = 64

    total_epoch = 20
    checkpoint_epoch = 10
    eval_epoch = 10

    batch_size = 8
    lr = 1e-4
    cos_end_lr = 1e-6

    config = {
        "Base": Base,
        "base_config": {
            "seed": 1234,
            "use_compile": False,
            "Net": Diffusion,
            "Logger": MetricsLogger,
            "Loss": Loss,
            "Opt": Adam,
            "Trainer": Trainer,
            "logger_path": f"{save_path}/log",
            "net_kwargs": {
                "model_kwargs": {
                    "image_size":256,
                    "num_channels":256,
                    "num_res_blocks":2,
                    "channel_mult":"",
                    "learn_sigma":True,
                    "class_cond":False,
                    "use_checkpoint":False,
                    "attention_resolutions":"32,16,8",
                    "num_heads":4,
                    "num_head_channels":64,
                    "num_heads_upsample":-1,
                    "use_scale_shift_norm":True,
                    "dropout":0.0,
                    "resblock_updown":True,
                    "use_fp16":True,
                    "use_new_attention_order":False,
                },
                "model_path": "pretrained_guided_diffusion/256x256_diffusion_uncond.pt",
                "sensing_rate": cs_ratio,
                "phi_size": phi_size,
                "time_step": 25,
                "start_time": 100,
                "res_inner_channels": 32,
                "res_inner_num": 4
            },
            "loss_kwargs": {
            },
            "opt_kwargs": {
                "lr": lr,
                "betas": (0.9, 0.999)
            },
            "trainer_kwargs": {
                "data_prepare_output_keys": ["x_true"],
                "net_forward_output_keys": ["x_pred"],
                "to_dev_keys": ["x_true"],
                "batch_size_sample_key": "x_true",
                "net_kwargs_map": {"x": "x_true"},
                "loss_kwargs_map": {
                    "data_prepare_output": {"x_true": "x_true"},
                    "net_forward_output": {"x_pred": "x_pred"},
                },
                "other_data_dict": {
                    "psnr_kwargs": {
                        "data_range": 1
                    }, 
                    "ssim_kwargs": {
                        "addition_func": lambda x: x * 255,
                        "data_range": 255,
                    }
                }
            },
            "LR_Scheduler_list": [
                CosineAnnealingLR
            ],
            "lr_scheduler_kwargs_list": [
                {
                    "T_max": total_epoch,
                    "eta_min": cos_end_lr
                }
            ]
        },

        ######### train dataset ########
        "dataset_train_config": {
            "Dataset": LabDataset,
            "dataset_kwargs": {
                "dataset_root": "./datasets/Coco2017",
                "dataset_len": -1,
                "crop_size": 64,
                "dtype": "train",
            },
            "dtype": "train"
        },
        "dataloader_train_config": {
            "DataLoader": DataLoader,
            "dataloader_kwargs": {
                "batch_size": batch_size,
                "shuffle": True,
                "num_workers": 8,
            },
            "dtype": "train"
        },

        ######### val dataset ########
        "dataset_val_config": {
            "Dataset": LabDataset,
            "dataset_kwargs": {
                "dataset_root": "./datasets/BSDS500/val",
                "dtype": "val",
            },
            "dtype": "val"
        },
        "dataloader_val_config": {
            "DataLoader": DataLoader,
            "dataloader_kwargs": {
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 1,
            },
            "dtype": "val"
        },

        "train_config": {
            "metrics_func_dict": {
                "psnr": psnr,
                "ssim": ssim
            },
            "checkpoint_epoch": checkpoint_epoch,
            "eval_epoch": eval_epoch,
            "epochs": total_epoch
        }, 
    
        "addition_params": {
            "save_path": save_path,
            "phi_size": phi_size
        }
    }

    return config
    