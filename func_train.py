"""train and test GuidedDiffusion using argv"""
import gc
from sys import argv

import torch

from api import api
from imageCS_utils.base_utils.status import QueueStateManager
from imageCS_utils.file_writer import Log_Writer

def train(cfg_name:str, gpu_num:int, cs_ratios:list[str]):
    # Init Logger and Queue
    log_writer = Log_Writer("log", "finished_work.log")
    task_name = f"train-{cfg_name}-{cs_ratios}"
    qma = QueueStateManager(
        name=task_name,
        gpu=gpu_num
    )
    qma.start()

    # Train Loop
    for cs_ratio in cs_ratios:
        base = api(
            cfg_name=cfg_name,
            gpu_num=gpu_num,
            img_channels=1,
            cs_ratio = cs_ratio,
            if_train = True,
        )

    # Release Logger and Queue
    qma.stop()
    log_writer.write(task_name)
