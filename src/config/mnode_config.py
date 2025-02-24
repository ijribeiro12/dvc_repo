import os
import glob
import math
import numpy as np
import pandas as pd
import configparser
import random
import socket
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
##############################
# Funções de Setup DDP
##############################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_ddp():
    # Assume que SLURM_PROCID, WORLD_SIZE, MASTER_ADDR e MASTER_PORT estão definidas
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')

    set_seed(42 + rank)
    return device, rank, world_size, device_id

def cleanup_ddp():
    dist.destroy_process_group()