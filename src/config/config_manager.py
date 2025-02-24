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
# ConfigManager
##############################
class ConfigManager:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self._check_sections()

    def _check_sections(self):
        required_sections = ['paths', 'transformer', 'train']
        for section in required_sections:
            if section not in self.config:
                raise configparser.NoSectionError(section)

    def get_path(self, key):
        return self.config.get('paths', key)
    
    def get_sensor_config(self):
        return {
            'sensor_end': self.config.getint('sensor_selected', 'sensor_end'),
        }

    def get_transformer_config(self):
        return {
            'T': self.config.getint('transformer', 'T'),
            'batch_size': self.config.getint('transformer', 'batch_size'),
            'learning_rate': self.config.getfloat('transformer', 'learning_rate')
        }

    def get_train_config(self):
        return {
            'epochs': self.config.getint('train', 'epochs'),
            'validation_split': self.config.getfloat('train', 'validation_split'),
            'patience': self.config.getint('train', 'patience'),
            'min_delta': self.config.getfloat('train', 'min_delta')
        }
    
    def get_lr_scheduler_config(self):
        return {
            'monitor': self.config.get('lr_scheduler', 'monitor'),
            'factor': self.config.getfloat('lr_scheduler', 'factor'),
            'patience': self.config.getint('lr_scheduler', 'patience'),
            'min_lr': self.config.getfloat('lr_scheduler', 'min_lr'),
            'mode': self.config.get('lr_scheduler', 'mode')
        }
