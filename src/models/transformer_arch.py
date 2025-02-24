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
# Positional Encoding
##############################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, T, d_model)
        # pe shape: (1, max_len, d_model)
        return x + self.pe[:, :x.size(1)]

##############################
# Modelo Transformer Multihead
##############################
class TransformerMultihead(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, T,
                 output_size=1):
        super(TransformerMultihead, self).__init__()
        self.T = T
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=T)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x: (batch_size, T, input_size)
        returns: (batch_size, T, output_size)
        """
        # Project input
        x = self.input_projection(x)         # (batch_size, T, d_model)
        # Add positional encoding
        x = self.positional_encoding(x)      # (batch_size, T, d_model)
        # Transformer expects shape (T, batch_size, d_model)
        x = x.transpose(0, 1)               # (T, batch_size, d_model)
        x = self.transformer_encoder(x)     # (T, batch_size, d_model)
        # Switch back
        x = x.transpose(0, 1)               # (batch_size, T, d_model)
        # Last timestep
        x_last = x[:, -1, :]                # (batch_size, d_model)
        x_last = self.bn(x_last)
        out = self.output_layer(x_last)     # (batch_size, output_size)
        # replicate across all timesteps if you are reconstructing
        out = out.unsqueeze(1).repeat(1, self.T, 1)  # (batch_size, T, output_size)
        return out