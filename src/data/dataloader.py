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
############################################
# Data Preprocessor
############################################
class DataPreprocessor:
    """Handles data loading, preprocessing, and normalization."""
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def carregar_dataframe(self, caminho_arquivo):
        """Loads a CSV file into a DataFrame and cleans column names."""
        try:
            df = pd.read_csv(caminho_arquivo)
            df.columns = df.columns.str.replace("'", "").str.replace(" ", "")
            return df
        except Exception as e:
            print(f"Erro ao carregar {caminho_arquivo}: {str(e)}")
            return None

    def processar_dataframe(self, df):
        """Selects columns ending with 'A' and drops NaNs."""
        novo_df = pd.DataFrame()
        for chave_interna in df.columns:
            if chave_interna.endswith('A'):
                novo_df[chave_interna] = df[chave_interna]
        novo_df = novo_df.dropna(axis=1)
        return novo_df

    def normalize_dataframe(self, df):
        """Normalizes a single dataframe."""
        if not df.empty and np.issubdtype(df.dtypes[0], np.number):
            normalized_data = self.scaler.fit_transform(df)
            normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
            return normalized_df
        else:
            print("Skipping normalization: DataFrame empty or contains non-numerical data")
            return pd.DataFrame()

    def normalize_dataframes(self, datasets):
        """Normalizes multiple dataframes in a dict."""
        normalized_datasets = {}
        for name, df in datasets.items():
            if not df.empty and np.issubdtype(df.dtypes[0], np.number):
                normalized_data = self.scaler.fit_transform(df)
                normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
                normalized_datasets[name] = normalized_df
            else:
                print(f"Skipping dataset {name}: empty or non-numerical")
        return normalized_datasets

    def build_dataset(self, df, T):
        """Builds a dataset for Transformer training from DataFrame."""
        X, Y = [], []
        for column in df.columns:
            for t in range(len(df[column]) - T):
                x = df[column][t:t+T].values
                y_val = df[column][t+T]
                X.append(x)
                Y.append(y_val)
        
        X, Y = np.array(X), np.array(Y)
        # Reshape X to (num_samples, T, 1)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        return X, Y, len(X)

##############################
# Dataset para Séries Temporais
##############################
class TimeSeriesDataset(Dataset):
    def __init__(self, data, T):
        self.data = data
        self.T = T

    def __len__(self):
        return len(self.data) - self.T
    
    def __getitem__(self, index):
        x = self.data[index:index + self.T]
        y = self.data[index:index + self.T]  # for reconstruction tasks
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        return x, y

##############################
# Helper para seleção de sensores
##############################
def select_sensors(df, select_sensors):
    columns = [col for col in df.columns if col.endswith(select_sensors)]
    print(f"Selected sensors: {columns}")
    return columns