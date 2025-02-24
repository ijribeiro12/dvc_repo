# src/main.py

import os
import glob
import socket
import traceback

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from src.config.config_manager import ConfigManager
from src.config.mnode_config import setup_ddp, cleanup_ddp
from src.data.dataloader import DataPreprocessor, select_sensors,TimeSeriesDataset
from src.models.transformer_arch import TransformerMultihead
from src.models.train import train_model_ddp

def main_ddp():
    try:
        # Setup Distributed Data Parallel (DDP)
        device, rank, world_size, device_id = setup_ddp()
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is not available. DDP requires GPUs.")

        # Prevent running on a login node
        computer_name = socket.gethostname()
        if "login" in computer_name:
            raise ValueError(f"Running on a login node ({computer_name}) is not allowed. Use a compute node.")

        # Load configuration and initialize the preprocessor
        config_file_path = '/home/igor.jose/igor_doutorado/Codes/config.ini'
        config = ConfigManager(config_file=config_file_path)
        preprocessor = DataPreprocessor()

        # Gather baseline and damaged CSV file paths
        baseline_files = glob.glob(
            os.path.join(config.get_path('base_pasta'), '**', '*_predat_sg.csv'),
            recursive=True
        )
        if len(baseline_files) < 4:
            if rank == 0:
                raise ValueError(f"Not enough baseline files in {config.get_path('base_pasta')}. Need >= 4.")
        baseline_files = baseline_files[:5]
        if rank == 0:
            print(f"Baseline files (limited to 5): {baseline_files}")

        damaged_files = glob.glob(
            os.path.join(config.get_path('damaged_pasta'), '**', '*_predat_sg.csv'),
            recursive=True
        )
        if not damaged_files:
            if rank == 0:
                raise ValueError(f"No damaged files in {config.get_path('damaged_pasta')}.")
        damaged_files = damaged_files[:5]
        if rank == 0:
            print(f"Damaged files (limited to 5): {damaged_files}")

        # Determine available sensors from the first baseline file
        first_baseline_file = baseline_files[0]
        df = preprocessor.carregar_dataframe(first_baseline_file)
        df = preprocessor.processar_dataframe(df)
        sensor_end = config.get_sensor_end()
        sensors = select_sensors(df, sensor_end)
        if not sensors and rank == 0:
            raise ValueError("No sensors matching the criteria found.")

        # Loop over each sensor for training
        for sensor in sensors:
            if rank == 0:
                print(f"Processing sensor: {sensor}")

            # Train on Baseline Files
            for i, file in enumerate(baseline_files):
                df = preprocessor.carregar_dataframe(file)
                df = preprocessor.processar_dataframe(df)
                if sensor not in df.columns:
                    if rank == 0:
                        print(f"Sensor {sensor} not found in file: {file}")
                    continue

                series_data = df[sensor].values
                transformer_config = config.get_transformer_config()
                T = transformer_config['T']
                dataset = TimeSeriesDataset(series_data, T)

                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=transformer_config['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    sampler=train_sampler,
                    pin_memory=True,
                    drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=transformer_config['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    sampler=val_sampler,
                    pin_memory=True,
                    drop_last=True
                )

                baseline_weights_path = os.path.join(
                    config.get_path('base_output'),
                    f"{sensor}_baseline_file_{i+1}_weights_mse_ssim.pth"
                )
                if os.path.exists(baseline_weights_path):
                    if rank == 0:
                        print(f"Baseline weights exist for sensor {sensor}, file {i+1}, skipping.")
                    continue

                if rank == 0:
                    print(f"Training baseline model for sensor {sensor} on file #{i+1}")

                model = TransformerMultihead(
                    input_size=1,
                    d_model=256,
                    nhead=4,
                    num_encoder_layers=1,
                    dim_feedforward=128,
                    T=T,
                    output_size=1
                ).to(device)

                train_model_ddp(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    train_sampler=train_sampler,
                    epochs=config.get_train_config()['epochs'],
                    learning_rate=transformer_config['learning_rate'],
                    patience=config.get_train_config()['patience'],
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    device_id=device_id,
                    save_path=baseline_weights_path,
                    sensor=sensor,
                    min_delta=config.get_train_config()['min_delta']
                )

            # Train on Damaged Files
            for damaged_file in damaged_files:
                df = preprocessor.carregar_dataframe(damaged_file)
                df = preprocessor.processar_dataframe(df)
                if sensor not in df.columns:
                    if rank == 0:
                        print(f"Sensor {sensor} not found in damaged file: {damaged_file}")
                    continue

                series_data = df[sensor].values
                transformer_config = config.get_transformer_config()
                T = transformer_config['T']
                dataset = TimeSeriesDataset(series_data, T)

                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=transformer_config['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    sampler=train_sampler,
                    pin_memory=True,
                    drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=transformer_config['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    sampler=val_sampler,
                    pin_memory=True,
                    drop_last=True
                )

                damaged_weights_path = os.path.join(
                    config.get_path('damaged_output'),
                    f"{sensor}_damaged_weights_mse_ssim{os.path.basename(damaged_file)}.pth"
                )
                if os.path.exists(damaged_weights_path):
                    if rank == 0:
                        print(f"Damaged weights exist for sensor {sensor} in file {damaged_file}, skipping.")
                    continue

                if rank == 0:
                    print(f"Training damaged model for sensor {sensor} on file {damaged_file}")

                model = TransformerMultihead(
                    input_size=1,
                    d_model=256,
                    nhead=4,
                    num_encoder_layers=1,
                    dim_feedforward=128,
                    T=T,
                    output_size=1
                ).to(device)

                train_model_ddp(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    train_sampler=train_sampler,
                    epochs=config.get_train_config()['epochs'],
                    learning_rate=transformer_config['learning_rate'],
                    patience=config.get_train_config()['patience'],
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    device_id=device_id,
                    save_path=damaged_weights_path,
                    sensor=sensor,
                    min_delta=config.get_train_config()['min_delta']
                )
    except Exception as e:
        if rank == 0:
            print(f"Error in main_ddp: {e}")
            traceback.print_exc()
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main_ddp()
