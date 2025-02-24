# src/train.py

import os
import socket
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from src.metrics.ssim import ssim_1d

def train_model_ddp(model,
                    train_loader,
                    val_loader,
                    train_sampler,
                    epochs,
                    learning_rate,
                    patience,
                    device,
                    rank,
                    world_size,
                    device_id,
                    save_path,
                    sensor,
                    min_delta=0.0):
    """
    Distributed Data Parallel (DDP) training loop using MSE loss for optimization and SSIM as
    the validation metric for saving the best model and early stopping.

    Args:
        model (nn.Module): The Transformer model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        train_sampler (DistributedSampler): Sampler for the training dataset.
        epochs (int): Maximum number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs with no improvement to wait before stopping.
        device (torch.device): Device to run training on.
        rank (int): Process rank.
        world_size (int): Total number of processes.
        device_id (int): GPU device id.
        save_path (str): Path to save the best model weights.
        sensor (str): Sensor identifier used for logging.
        min_delta (float, optional): Minimum change to qualify as an improvement. Defaults to 0.0.
    """
    # Convert model to use synchronized BatchNorm and wrap with DDP
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_criterion = nn.MSELoss()

    best_val_ssim = -float('inf')
    epochs_no_improve = 0
    early_stop = False

    writer = None
    if rank == 0:
        log_dir = os.path.join(
            "/home/igor.jose/tensorboard_training/runs",
            f"{datetime.now().strftime('%Y-%m-%d')}_transformer_{sensor}"
        )
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("Hyperparameters",
                        f"Loss=MSE, Save=SSIM, Sensor={sensor}, LR={learning_rate}, "
                        f"Patience={patience}, Epochs={epochs}, MinDelta={min_delta}")

    try:
        for epoch in range(epochs):
            # Ensure proper shuffling for distributed training
            train_sampler.set_epoch(epoch)
            model.train()
            
            train_mse = 0.0
            for X, Y in train_loader:
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                optimizer.zero_grad()
                output = model(X)
                loss = mse_criterion(output, Y)
                loss.backward()
                
                # Log gradients (only rank 0 logs)
                if writer is not None:
                    target_model = model.module if hasattr(model, "module") else model
                    for name, param in target_model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f"{name}_{sensor}_grad", param.grad, epoch)

                optimizer.step()
                train_mse += loss.item()

            # Average training MSE across processes
            train_mse_tensor = torch.tensor(train_mse / len(train_loader), device=device)
            dist.all_reduce(train_mse_tensor, op=dist.ReduceOp.SUM)
            train_mse_avg = train_mse_tensor.item() / world_size

            # Validation phase
            model.eval()
            val_mse = 0.0
            val_ssim = 0.0
            with torch.no_grad():
                for X, Y in val_loader:
                    X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                    output = model(X)
                    mse_val = mse_criterion(output, Y)
                    val_mse += mse_val.item()
                    ssim_val = ssim_1d(output, Y)
                    val_ssim += ssim_val.item()

            val_mse_tensor = torch.tensor(val_mse / len(val_loader), device=device)
            val_ssim_tensor = torch.tensor(val_ssim / len(val_loader), device=device)
            dist.all_reduce(val_mse_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_ssim_tensor, op=dist.ReduceOp.SUM)
            val_mse_avg = val_mse_tensor.item() / world_size
            val_ssim_avg = val_ssim_tensor.item() / world_size

            # Logging and early stopping checks (only rank 0)
            if rank == 0:
                print(
                    f"[Epoch {epoch+1:03d} - Sensor {sensor}] "
                    f"Train MSE: {train_mse_avg:.6f} | "
                    f"Val MSE: {val_mse_avg:.6f} | "
                    f"Val SSIM: {val_ssim_avg:.6f}"
                )
                writer.add_scalar(f"MSE/Train_{sensor}", train_mse_avg, epoch)
                writer.add_scalar(f"MSE/Val_{sensor}", val_mse_avg, epoch)
                writer.add_scalar(f"SSIM/Val_{sensor}", val_ssim_avg, epoch)
                
                target_model = model.module if hasattr(model, "module") else model
                for name, param in target_model.named_parameters():
                    writer.add_histogram(f"{name}_{sensor}", param, epoch)
                
                if (val_ssim_avg - best_val_ssim) > min_delta:
                    best_val_ssim = val_ssim_avg
                    epochs_no_improve = 0
                    torch.save(model.module.state_dict(), save_path)
                    print(f"  >> Saved new best model [SSIM={val_ssim_avg:.6f}] at epoch {epoch+1}.")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("Early stopping triggered (SSIM not improving).")
                        early_stop = True
            else:
                early_stop = False

            # Broadcast early_stop flag to all processes
            dist.barrier()
            early_stop_tensor = torch.tensor([1 if early_stop else 0], dtype=torch.int, device=device)
            dist.broadcast(early_stop_tensor, src=0)
            early_stop = bool(early_stop_tensor.item())

            if early_stop:
                if rank == 0:
                    print(f"Early stopping at epoch {epoch+1} for sensor {sensor}!")
                break

            dist.barrier()

    except Exception as e:
        if rank == 0:
            print(f"Rank {rank} - Sensor {sensor}: Exception during training: {e}")
            traceback.print_exc()
        raise e
    finally:
        if writer is not None:
            writer.close()
        dist.barrier()
