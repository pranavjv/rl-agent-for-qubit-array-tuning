#!/usr/bin/env python3
"""
Training script for capacitance prediction model.

Trains a MobileNetV3-based model to predict Cgd[1,2], Cgd[1,3], and Cgd[0,2]
from single-channel capacitance images with uncertainty estimation.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

# Local imports
from CapacitancePrediction import create_model, create_loss_function
from dataset import create_data_loaders, get_transforms


def setup_device() -> torch.device:
    """Setup GPU device if available"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def calculate_metrics(predictions: Tuple[torch.Tensor, torch.Tensor], 
                     targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate detailed metrics for evaluation.
    
    Args:
        predictions: Tuple of (values, log_vars) from model
        targets: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    values, log_vars = predictions
    values = values.detach().cpu()
    targets = targets.detach().cpu()
    log_vars = log_vars.detach().cpu()
    
    # Overall metrics
    mse = nn.functional.mse_loss(values, targets).item()
    mae = nn.functional.l1_loss(values, targets).item()
    
    # Per-target metrics
    per_target_mse = []
    per_target_mae = []
    target_names = ['Cgd_1_2', 'Cgd_1_3', 'Cgd_0_2']
    
    metrics = {'mse': mse, 'mae': mae}
    
    for i, name in enumerate(target_names):
        target_mse = nn.functional.mse_loss(values[:, i], targets[:, i]).item()
        target_mae = nn.functional.l1_loss(values[:, i], targets[:, i]).item()
        
        metrics[f'{name}_mse'] = target_mse
        metrics[f'{name}_mae'] = target_mae
        
        per_target_mse.append(target_mse)
        per_target_mae.append(target_mae)
    
    # Uncertainty metrics
    uncertainties = torch.exp(0.5 * log_vars)
    metrics['mean_uncertainty'] = uncertainties.mean().item()
    metrics['std_uncertainty'] = uncertainties.std().item()
    
    return metrics


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                loss_fn: nn.Module,
                device: torch.device,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_mse = 0.0
    total_nll = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Compute loss
        total_loss_batch, mse_loss, nll_loss = loss_fn(predictions, targets)
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_mse += mse_loss.item()
        total_nll += nll_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'MSE': f'{mse_loss.item():.4f}',
            'NLL': f'{nll_loss.item():.4f}'
        })
    
    return {
        'train_loss': total_loss / num_batches,
        'train_mse': total_mse / num_batches,
        'train_nll': total_nll / num_batches
    }


def validate_epoch(model: nn.Module, 
                   val_loader: DataLoader, 
                   loss_fn: nn.Module,
                   device: torch.device) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_nll = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            total_loss_batch, mse_loss, nll_loss = loss_fn(predictions, targets)
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_nll += nll_loss.item()
            
            # Store for detailed metrics
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    num_batches = len(val_loader)
    
    # Calculate detailed metrics on all validation data
    all_pred_values = torch.cat([pred[0] for pred in all_predictions])
    all_pred_logvars = torch.cat([pred[1] for pred in all_predictions])
    all_targ = torch.cat(all_targets)
    
    detailed_metrics = calculate_metrics((all_pred_values, all_pred_logvars), all_targ)
    
    # Base validation metrics
    val_metrics = {
        'val_loss': total_loss / num_batches,
        'val_mse': total_mse / num_batches,
        'val_nll': total_nll / num_batches
    }
    
    # Add detailed metrics
    val_metrics.update(detailed_metrics)
    
    return val_metrics


def save_checkpoint(model: nn.Module, 
                   optimizer: optim.Optimizer, 
                   epoch: int, 
                   best_loss: float,
                   save_dir: Path,
                   is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = save_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Capacitance Prediction Model')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/rahul/Summer2025/rl-agent-for-qubit-array-tuning/Swarm/Qarray/example_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split fraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--load_to_memory', action='store_true',
                       help='Load all data to memory (requires ~11GB RAM)')
    parser.add_argument('--mse_weight', type=float, default=1.0,
                       help='Weight for MSE loss component')
    parser.add_argument('--nll_weight', type=float, default=0.1,
                       help='Weight for NLL loss component (default 0.1 to balance scales)')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--wandb_project', type=str, default='capacitance-prediction',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args
        )
    
    # Create data loaders
    print("Creating data loaders...")
    transform = get_transforms(normalize=True)
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        load_to_memory=args.load_to_memory,
        transform=transform
    )
    
    # Create model and loss function
    print("Creating model...")
    model = create_model()
    model = model.to(device)
    
    loss_fn = create_loss_function(
        mse_weight=args.mse_weight,
        nll_weight=args.nll_weight
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if not args.no_wandb:
        wandb.watch(model, log_freq=100)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['val_loss'])
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
              f"MSE: {train_metrics['train_mse']:.4f}, "
              f"NLL: {train_metrics['train_nll']:.4f}")
        print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
              f"MSE: {val_metrics['val_mse']:.4f}, "
              f"MAE: {val_metrics['mae']:.4f}")
        print(f"Val Targets - Cgd_1_2: {val_metrics['Cgd_1_2_mae']:.4f}, "
              f"Cgd_1_3: {val_metrics['Cgd_1_3_mae']:.4f}, "
              f"Cgd_0_2: {val_metrics['Cgd_0_2_mae']:.4f}")
        
        # Log to wandb
        if not args.no_wandb:
            log_dict = {**train_metrics, **val_metrics}
            log_dict['epoch'] = epoch
            log_dict['lr'] = optimizer.param_groups[0]['lr']
            log_dict['epoch_time'] = epoch_time
            wandb.log(log_dict)
        
        # Save checkpoints
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
        
        if epoch % args.save_freq == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, 
                output_dir, is_best
            )
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 