# Capacitance Prediction Model Training

This directory contains a complete training pipeline for predicting capacitance values (Cgd[1,2], Cgd[1,3], Cgd[0,2]) from single-channel images using a MobileNetV3-based neural network with uncertainty estimation.

## Overview

- **Model**: MobileNetV3-small backbone with custom prediction heads
- **Input**: Single-channel 128×128 images  
- **Output**: 3 capacitance values + uncertainty estimates
- **Dataset**: ~200K samples from NPY batch files
- **Loss**: Combined MSE + Negative Log-Likelihood for uncertainty

## Files

- `CapacitancePrediction.py` - Model architecture and loss function
- `dataset.py` - PyTorch dataset class for loading NPY batch files  
- `train_capacitance_model.py` - Main training script with Wandb integration
- `requirements.txt` - Required Python packages
- `README.md` - This documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Training

```bash
# Quick test run (1 epoch, small batch)
python train_capacitance_model.py \
    --epochs 1 \
    --batch_size 8 \
    --no_wandb \
    --output_dir ./test_outputs

# Full training with Wandb logging
python train_capacitance_model.py \
    --epochs 50 \
    --batch_size 32 \
    --wandb_project "capacitance-prediction" \
    --wandb_run_name "mobilenetv3_baseline" \
    --output_dir ./outputs
```

### 3. Training with Memory Loading (faster, requires ~11GB RAM)

```bash
python train_capacitance_model.py \
    --epochs 50 \
    --batch_size 64 \
    --load_to_memory \
    --wandb_project "capacitance-prediction" \
    --output_dir ./outputs
```

## Command Line Arguments

### Data & I/O
- `--data_dir`: Path to dataset directory (default: example_dataset path)
- `--output_dir`: Directory to save checkpoints (default: ./outputs)
- `--load_to_memory`: Load all data to memory for faster training (~11GB RAM)

### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--val_split`: Validation split fraction (default: 0.2)
- `--num_workers`: Data loading workers (default: 4)

### Loss Weights
- `--mse_weight`: Weight for MSE loss component (default: 1.0)
- `--nll_weight`: Weight for NLL uncertainty loss (default: 0.1 for balanced scales)

### Logging & Checkpoints
- `--save_freq`: Save checkpoint every N epochs (default: 10)
- `--wandb_project`: Wandb project name (default: capacitance-prediction)
- `--wandb_run_name`: Wandb run name (default: auto-generated)
- `--no_wandb`: Disable Wandb logging

## Dataset Structure

The training script expects the following dataset structure:

```
example_dataset/
├── images/           # Image batch files
│   ├── batch_000271.npy  (500 samples × 128×128)
│   ├── batch_000272.npy
│   └── ...
├── parameters/       # Parameter batch files  
│   ├── batch_000271.npy  (500 parameter dicts)
│   ├── batch_000272.npy
│   └── ...
└── metadata/         # Dataset metadata
    └── dataset_info.json
```

Each image batch contains 500 single-channel 128×128 images.
Each parameter batch contains 500 dictionaries with capacitance matrices in `['model_params']['Cgd']`.

## Model Architecture

### Backbone
- **MobileNetV3-small** pretrained on ImageNet
- Modified first conv layer: 3→1 channel input
- Classification head replaced with custom heads

### Prediction Heads
- **Value Head**: 576 → 256 → 128 → 3 (capacitance values)
- **Confidence Head**: 576 → 256 → 128 → 3 (log variances for uncertainty)

### Loss Function
Combined loss with two components:
1. **MSE Loss**: Standard regression loss on predicted values
2. **NLL Loss**: Negative log-likelihood incorporating predicted uncertainty

```
Total Loss = MSE_weight × MSE + NLL_weight × NLL
NLL = 0.5 × (log(2π) + log_var + (target - pred)² / exp(log_var))
```

## Training Metrics

The training script logs comprehensive metrics:

### Per-Epoch Metrics
- `train_loss`, `train_mse`, `train_nll` - Training losses
- `val_loss`, `val_mse`, `val_nll` - Validation losses  
- `mae` - Overall mean absolute error
- `Cgd_1_2_mae`, `Cgd_1_3_mae`, `Cgd_0_2_mae` - Per-target MAE
- `mean_uncertainty`, `std_uncertainty` - Uncertainty statistics

### Wandb Integration
- Real-time metric tracking
- Model parameter monitoring (`wandb.watch`)
- Checkpoint artifact saving
- Hyperparameter logging

## Performance Considerations

### Memory Usage
- **Disk Loading**: ~2GB GPU memory, slower data loading
- **Memory Loading**: ~11GB RAM + ~4GB GPU memory, faster training
- **Batch Size**: Start with 32, increase based on GPU memory

### Training Speed
- With disk loading: ~15 batches/sec on RTX 4000 Ada
- With memory loading: ~25-30 batches/sec (estimated)
- Full epoch: ~20-25 minutes (disk) / ~12-15 minutes (memory)

### Recommended Settings

**Development/Testing:**
```bash
--epochs 5 --batch_size 16 --no_wandb
```

**Production Training:**
```bash
--epochs 50 --batch_size 32 --load_to_memory --wandb_project "production-run"
```

## Model Outputs

### Training Artifacts
- `best_model.pth` - Best model checkpoint (based on validation loss)
- `checkpoint_epoch_N.pth` - Regular epoch checkpoints
- Wandb artifacts - Model checkpoints and metrics

### Predictions
The trained model outputs:
- **Values**: 3 predicted capacitance values [Cgd_1_2, Cgd_1_3, Cgd_0_2]
- **Uncertainties**: 3 uncertainty estimates (standard deviations)

```python
# Load and use trained model
model = create_model()
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

values, log_vars = model(input_images)
uncertainties = torch.exp(0.5 * log_vars)  # Convert to std dev
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `--batch_size` to 16 or 8
   - Ensure other GPU processes are stopped

2. **Slow data loading**
   - Use `--load_to_memory` if you have sufficient RAM
   - Increase `--num_workers` (but watch CPU usage)

3. **Wandb connection issues**
   - Use `--no_wandb` for offline training
   - Check internet connection and wandb login

4. **Model not converging**
   - Check data normalization (enabled by default)
   - Adjust loss weights: `--mse_weight 1.0 --nll_weight 0.05` (reduce NLL weight)
   - Reduce learning rate: `--lr 1e-4`

### Hardware Requirements

**Minimum:**
- 8GB GPU memory
- 16GB system RAM  
- 4 CPU cores

**Recommended:**
- 16GB+ GPU memory
- 32GB+ system RAM (for memory loading)
- 8+ CPU cores

## Dataset Statistics

- **Total Samples**: ~200,500
- **Train/Val Split**: 80%/20%
- **Image Size**: 128×128 single-channel
- **Value Range**: Images [0.25, 1.14], Targets ~[0.0, 1.0]
- **Storage**: ~11.5GB total dataset size 