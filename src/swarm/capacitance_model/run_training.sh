#!/bin/bash

# Capacitance Prediction Model Training Script Examples
# Choose one of the commands below based on your needs

echo "=== Capacitance Prediction Model Training ==="
echo "Choose your training configuration:"
echo ""

# Test run - quick verification that everything works
echo "1. TEST RUN (1 epoch, no Wandb):"
echo "python train_capacitance_model.py --epochs 1 --batch_size 8 --no_wandb --output_dir ./test_outputs"
echo ""

# Development run - short training for development
echo "2. DEVELOPMENT RUN (5 epochs, no Wandb):"
echo "python train_capacitance_model.py --epochs 5 --batch_size 16 --no_wandb --output_dir ./dev_outputs"
echo ""

# Production run - full training with Wandb
echo "3. PRODUCTION RUN (50 epochs, with Wandb):"
echo "python train_capacitance_model.py --epochs 50 --batch_size 32 --wandb_project 'capacitance-prediction' --wandb_run_name 'baseline_run' --output_dir ./outputs"
echo ""

# Fast training - using memory loading for speed
echo "4. FAST TRAINING (with memory loading, requires ~11GB RAM):"
echo "python train_capacitance_model.py --epochs 50 --batch_size 64 --load_to_memory --wandb_project 'capacitance-prediction' --wandb_run_name 'fast_training' --output_dir ./fast_outputs"
echo ""

# Custom hyperparameters
echo "5. CUSTOM HYPERPARAMETERS:"
echo "python train_capacitance_model.py --epochs 30 --batch_size 32 --lr 5e-4 --mse_weight 1.0 --nll_weight 0.5 --wandb_project 'capacitance-prediction' --wandb_run_name 'custom_hp' --output_dir ./custom_outputs"
echo ""

echo "=== Usage Instructions ==="
echo "1. First install dependencies: pip install -r requirements.txt"
echo "2. Choose a command above and run it"
echo "3. Monitor training progress in terminal or Wandb dashboard"
echo "4. Find best model in the output directory as 'best_model.pth'"
echo ""

echo "=== Hardware Requirements ==="
echo "- Minimum: 8GB GPU memory, 16GB RAM"
echo "- Recommended: 16GB+ GPU memory, 32GB+ RAM"
echo "- With --load_to_memory: Additional 11GB RAM required"
echo ""

echo "=== Quick Wandb Setup ==="
echo "1. Create account at https://wandb.ai"
echo "2. Run: wandb login"
echo "3. Use your training command with --wandb_project"
echo ""

# Uncomment one of the commands below to run automatically:

# Test run
# python train_capacitance_model.py --epochs 1 --batch_size 8 --no_wandb --output_dir ./test_outputs

# Development run  
# python train_capacitance_model.py --epochs 5 --batch_size 16 --no_wandb --output_dir ./dev_outputs

# Production run
# python train_capacitance_model.py --epochs 50 --batch_size 32 --wandb_project "capacitance-prediction" --wandb_run_name "baseline_run" --output_dir ./outputs 