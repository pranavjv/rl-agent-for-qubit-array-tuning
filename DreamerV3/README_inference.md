# DreamerV3 Inference for Quantum Device Environment

This directory contains scripts for running inference with trained DreamerV3 agents on the quantum device environment and creating visualizations.

## Files

- `inference.py` - Main inference script for running trained agents
- `create_gif.py` - Script to create GIF animations from inference frames
- `test_gif_settings.py` - Test script to demonstrate different GIF settings
- `README_inference.md` - This documentation file

## Quick Start

### 1. Run Inference

```bash
# Activate the conda environment
conda activate rl_agent_py311

# Run inference with a trained checkpoint
python inference.py --checkpoint /path/to/checkpoint --episodes 5
```

### 2. Create GIFs

```bash
# Create GIFs from the generated frames
python create_gif.py --fps 1 --end-pause 2000
```

## Detailed Usage

### Inference Script (`inference.py`)

The inference script loads a trained DreamerV3 agent and runs it on the quantum device environment, saving visualization frames for each step.

#### Arguments

- `--checkpoint` (required): Path to the checkpoint directory
- `--episodes` (default: 5): Number of episodes to run
- `--max-steps` (default: 50): Maximum steps per episode
- `--output-dir` (default: 'inference_frames'): Directory to save frames
- `--config` (default: 'dreamerv3/configs.yaml'): Path to DreamerV3 config file

#### Example

```bash
python inference.py \
    --checkpoint /home/rahul/logdir/20250724T192626/ckpt/20250724T220834F902303 \
    --episodes 3 \
    --max-steps 30 \
    --output-dir my_inference_frames
```

#### Output

The script creates a directory structure like:
```
inference_frames/
├── episode_1/
│   ├── step_0001.png
│   ├── step_0002.png
│   └── ...
├── episode_2/
│   ├── step_0001.png
│   └── ...
└── ...
```

### GIF Creation Script (`create_gif.py`)

The GIF creation script converts the saved frames into animated GIFs with configurable timing and effects.

#### Arguments

- `--frames-dir` (default: 'inference_frames'): Directory containing inference frames
- `--output-dir` (default: same as frames-dir): Directory to save GIFs
- `--fps` (default: 1): Frames per second for GIFs
- `--duration` (optional): Duration per frame in milliseconds (overrides fps)
- `--end-pause` (default: 1000): Duration to pause at the end in milliseconds
- `--no-repeat-first`: Disable repeating the first frame at the end
- `--episode` (optional): Specific episode to process

#### Examples

```bash
# Create slow GIF with long end pause
python create_gif.py --fps 0.5 --end-pause 3000

# Create fast GIF with short end pause
python create_gif.py --fps 2 --end-pause 500

# Create GIF for specific episode
python create_gif.py --episode episode_1 --fps 1 --end-pause 2000

# Create GIF without repeating first frame
python create_gif.py --fps 1 --end-pause 1000 --no-repeat-first

# Use custom duration instead of fps
python create_gif.py --duration 800 --end-pause 2000
```

#### GIF Features

1. **End Pause**: The last frame is duplicated with a longer duration to create a clear pause at the end
2. **First Frame Repeat**: The first frame is added at the end to create a smooth loop
3. **Configurable Timing**: Control frame rate and pause duration
4. **Optimization**: GIFs are optimized for size while maintaining quality

### Test Script (`test_gif_settings.py`)

The test script demonstrates different GIF settings and their effects.

```bash
python test_gif_settings.py
```

This creates multiple GIFs with different settings for comparison.

## Troubleshooting

### Device Configuration Issues

If you encounter device-related errors like "index 6 is out of bounds for axis 1 with size 1", the inference script automatically handles this by:

1. Setting the platform to CPU
2. Using device index 0 instead of the training device index
3. Using float32 instead of bfloat16 for CPU compatibility

### Memory Issues

If you encounter memory issues:

1. Reduce the number of episodes: `--episodes 1`
2. Reduce max steps per episode: `--max-steps 20`
3. Use a smaller batch size in the config if needed

### Frame Generation Issues

If frames are not being generated:

1. Check that the checkpoint path is correct
2. Verify the environment is working by running a simple test
3. Check the console output for error messages

## Output Interpretation

### Episode Information

Each episode shows:
- **Steps**: Number of steps taken
- **Total reward**: Cumulative reward for the episode
- **Final step reward**: Reward of the last step
- **Termination reason**: Whether the episode reached the target or was truncated

### Frame Information

Each frame shows:
- **Step number**: Current step in the episode
- **Reward**: Reward for the current step
- **Visualization**: The quantum device state visualization

### GIF Information

Each GIF shows:
- **Total frames**: Number of frames in the GIF
- **Total duration**: Total duration of the GIF in seconds
- **Frame timing**: Individual frame durations

## Example Workflow

```bash
# 1. Run inference to generate frames
python inference.py --checkpoint /path/to/checkpoint --episodes 3

# 2. Create a slow, clear GIF
python create_gif.py --fps 0.5 --end-pause 3000

# 3. Create a faster GIF for quick viewing
python create_gif.py --fps 2 --end-pause 500

# 4. View the results
ls -la inference_frames/*.gif
```

## Notes

- The inference script automatically handles device configuration differences between training and inference environments
- GIFs are created with optimization enabled to reduce file size
- The end pause and first frame repeat features make it easier to understand when the episode ends and loops
- All scripts include detailed progress information and error handling 