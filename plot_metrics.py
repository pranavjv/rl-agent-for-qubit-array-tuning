#!/usr/bin/env python3
"""
Script to plot training metrics from JSONL file.
Generates plots of episode/score and episode/length against training step.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_metrics_file(file_path):
    """Parse the JSONL metrics file and extract relevant data."""
    steps = []
    scores = []
    lengths = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'step' in data:
                    steps.append(data['step'])
                    # Extract episode/score if available
                    if 'episode/score' in data:
                        scores.append(data['episode/score'])
                    else:
                        scores.append(None)
                    # Extract episode/length if available
                    if 'episode/length' in data:
                        lengths.append(data['episode/length'])
                    else:
                        lengths.append(None)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")
                continue
    
    return steps, scores, lengths

def calculate_rolling_average(values, window_size=50):
    """Calculate rolling average with specified window size."""
    if len(values) < window_size:
        return values, list(range(len(values)))
    
    rolling_avg = []
    rolling_indices = []
    
    for i in range(window_size - 1, len(values)):
        window = values[i - window_size + 1:i + 1]
        rolling_avg.append(np.mean(window))
        rolling_indices.append(i)
    
    return rolling_avg, rolling_indices

def create_plots(steps, scores, lengths, output_dir="plots"):
    """Create and save the plots."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Filter out None values
    valid_data = [(s, sc, l) for s, sc, l in zip(steps, scores, lengths) 
                  if sc is not None or l is not None]
    
    if not valid_data:
        print("No valid data found!")
        return
    
    valid_steps, valid_scores, valid_lengths = zip(*valid_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Episode Score vs Training Step
    if any(sc is not None for sc in valid_scores):
        score_steps = [s for s, sc, l in valid_data if sc is not None]
        score_values = [sc for s, sc, l in valid_data if sc is not None]
        
        # Plot raw data
        ax1.plot(score_steps, score_values, 'b-', alpha=0.3, linewidth=0.5, label='Raw Data')
        ax1.scatter(score_steps, score_values, c='blue', s=10, alpha=0.4)
        
        # Calculate and plot rolling averages with different window sizes
        if len(score_values) >= 20:
            # Short-term trend (20 episodes)
            rolling_avg_20, rolling_indices_20 = calculate_rolling_average(score_values, 20)
            if rolling_avg_20:
                rolling_steps_20 = [score_steps[i] for i in rolling_indices_20]
                ax1.plot(rolling_steps_20, rolling_avg_20, 'orange', linewidth=2, 
                        label='Rolling Avg (20 episodes)', alpha=0.8)
        
        if len(score_values) >= 100:
            # Medium-term trend (100 episodes)
            rolling_avg_100, rolling_indices_100 = calculate_rolling_average(score_values, 100)
            if rolling_avg_100:
                rolling_steps_100 = [score_steps[i] for i in rolling_indices_100]
                ax1.plot(rolling_steps_100, rolling_avg_100, 'red', linewidth=3, 
                        label='Rolling Avg (100 episodes)', alpha=0.9)
        
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Episode Score')
        ax1.set_title('Episode Score vs Training Step (with Rolling Averages)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Episode Length vs Training Step
    if any(l is not None for l in valid_lengths):
        length_steps = [s for s, sc, l in valid_data if l is not None]
        length_values = [l for s, sc, l in valid_data if l is not None]
        
        # Plot raw data
        ax2.plot(length_steps, length_values, 'g-', alpha=0.3, linewidth=0.5, label='Raw Data')
        ax2.scatter(length_steps, length_values, c='green', s=10, alpha=0.4)
        
        # Calculate and plot rolling averages with different window sizes
        if len(length_values) >= 20:
            # Short-term trend (20 episodes)
            rolling_avg_20, rolling_indices_20 = calculate_rolling_average(length_values, 20)
            if rolling_avg_20:
                rolling_steps_20 = [length_steps[i] for i in rolling_indices_20]
                ax2.plot(rolling_steps_20, rolling_avg_20, 'orange', linewidth=2, 
                        label='Rolling Avg (20 episodes)', alpha=0.8)
        
        if len(length_values) >= 100:
            # Medium-term trend (100 episodes)
            rolling_avg_100, rolling_indices_100 = calculate_rolling_average(length_values, 100)
            if rolling_avg_100:
                rolling_steps_100 = [length_steps[i] for i in rolling_indices_100]
                ax2.plot(rolling_steps_100, rolling_avg_100, 'red', linewidth=3, 
                        label='Rolling Avg (100 episodes)', alpha=0.9)
        
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length vs Training Step (with Rolling Averages)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / "training_metrics_smoothed.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Smoothed plots saved to: {plot_path}")
    
    # Also save individual plots with rolling averages
    if any(sc is not None for sc in valid_scores):
        plt.figure(figsize=(12, 8))
        
        # Plot raw data
        plt.plot(score_steps, score_values, 'b-', alpha=0.3, linewidth=0.5, label='Raw Data')
        plt.scatter(score_steps, score_values, c='blue', s=10, alpha=0.4)
        
        # Add rolling averages
        if len(score_values) >= 20:
            rolling_avg_20, rolling_indices_20 = calculate_rolling_average(score_values, 20)
            if rolling_avg_20:
                rolling_steps_20 = [score_steps[i] for i in rolling_indices_20]
                plt.plot(rolling_steps_20, rolling_avg_20, 'orange', linewidth=2, 
                        label='Rolling Avg (20 episodes)', alpha=0.8)
        
        if len(score_values) >= 100:
            rolling_avg_100, rolling_indices_100 = calculate_rolling_average(score_values, 100)
            if rolling_avg_100:
                rolling_steps_100 = [score_steps[i] for i in rolling_indices_100]
                plt.plot(rolling_steps_100, rolling_avg_100, 'red', linewidth=3, 
                        label='Rolling Avg (100 episodes)', alpha=0.9)
        
        plt.xlabel('Training Step')
        plt.ylabel('Episode Score')
        plt.title('Episode Score vs Training Step (with Rolling Averages)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        score_plot_path = Path(output_dir) / "episode_score_smoothed.png"
        plt.savefig(score_plot_path, dpi=300, bbox_inches='tight')
        print(f"Smoothed score plot saved to: {score_plot_path}")
        plt.close()
    
    if any(l is not None for l in valid_lengths):
        plt.figure(figsize=(12, 8))
        
        # Plot raw data
        plt.plot(length_steps, length_values, 'g-', alpha=0.3, linewidth=0.5, label='Raw Data')
        plt.scatter(length_steps, length_values, c='green', s=10, alpha=0.4)
        
        # Add rolling averages
        if len(length_values) >= 20:
            rolling_avg_20, rolling_indices_20 = calculate_rolling_average(length_values, 20)
            if rolling_avg_20:
                rolling_steps_20 = [length_steps[i] for i in rolling_indices_20]
                plt.plot(rolling_steps_20, rolling_avg_20, 'orange', linewidth=2, 
                        label='Rolling Avg (20 episodes)', alpha=0.8)
        
        if len(length_values) >= 100:
            rolling_avg_100, rolling_indices_100 = calculate_rolling_average(length_values, 100)
            if rolling_avg_100:
                rolling_steps_100 = [length_steps[i] for i in rolling_indices_100]
                plt.plot(rolling_steps_100, rolling_avg_100, 'red', linewidth=3, 
                        label='Rolling Avg (100 episodes)', alpha=0.9)
        
        plt.xlabel('Training Step')
        plt.ylabel('Episode Length')
        plt.title('Episode Length vs Training Step (with Rolling Averages)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        length_plot_path = Path(output_dir) / "episode_length_smoothed.png"
        plt.savefig(length_plot_path, dpi=300, bbox_inches='tight')
        print(f"Smoothed length plot saved to: {length_plot_path}")
        plt.close()
    
    plt.close()

def main():
    """Main function to run the plotting script."""
    # Path to the metrics file
    metrics_file = "/home/rahul/logdir/20250725T195221/metrics.jsonl"
    
    print(f"Parsing metrics file: {metrics_file}")
    
    # Parse the file
    steps, scores, lengths = parse_metrics_file(metrics_file)
    
    print(f"Parsed {len(steps)} data points")
    print(f"Found {sum(1 for s in scores if s is not None)} score entries")
    print(f"Found {sum(1 for l in lengths if l is not None)} length entries")
    
    # Create plots
    create_plots(steps, scores, lengths)
    
    # Print some statistics
    valid_scores = [s for s in scores if s is not None]
    valid_lengths = [l for l in lengths if l is not None]
    
    if valid_scores:
        print(f"\nScore Statistics:")
        print(f"  Min: {min(valid_scores):.2f}")
        print(f"  Max: {max(valid_scores):.2f}")
        print(f"  Mean: {np.mean(valid_scores):.2f}")
        print(f"  Std: {np.std(valid_scores):.2f}")
    
    if valid_lengths:
        print(f"\nLength Statistics:")
        print(f"  Min: {min(valid_lengths):.2f}")
        print(f"  Max: {max(valid_lengths):.2f}")
        print(f"  Mean: {np.mean(valid_lengths):.2f}")
        print(f"  Std: {np.std(valid_lengths):.2f}")

if __name__ == "__main__":
    main() 