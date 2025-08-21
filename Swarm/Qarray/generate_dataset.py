#!/usr/bin/env python3
"""
Example script to generate a small quantum device dataset

This script demonstrates how to use the dataset generator to create
a small dataset for testing and development purposes.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_generator import DatasetGenerator, GenerationConfig


def main():
    """Generate a small example dataset"""
    
    # Configuration for a dataset
    config = GenerationConfig(
        total_samples=500000,       # dataset size
        batch_size= 500,            # batch size
        workers=10,                 # number of workers
        output_dir='./example_dataset',
        config_path='qarray_config.yaml',
        voltage_range=(-1.0, 1.0),
        resolution=128,
        seed=42,
        resume=True
    )
    
    print("Generating example quantum device dataset...")
    print(f"Total samples: {config.total_samples}")
    print(f"Batch size: {config.batch_size}")
    print(f"Workers: {config.workers}")
    print(f"Output directory: {config.output_dir}")
    print(f"Image resolution: {config.resolution}x{config.resolution}")
    print()
    
    try:
        # Create dataset generator
        generator = DatasetGenerator(config)
        
        # Generate the dataset
        generator.generate_dataset()
        
        print("\nDataset generation completed successfully!")
        print(f"Dataset saved to: {config.output_dir}")
        
        # Show dataset structure
        output_path = Path(config.output_dir)
        print("\nDataset structure:")
        for item in sorted(output_path.rglob('*')):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(output_path)} ({size_mb:.1f} MB)")
                
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


if __name__ == '__main__':
    main() 