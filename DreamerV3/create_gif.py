#!/usr/bin/env python3
"""
Script to create GIFs from inference frames.
"""

import os
import argparse
import glob
from pathlib import Path
from PIL import Image


def create_gif_from_frames(frames_dir, output_gif_path, fps=2, duration=None, end_pause_ms=1000, repeat_first_frame=True):
    """
    Create a GIF from PNG frames in the given directory.
    
    Args:
        frames_dir: Directory containing PNG frames
        output_gif_path: Path for the output GIF file
        fps: Frames per second for the GIF
        duration: Duration per frame in milliseconds (overrides fps if provided)
        end_pause_ms: Duration to pause at the end in milliseconds
        repeat_first_frame: Whether to repeat the first frame at the end for clear looping
    """
    # Get all PNG files and sort them
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    
    if not frame_files:
        print(f"No PNG files found in {frames_dir}")
        return
    
    print(f"Found {len(frame_files)} frames in {frames_dir}")
    
    # Calculate duration per frame
    if duration is None:
        duration = int(1000 / fps)  # Convert fps to milliseconds
    
    # Load all frames
    frames = []
    durations = []
    
    for i, frame_file in enumerate(frame_files):
        try:
            frame = Image.open(frame_file)
            # Convert to RGB if necessary
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            frames.append(frame)
            durations.append(duration)
            print(f"Loaded frame {i+1}/{len(frame_files)}: {os.path.basename(frame_file)}")
        except Exception as e:
            print(f"Warning: Could not read frame {frame_file}: {e}")
    
    if not frames:
        print("No frames could be loaded")
        return
    
    # Add end pause by duplicating the last frame with longer duration
    if end_pause_ms > 0:
        frames.append(frames[-1].copy())  # Duplicate last frame
        durations.append(end_pause_ms)
        print(f"Added end pause frame with {end_pause_ms}ms duration")
    
    # Add first frame at the end for clear looping if requested
    if repeat_first_frame and len(frames) > 1:
        frames.append(frames[0].copy())  # Add first frame at the end
        durations.append(duration)
        print(f"Added first frame at the end for clear looping")
    
    # Save as GIF
    try:
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,  # Loop forever
            optimize=True
        )
        print(f"GIF saved to: {output_gif_path}")
        print(f"Total frames: {len(frames)}, Total duration: {sum(durations)/1000:.1f}s")
    except Exception as e:
        print(f"Error saving GIF: {e}")


def create_episode_gifs(inference_dir, output_dir=None, fps=2, duration=None, end_pause_ms=1000, repeat_first_frame=True):
    """
    Create GIFs for all episodes in the inference directory.
    
    Args:
        inference_dir: Directory containing episode folders
        output_dir: Directory to save GIFs (defaults to inference_dir)
        fps: Frames per second for the GIFs
        duration: Duration per frame in milliseconds (overrides fps if provided)
        end_pause_ms: Duration to pause at the end in milliseconds
        repeat_first_frame: Whether to repeat the first frame at the end for clear looping
    """
    if output_dir is None:
        output_dir = inference_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all episode directories
    episode_dirs = sorted(glob.glob(os.path.join(inference_dir, "episode_*")))
    
    if not episode_dirs:
        print(f"No episode directories found in {inference_dir}")
        return
    
    print(f"Found {len(episode_dirs)} episode directories")
    
    for episode_dir in episode_dirs:
        episode_name = os.path.basename(episode_dir)
        output_gif = os.path.join(output_dir, f"{episode_name}.gif")
        
        print(f"\nProcessing {episode_name}...")
        create_gif_from_frames(episode_dir, output_gif, fps, duration, end_pause_ms, repeat_first_frame)
    
    print(f"\nAll GIFs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create GIFs from inference frames')
    parser.add_argument('--frames-dir', type=str, default='inference_frames',
                       help='Directory containing inference frames (default: inference_frames)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save GIFs (default: same as frames-dir)')
    parser.add_argument('--fps', type=int, default=1,
                       help='Frames per second for GIFs (default: 1)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration per frame in milliseconds (overrides fps if provided)')
    parser.add_argument('--end-pause', type=int, default=1000,
                       help='Duration to pause at the end in milliseconds (default: 1000)')
    parser.add_argument('--no-repeat-first', action='store_true',
                       help='Disable repeating the first frame at the end')
    parser.add_argument('--episode', type=str, default=None,
                       help='Specific episode to process (e.g., episode_1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.frames_dir):
        print(f"Error: Frames directory does not exist: {args.frames_dir}")
        return
    
    if args.episode:
        # Process specific episode
        episode_dir = os.path.join(args.frames_dir, args.episode)
        if not os.path.exists(episode_dir):
            print(f"Error: Episode directory does not exist: {episode_dir}")
            return
        
        output_gif = os.path.join(args.output_dir or args.frames_dir, f"{args.episode}.gif")
        create_gif_from_frames(episode_dir, output_gif, args.fps, args.duration, 
                              args.end_pause, not args.no_repeat_first)
    else:
        # Process all episodes
        create_episode_gifs(args.frames_dir, args.output_dir, args.fps, args.duration,
                           args.end_pause, not args.no_repeat_first)


if __name__ == '__main__':
    main() 