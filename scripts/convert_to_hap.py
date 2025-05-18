"""
Converts video files to HAP format for TouchDesigner playback.
Uses paths and settings from config.py.
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import config

def convert_to_hap(input_path, output_path):
    """Convert a single video file to HAP format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-c:v", "hap",
            "-format", "hap",
            "-quality", config.HAP_QUALITY,
            "-y",  # Overwrite output file if it exists
            str(output_path)
        ]
        
        # Run conversion
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting {input_path}:")
            print(result.stderr)
            return False
            
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def main():
    """Convert all scene videos to HAP format."""
    # Get list of scene files
    scene_files = []
    for ext in [".mov", ".mp4"]:
        scene_files.extend(list(config.SCENES_DIR.glob(f"*{ext}")))
    
    if not scene_files:
        print(f"No scene files found in {config.SCENES_DIR}")
        return

    # Convert each file
    print(f"Converting {len(scene_files)} files to HAP format...")
    for input_path in tqdm(scene_files, desc="Converting to HAP"):
        output_path = config.PLAYBACK_DIR / f"{input_path.stem}_hap.mov"
        
        # Skip if output file already exists
        if output_path.exists():
            print(f"Skipping {input_path.name} - HAP file already exists")
            continue
            
        convert_to_hap(input_path, output_path)

if __name__ == "__main__":
    main()