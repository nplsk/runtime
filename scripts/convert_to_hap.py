"""
Converts video files to HAP format for TouchDesigner playback.
Uses paths and settings from config.py.
"""

import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import config

def convert_to_hap(input_path, output_path):
    """Convert a single video file to HAP format."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        print(f"\nProcessing file: {input_path}")
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", "scale=trunc(iw/4)*4:trunc(ih/4)*4", # Ensure dimensions are multiples of 4
            "-c:v", "hap",
            "-format", "hap",
            "-quality", config.HAP_QUALITY,
            "-y",  # Overwrite output file if it exists
            str(output_path)
        ]
        
        # Run conversion
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting {input_path}:")
            print(result.stderr)
            return False
            
        # Verify the output file has video
        if output_path.exists():
            verify_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
                         "stream=codec_type", "-of", "csv=p=0", str(output_path)]
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            
            if verify_result.stdout.strip() != "video":
                print(f"WARNING: Output file {output_path} doesn't appear to have a valid video stream!")
                return False
            else:
                print(f"✓ Successfully created HAP file with valid video stream: {output_path}")
                return True
        else:
            print(f"ERROR: Output file {output_path} was not created!")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def main():
    """Convert all scene videos to HAP format."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert video files to HAP format")
    parser.add_argument("--force", action="store_true", help="Force re-encoding of existing files")
    args = parser.parse_args()
    
    # Get list of scene files
    scene_files = []
    for ext in [".mov", ".mp4"]:
        scene_files.extend(list(config.SCENES_DIR.glob(f"*{ext}")))
    
    if not scene_files:
        print(f"No scene files found in {config.SCENES_DIR}")
        return

    # Convert each file
    print(f"Converting {len(scene_files)} files to HAP format...")
    
    success_count = 0
    failed_files = []
    
    for input_path in tqdm(scene_files, desc="Converting to HAP"):
        output_path = config.PLAYBACK_DIR / f"{input_path.stem}_hap.mov"
        
        # Skip if output file already exists and not forcing
        if output_path.exists() and not args.force:
            print(f"Skipping {input_path.name} - HAP file already exists (use --force to re-encode)")
            continue
        
        success = convert_to_hap(input_path, output_path)
        if success:
            success_count += 1
        else:
            failed_files.append(input_path.name)
    
    print(f"\nConversion summary:")
    print(f"  - Successfully converted: {success_count}/{len(scene_files)}")
    
    if failed_files:
        print(f"  - Failed files ({len(failed_files)}):")
        for fname in failed_files:
            print(f"    - {fname}")

if __name__ == "__main__":
    main()