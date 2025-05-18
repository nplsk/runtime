"""
This script converts video files to HAP format, which is optimized for real-time video playback.
It processes ProRes, MP4, and MXF files, converting them to HAP-encoded MOV files with specific
settings for optimal performance in video playback systems.

Key features:
- Parallel processing with configurable worker count
- Automatic cooldown periods to prevent system overload
- Graceful interruption handling
- Progress tracking with tqdm
- Error logging and resume capability
- Input validation and format checking
"""

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime
import signal
import sys
import argparse

# Configuration constants
INPUT_DIR = "/Volumes/RUNTIME/JOSH CLIPS/FINAL FRANKIE CLIPS"
OUTPUT_DIR = "/Volumes/CORSAIR/THREAD2-HAP/FINAL CLIPS"
LOG_FILE = "conversion_errors.log"
MAX_WORKERS = 16  # Maximum number of parallel conversion processes
COOLDOWN_INTERVAL = 100  # Number of files to process before cooldown
COOLDOWN_DURATION = 60  # Cooldown duration in seconds

# Global state for graceful shutdown
should_exit = False
futures = []

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_error(msg):
    """
    Log error messages to the error log file with timestamps.
    
    Args:
        msg: Error message to log
    """
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} | {msg}\n")

def handle_exit(signum, frame):
    """
    Handle graceful shutdown on SIGINT or SIGTERM.
    Sets global flag to stop accepting new tasks but allows current tasks to complete.
    """
    global should_exit
    print("\nGracefully stopping... finishing current tasks.")
    should_exit = True

def convert_file(filename):
    """
    Convert a single video file to HAP format using ffmpeg.
    
    Args:
        filename: Name of the input video file
        
    Returns:
        Status message indicating conversion result
    """
    if should_exit:
        return f"Skipped (interrupted): {filename}"
    
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = os.path.splitext(filename)[0] + "_hap.mov"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Skip if output file already exists (resume capability)
    if os.path.exists(output_path):
        return f"Skipped (already exists): {filename}"

    # Construct ffmpeg command with HAP-specific settings
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        # Scale and crop to ensure dimensions are multiples of 4 (HAP requirement)
        "-vf", "scale=ceil(iw/4)*4:ceil(ih/4)*4,crop=trunc(iw/4)*4:trunc(ih/4)*4",
        "-c:v", "hap",  # Use HAP codec
        "-format", "hap_q",  # Use HAP Q format for better quality
        "-video_track_timescale", "600",  # Set timescale for better frame accuracy
        "-c:a", "aac",  # Convert audio to AAC
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        return f"Converted: {filename}"
    except subprocess.CalledProcessError as e:
        log_error(f"Failed: {filename} — {str(e)}")
        return f"Error: {filename}"

def main():
    """
    Main entry point for the conversion process.
    Sets up signal handlers, processes command line arguments,
    and manages the parallel conversion of video files.
    """
    global should_exit
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert ProRes to HAP")
    parser.add_argument("--dry-run", action="store_true", help="Only process the first 100 files for testing")
    args = parser.parse_args()

    # Get list of video files to process
    all_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.mov', '.mp4', '.mxf')) and not f.startswith("._")
    ]
    if args.dry_run:
        all_files = all_files[:100]

    try:
        # Process files in parallel with progress tracking
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            with tqdm(total=len(all_files)) as pbar:
                for i, filename in enumerate(all_files):
                    if should_exit:
                        break
                    future = executor.submit(convert_file, filename)
                    futures.append(future)

                    # Implement cooldown period to prevent system overload
                    if (i + 1) % COOLDOWN_INTERVAL == 0:
                        time.sleep(COOLDOWN_DURATION)

                # Process results as they complete
                for future in as_completed(futures):
                    if should_exit:
                        break
                    print(future.result())
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\nCancelling remaining tasks...")
        for future in futures:
            future.cancel()

if __name__ == "__main__":
    main()