"""
Converts source videos to ProRes format for high-quality processing.
Uses paths and settings from config.py.

Key features:
- Parallel processing using CPU cores
- Automatic error detection and handling
- Progress tracking with tqdm
- Color-coded console output
- Error logging and file categorization
- Support for various input formats
- Audio normalization and AAC encoding
"""

import os
import sys
from pathlib import Path
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import config
from config import (
    DATA_DIR,
    SOURCE_DIR,
    PRORES_DIR,
    VIDEO_EXTENSIONS
)

def color(text, c):
    """
    Add ANSI color codes to text for console output.
    
    Args:
        text: Text to colorize
        c: Color name ("green", "yellow", "red", "cyan")
        
    Returns:
        Colorized text string
    """
    colors = {"green":32, "yellow":33, "red":31, "cyan":36}
    return f"\033[{colors.get(c,37)}m{text}\033[0m"

def log_error(message):
    """
    Log error messages to the error log file.
    
    Args:
        message: Error message to log
    """
    error_log = DATA_DIR / "conversion_errors.txt"
    with open(error_log, "a") as f:
        f.write(message + "\n")

def move_to_skipped(input_path, reason):
    """
    Copy a file to a skipped/corrupted directory with a specific reason.
    Preserves the original file by copying instead of moving.
    
    Args:
        input_path: Path to the input file
        reason: Reason for skipping (e.g., "MOOV_MISSING", "BITSTREAM_ERROR")
    """
    skip_dir = DATA_DIR / "_SKIPPED_OR_CORRUPTED" / reason
    os.makedirs(skip_dir, exist_ok=True)
    dest_path = skip_dir / Path(input_path).name
    try:
        shutil.copy2(input_path, dest_path)  # Copy instead of move to preserve original
    except Exception as e:
        log_error(f"[COPY ERROR] Failed to copy {input_path} to {dest_path}: {e}")

def convert_to_prores_task(args):
    """
    Convert a single video file to ProRes format.
    Handles various error cases and retries with different settings.
    
    Args:
        args: Tuple of (input_path, output_path)
    """
    input_path, output_path = args
    ext = os.path.splitext(input_path)[1].lower()

    # Skip if already converted
    if os.path.exists(output_path):
        print(color(f"Skipping already converted: {output_path}", "cyan"))
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Base ffmpeg command with ProRes settings
    base_command = [
        "ffmpeg", "-hide_banner", "-i", str(input_path),
        "-c:v", "prores_ks", "-profile:v", "3",  # ProRes HQ profile
        "-video_track_timescale", "600",  # Standard timescale for ProRes
        "-af", "loudnorm",  # Normalize audio levels
        "-c:a", "aac",  # Convert audio to AAC
        str(output_path)
    ]

    # Add deinterlacing for DV format
    if ext == ".dv":
        base_command.insert(4, "-vf")
        base_command.insert(5, "yadif=mode=send_frame,scale=720:480")

    print(color(f"Converting {input_path} → {output_path}", "green"))

    # Retry logic with error detection
    MAX_RETRIES = 2
    for attempt in range(1, MAX_RETRIES + 1):
        command = base_command.copy()
        if attempt == 2:
            command.insert(3, "-err_detect")
            command.insert(4, "ignore_err")

        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        if result.returncode == 0:
            return

        # Handle specific error cases
        stderr_text = result.stderr.lower()
        if "moov atom not found" in stderr_text:
            log_error(f"[MOOV MISSING] {input_path}")
            move_to_skipped(input_path, "MOOV_MISSING")
            print(color(f"❌ MOOV atom missing: {input_path}", "red"))
            return
        elif "concealing bitstream errors" in stderr_text or "decoding error" in stderr_text:
            log_error(f"[BITSTREAM ERROR] {input_path}")
            move_to_skipped(input_path, "BITSTREAM_ERROR")
            print(color(f"⚠️ Bitstream warnings: {input_path}", "yellow"))
            return
        else:
            print(color(f"Attempt {attempt} failed for {input_path}", "yellow"))
            if attempt == MAX_RETRIES:
                log_error(f"[FAILED FINAL] {input_path}")
                move_to_skipped(input_path, "FAILED_FINAL")
                print(color(f"❌ Failed permanently: {input_path}", "red"))

def main():
    """
    Main entry point for the conversion process.
    Scans source directory, creates conversion tasks,
    and processes them in parallel using CPU cores.
    """
    # Get list of source files (case-insensitive)
    source_files = []
    for ext in VIDEO_EXTENSIONS:
        # Add both lowercase and uppercase extensions
        source_files.extend(list(SOURCE_DIR.glob(f"*{ext.lower()}")))
        source_files.extend(list(SOURCE_DIR.glob(f"*{ext.upper()}")))
    
    if not source_files:
        print(f"No source video files found in {SOURCE_DIR}")
        return

    # Create conversion tasks
    tasks = []
    for input_path in source_files:
        # Skip .gitkeep and other hidden files
        if input_path.name.startswith('.'):
            continue
        output_path = PRORES_DIR / f"{input_path.stem}_prores.mov"
        tasks.append((input_path, output_path))

    # Process files in parallel
    print(f"Found {len(tasks)} files to process. Starting conversion using {cpu_count()} cores...")
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(convert_to_prores_task, tasks), total=len(tasks)):
            pass
    
    # Print summary
    print("\nConversion Summary:")
    print(f"Total files found: {len(source_files)}")
    print(f"Files processed: {len(tasks)}")
    print(f"Files skipped: {len(source_files) - len(tasks)}")
    
    # Check for error log
    error_log = DATA_DIR / "conversion_errors.txt"
    if error_log.exists():
        print(f"\nErrors were logged to: {error_log}")
    
    # Check for skipped files
    skip_dir = DATA_DIR / "_SKIPPED_OR_CORRUPTED"
    if skip_dir.exists():
        for reason_dir in skip_dir.iterdir():
            if reason_dir.is_dir():
                skipped_files = list(reason_dir.glob("*"))
                if skipped_files:
                    print(f"\nFiles skipped due to {reason_dir.name}:")
                    for file in skipped_files:
                        print(f"  - {file.name}")

    # Print next steps
    print("\nNext Steps:")
    print("1. Review any errors or skipped files above")
    print("2. Once all videos are successfully converted to ProRes, run:")
    print("   python split_scenes.py")
    print("   python scripts/process_videos.py")
    print("\nNote: Make sure to address any conversion errors before proceeding to the next step.")

if __name__ == "__main__":
    main() 