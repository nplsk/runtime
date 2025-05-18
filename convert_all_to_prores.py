"""
This script converts various video formats to ProRes format, which is optimized for professional video editing.
It processes videos from multiple source directories, handling different input formats and error cases.

Key features:
- Multi-directory source scanning
- Parallel processing using CPU cores
- Automatic error detection and handling
- Progress tracking with tqdm
- Color-coded console output
- Error logging and file categorization
- Support for various input formats (M4V, MP4, 3GP, MOV, MXF, AVI, DV)
- Special handling for DV format with deinterlacing
- Audio normalization and AAC encoding
"""

import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil

# List of source directories to scan for videos
INPUT_DIRS = [
    "/Users/franknapolski/Movies/_____RUNTIME",
    "/Users/franknapolski/Library/CloudStorage/Dropbox/_MEMORIES VIDEOS",
    "/Volumes/LaCie/RUNTIMEVIDEOS",
    "/Volumes/500GB-NPLSK/Video/14th Street",
    "/Volumes/500GB-NPLSK/Video/Rea/Media",
    "/Volumes/500GB-NPLSK/Video/Utah",
    "/Volumes/500GB-NPLSK/Video/West Virginia",
    "/Volumes/500GB-NPLSK/Photos/Squeakeasy 2014",
    "/Volumes/500GB-NPLSK/Frank 17 iMac",
    "/Volumes/500GB-NPLSK/_STRG01 Backup/STRG01/Untitled Project 1",
    "/Volumes/500GB-NPLSK/_STRG01 Backup/STRG01/Videos/Forest Walk",
    "/Users/franknapolski/Library/CloudStorage/Dropbox/_MEMORIES VIDEOS/8MM Conversion",
    "/Volumes/4TB-NPLSK-EXT/Video",
    "/Volumes/4TB-NPLSK-EXT/Johns\ JMM",
    "/Volumes/4TB-NPLSK-EXT/Final\ Cut\ Pro\ Documents/Capture\ Scratch",
    "/Volumes/4TB-NPLSK-EXT/Backup\ Little\ Disk",
    "/Volumes/1TB-DAD/ALL\ PHOTOS\ LIBRARY\ VIDEOS",
    "/Volumes/1TB-DAD/Silence",
    "/Volumes/1TB-DAD/Phone\ Videos",
    "/Volumes/2TB-NPLSK/Backups.backupdb",
    "/Volumes/2TB-NPLSK/iPhone\ Backup",
    "/Volumes/2TB-NPLSK/Storehouse",
    "/Volumes/4TB-DESKTOP/_GoPro",
    "/Volumes/4TB-DESKTOP/_Video Edits",
    "/Volumes/4TB-DESKTOP/_VJ/Static and Dust"
]

# Output directory for converted files
OUTPUT_DIR = "/Volumes/LaCie/CONVERTED"
ERROR_LOG = os.path.join(OUTPUT_DIR, "conversion_errors.txt")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_convertible(filename):
    """
    Check if a file should be converted based on its extension.
    Excludes files that are already in ProRes or HAP format.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        Boolean indicating if the file should be converted
    """
    ext = filename.lower()
    return (ext.endswith((".m4v", ".mp4", ".3gp", ".mov", ".mxf", ".avi", ".dv")) and
            "prores" not in ext and
            "hap" not in ext)

def log_error(message):
    """
    Log error messages to the error log file.
    
    Args:
        message: Error message to log
    """
    with open(ERROR_LOG, "a") as f:
        f.write(message + "\n")

def move_to_skipped(input_path, reason):
    """
    Copy a file to a skipped/corrupted directory with a specific reason.
    Preserves the original file by copying instead of moving.
    
    Args:
        input_path: Path to the input file
        reason: Reason for skipping (e.g., "MOOV_MISSING", "BITSTREAM_ERROR")
    """
    skip_dir = os.path.join(OUTPUT_DIR, "_SKIPPED_OR_CORRUPTED", reason)
    os.makedirs(skip_dir, exist_ok=True)
    basename = os.path.basename(input_path)
    dest_path = os.path.join(skip_dir, basename)
    try:
        shutil.copy2(input_path, dest_path)  # Copy instead of move to preserve original
    except Exception as e:
        log_error(f"[COPY ERROR] Failed to copy {input_path} to {dest_path}: {e}")

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
        "ffmpeg", "-hide_banner", "-i", input_path,
        "-c:v", "prores_ks", "-profile:v", "3",  # ProRes HQ profile
        "-video_track_timescale", "600",  # Standard timescale for ProRes
        "-af", "loudnorm",  # Normalize audio levels
        "-c:a", "aac",  # Convert audio to AAC
        output_path
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
    Scans input directories, creates conversion tasks,
    and processes them in parallel using CPU cores.
    """
    tasks = []
    for input_dir in INPUT_DIRS:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if is_convertible(filename):
                    input_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(input_path, input_dir)
                    output_filename = os.path.splitext(rel_path)[0] + "_prores.mov"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    tasks.append((input_path, output_path))

    if tasks:
        print(f"Found {len(tasks)} files to process. Starting conversion using {cpu_count()} cores...")
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(convert_to_prores_task, tasks), total=len(tasks)):
                pass
        print(color(f"✅ Completed conversion of {len(tasks)} files.", "green"))
    else:
        print("No convertible files found.")

if __name__ == "__main__":
    main()
