"""
This script identifies and regenerates poor quality video captions.
It scans through processed video metadata files and identifies captions that are:
1. Too short (less than 5 words)
2. Missing or empty
3. Using generic phrases like "no caption" or "screenshot"
4. Starting with "image of"

The script then reprocesses these videos using the process_video function
to generate new, higher quality captions.

Input:
- Directory of JSON files containing video metadata
- Each JSON file should contain a "semantic_caption" field

Output:
- Regenerated captions in the same JSON files
- Progress bar showing reprocessing status
- Error messages for any failed reprocessing attempts
"""

import os
import json
from process_videos import process_video
from tqdm import tqdm

# Directory containing processed video metadata files
INPUT_DIR = "/Volumes/RUNTIME/PROCESSED"

def is_bad_caption(caption):
    """
    Check if a caption is of poor quality.
    
    Args:
        caption: The caption text to evaluate
        
    Returns:
        bool: True if the caption is considered bad quality
    """
    # Check for empty or very short captions
    if not caption or len(caption.strip().split()) < 5:
        return True
        
    # Check for generic or problematic phrases
    lowered = caption.lower()
    return (
        lowered.startswith("no caption") or 
        "screenshot" in lowered or 
        "image of" in lowered
    )

# Find all files with bad captions
bad_files = []

# Scan through all JSON files in the input directory
for file in os.listdir(INPUT_DIR):
    # Skip non-JSON files and macOS metadata files
    if not file.endswith(".json") or file.startswith("._"):
        continue
        
    try:
        # Load and check each file's caption
        with open(os.path.join(INPUT_DIR, file), "r") as f:
            data = json.load(f)
        if is_bad_caption(data.get("semantic_caption", "")):
            bad_files.append(data["file_path"])
    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"Found {len(bad_files)} bad captions. Reprocessing...")

# Reprocess each file with a bad caption
for path in tqdm(bad_files):
    try:
        process_video(path)
    except Exception as e:
        print(f"Failed to process {path}: {e}")