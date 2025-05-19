"""
Updates JSON metadata files to add playback paths for HAP-encoded video files.

This script:
1. Reads JSON metadata files from the descriptions directory
2. Adds a playback_path field pointing to the corresponding HAP file
3. Saves the updated JSON files back to the same location
"""

import os
import json
from pathlib import Path

# Get the project root directory (parent of scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Define paths relative to project root
DESCRIPTIONS_DIR = PROJECT_ROOT / "data" / "descriptions"
PLAYBACK_DIR = PROJECT_ROOT / "data" / "playback"

print(f"Processing JSON files in: {DESCRIPTIONS_DIR}")
print(f"Looking for HAP files in: {PLAYBACK_DIR}")

if not os.path.exists(DESCRIPTIONS_DIR):
    print(f"Error: Descriptions directory does not exist: {DESCRIPTIONS_DIR}")
    exit(1)

if not os.path.exists(PLAYBACK_DIR):
    print(f"Error: Playback directory does not exist: {PLAYBACK_DIR}")
    exit(1)

for filename in os.listdir(DESCRIPTIONS_DIR):
    if filename.startswith(".") or not filename.endswith(".json"):
        continue

    file_path = os.path.join(DESCRIPTIONS_DIR, filename)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON: {filename}")
        continue

    # Get the base name without extension
    base_name = os.path.splitext(filename)[0]
    hap_filename = f"{base_name}_hap.mov"
    hap_path = os.path.join(PLAYBACK_DIR, hap_filename)

    if os.path.exists(hap_path):
        # Add the playback_path field
        data["playback_path"] = str(hap_path)
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Added playback path to: {filename}")
    else:
        print(f"⚠️ Missing HAP file for {filename}: {hap_path}") 