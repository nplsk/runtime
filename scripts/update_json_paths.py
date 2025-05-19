"""
Updates JSON metadata files to point to HAP-encoded video files.

This script:
1. Reads JSON metadata files from the source directory
2. Updates video file paths to point to HAP-encoded versions
3. Saves the updated JSON files to the destination directory

The script preserves all metadata while ensuring that file paths reference 
the optimized HAP video files needed for TouchDesigner playback.
"""

import os
import json
import shutil

SOURCE_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"
DEST_DIR = "/Volumes/CORSAIR/DESCRIPTIONS"
VIDEO_DIR = "/Volumes/CORSAIR/SCENES"

os.makedirs(DEST_DIR, exist_ok=True)

for filename in os.listdir(SOURCE_DIR):
    if filename.startswith(".") or not filename.endswith(".json"):
        continue

    source_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(DEST_DIR, filename)

    try:
        with open(source_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON: {filename}")
        continue

    updated = False
    for key in ["file_path", "video_path"]:
        if key in data and isinstance(data[key], str):
            original_basename = os.path.basename(data[key])
            hap_filename = os.path.splitext(original_basename)[0] + "_hap.mov"
            hap_path = os.path.join(VIDEO_DIR, hap_filename)
            if os.path.exists(hap_path):
                data[key] = hap_path
                updated = True
            else:
                print(f"⚠️ Missing HAP video for {filename}: {hap_path}")

    if updated:
        with open(dest_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Updated and copied: {filename}")
    else:
        print(f"➖ No updates made to: {filename}") 