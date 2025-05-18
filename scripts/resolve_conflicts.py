"""
This script resolves semantic conflicts in video metadata tags.
It scans through processed video metadata files and removes tags that conflict with the assigned motion category.

For example:
- If a video is tagged as "dynamic", it should not also have tags like "stillness" or "static" in its formal or emotional tags.
- The script uses a predefined set of motion conflict rules to clean up these inconsistencies.

Input:
- Directory of JSON files containing video metadata
- Each file should contain a "motion_tag" and tag lists ("formal_tags", "emotional_tags")

Output:
- Updated JSON files with conflicting tags removed
- Console output indicating which files were processed or if any errors occurred
"""

import os
import json

# Directory containing processed video metadata files
OUTPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"

# Define conflict lists for each motion category
MOTION_CONFLICTS = {
    "dynamic": ["stillness", "static", "calm", "quietude"],
    "active": ["stillness", "static", "calm"],
    "still": ["movement", "chaotic_motion", "activity", "rush"],
}

# Process each JSON file in the directory
for filename in os.listdir(OUTPUT_DIR):
    # Skip non-JSON files and macOS metadata files
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        # Load the video metadata
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Get the motion tag for this video
        motion_tag = data.get("motion_tag", "")

        # Get the list of tags that conflict with this motion category
        conflicts = MOTION_CONFLICTS.get(motion_tag, [])
        if conflicts:
            # Remove conflicting tags from formal and emotional tags
            for key in ["formal_tags", "emotional_tags"]:
                if key in data:
                    data[key] = [tag for tag in data[key] if tag not in conflicts]

        # Save the updated metadata
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Resolved conflicts in {filename}")
    except Exception as e:
        print(f"Failed resolving {filename}: {e}")