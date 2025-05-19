"""
This script resolves semantic conflicts in video metadata tags.
It uses both motion tags and motion scores to determine which tags should be removed.

For example:
- If a video has a high motion_score (>50), it should not have tags like "stillness" or "static"
- If a video has a low motion_score (<20), it should not have tags like "movement" or "chaotic_motion"
- The script uses predefined thresholds and conflict rules to clean up these inconsistencies.

Input:
- Directory of JSON files containing video metadata
- Each file should contain a "motion_tag", "motion_score" in metadata_payload, and tag lists

Output:
- Updated JSON files with conflicting tags removed
- Console output indicating which files were processed or if any errors occurred
"""

import os
import json
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import config

# Motion score thresholds
HIGH_MOTION_THRESHOLD = 50
LOW_MOTION_THRESHOLD = 20

# Define conflict lists for each motion category
MOTION_CONFLICTS = {
    "high_motion": [
        "stillness", "static", "calm", "quietude", "peaceful", 
        "serene", "tranquil", "stationary"
    ],
    "low_motion": [
        "movement", "chaotic_motion", "activity", "rush",
        "explosive", "burst", "sudden", "rapid",
        "dynamic", "active", "flowing"
    ],
}

def get_motion_category(motion_score):
    """Determine motion category based on motion score."""
    if motion_score >= HIGH_MOTION_THRESHOLD:
        return "high_motion"
    elif motion_score <= LOW_MOTION_THRESHOLD:
        return "low_motion"
    return None

# Process each JSON file in the directory
for filename in os.listdir(config.OUTPUT_DIR):
    # Skip non-JSON files and macOS metadata files
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(config.OUTPUT_DIR, filename)
    try:
        # Load the video metadata
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Get the motion score from metadata_payload and determine category
        motion_score = data.get("metadata_payload", {}).get("motion_score", 0)
        motion_category = get_motion_category(motion_score)
        tags_removed = False
        removal_log = []

        if motion_category:
            # Get the list of tags that conflict with this motion category
            conflicts = MOTION_CONFLICTS.get(motion_category, [])
            if conflicts:
                # Remove conflicting tags from formal and emotional tags
                for key in ["formal_tags", "emotional_tags"]:
                    if key in data.get("metadata_payload", {}):
                        tags = data["metadata_payload"][key]
                        to_remove = [tag for tag in tags if tag in conflicts]
                        if to_remove:
                            removal_log.append((key, to_remove))
                if removal_log:
                    print(f"\n{filename} (motion_score: {motion_score}) has conflicting tags:")
                    for key, tags in removal_log:
                        print(f"  {key}: {tags}")
                    confirm = input("Remove these tags? [y/N]: ").strip().lower()
                    if confirm == "y":
                        for key, tags in removal_log:
                            data["metadata_payload"][key] = [tag for tag in data["metadata_payload"][key] if tag not in tags]
                        tags_removed = True
                        print(f"  Tags removed from {filename}.")
                    else:
                        print(f"  No changes made to {filename}.")
        if tags_removed:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        print(f"Processed {filename} (motion_score: {motion_score})")
    except Exception as e:
        print(f"Failed processing {filename}: {e}")