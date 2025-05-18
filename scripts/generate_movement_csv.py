"""
This script generates CSV files for each movement phase by processing video metadata JSON files.
It creates separate CSV files for each movement (orientation, elemental, built, people, blur)
with carefully selected clips that meet specific criteria for each phase.

Key features:
- Limits the number of clips from the same source video per movement
- Ensures diversity in clip selection
- Handles special cases for the orientation phase
- Maintains metadata consistency across movements
"""

# Track root usage counts per movement to prevent overuse of clips from same source
root_usage_counts = {}
ROOT_CLIP_CAP = 2  # Maximum number of clips allowed per root video per movement

import os
import json
import csv
from pathlib import Path

# Directory containing processed video JSON files with metadata
INPUT_DIR = Path("/Volumes/CORSAIR/DESCRIPTIONS")
OUTPUT_DIR = Path("./movement_csvs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define the five main movement phases
MOVEMENTS = ["orientation", "elemental", "built", "people", "blur"]

# CSV headers for the output files, excluding internal processing fields
HEADERS = [
    "video_id", "file_path", "duration", "frame_rate", "resolution",
    "motion_score", "motion_variance", "motion_tag", "mood_tag",
    "semantic_tags", "formal_tags", "emotional_tags", "material_tags",
    "dominant_colors", "semantic_caption", "ai_description"
]

# Initialize CSV writers for each movement
writers = {}
files = {}
for movement in MOVEMENTS:
    f = open(OUTPUT_DIR / f"{movement}.csv", "w", newline='', encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=HEADERS)
    writer.writeheader()
    writers[movement] = writer
    files[movement] = f

# Track processed files to avoid duplicates
seen_files = set()

# Process each JSON file and assign to appropriate movement CSV
for file in INPUT_DIR.glob("*.json"):
    if file.name.startswith("._"):
        continue
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Get the suggested phase from the JSON metadata
        movement = data.get("suggested_phase", "").lower()
        if movement not in MOVEMENTS:
            continue

        # Skip duplicates except for orientation phase
        if data.get("video_id") in seen_files and movement != "orientation":
            continue
        seen_files.add(data.get("video_id"))

        # Prepare row data from JSON metadata
        row = {
            "video_id": data.get("video_id"),
            "file_path": data.get("file_path"),
            "duration": data.get("duration"),
            "frame_rate": data.get("frame_rate"),
            "resolution": "x".join(map(str, data.get("resolution", []))),
            "motion_score": data.get("motion_score"),
            "motion_variance": data.get("motion_variance"),
            "motion_tag": data.get("motion_tag"),
            "mood_tag": data.get("mood_tag"),
            "semantic_tags": ", ".join(data.get("semantic_tags", [])),
            "formal_tags": ", ".join(data.get("formal_tags", [])),
            "emotional_tags": ", ".join(data.get("emotional_tags", [])),
            "material_tags": ", ".join(data.get("material_tags", [])),
            "dominant_colors": ", ".join(data.get("dominant_colors", [])),
            "semantic_caption": data.get("semantic_caption", ""),
            "ai_description": data.get("ai_description", "")
        }

        if movement != "orientation":
            # Limit the number of clips from the same source video
            root_id = data.get("video_id", "").split("_scene_")[0]
            root_usage_counts.setdefault(movement, {})
            count = root_usage_counts[movement].get(root_id, 0)
            if count >= ROOT_CLIP_CAP and movement != "orientation":
                continue
            root_usage_counts[movement][root_id] = count + 1
            writers[movement].writerow(row)

    except Exception as e:
        print(f"⚠️ Error processing {file.name}: {e}")

# Special handling for orientation phase
import random

# Collect all non-orientation clips for potential orientation use
all_other_rows = []
for movement in MOVEMENTS:
    if movement == "orientation":
        continue
    with open(OUTPUT_DIR / f"{movement}.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_other_rows.extend(list(reader))

# Group clips by their source video
base_seen = set()
candidates = [r for r in all_other_rows if r["video_id"].split("_scene_")[0] not in base_seen]
random.shuffle(candidates)

from collections import defaultdict

# Group clips by their root video ID
grouped = defaultdict(list)
for r in candidates:
    base = r["video_id"].split("_scene_")[0]
    grouped[base].append(r)

# Select clips for orientation phase with specific criteria
used_video_ids = set()
selected_orientation = []
for group in grouped.values():
    # Prefer medium-length clips (8-20 seconds) for orientation
    medium = [r for r in group if 8 <= float(r.get("duration", 0)) <= 20]
    selection = medium if medium else group
    if selection:
        candidates = [r for r in selection if r["video_id"] not in used_video_ids]
        if candidates:
            pick = random.choice(candidates)
            selected_orientation.append(pick)
            used_video_ids.add(pick["video_id"])

    # Further reduce overrepresentation by limiting clips per root
    seen_roots = {}
    filtered_orientation = []
    for row in selected_orientation:
        root = row["video_id"].split("_scene_")[0]
        # Allow at most one scene per source video
        if seen_roots.get(root, 0) < 1:
            filtered_orientation.append(row)
            seen_roots[root] = seen_roots.get(root, 0) + 1
    selected_orientation = filtered_orientation

# Limit orientation phase to 150 clips
selected_orientation = selected_orientation[:150]
for row in selected_orientation:
    writers["orientation"].writerow(row)

# Close all file handles
for f in files.values():
    f.close()

print("✅ Movement CSVs generated successfully.")