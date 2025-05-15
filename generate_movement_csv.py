import os
import json
import csv
from pathlib import Path

# Directory containing processed video JSON files
INPUT_DIR = Path("/Volumes/CORSAIR/DESCRIPTIONS")
OUTPUT_DIR = Path("./movement_csvs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define the five movements
MOVEMENTS = ["orientation", "elemental", "built", "people", "blur"]

# Final headers (excluding 'theme_anchors' and 'segments')
HEADERS = [
    "video_id", "file_path", "duration", "frame_rate", "resolution",
    "motion_score", "motion_variance", "motion_tag", "mood_tag",
    "semantic_tags", "formal_tags", "emotional_tags", "material_tags",
    "dominant_colors", "semantic_caption", "ai_description"
]

# Open one CSV per movement
writers = {}
files = {}
for movement in MOVEMENTS:
    f = open(OUTPUT_DIR / f"{movement}.csv", "w", newline='', encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=HEADERS)
    writer.writeheader()
    writers[movement] = writer
    files[movement] = f

seen_files = set()

# Process each JSON file
for file in INPUT_DIR.glob("*.json"):
    if file.name.startswith("._"):
        continue
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        movement = data.get("suggested_phase", "").lower()
        if movement not in MOVEMENTS:
            continue

        if data.get("video_id") in seen_files and movement != "orientation":
            continue
        seen_files.add(data.get("video_id"))

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
            writers[movement].writerow(row)

    except Exception as e:
        print(f"⚠️ Error processing {file.name}: {e}")


import random

all_other_rows = []
for movement in MOVEMENTS:
    if movement == "orientation":
        continue
    with open(OUTPUT_DIR / f"{movement}.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_other_rows.extend(list(reader))

base_seen = set()
candidates = [r for r in all_other_rows if r["video_id"].split("_scene_")[0] not in base_seen]
random.shuffle(candidates)

from collections import defaultdict

grouped = defaultdict(list)
for r in candidates:
    base = r["video_id"].split("_scene_")[0]
    grouped[base].append(r)

selected_orientation = []
for group in grouped.values():
    medium = [r for r in group if 8 <= float(r.get("duration", 0)) <= 20]
    selection = medium if medium else group
    if selection:
        selected_orientation.append(random.choice(selection))

selected_orientation = selected_orientation[:150]
for row in selected_orientation:
    writers["orientation"].writerow(row)


# Close all file handles
for f in files.values():
    f.close()

print("✅ Movement CSVs generated successfully.")