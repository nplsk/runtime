# resolve_conflicts.py
import os
import json

OUTPUT_DIR = "/Volumes/RUNTIME/PROCESSED_DESCRIPTIONS"

# Define conflict lists
MOTION_CONFLICTS = {
    "dynamic": ["stillness", "static", "calm", "quietude"],
    "active": ["stillness", "static", "calm"],
    "still": ["movement", "chaotic_motion", "activity", "rush"],
}

for filename in os.listdir(OUTPUT_DIR):
    if not filename.endswith(".json") or filename.startswith("._"):
        continue
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        motion_tag = data.get("motion_tag", "")

        conflicts = MOTION_CONFLICTS.get(motion_tag, [])
        if conflicts:
            for key in ["formal_tags", "emotional_tags"]:
                if key in data:
                    data[key] = [tag for tag in data[key] if tag not in conflicts]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Resolved conflicts in {filename}")
    except Exception as e:
        print(f"Failed resolving {filename}: {e}")